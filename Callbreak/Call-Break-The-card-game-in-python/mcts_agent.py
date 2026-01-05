import gymnasium as gym
import numpy as np
import math
import copy
import random
import torch
from sb3_contrib import MaskablePPO
from callbreak_env import CallbreakEnv

# --- CONFIG ---
PPO_MODEL_PATH = "ppo_callbreak_zero_pro.zip"
#PPO_MODEL_PATH = "ppo_callbreak_zero.zip"
#PPO_MODEL_PATH = "ppo_callbreak_zero_pro_x.zip"
#PPO_MODEL_PATH = "ppo_zero_epoch_784.zip"


SIMULATIONS = 50   # MCTS iterations per move (Higher = Smarter but Slower)
C_PUCT = 1.0       # Exploration constant

class Node:
    def __init__(self, parent=None, prior=1.0):
        self.parent = parent
        self.children = {}  # Map: action_idx -> Node
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior  # Probability from PPO Policy
        
    def value(self):
        if self.visit_count == 0: return 0
        return self.value_sum / self.visit_count

    def ucb_score(self, parent_visits):
        # PUCT Formula: Q(s,a) + U(s,a)
        # U(s,a) = C * P(s,a) * sqrt(parent_visits) / (1 + visit_count)
        u = C_PUCT * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.value() + u

class ISMCTSAgent:
    def __init__(self):
        print(f"Loading PPO Policy from {PPO_MODEL_PATH}...")
        try:
            self.model = MaskablePPO.load(PPO_MODEL_PATH, device="cpu")
        except:
            print("Error: Could not load PPO model. Ensure training is done.")
            self.model = None

    def determinize(self, known_hand, history_set, leader_idx, current_trick, tricks_won):
        """
        Creates a 'Hypothetical World' (GameState) where unknown cards are 
        randomly dealt to opponents.
        """
        # 1. Identify all unknown cards
        all_cards = set(range(52))
        
        # Known cards = My Hand + History + Current Trick
        known_cards = set()
        
        # Add my hand
        for c in known_hand:
            idx = (c.rank - 2) * 4 + c.suit
            known_cards.add(idx)
            
        # Add history (cards played in previous tricks)
        for idx in history_set:
            known_cards.add(idx)
            
        # Add current trick cards
        for c in current_trick:
            idx = (c.rank - 2) * 4 + c.suit
            known_cards.add(idx)

        unknown_cards = list(all_cards - known_cards)
        random.shuffle(unknown_cards)

        # 2. Reconstruct Environment State
        # We create a temporary CallbreakEnv to run simulations
        sim_env = CallbreakEnv()
        sim_env.reset() # This shuffles deck, we need to override it manually
        
        # Override Hands
        # Player 0 gets the 'known_hand'
        sim_env.hands[0] = [c for c in known_hand] # Deep copy
        
        # Distribute unknown cards to Players 1, 2, 3
        # Logic: We need to respect hand sizes based on history?
        # Simplification: Distribute remaining cards evenly.
        # In a real game, opponents might have uneven cards during a trick, 
        # but at start of a trick, everyone has same count.
        
        # Count how many cards each opponent *should* have
        # Total played so far = len(history) + len(current_trick)
        # Cards per player = (52 - len(history) - len(current_trick)) / 4 ?? 
        # Actually easier: Just fill hands until empty.
        
        chunk_size = len(unknown_cards) // 3
        sim_env.hands[1] = self._indices_to_cards(unknown_cards[0:chunk_size])
        sim_env.hands[2] = self._indices_to_cards(unknown_cards[chunk_size:2*chunk_size])
        sim_env.hands[3] = self._indices_to_cards(unknown_cards[2*chunk_size:])
        
        # Restore Game Context
        sim_env.cards_played_history = copy.deepcopy(history_set)
        sim_env.trick_cards = [c for c in current_trick] # Deep copy
        sim_env.tricks_won = list(tricks_won)
        sim_env.leader_idx = leader_idx
        
        # Set current player (It's my turn, so Player 0)
        sim_env.current_player = 0
        
        return sim_env

    def _indices_to_cards(self, indices):
        cards = []
        for idx in indices:
            r = (idx // 4) + 2
            s = idx % 4
            # We need to instantiate the Card class used by Env
            # Quick hack: Use the fallback class or import
            cards.append(self._create_card(r, s))
        return cards

    def _create_card(self, r, s):
        # Helper to match Env's Card class
        from callbreak_env import Card 
        return Card(r, s)

    def select_move(self, real_env):
        """
        The main entry point. 
        real_env: The actual game environment at the start of your turn.
        """
        root = Node()
        
        # Extract observable info from real_env
        my_hand = real_env.hands[0]
        history = real_env.cards_played_history
        current_trick = real_env.trick_cards
        leader = real_env.leader_idx
        tricks_won = real_env.tricks_won
        
        for _ in range(SIMULATIONS):
            node = root
            
            # 1. Determinize (Sample a world)
            sim_env = self.determinize(my_hand, history, leader, current_trick, tricks_won)
            
            # 2. Select (Traverse Tree)
            path = [] # Store path to backpropagate
            
            # While we are in known nodes and not terminal
            terminated = False
            while node.children and not terminated:
                # Pick best child
                action_idx, node = max(
                    node.children.items(),
                    key=lambda item: item[1].ucb_score(node.visit_count)
                )
                
                # Apply move in sim
                obs, reward, terminated, _, info = sim_env.step(action_idx)
                path.append(node)
                
                # If step caused termination or error, break
                if terminated or 'error' in info: break

            # 3. Expand & Evaluate (Leaf Node)
            if not terminated:
                # Use PPO to get Policy (Priors) and Value
                obs = sim_env._get_obs()
                mask = sim_env._get_valid_mask()
                
                # Forward Pass through PPO
                # We need raw logits/values. 
                # Helper: use model.policy.evaluate_actions or just predict
                
                # SB3 Trick: Get distribution
                with torch.no_grad():
                    obs_tensor = torch.as_tensor(obs).unsqueeze(0)
                    # Get features
                    features = self.model.policy.extract_features(obs_tensor)
                    latent_pi = self.model.policy.mlp_extractor.forward_actor(features)
                    latent_vf = self.model.policy.mlp_extractor.forward_critic(features)
                    
                    # Get Value
                    value_est = self.model.policy.value_net(latent_vf).item()
                    
                    # Get Policy (Logits)
                    logits = self.model.policy.action_net(latent_pi)
                    
                    # Apply Mask
                    # Set invalid logits to -infinity
                    mask_tensor = torch.as_tensor(mask).unsqueeze(0)
                    HUGE_NEG = -1e8
                    logits[mask_tensor == 0] = HUGE_NEG
                    
                    probs = torch.softmax(logits, dim=1).numpy()[0]

                # Expand children
                valid_indices = [i for i, m in enumerate(mask) if m == 1]
                for idx in valid_indices:
                    node.children[idx] = Node(parent=node, prior=probs[idx])
                
                # Rollout Value (Simulation to end) - OPTIONAL
                # Since we have a Value Network, we use value_est directly!
                # This is AlphaZero style (No random rollout needed)
                leaf_value = value_est 
                
            else:
                # Terminal State: Calculate actual reward
                # Reward is usually 0 or 1 in Step. 
                # We need game Score?
                # Simplified: Did we win this trick? Or total tricks?
                # Let's trust the Value Net for intermediate states.
                # For terminal, use the result.
                leaf_value = 1.0 if sim_env.tricks_won[0] > tricks_won[0] else 0.0

            # 4. Backpropagate
            # Update path nodes
            # Note: In 4-player, value perspective flips?
            # Simpler: We are Player 0. Value Net predicts Player 0's win prob.
            # So we add leaf_value directly.
            node.visit_count += 1
            node.value_sum += leaf_value
            
            for p_node in reversed(path):
                p_node.visit_count += 1
                p_node.value_sum += leaf_value
                
        # 5. Return Best Move
        # Robust selection: Most visited child
        if not root.children:
            # Fallback if no expansion happened (e.g. strict terminal)
            return self._random_legal(real_env)
            
        best_idx, _ = max(root.children.items(), key=lambda item: item[1].visit_count)
        return best_idx

    def _random_legal(self, env):
        mask = env._get_valid_mask()
        indices = [i for i, m in enumerate(mask) if m == 1]
        return random.choice(indices) if indices else 0

# --- TEST ---
if __name__ == "__main__":
    # Setup a dummy game to test MCTS
    env = CallbreakEnv()
    env.reset()
    
    agent = ISMCTSAgent()
    
    print("Thinking...")
    action = agent.select_move(env)
    
    # Print Card
    r = (action // 4) + 2
    s = action % 4
    print(f"MCTS Selected Action Index: {action} (Rank {r}, Suit {s})")