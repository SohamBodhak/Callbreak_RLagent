import gymnasium as gym
import numpy as np
import math
import copy
import random
import torch
from stable_baselines3 import A2C  # Changed from MaskablePPO
from callbreak_env import CallbreakEnv

# --- CONFIG ---
A2C_MODEL_PATH = "a2c_callbreak_final.zip"
SIMULATIONS = 50   # Adjust based on speed requirements
C_PUCT = 1.0       

class Node:
    def __init__(self, parent=None, prior=1.0):
        self.parent = parent
        self.children = {}  
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior  
        
    def value(self):
        if self.visit_count == 0: return 0
        return self.value_sum / self.visit_count

    def ucb_score(self, parent_visits):
        u = C_PUCT * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.value() + u

class ISMCTSAgentA2C:
    def __init__(self):
        print(f"Loading MCTS (A2C) Brain...")
        try:
            # Load the A2C model
            self.model = A2C.load(A2C_MODEL_PATH, device="cpu")
            print("Loaded A2C Model.")
        except Exception as e:
            print(f"Error: Could not load A2C model ({e}). MCTS will fail.")
            self.model = None

    def determinize(self, real_hand, history_set, leader_idx, current_trick, tricks_won, player_perspective):
        """
        Creates a 'Hypothetical World' moved to Player 0 perspective.
        """
        all_cards = set(range(52))
        
        known_cards = set()
        for c in real_hand:
            idx = (c.rank - 2) * 4 + c.suit
            known_cards.add(idx)
        for idx in history_set:
            known_cards.add(idx)
        for c in current_trick:
            idx = (c.rank - 2) * 4 + c.suit
            known_cards.add(idx)

        unknown_cards = list(all_cards - known_cards)
        random.shuffle(unknown_cards)

        sim_env = CallbreakEnv()
        sim_env.reset()
        
        # Deep copy and Rotate Perspective
        sim_env.hands[0] = [c for c in real_hand]
        
        chunk_size = len(unknown_cards) // 3
        sim_env.hands[1] = self._indices_to_cards(unknown_cards[0:chunk_size])
        sim_env.hands[2] = self._indices_to_cards(unknown_cards[chunk_size:2*chunk_size])
        sim_env.hands[3] = self._indices_to_cards(unknown_cards[2*chunk_size:])
        
        sim_env.cards_played_history = copy.deepcopy(history_set)
        sim_env.trick_cards = [c for c in current_trick]
        
        relative_leader = (leader_idx - player_perspective) % 4
        sim_env.leader_idx = relative_leader
        
        sim_env.tricks_won = [0]*4
        for i in range(4):
            rel_idx = (i - player_perspective) % 4
            sim_env.tricks_won[rel_idx] = tricks_won[i]

        sim_env.current_player = 0
        return sim_env

    def _indices_to_cards(self, indices):
        from callbreak_env import Card
        cards = []
        for idx in indices:
            r = (idx // 4) + 2
            s = idx % 4
            cards.append(Card(r, s))
        return cards

    def select_move(self, real_env):
        root = Node()
        
        current_player = real_env.current_player
        my_hand = real_env.hands[current_player]
        history = real_env.cards_played_history
        current_trick = real_env.trick_cards
        leader = real_env.leader_idx
        tricks_won = real_env.tricks_won
        
        for _ in range(SIMULATIONS):
            node = root
            
            sim_env = self.determinize(my_hand, history, leader, current_trick, tricks_won, current_player)
            
            # Selection
            path = []
            terminated = False
            while node.children and not terminated:
                action_idx, node = max(
                    node.children.items(),
                    key=lambda item: item[1].ucb_score(node.visit_count)
                )
                _, _, terminated, _, info = sim_env.step(action_idx)
                path.append(node)
                if terminated or 'error' in info: break

            # Expansion & Evaluation
            if not terminated:
                obs = sim_env._get_obs()
                mask = sim_env._get_valid_mask()
                
                # --- A2C INFERENCE ---
                with torch.no_grad():
                    obs_tensor = torch.as_tensor(obs).unsqueeze(0)
                    
                    # Extract Features (Same structure for A2C and PPO in SB3)
                    features = self.model.policy.extract_features(obs_tensor)
                    latent_pi = self.model.policy.mlp_extractor.forward_actor(features)
                    latent_vf = self.model.policy.mlp_extractor.forward_critic(features)
                    
                    # Get Value Estimate
                    value_est = self.model.policy.value_net(latent_vf).item()
                    
                    # Get Policy Logits
                    logits = self.model.policy.action_net(latent_pi)
                    
                    # Manual Masking (Crucial for A2C since it wasn't trained with MaskablePPO)
                    HUGE_NEG = -1e8
                    mask_tensor = torch.as_tensor(mask).unsqueeze(0)
                    logits[mask_tensor == 0] = HUGE_NEG
                    probs = torch.softmax(logits, dim=1).numpy()[0]

                valid_indices = [i for i, m in enumerate(mask) if m == 1]
                for idx in valid_indices:
                    node.children[idx] = Node(parent=node, prior=probs[idx])
                
                leaf_value = value_est 
            else:
                leaf_value = 1.0 if sim_env.tricks_won[0] > sim_env.tricks_won[1] else 0.0

            # Backpropagation
            node.visit_count += 1
            node.value_sum += leaf_value
            for p_node in reversed(path):
                p_node.visit_count += 1
                p_node.value_sum += leaf_value
                
        if not root.children:
            return self._random_legal(real_env, current_player)
            
        best_idx, _ = max(root.children.items(), key=lambda item: item[1].visit_count)
        return best_idx

    def _random_legal(self, env, player_idx):
        hand = env.hands[player_idx]
        trick = env.trick_cards
        lead_suit = trick[0].suit if trick else None
        valid = hand
        if lead_suit:
            s = [c for c in hand if c.suit == lead_suit]
            if s: valid = s
            else:
                sp = [c for c in hand if c.suit == 3]
                if sp: valid = sp
        
        if not valid: valid = hand
        if not valid: return 0
        c = random.choice(valid)
        return (c.rank - 2) * 4 + c.suit

if __name__ == "__main__":
    agent = ISMCTSAgentA2C()
    print("MCTS with A2C Ready.")