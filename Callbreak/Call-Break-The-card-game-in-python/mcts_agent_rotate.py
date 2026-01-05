import gymnasium as gym
import numpy as np
import math
import copy
import random
import torch
from sb3_contrib import MaskablePPO
from callbreak_env import CallbreakEnv

# --- CONFIG ---
PPO_MODEL_PATH = "ppo_callbreak_pro_without_self_play.zip"
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
        # PUCT Formula
        u = C_PUCT * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.value() + u

class ISMCTSAgent:
    def __init__(self):
        print(f"Loading MCTS Brain...")
        try:
            # Try loading the latest model first, fallback to pro
            if self._file_exists("ppo_callbreak_zero.zip"):
                self.model = MaskablePPO.load("ppo_callbreak_zero.zip", device="cpu")
                print("Loaded AlphaZero Model.")
            else:
                self.model = MaskablePPO.load("ppo_callbreak_pro.zip", device="cpu")
                print("Loaded PPO Pro Model.")
        except:
            print("Error: Could not load PPO model.")
            self.model = None

    def _file_exists(self, path):
        import os
        return os.path.exists(path)

    def determinize(self, real_hand, history_set, leader_idx, current_trick, tricks_won, player_perspective):
        """
        Creates a 'Hypothetical World' where the 'player_perspective' is moved to Seat 0.
        All unknown cards are shuffled and distributed to opponents (Seat 1, 2, 3).
        """
        # 1. Identify all unknown cards
        all_cards = set(range(52))
        
        # Known cards = My Hand + History + Current Trick
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

        # 2. Setup Sim Environment
        sim_env = CallbreakEnv()
        sim_env.reset()
        
        # --- CRITICAL FIX: ROTATION ---
        # The AI (PPO) expects to be Player 0.
        # We give the real_hand to Player 0 in the sim.
        sim_env.hands[0] = [c for c in real_hand] # Deep copy
        
        # Distribute unknown cards to others (1, 2, 3)
        chunk_size = len(unknown_cards) // 3
        # Handle edge case where division isn't perfect (game end)
        p1_cards = unknown_cards[0:chunk_size]
        p2_cards = unknown_cards[chunk_size:2*chunk_size]
        p3_cards = unknown_cards[2*chunk_size:]
        
        sim_env.hands[1] = self._indices_to_cards(p1_cards)
        sim_env.hands[2] = self._indices_to_cards(p2_cards)
        sim_env.hands[3] = self._indices_to_cards(p3_cards)
        
        # Restore Context (Rotated)
        sim_env.cards_played_history = copy.deepcopy(history_set)
        sim_env.trick_cards = [c for c in current_trick] # This needs logical rotation? 
        # Actually, trick_cards are absolute objects. The environment logic handles order.
        # BUT, we need to know who played them relative to us.
        
        # Rotate Leader: If I am Player 'p', and leader was 'L', 
        # In my relative world (where I am 0), the relative leader is (L - p) % 4
        relative_leader = (leader_idx - player_perspective) % 4
        sim_env.leader_idx = relative_leader
        
        # Rotate Tricks Won
        # sim_env.tricks_won[0] should be MY tricks
        sim_env.tricks_won = [0]*4
        for i in range(4):
            rel_idx = (i - player_perspective) % 4
            sim_env.tricks_won[rel_idx] = tricks_won[i]

        # It is "my" turn in the sim, so current_player = 0
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
        """
        Main entry point. Automatically handles perspective rotation.
        """
        root = Node()
        
        # 1. Capture State from Real Env
        current_player = real_env.current_player
        my_hand = real_env.hands[current_player] # FIX: Read correct hand
        history = real_env.cards_played_history
        current_trick = real_env.trick_cards
        leader = real_env.leader_idx
        tricks_won = real_env.tricks_won
        
        for _ in range(SIMULATIONS):
            node = root
            
            # Determinize with Rotation
            sim_env = self.determinize(my_hand, history, leader, current_trick, tricks_won, current_player)
            
            # Select
            path = []
            terminated = False
            while node.children and not terminated:
                action_idx, node = max(
                    node.children.items(),
                    key=lambda item: item[1].ucb_score(node.visit_count)
                )
                obs, reward, terminated, _, info = sim_env.step(action_idx)
                path.append(node)
                if terminated or 'error' in info: break

            # Expand & Evaluate
            if not terminated:
                # PPO Inference (sim_env is already rotated to P0 perspective)
                obs = sim_env._get_obs()
                mask = sim_env._get_valid_mask()
                
                # Get PPO prediction
                with torch.no_grad():
                    obs_tensor = torch.as_tensor(obs).unsqueeze(0)
                    features = self.model.policy.extract_features(obs_tensor)
                    latent_pi = self.model.policy.mlp_extractor.forward_actor(features)
                    latent_vf = self.model.policy.mlp_extractor.forward_critic(features)
                    
                    value_est = self.model.policy.value_net(latent_vf).item()
                    logits = self.model.policy.action_net(latent_pi)
                    
                    # Masking
                    HUGE_NEG = -1e8
                    mask_tensor = torch.as_tensor(mask).unsqueeze(0)
                    logits[mask_tensor == 0] = HUGE_NEG
                    probs = torch.softmax(logits, dim=1).numpy()[0]

                # Expand
                valid_indices = [i for i, m in enumerate(mask) if m == 1]
                for idx in valid_indices:
                    node.children[idx] = Node(parent=node, prior=probs[idx])
                
                leaf_value = value_est 
            else:
                # Terminal
                leaf_value = 1.0 if sim_env.tricks_won[0] > sim_env.tricks_won[1] else 0.0 # Heuristic win check

            # Backprop
            node.visit_count += 1
            node.value_sum += leaf_value
            for p_node in reversed(path):
                p_node.visit_count += 1
                p_node.value_sum += leaf_value
                
        # Return Best Move
        if not root.children:
            return self._random_legal(real_env, current_player)
            
        best_idx, _ = max(root.children.items(), key=lambda item: item[1].visit_count)
        return best_idx

    def _random_legal(self, env, player_idx):
        # Fallback helper
        # We need a mask for the SPECIFIC player, not just P0
        # But env._get_valid_mask() is hardcoded for hands[0]
        # So we manually calc valid moves
        hand = env.hands[player_idx]
        trick = env.trick_cards
        
        # Re-implement mask logic quickly for fallback
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

# --- TEST ---
if __name__ == "__main__":
    env = CallbreakEnv()
    env.reset()
    agent = ISMCTSAgent()
    print("MCTS initialized.")