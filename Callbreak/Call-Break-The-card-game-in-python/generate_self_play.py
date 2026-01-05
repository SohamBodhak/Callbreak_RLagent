import numpy as np
import time
import os
from callbreak_env import CallbreakEnv
from mcts_agent import ISMCTSAgent, Node

# --- CONFIGURATION ---
NUM_GAMES = 1000          # How many self-play games to generate
SIMULATIONS = 500         # MCTS Sims (Higher = Better Quality Data, Slower)
OUTPUT_FILE = "callbreak_selfplay_data_pro.npz"
TEMP = 1.0               # Temperature for exploration (1.0 = Proportional to visits)

class DataCollectorMCTS(ISMCTSAgent):
    """
    Extended MCTS Agent that returns the Probability Distribution (Thinking)
    instead of just the final move index.
    """
    def get_action_distribution(self, real_env):
        """
        Runs MCTS and returns:
        1. best_action_idx (int)
        2. probabilities (numpy array of shape 52) - The 'Thinking'
        """
        root = Node()
        
        # Extract observable info
        my_hand = real_env.hands[real_env.current_player]
        history = real_env.cards_played_history
        current_trick = real_env.trick_cards
        leader = real_env.leader_idx
        tricks_won = real_env.tricks_won
        
        # Run Simulations
        for _ in range(SIMULATIONS):
            node = root
            sim_env = self.determinize(my_hand, history, leader, current_trick, tricks_won)
            
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
                obs = sim_env._get_obs()
                mask = sim_env._get_valid_mask()
                
                # Get Priors from PPO
                probs = self._get_ppo_probs(obs, mask)
                
                valid_indices = [i for i, m in enumerate(mask) if m == 1]
                for idx in valid_indices:
                    node.children[idx] = Node(parent=node, prior=probs[idx])
                
                # Value (Using Network Estimation)
                leaf_value = self._get_ppo_value(obs)
            else:
                leaf_value = 1.0 if sim_env.tricks_won[0] > tricks_won[0] else 0.0

            # Backprop
            node.visit_count += 1
            node.value_sum += leaf_value
            for p_node in reversed(path):
                p_node.visit_count += 1
                p_node.value_sum += leaf_value

        # --- CALCULATE DISTRIBUTION (THINKING) ---
        # The 'thinking' is the visit counts of the root children
        policy_probs = np.zeros(52, dtype=np.float32)
        
        if root.children:
            visits = {a: n.visit_count for a, n in root.children.items()}
            sum_visits = sum(visits.values())
            
            if sum_visits > 0:
                for a, v in visits.items():
                    # Temperature scaling (optional, keeps exploration)
                    policy_probs[a] = (v ** (1.0/TEMP))
                
                # Normalize to sum to 1
                policy_probs /= np.sum(policy_probs)
            else:
                # Fallback to uniform random
                valid = list(visits.keys())
                policy_probs[valid] = 1.0 / len(valid)
                
            # Select Action based on distribution
            action = np.random.choice(52, p=policy_probs)
        else:
            # Fallback if no search
            action = self._random_legal(real_env)
            policy_probs[action] = 1.0

        return action, policy_probs

    def _get_ppo_probs(self, obs, mask):
        import torch
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs).unsqueeze(0)
            features = self.model.policy.extract_features(obs_tensor)
            latent_pi = self.model.policy.mlp_extractor.forward_actor(features)
            logits = self.model.policy.action_net(latent_pi)
            
            # Manual Masking
            HUGE_NEG = -1e8
            mask_tensor = torch.as_tensor(mask).unsqueeze(0)
            logits[mask_tensor == 0] = HUGE_NEG
            
            probs = torch.softmax(logits, dim=1).numpy()[0]
        return probs

    def _get_ppo_value(self, obs):
        import torch
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs).unsqueeze(0)
            features = self.model.policy.extract_features(obs_tensor)
            latent_vf = self.model.policy.mlp_extractor.forward_critic(features)
            value = self.model.policy.value_net(latent_vf).item()
        return value

def generate_data():
    print(f"--- GENERATING SELF-PLAY DATA ---")
    print(f"Games: {NUM_GAMES} | Simulations: {SIMULATIONS}")
    
    env = CallbreakEnv()
    agent = DataCollectorMCTS()
    
    if agent.model is None:
        print("Error: Train your PPO model first!")
        return

    # Data Buffers
    # X: Observations (156 features)
    # Y_policy: The MCTS "Thinking" (52 probabilities)
    # Y_value: The final game result (-1 to 1)
    dataset_obs = []
    dataset_probs = []
    dataset_values = []
    
    start_time = time.time()

    for i in range(1, NUM_GAMES + 1):
        obs, _ = env.reset()
        terminated = False
        
        # Temp buffer for this specific game
        game_obs = []
        game_probs = []
        game_player_indices = [] # Track which player made the move
        
        while not terminated:
            current_player = env.current_player
            
            # Get MCTS Decision & Thinking Distribution
            action, probs = agent.get_action_distribution(env)
            
            # Store Data
            # Note: We must store the observation BEFORE the step
            current_obs = env._get_obs() 
            game_obs.append(current_obs)
            game_probs.append(probs)
            game_player_indices.append(current_player)
            
            # Step
            obs, reward, terminated, _, _ = env.step(action)
            
        # Game Over - Assign Values (Who won?)
        # 1.0 for Winner, 0.0 (or -1.0) for Losers
        winner_idx = np.argmax(env.tricks_won)
        
        for p_idx in game_player_indices:
            # Standard AlphaZero Value:
            # +1 if this move led to a win (was made by the winner)
            # -1 if this move led to a loss
            val = 1.0 if p_idx == winner_idx else -1.0
            dataset_values.append(val)
            
        # Add game data to main buffer
        dataset_obs.extend(game_obs)
        dataset_probs.extend(game_probs)
        
        if i % 10 == 0:
            print(f"Generated {i}/{NUM_GAMES} games... ({len(dataset_obs)} total samples)")

    # Save to File
    print(f"Saving {len(dataset_obs)} samples to {OUTPUT_FILE}...")
    np.savez_compressed(
        OUTPUT_FILE, 
        obs=np.array(dataset_obs, dtype=np.float32), 
        probs=np.array(dataset_probs, dtype=np.float32),
        values=np.array(dataset_values, dtype=np.float32)
    )
    print("Done! You can now use this file to retrain the PPO model.")

if __name__ == "__main__":
    generate_data()