import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sb3_contrib import MaskablePPO
from stable_baselines3 import A2C

# Import the original env to inherit from
from callbreak_env import CallbreakEnv

# --- CONFIGURATION ---
NUM_GAMES = 10000
MODEL_PATHS = {
    "PPO": "ppo_callbreak_pro_without_self_play.zip",
    "A2C": "a2c_callbreak_final.zip",
    "A3C": "a3c_callbreak_final.pth"
}

# --- 1. SPECIAL TOURNAMENT ENVIRONMENT ---
# This overrides the standard env to allow manual control for ALL players
class TournamentEnv(CallbreakEnv):
    def get_obs_for_player(self, p_idx):
        """Generates the observation vector from p_idx's perspective."""
        # 1. One-Hot Hand
        hand_obs = np.zeros(52, dtype=np.float32)
        for c in self.hands[p_idx]: 
            idx = (c.rank - 2) * 4 + c.suit
            if 0 <= idx < 52: hand_obs[idx] = 1.0
        
        # 2. One-Hot Board (Same for everyone)
        board_obs = np.zeros(52, dtype=np.float32)
        for c in self.trick_cards:
            idx = (c.rank - 2) * 4 + c.suit
            if 0 <= idx < 52: board_obs[idx] = 1.0
            
        # 3. History (Same for everyone)
        history_obs = np.zeros(52, dtype=np.float32)
        for idx in self.cards_played_history:
            if 0 <= idx < 52: history_obs[idx] = 1.0
            
        return np.concatenate([hand_obs, board_obs, history_obs])

    def get_valid_mask_for_player(self, p_idx):
        """Calculates valid moves for p_idx."""
        mask = np.zeros(52, dtype=np.float32)
        if not self.hands[p_idx]: return mask
        
        lead_card = self.trick_cards[0] if self.trick_cards else None
        hand = self.hands[p_idx]
        valid_cards = hand
        
        if lead_card:
            same_suit = [c for c in hand if c.suit == lead_card.suit]
            if same_suit: valid_cards = same_suit
            else:
                spades = [c for c in hand if c.suit == 3]
                if spades: valid_cards = spades
        
        for c in valid_cards:
            idx = (c.rank - 2) * 4 + c.suit
            if 0 <= idx < 52: mask[idx] = 1.0
        return mask

    def step_manual(self, action_idx):
        """Executes one move for current_player and stops (No Auto-Play)."""
        p_idx = self.current_player
        
        # Find card object
        card_obj = None
        for c in self.hands[p_idx]:
            if ((c.rank - 2) * 4 + c.suit) == action_idx: card_obj = c; break
        
        # Safety fallback
        if card_obj is None:
            # print(f"Agent {p_idx} tried invalid move {action_idx}. Playing Random.")
            mask = self.get_valid_mask_for_player(p_idx)
            valid_indices = np.where(mask == 1)[0]
            if len(valid_indices) > 0:
                action_idx = np.random.choice(valid_indices)
                for c in self.hands[p_idx]:
                    if ((c.rank - 2) * 4 + c.suit) == action_idx: card_obj = c; break
        
        if card_obj:
            self.hands[p_idx].remove(card_obj)
            self.trick_cards.append(card_obj)
            self.cards_played_history.add(action_idx)

        # Check Trick End
        trick_over = False
        if len(self.trick_cards) == 4:
            trick_over = True
            winner_rel = self._get_winner_idx(self.trick_cards)
            winner_abs = (self.leader_idx + winner_rel) % 4
            self.tricks_won[winner_abs] += 1
            self.leader_idx = winner_abs
            self.current_player = winner_abs
            self.trick_cards = []
        else:
            self.current_player = (self.current_player + 1) % 4
            
        terminated = sum(self.tricks_won) == 13
        return terminated


# --- 2. ARCHITECTURE & AGENTS ---
class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.l1 = nn.Linear(input_dim, 256)
        self.l2 = nn.Linear(256, 128)
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.actor(x), self.critic(x)

class PPOBot:
    def __init__(self, path):
        try: self.model = MaskablePPO.load(path)
        except: self.model = None
    def select_move(self, env, p_idx):
        if self.model:
            obs = env.get_obs_for_player(p_idx)
            mask = env.get_valid_mask_for_player(p_idx)
            action, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
            return int(action)
        return self.random_move(env, p_idx)
    def random_move(self, env, p_idx):
        mask = env.get_valid_mask_for_player(p_idx)
        return int(np.random.choice(np.where(mask==1)[0]))

class A2CBot:
    def __init__(self, path):
        try: self.model = A2C.load(path)
        except: self.model = None
    def select_move(self, env, p_idx):
        if self.model:
            obs = env.get_obs_for_player(p_idx)
            action, _ = self.model.predict(obs, deterministic=True)
            mask = env.get_valid_mask_for_player(p_idx)
            if mask[action] == 1: return int(action)
        return self.random_move(env, p_idx)
    def random_move(self, env, p_idx):
        mask = env.get_valid_mask_for_player(p_idx)
        return int(np.random.choice(np.where(mask==1)[0]))

class A3CBot:
    def __init__(self, path):
        self.model = ActorCritic(156, 52)
        try:
            self.model.load_state_dict(torch.load(path))
            self.model.eval()
        except: self.model = None
    def select_move(self, env, p_idx):
        if self.model:
            obs = torch.FloatTensor(env.get_obs_for_player(p_idx)).unsqueeze(0)
            mask = torch.FloatTensor(env.get_valid_mask_for_player(p_idx)).unsqueeze(0)
            with torch.no_grad():
                logits, _ = self.model(obs)
            logits[mask == 0] = -1e9
            return torch.argmax(logits, dim=1).item()
        return 0

class HeuristicBot:
    def select_move(self, env, p_idx):
        mask = env.get_valid_mask_for_player(p_idx)
        valid = np.where(mask == 1)[0]
        if len(valid) == 0: return 0
        return int(valid[-1]) # Play highest index valid card (Rank based greedy)

# --- 3. TOURNAMENT MAIN LOOP ---
def run_tournament():
    print(f"\n--- STARTING TOURNAMENT (Manual Control Env) ---")
    print(f"PPO vs A2C vs A3C vs Heuristic")
    
    # Initialize the Manual Environment
    env = TournamentEnv()
    
    agents = {
        0: PPOBot(MODEL_PATHS["PPO"]),
        1: A2CBot(MODEL_PATHS["A2C"]),
        2: A3CBot(MODEL_PATHS["A3C"]),
        #2:HeuristicBot(),
        3: HeuristicBot()
    }
    names = ["PPO", "A2C", "A3C", "Heuristic"]
    
    total_wins = [0, 0, 0, 0]
    total_tricks = [0, 0, 0, 0]
    
    start_t = time.time()
    
    for game in range(1, NUM_GAMES + 1):
        # Reset but don't auto-play
        env.reset() 
        terminated = False
        
        while not terminated:
            p = env.current_player
            # Ask specific agent for move using THEIR perspective
            action = agents[p].select_move(env, p)
            # Step manually
            terminated = env.step_manual(action)
            
        # Game Over
        winner = np.argmax(env.tricks_won)
        total_wins[winner] += 1
        for i in range(4): total_tricks[i] += env.tricks_won[i]
        
        if game % 50 == 0:
            print(f"Game {game} | Wins: {total_wins}")

    print("\n" + "="*50)
    print(f"{'Rank':<5} | {'Agent':<10} | {'Wins':<8} | {'Avg Tricks'}")
    print("-" * 50)
    
    ranking = []
    for i in range(4):
        ranking.append((names[i], total_wins[i], total_tricks[i]/NUM_GAMES))
    ranking.sort(key=lambda x: x[1], reverse=True)
    
    for r, (n, w, avg) in enumerate(ranking, 1):
        print(f"#{r:<4} | {n:<10} | {w:<8} | {avg:.2f}")

if __name__ == "__main__":
    run_tournament()