import gymnasium as gym
import numpy as np
import random
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from callbreak_env import CallbreakEnv

# CONFIG
MODEL_PATH = "ppo_callbreak_pro.zip"
NUM_GAMES = 10000 # 1000 games is enough for a solid comparison

# --- 1. DEFINE THE SIMPLE STRATEGY ---
def simple_heuristic_strategy(hand, trick_cards, lead_suit=None):
    """
    A basic 'Common Sense' strategy for opponents:
    1. If you can win the trick cheaply, do it.
    2. If you can't win, throw the smallest garbage card.
    """
    # Filter valid moves logic (Replicating env logic for the bot)
    valid = hand
    if trick_cards:
        lead_suit = trick_cards[0].suit
        s = [c for c in hand if c.suit == lead_suit]
        if s: 
            valid = s
        else:
            sp = [c for c in hand if c.suit == 3] # Spades = 3
            if sp: 
                valid = sp
    
    # ANALYSIS
    current_winner_is_spade = False
    current_best_rank = -1
    
    if trick_cards:
        # Who is winning currently?
        best_card = trick_cards[0]
        lead_suit = trick_cards[0].suit
        
        for i, c in enumerate(trick_cards):
            if i == 0: continue
            if c.suit == 3 and best_card.suit != 3: 
                best_card = c
            elif c.suit == 3 and best_card.suit == 3:
                if c.rank > best_card.rank: best_card = c
            elif c.suit == lead_suit and best_card.suit != 3:
                if c.rank > best_card.rank: best_card = c
        
        if best_card.suit == 3: current_winner_is_spade = True
        current_best_rank = best_card.rank

    # DECISION LOGIC
    winning_moves = []
    
    for c in valid:
        # Can we beat the current best card?
        is_winner = False
        if not trick_cards:
            # Leading: High cards are generally good
            is_winner = True 
        else:
            if c.suit == lead_suit and not current_winner_is_spade:
                if c.rank > current_best_rank: is_winner = True
            elif c.suit == 3 and not current_winner_is_spade:
                is_winner = True # First spade cuts
            elif c.suit == 3 and current_winner_is_spade:
                if c.rank > current_best_rank: is_winner = True # Over-cut
        
        if is_winner:
            winning_moves.append(c)

    if winning_moves:
        # Win as cheaply as possible (smallest winning card)
        winning_moves.sort(key=lambda x: x.rank)
        return winning_moves[0]
    else:
        # Can't win? Throw smallest garbage (save high cards for later? No, simple bot dumps low)
        valid.sort(key=lambda x: x.rank)
        return valid[0]

# --- 2. HELPER TO OVERRIDE ENV ---
def patch_env_with_heuristic(env):
    """Replaces the random bot logic in the environment with our heuristic."""
    
    # We define a new method bound to the environment instance
    def smart_bot_play(self, player_idx):
        if not self.hands[player_idx]: return

        # Get inputs for decision
        hand = self.hands[player_idx]
        trick = self.trick_cards
        
        # Use Strategy
        chosen_card = simple_heuristic_strategy(hand, trick)
        
        # Execute (standard env logic)
        self.hands[player_idx].remove(chosen_card)
        self.trick_cards.append(chosen_card)
        self.cards_played_history.add((chosen_card.rank-2)*4 + chosen_card.suit)

    # MONKEY PATCH: Replace the method on this specific instance
    # We need to bind it to the unwrapped env (the base CallbreakEnv)
    env.unwrapped._bot_play = smart_bot_play.__get__(env.unwrapped, CallbreakEnv)
    return env

# --- 3. EVALUATION LOOP ---
def mask_fn(env):
    return env.unwrapped._get_valid_mask()

def evaluate():
    print(f"Loading {MODEL_PATH}...")
    try:
        model = MaskablePPO.load(MODEL_PATH)
    except:
        print("Model not found. Train it first!")
        return

    # Create & Patch Environment
    env = CallbreakEnv()
    env = patch_env_with_heuristic(env) # <--- UPGRADE OPPONENTS
    env = ActionMasker(env, mask_fn)

    print(f"Simulating {NUM_GAMES} games against HEURISTIC BOTS (Simple Strategy)...")
    
    scores = []
    
    for i in range(NUM_GAMES):
        obs, _ = env.reset()
        done = False
        tricks = 0
        
        while not done:
            action_masks = get_action_masks(env)
            # Deterministic=True for fair skill evaluation
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            
            obs, reward, done, _, _ = env.step(action)
            
            if reward > 0:
                tricks += 1
        
        scores.append(tricks)
        
        if (i+1) % 100 == 0:
            print(f"Game {i+1} | Avg Tricks: {np.mean(scores):.2f}")

    avg_score = np.mean(scores)
    print(f"\nFINAL RESULT vs HEURISTIC BOTS ({NUM_GAMES} games):")
    print(f"Average Tricks Won: {avg_score:.2f}")
    
    # Comparison
    print("-" * 30)
    if avg_score > 3.25:
        print("Verdict: SUPERHUMAN (Beats the heuristic consistently)")
    elif avg_score > 3.00:
        print("Verdict: COMPETITIVE (Matches the heuristic strategy)")
    else:
        print("Verdict: WEAK (Losing to simple logic. Needs more training)")
    print("-" * 30)

if __name__ == "__main__":
    evaluate()