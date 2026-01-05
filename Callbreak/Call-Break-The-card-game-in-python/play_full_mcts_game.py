import gymnasium as gym
from callbreak_env import CallbreakEnv
from mcts_agent import ISMCTSAgent # Import your MCTS agent

def play_full_game():
    # 1. Initialize Environment & Agent
    print("Initializing Game and MCTS Agent...")
    env = CallbreakEnv()
    env.reset() # Deals cards
    
    agent = ISMCTSAgent()
    
    # Check if model loaded
    if agent.model is None:
        print("Model failed to load. Exiting.")
        return

    print("\n--- GAME START ---")
    
    # 2. Loop through 13 Tricks
    total_tricks_won = 0
    
    # We must loop until game is 'terminated'
    terminated = False
    trick_count = 1
    
    while not terminated:
        print(f"\n[Trick {trick_count}]")
        print(f"My Hand ({len(env.hands[0])} cards): {[f'{c.rank}-{c.suit}' for c in env.hands[0]]}")
        
        # Is it my turn?
        # The env automatically fast-forwards to your turn in step(), 
        # but we need to check if we are the leader or following.
        print(f"Current Board: {[f'{c.rank}-{c.suit}' for c in env.trick_cards]}")
        
        # --- 3. ASK MCTS FOR THE MOVE ---
        print("MCTS is thinking...")
        action_idx = agent.select_move(env)
        
        # Decode for display
        r, s = (action_idx // 4) + 2, action_idx % 4
        print(f"MCTS plays: Rank {r}, Suit {s}")
        
        # --- 4. APPLY MOVE ---
        # This updates the hand (removes card) and history
        obs, reward, terminated, _, info = env.step(action_idx)
        
        if reward > 0:
            print(">>> You WON this trick! (+1)")
            total_tricks_won += 1
        else:
            print("--- Trick lost.")
            
        trick_count += 1
        
    print("\n--- GAME OVER ---")
    print(f"Total Tricks Won by MCTS: {total_tricks_won}")

if __name__ == "__main__":
    play_full_game()