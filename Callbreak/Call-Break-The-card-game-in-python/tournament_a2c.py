import time
import numpy as np
from callbreak_env import CallbreakEnv
# Assuming you saved the A2C MCTS code in mcts_agent_a2c.py
from mcts_agent_a2c import ISMCTSAgentA2C 

# --- CONFIGURATION ---
NUM_GAMES = 1000
MCTS_SIMULATIONS = 50 

def run_tournament():
    print(f"--- STARTING TOURNAMENT: MCTS (A2C) vs 3 HEURISTIC BOTS ---")
    print(f"Goal: Highest TOTAL SCORE after {NUM_GAMES} games.")
    
    # 1. Initialize
    env = CallbreakEnv()
    
    mcts_bot = ISMCTSAgentA2C()
    
    if mcts_bot.model is None:
        print("CRITICAL ERROR: A2C Model not found. Train A2C first!")
        return

    # THE SCOREBOARD
    total_tricks_won = [0, 0, 0, 0] # Total tricks over all games
    total_games_won = [0, 0, 0, 0]  # Total distinct games won
    
    start_time = time.time()

    # 2. Game Loop
    for game_idx in range(1, NUM_GAMES + 1):
        obs, _ = env.reset()
        terminated = False
        
        while not terminated:
            action = mcts_bot.select_move(env)
            obs, reward, terminated, _, info = env.step(action)
            
        # GAME OVER: Update Totals
        
        # 1. Add tricks to total count
        for i in range(4):
            total_tricks_won[i] += env.tricks_won[i]
            
        # 2. Determine who won THIS specific game and increment their game counter
        winner_idx = np.argmax(env.tricks_won)
        total_games_won[winner_idx] += 1
        
        # Logging
        elapsed = time.time() - start_time
        avg_time = elapsed / game_idx
        eta = avg_time * (NUM_GAMES - game_idx)
        
        winner_name = "MCTS-A2C" if winner_idx == 0 else f"Bot {winner_idx}"
        
        if game_idx % 10 == 0: 
            print(f"Game {game_idx}/{NUM_GAMES} | "
                  f"Winner: {winner_name} ({env.tricks_won[winner_idx]} tricks) | "
                  f"MCTS Total Tricks: {total_tricks_won[0]} | "
                  f"ETA: {eta/60:.1f} min")

    # 3. FINAL RESULTS
    print("\n" + "="*60)
    print("           FINAL TOURNAMENT SCOREBOARD           ")
    print("="*60)
    # Changed header: 'Type' -> 'GAMES WON'
    print(f"{'Rank':<6} | {'Player Name':<15} | {'GAMES WON':<12} | {'TOTAL TRICKS':<12}")
    print("-" * 65)
    
    # Results tuple: (PlayerIndex, TotalTricks, Name, GamesWon)
    results = [
        (0, total_tricks_won[0], "MCTS (A2C)", total_games_won[0]),
        (1, total_tricks_won[1], "Bot 1", total_games_won[1]),
        (2, total_tricks_won[2], "Bot 2", total_games_won[2]),
        (3, total_tricks_won[3], "Bot 3", total_games_won[3])
    ]
    
    # Sort by TOTAL TRICKS (Standard Callbreak ranking)
    # If you prefer to rank by Games Won, change x[1] to x[3]
    results.sort(key=lambda x: x[1], reverse=True)
    
    for rank, (pid, tricks, name, games) in enumerate(results, 1):
        print(f"#{rank:<5} | {name:<15} | {games:<12} | {tricks:<12}")
        
    print("-" * 65)
    
    # Final Verdict
    winner = results[0]
    diff = winner[1] - results[1][1]
    
    if winner[0] == 0:
        print(f"\n>>> VICTORY! The A2C-MCTS Agent won by {diff} tricks!")
    else:
        diff_to_winner = results[0][1] - total_tricks_won[0]
        print(f"\n>>> DEFEAT. The A2C-MCTS Agent lost by {diff_to_winner} tricks.")

if __name__ == "__main__":
    run_tournament()