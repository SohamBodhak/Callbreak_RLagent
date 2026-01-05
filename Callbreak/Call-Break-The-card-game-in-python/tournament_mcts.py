import time
import numpy as np
from callbreak_env import CallbreakEnv
from mcts_agent import ISMCTSAgent

# --- CONFIGURATION ---
NUM_GAMES = 1000
MCTS_SIMULATIONS = 50  # Adjust based on speed preference

def run_tournament():
    print(f"--- STARTING TOURNAMENT: MCTS (PPO) vs 3 HEURISTIC BOTS ---")
    print(f"Goal: Highest TOTAL SCORE after {NUM_GAMES} games.")
    
    # 1. Initialize
    env = CallbreakEnv()
    
    # Initialize Agent (Player 0)
    mcts_bot = ISMCTSAgent()
    
    if mcts_bot.model is None:
        print("CRITICAL ERROR: PPO Model not found. Train Phase 2 Pro first!")
        return

    # THE SCOREBOARD
    total_tricks_won = [0, 0, 0, 0] # Total tricks collected across all games
    total_games_won = [0, 0, 0, 0]  # Count of games where this player had the highest score
    
    start_time = time.time()

    # 2. Game Loop
    for game_idx in range(1, NUM_GAMES + 1):
        obs, _ = env.reset()
        terminated = False
        
        while not terminated:
            # Player 0 (MCTS-PPO) thinks and plays
            action = mcts_bot.select_move(env)
            
            # Environment plays P1, P2, P3 automatically using Heuristic
            obs, reward, terminated, _, info = env.step(action)
            
        # GAME OVER: Update Totals
        
        # 1. Add tricks to total count
        for i in range(4):
            total_tricks_won[i] += env.tricks_won[i]
            
        # 2. Determine who won THIS specific game
        winner_idx = np.argmax(env.tricks_won)
        total_games_won[winner_idx] += 1
            
        # Live Progress Update
        elapsed = time.time() - start_time
        avg_time = elapsed / game_idx
        eta = avg_time * (NUM_GAMES - game_idx)
        
        winner_name = "MCTS-PPO" if winner_idx == 0 else f"Bot {winner_idx}"
        
        if game_idx % 10 == 0:
            print(f"Game {game_idx}/{NUM_GAMES} | "
                  f"Winner: {winner_name} ({env.tricks_won[winner_idx]} tricks) | "
                  f"MCTS Total Tricks: {total_tricks_won[0]} | "
                  f"ETA: {eta/60:.1f} min")

    # 3. FINAL RESULTS
    print("\n" + "="*60)
    print("           FINAL TOURNAMENT SCOREBOARD           ")
    print("="*60)
    # Replaced 'Type' with 'GAMES WON'
    print(f"{'Rank':<6} | {'Player Name':<15} | {'GAMES WON':<12} | {'TOTAL TRICKS':<12}")
    print("-" * 65)
    
    # Results tuple: (PlayerIndex, TotalTricks, Name, GamesWon)
    results = [
        (0, total_tricks_won[0], "MCTS (PPO)", total_games_won[0]),
        (1, total_tricks_won[1], "Bot 1", total_games_won[1]),
        (2, total_tricks_won[2], "Bot 2", total_games_won[2]),
        (3, total_tricks_won[3], "Bot 3", total_games_won[3])
    ]
    
    # Sort by Total Tricks (Standard Callbreak ranking)
    # If you prefer ranking by Games Won, change x[1] to x[3]
    results.sort(key=lambda x: x[1], reverse=True)
    
    for rank, (pid, tricks, name, games) in enumerate(results, 1):
        print(f"#{rank:<5} | {name:<15} | {games:<12} | {tricks:<12}")
        
    print("-" * 65)
    
    # Final Verdict
    winner = results[0]
    diff = winner[1] - results[1][1]
    
    if winner[0] == 0:
        print(f"\n>>> VICTORY! The PPO-MCTS Agent won by {diff} tricks!")
    else:
        diff_to_winner = results[0][1] - total_tricks_won[0]
        print(f"\n>>> DEFEAT. The PPO-MCTS Agent lost by {diff_to_winner} tricks.")

if __name__ == "__main__":
    run_tournament()