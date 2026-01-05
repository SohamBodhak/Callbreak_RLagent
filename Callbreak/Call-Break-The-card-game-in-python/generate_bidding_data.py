import torch
import numpy as np
import csv
import random
from train_ppo import PPOAgent, state_to_tensor, get_valid_moves, get_winner_idx, Memory
from card import Deck

# CONFIG
PPO_CHECKPOINT = "ppo_policy.pth"
OUTPUT_FILE = "bidding_data.csv"
GAMES_TO_PLAY = 2000000

def generate_bidding_data():
    device = torch.device('cpu') # CPU is fine for inference
    
    # Load your trained PPO "Player"
    agent = PPOAgent()
    agent.load_checkpoint(PPO_CHECKPOINT)
    agent.policy.eval() # Set to eval mode
    memory = Memory() # Dummy memory

    print(f"Generating {GAMES_TO_PLAY} games to learn Bidding...")
    
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["hand_cards", "tricks_won"])

        for g in range(GAMES_TO_PLAY):
            deck = Deck()
            hands = [[], [], [], []]
            for i in range(52): hands[i%4].append(deck.pop_card())
            for h in hands: h.sort()
            
            # Save the PPO Agent's hand (Player 0) BEFORE playing
            # Format: "Rank-Suit Rank-Suit..."
            p0_hand_str = " ".join([f"{c.rank}-{c.suit}" for c in hands[0]])
            
            leader_idx = random.randint(0, 3)
            player_tricks = [0, 0, 0, 0]
            
            # Play the full game
            for trick_num in range(13):
                trick_cards = []
                play_order = [(leader_idx + k) % 4 for k in range(4)]
                lead_card = None
                
                for p_idx in play_order:
                    valid = get_valid_moves(hands[p_idx], lead_card)
                    
                    if p_idx == 0:
                        # Use your TRAINED PPO Agent
                        card = agent.select_action(hands[p_idx], trick_cards, valid, memory)
                    else:
                        # Opponents play random valid (or use heuristic)
                        card = random.choice(valid)
                    
                    hands[p_idx].remove(card)
                    trick_cards.append(card)
                    if not lead_card: lead_card = card
                
                winner_idx = (leader_idx + get_winner_idx(trick_cards)) % 4
                player_tricks[winner_idx] += 1
                leader_idx = winner_idx
                
            # Log result: Hand -> Tricks Won
            writer.writerow([p0_hand_str, player_tricks[0]])
            
            if (g+1) % 1000 == 0:
                print(f"Played {g+1} games...", end='\r')
                
    print(f"\nData saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_bidding_data()