import csv
import random
import sys

# --- IMPORT THE HERESHEM ENGINE ---
try:
    from card import Deck, Card
except ImportError:
    print("Error: Could not import 'card.py'. Ensure this is in the Hereshem folder.")
    sys.exit(1)

# --- CONFIGURATION ---
NUM_GAMES = 100000 
OUTPUT_FILE = "callbreak_expert_data.csv"

# --- SMART BOT LOGIC ---
def get_winner_of_trick(trick_cards):
    """
    Returns the index of the winning card in the trick list.
    Rules: Spades > Lead Suit > Others. High rank wins.
    """
    if not trick_cards: return -1
    
    lead_suit = trick_cards[0].suit
    best_card = trick_cards[0]
    best_idx = 0
    
    for i, c in enumerate(trick_cards):
        if i == 0: continue
        
        # 1. If new card is Spade and current best is NOT Spade -> New wins
        if c.suit == 3 and best_card.suit != 3:
            best_card = c
            best_idx = i
        # 2. If both are Spades -> Higher rank wins
        elif c.suit == 3 and best_card.suit == 3:
            if c.rank > best_card.rank:
                best_card = c
                best_idx = i
        # 3. If neither is Spade, check Lead Suit
        elif c.suit == lead_suit and best_card.suit != 3:
            # If current best is not lead suit (impossible for index 0 but logic holds)
            if best_card.suit != lead_suit: 
                best_card = c
                best_idx = i
            # If both lead suit -> Higher rank wins
            elif c.rank > best_card.rank:
                best_card = c
                best_idx = i
                
    return best_idx

def pick_smart_move(valid_moves, board_cards):
    """
    Selects the best move from valid_moves based on a simple heuristic.
    """
    # 1. IF LEADING (Board is empty)
    if not board_cards:
        # Strategy: Play High Non-Spades (Aces/Kings) to secure tricks
        # Sort by Rank Descending
        valid_moves.sort(key=lambda x: x.rank, reverse=True)
        
        # Try to find Ace or King of non-spade
        for c in valid_moves:
            if c.suit != 3 and c.rank >= 12: # 12=Q, 13=K, 14=A
                return c
        
        # If no high cards, play lowest non-spade (save spades)
        non_spades = [c for c in valid_moves if c.suit != 3]
        if non_spades:
            non_spades.sort(key=lambda x: x.rank) # Lowest first
            return non_spades[0]
            
        # If only spades left, play lowest spade
        return valid_moves[-1] # valid_moves is sorted Desc, so last is lowest

    # 2. IF FOLLOWING
    # Determine who is currently winning
    current_winner_idx = get_winner_of_trick(board_cards)
    current_best_card = board_cards[current_winner_idx]
    lead_suit = board_cards[0].suit
    
    # Identify moves that would WIN the trick
    winning_moves = []
    
    for move in valid_moves:
        # Temporarily append to check if it wins
        temp_trick = board_cards + [move]
        winner_idx = get_winner_of_trick(temp_trick)
        
        # If the winner is the card we just added (last index)
        if winner_idx == len(temp_trick) - 1:
            winning_moves.append(move)
            
    # DECISION:
    if winning_moves:
        # STRATEGY: Win as cheaply as possible.
        # Pick the LOWEST ranking card that still wins.
        winning_moves.sort(key=lambda x: x.rank)
        return winning_moves[0]
    else:
        # STRATEGY: We can't win. Throw trash.
        # Pick the LOWEST ranking card available.
        valid_moves.sort(key=lambda x: x.rank)
        return valid_moves[0]

def get_valid_moves(hand, lead_card=None):
    """Standard Callbreak Rules Logic"""
    if not lead_card:
        return hand

    # 1. Follow Suit
    same_suit_cards = [c for c in hand if c.suit == lead_card.suit]
    if same_suit_cards:
        # Strict Rule: Must beat current winner if possible (Not implemented fully here to keep speed, 
        # but Hereshem engine might enforce it. We stick to suit constraint).
        return same_suit_cards

    # 2. Trump (Spade) if out of suit
    spades = [c for c in hand if c.suit == 3]
    if spades:
        return spades
        
    # 3. Any card
    return hand

# --- SIMULATION LOOP ---
def run_simulation():
    print(f"Generating {NUM_GAMES} EXPERT games...")

    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "game_id", "round_num", "trick_num", "player_id", 
            "hand_cards", "board_cards", "action_played", "winner_id"
        ])

        game_id = 0
        while game_id < NUM_GAMES:
            deck = Deck()
            hands = [[], [], [], []]
            for i in range(52):
                hands[i % 4].append(deck.pop_card())
            for h in hands: h.sort()

            leader_idx = random.randint(0, 3)

            for trick_num in range(1, 14):
                current_trick_cards = []
                trick_data_buffer = [] 
                
                play_order = [(leader_idx + i) % 4 for i in range(4)]
                lead_card = None

                for p_idx in play_order:
                    player_hand = hands[p_idx]
                    valid_moves = get_valid_moves(player_hand, lead_card)
                    
                    # --- CHANGE: USE SMART BOT INSTEAD OF RANDOM ---
                    # 90% chance to play smart (simulate human errors/variety)
                    if random.random() < 0.9:
                        played_card = pick_smart_move(valid_moves, current_trick_cards)
                    else:
                        played_card = random.choice(valid_moves)
                    
                    player_hand.remove(played_card) 
                    if lead_card is None: lead_card = played_card

                    # Log
                    hand_str = " ".join([str(c) for c in player_hand + [played_card]]) # Log full hand
                    board_str = " ".join([str(c) for c in current_trick_cards])
                    
                    trick_data_buffer.append({
                        "player_id": p_idx,
                        "hand": hand_str,
                        "board": board_str,
                        "action": str(played_card)
                    })
                    
                    current_trick_cards.append(played_card)

                # Determine Trick Winner
                winner_rel_idx = get_winner_of_trick(current_trick_cards)
                winner_p_idx = play_order[winner_rel_idx]
                
                # Write to CSV
                for log in trick_data_buffer:
                    writer.writerow([
                        game_id, 1, trick_num,
                        log["player_id"], log["hand"], log["board"], 
                        log["action"], winner_p_idx
                    ])

                leader_idx = winner_p_idx
            
            game_id += 1
            if game_id % 1000 == 0:
                print(f"Generated {game_id} games...", end='\r')

    print(f"\nDone! Expert data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_simulation()