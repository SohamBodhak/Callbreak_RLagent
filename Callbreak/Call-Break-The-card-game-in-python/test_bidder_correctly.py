import torch
import random
from train_bidder import BiddingNetwork  # Ensure this matches your file name

def test_bidder_correctly():
    # 1. Load Model
    model = BiddingNetwork()
    try:
        model.load_state_dict(torch.load("bidder_network.pth"))
    except:
        print("Train the model first!")
        return
    model.eval()

    # 2. Construct a REAL hand (13 Cards)
    # We fix A(51), K(47), Q(43) of Spades
    high_spades = [51, 47, 43]
    
    # We need 10 more random cards that are NOT the ones we picked
    all_indices = list(range(52))
    for idx in high_spades:
        all_indices.remove(idx)
        
    # Pick 10 random "filler" cards
    filler_cards = random.sample(all_indices, 10)
    
    # Combine
    full_hand_indices = high_spades + filler_cards
    
    # 3. Create Tensor
    hand_tensor = torch.zeros(52)
    for idx in full_hand_indices:
        hand_tensor[idx] = 1.0
        
    # 4. Predict
    with torch.no_grad():
        pred = model(hand_tensor.unsqueeze(0)).item()
        
    print(f"Hand: A-K-Q Spades + {10} random cards")
    print(f"Predicted Bid: {pred:.2f}")

    # Interpretation
    if pred < 3.0:
        print("\n[Diagnosis] Bid is still low.")
        print("Reason: Your PPO Agent is likely not strong yet.")
        print("Solution: Train the PPO Agent for more episodes, then re-generate bidding data.")
    else:
        print("\n[Diagnosis] Bid looks healthy! The logic works.")

if __name__ == "__main__":
    test_bidder_correctly()