import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# CONFIG
CSV_FILE = "bidding_data.csv"
BIDDER_MODEL_PATH = "bidder_network.pth"
BATCH_SIZE = 64
EPOCHS = 200

# --- DATASET ---
class BiddingDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Parse Hand "Rank-Suit"
        hand_parts = row['hand_cards'].split()
        hand_tensor = torch.zeros(52, dtype=torch.float32)
        
        for p in hand_parts:
            r, s = map(int, p.split('-'))
            idx = (r - 2) * 4 + s
            if 0 <= idx < 52: hand_tensor[idx] = 1.0
            
        # Target: Tricks Won (Float for regression)
        label = torch.tensor(float(row['tricks_won']), dtype=torch.float32)
        return hand_tensor, label

# --- BIDDING NETWORK ---
class BiddingNetwork(nn.Module):
    def __init__(self):
        super(BiddingNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(52, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Output: Predicted Number of Tricks
        )

    def forward(self, x):
        return self.net(x)

# --- TRAIN ---
def train_bidder():
    dataset = BiddingDataset(CSV_FILE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = BiddingNetwork()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss() # Mean Squared Error for regression
    
    model.train()
    print("Training Bidding Network...")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for hands, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(hands).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(dataloader):.4f}")
        
    torch.save(model.state_dict(), BIDDER_MODEL_PATH)
    print(f"Bidder Model saved to {BIDDER_MODEL_PATH}")

    # Test
    test_hand = torch.zeros(52)
    # Add Ace, King, Queen of Spades (Indices 51, 47, 43)
    test_hand[51] = 1; test_hand[47] = 1; test_hand[43] = 1 
    with torch.no_grad():
        pred = model(test_hand.unsqueeze(0)).item()
    print(f"Predicted Bid for Hand with A,K,Q Spades: {pred:.2f} (Should be high)")

if __name__ == "__main__":
    train_bidder()