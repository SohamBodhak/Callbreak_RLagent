import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import glob
import re
from sb3_contrib import MaskablePPO
from callbreak_env import CallbreakEnv
from sb3_contrib.common.wrappers import ActionMasker

# --- CONFIGURATION ---
DATA_FILE = "callbreak_selfplay_data.npz"
MODEL_PATH = "ppo_callbreak_zero_pro.zip"     # The base model to update
OUTPUT_MODEL = "ppo_callbreak_zero_pro_x.zip"  # The new "Smarter" model output
CHECKPOINT_DIR = "x_zero_checkpoints"
BATCH_SIZE = 256
EPOCHS = 5000                              # How many times to learn from the data
LEARNING_RATE = 0.0003

def mask_fn(env): return env.unwrapped._get_valid_mask()

def find_latest_checkpoint(checkpoint_dir):
    """Scans for ppo_zero_epoch_X.zip and returns the latest."""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        return None, 0

    # Look for files like ppo_zero_epoch_5.zip
    files = glob.glob(os.path.join(checkpoint_dir, "ppo_zero_epoch_*.zip"))
    if not files: return None, 0

    latest_file = None
    max_epoch = -1
    
    for f in files:
        match = re.search(r"ppo_zero_epoch_(\d+).zip", f)
        if match:
            epoch = int(match.group(1))
            if epoch > max_epoch:
                max_epoch = epoch
                latest_file = f
                
    return latest_file, max_epoch

def train_network():
    print("--- ALPHAZERO RETRAINING LOOP ---")
    
    # 1. Load Data
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file {DATA_FILE} not found. Run generate_self_play.py first.")
        return

    print(f"Loading MCTS data from {DATA_FILE}...")
    data = np.load(DATA_FILE)
    
    # Convert numpy arrays to PyTorch tensors
    obs_data = torch.tensor(data['obs'], dtype=torch.float32)
    probs_data = torch.tensor(data['probs'], dtype=torch.float32)
    # Values need shape (N, 1) to match value net output
    vals_data = torch.tensor(data['values'], dtype=torch.float32).unsqueeze(1) 
    
    dataset_size = len(obs_data)
    print(f"Loaded {dataset_size} samples.")

    # 2. Load Model (Resume or Start Fresh update)
    latest_ckpt, start_epoch = find_latest_checkpoint(CHECKPOINT_DIR)
    
    if latest_ckpt:
        print(f"RESUMING from checkpoint: {latest_ckpt} (Epoch {start_epoch})")
        model = MaskablePPO.load(latest_ckpt, device="cpu")
    else:
        print(f"Starting update from base model: {MODEL_PATH}")
        try:
            model = MaskablePPO.load(MODEL_PATH, device="cpu")
        except:
            print(f"Base model {MODEL_PATH} not found! Please train phase 2 first.")
            return

    # Access Internal PyTorch Networks
    policy = model.policy
    policy.train() # Set PyTorch to training mode (enables dropout/batchnorm updates)
    
    # Optimizer for the shared feature extractor and heads
    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    
    # Loss Functions
    # Value Loss: Mean Squared Error between Predicted Value and Actual Outcome (+1/-1)
    val_loss_fn = nn.MSELoss()
    
    # 3. Training Loop
    print(f"Training for {EPOCHS} epochs...")
    
    for epoch in range(start_epoch + 1, start_epoch + EPOCHS + 1):
        total_p_loss = 0
        total_v_loss = 0
        
        # Shuffle data for this epoch
        permutation = torch.randperm(dataset_size)
        
        for i in range(0, dataset_size, BATCH_SIZE):
            indices = permutation[i:i+BATCH_SIZE]
            batch_obs = obs_data[indices]
            batch_probs = probs_data[indices] # Target Thinking (MCTS distribution)
            batch_vals = vals_data[indices]   # Target Winner (+1/-1)
            
            optimizer.zero_grad()
            
            # --- Forward Pass ---
            # Extract features from observation (Hand + Board + History)
            features = policy.extract_features(batch_obs)
            
            # Policy Head (Actor)
            latent_pi = policy.mlp_extractor.forward_actor(features)
            logits = policy.action_net(latent_pi)
            log_probs = torch.log_softmax(logits, dim=1)
            
            # Value Head (Critic)
            latent_vf = policy.mlp_extractor.forward_critic(features)
            value_pred = policy.value_net(latent_vf)
            
            # --- Calculate Losses ---
            # 1. Policy Loss (Cross Entropy)
            # We want the network's output (log_probs) to match the MCTS thinking (batch_probs)
            # Loss = - sum( Target_Prob * Log(Predicted_Prob) )
            p_loss = -torch.sum(batch_probs * log_probs) / BATCH_SIZE
            
            # 2. Value Loss (MSE)
            v_loss = val_loss_fn(value_pred, batch_vals)
            
            # Total Loss (AlphaZero standard combines them)
            loss = p_loss + v_loss
            
            loss.backward()
            optimizer.step()
            
            total_p_loss += p_loss.item()
            total_v_loss += v_loss.item()
            
        # Logging
        avg_p_loss = total_p_loss / (dataset_size / BATCH_SIZE)
        avg_v_loss = total_v_loss / (dataset_size / BATCH_SIZE)
        print(f"Epoch {epoch} | Policy Loss: {avg_p_loss:.4f} | Value Loss: {avg_v_loss:.4f}")
        
        # Checkpoint Save
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"ppo_zero_epoch_{epoch}.zip")
        model.save(ckpt_path)

    # 4. Final Save
    model.save(OUTPUT_MODEL)
    print(f"\nUpdate Complete. New model saved to {OUTPUT_MODEL}")
    print("Action Item: Rename this file to 'ppo_callbreak_pro.zip' (or update your config) so the MCTS agent uses this new brain.")

if __name__ == "__main__":
    train_network()