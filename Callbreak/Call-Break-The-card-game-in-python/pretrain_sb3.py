import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from callbreak_env import CallbreakEnv
import random

# CONFIG
PRETRAIN_GAMES = 50000  # 5k games is enough for initial behavior cloning
BATCH_SIZE = 64
EPOCHS = 50
MODEL_NAME = "ppo_callbreak_pretrained"

# --- EXPERT HEURISTIC BOT ---
def pick_expert_move(hand, trick_cards, valid_moves_mask):
    """
    Expert logic with safety checks.
    """
    # 1. Decode valid moves
    valid_indices = [i for i, m in enumerate(valid_moves_mask) if m == 1]
    
    # --- SAFETY FIX: Handle Empty Mask ---
    if not valid_indices:
        return None # Signal that no move is possible

    # If only one move, play it
    if len(valid_indices) == 1:
        return valid_indices[0]

    # Reconstruct card objects
    valid_cards = []
    for idx in valid_indices:
        r = (idx // 4) + 2
        s = idx % 4
        valid_cards.append({'idx': idx, 'rank': r, 'suit': s})

    # 2. Strategy Logic
    if not trick_cards:
        # LEAD: Play highest rank card
        valid_cards.sort(key=lambda x: x['rank'], reverse=True)
        return valid_cards[0]['idx']

    # FOLLOW:
    lead_suit = trick_cards[0].suit
    current_winner_is_spade = False
    for c in trick_cards:
        if c.suit == 3: current_winner_is_spade = True
        
    winning_moves = []
    for c in valid_cards:
        if c['suit'] == lead_suit and not current_winner_is_spade:
             winning_moves.append(c)
        elif c['suit'] == 3 and not current_winner_is_spade:
             winning_moves.append(c)

    if winning_moves:
        # Win cheaply
        winning_moves.sort(key=lambda x: x['rank'])
        return winning_moves[0]['idx']
    else:
        # Throw trash (lowest rank)
        valid_cards.sort(key=lambda x: x['rank'])
        return valid_cards[0]['idx']

# --- PRE-TRAINING LOOP ---
def pretrain():
    print("--- STARTING BEHAVIOR CLONING (Phase 1.5) ---")
    
    # 1. Setup Env
    def mask_fn(env): return env.unwrapped._get_valid_mask()
    env = CallbreakEnv()
    env = ActionMasker(env, mask_fn)
    
    # 2. Initialize PPO
    model = MaskablePPO(
        "MlpPolicy", 
        env, 
        learning_rate=0.0003,
        verbose=1
    )
    
    policy = model.policy.to("cpu")
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    # 3. Collect Data
    print(f"Generating data from {PRETRAIN_GAMES} expert games...")
    dataset_obs = []
    dataset_acts = []
    
    obs, _ = env.reset()
    total_moves = 0
    max_steps = PRETRAIN_GAMES * 13
    
    while total_moves < max_steps:
        mask = env.action_masks()
        
        # Get Expert Move
        unwrapped = env.unwrapped
        expert_action = pick_expert_move(unwrapped.hands[0], unwrapped.trick_cards, mask)
        
        # --- CRASH FIX: If expert returns None (empty mask), skip step ---
        if expert_action is None:
            obs, _ = env.reset()
            continue
            
        dataset_obs.append(obs)
        dataset_acts.append(expert_action)
        total_moves += 1
        
        obs, reward, done, _, _ = env.step(expert_action)
        if done:
            obs, _ = env.reset()
        
        if total_moves % 5000 == 0:
            print(f"Collected {total_moves} moves...", end='\r')

    # Convert to Tensor
    print(f"\nProcessing {len(dataset_obs)} samples...")
    tensor_obs = torch.tensor(np.array(dataset_obs), dtype=torch.float32)
    tensor_acts = torch.tensor(np.array(dataset_acts), dtype=torch.long)
    
    # 4. Train Loop
    print("Cloning behavior...")
    policy.train()
    
    dataset_size = len(dataset_obs)
    for epoch in range(EPOCHS):
        total_loss = 0
        permutation = torch.randperm(dataset_size)
        
        for i in range(0, dataset_size, BATCH_SIZE):
            indices = permutation[i:i+BATCH_SIZE]
            batch_obs = tensor_obs[indices]
            batch_acts = tensor_acts[indices]
            
            optimizer.zero_grad()
            
            features = policy.extract_features(batch_obs)
            latent_pi = policy.mlp_extractor.forward_actor(features)
            logits = policy.action_net(latent_pi)
            
            loss = loss_fn(logits, batch_acts)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss / (dataset_size/BATCH_SIZE):.4f}")

    # 5. Save
    model.save(MODEL_NAME)
    print(f"Pre-trained model saved to {MODEL_NAME}.zip")
    print("Now run train_phase2_pro.py (it will load this automatically).")

if __name__ == "__main__":
    pretrain()