import gymnasium as gym
import numpy as np
import os
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.callbacks import CheckpointCallback

# Import your environment
from callbreak_env import CallbreakEnv

# --- CONFIGURATION ---
TOTAL_TIMESTEPS = 2_000_000  # Train for 500k steps (approx 10k games)
CHECKPOINT_DIR = "phase2_checkpoints"
MODEL_NAME = "ppo_callbreak_phase2"

# --- MASKING FUNCTION ---
# This tells the PPO algorithm which moves are legal right now.
def mask_fn(env):
    # We access the internal method _get_valid_mask from your environment
    return env.unwrapped._get_valid_mask()

def train():
    print("Setting up Phase 2 Training...")
    
    # 1. Create Environment
    env = CallbreakEnv()
    
    # 2. Wrap it for Action Masking
    # This wrapper automatically fetches legal moves before the AI acts
    env = ActionMasker(env, mask_fn)
    
    # 3. Initialize MaskablePPO
    # MlpPolicy = Multi-Layer Perceptron (standard neural net)
    # ent_coef=0.01 adds curiosity (exploration)
    model = MaskablePPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=3e-4,
        gamma=0.99,
        ent_coef=0.01, # Encourage exploration
        tensorboard_log="./ppo_callbreak_logs/"
    )
    
    # 4. Setup Checkpointing (Save every 50k steps)
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_phase2"
    )

    # 5. Train
    print(f"Training for {TOTAL_TIMESTEPS} timesteps...")
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)
    except KeyboardInterrupt:
        print("\nTraining interrupted manually. Saving current model...")
    
    # 6. Save Final Model
    model.save(MODEL_NAME)
    print(f"Model saved to {MODEL_NAME}.zip")

    # --- TEST THE AGENT ---
    print("\nTesting the trained agent...")
    obs, _ = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        # Predict action using the mask to ensure legality
        action_masks = get_action_masks(env)
        action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
        
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        #env.render() # Uncomment if you implement render in callbreak_env
        
    print(f"Test Game Finished. Total Reward (Tricks Won): {total_reward}")

if __name__ == "__main__":
    train()