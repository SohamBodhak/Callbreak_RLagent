import gymnasium as gym
import numpy as np
import os
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

# Import your environment
from callbreak_env import CallbreakEnv

# --- CONFIGURATION ---
TOTAL_TIMESTEPS = 30_000_000  # 3 Million Steps (approx 60k games)
CHECKPOINT_DIR = "phase2_checkpoints"
LOG_DIR = "./ppo_callbreak_logs/"
#MODEL_NAME = "ppo_callbreak_pro"
MODEL_NAME = "ppo_callbreak_zero_pro"

def mask_fn(env):
    return env.unwrapped._get_valid_mask()

def make_env():
    env = CallbreakEnv()
    env = ActionMasker(env, mask_fn)
    env = Monitor(env) # Helps track stats like avg reward
    return env

def train():
    print("--- STARTING PRO TRAINING (3M Steps) ---")
    
    # 1. Setup Environment
    env = make_env()
    
    # 2. Setup Evaluation Environment (To test while training)
    eval_env = make_env()
    
    # 3. Callbacks
    # Save a checkpoint every 100k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_pro"
    )
    
    # Evaluate every 50k steps to see if we are improving
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./best_model/',
        log_path='./results/',
        eval_freq=50_000,
        deterministic=True,
        render=False
    )

    # NEW:
    print("Loading Pre-Trained Expert Model...")
    try:
        model = MaskablePPO.load("ppo_callbreak_zero.zip", env=env)
        
        # IMPORTANT: We must update the hyperparameters for RL 
        # (The loaded model keeps old settings, we want RL settings)
        model.learning_rate = 0.003
        model.ent_coef = 0.01  # Lower entropy now, we have knowledge
        model.n_steps = 2048
        model.tensorboard_log = LOG_DIR
        
    except FileNotFoundError:
        print("Pre-trained model not found! Run pretrain_sb3.py first.")
        return

    # Train
    print(f"Refining with RL for {TOTAL_TIMESTEPS} steps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[checkpoint_callback, eval_callback])
    
    # 6. Save Final
    model.save(MODEL_NAME)
    print(f"Training Complete. Model saved to {MODEL_NAME}.zip")

if __name__ == "__main__":
    train()