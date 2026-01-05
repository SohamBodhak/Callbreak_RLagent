import gymnasium as gym
import numpy as np
import os
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

# Import your environment
from callbreak_env import CallbreakEnv

# --- CONFIGURATION ---
TOTAL_TIMESTEPS = 30_000_000  # 30 Million Steps
CHECKPOINT_DIR = "phase2_checkpoints_a2c"
LOG_DIR = "./a2c_callbreak_logs/"
MODEL_NAME = "a2c_callbreak_final"

# A2C benefits from parallel environments
NUM_CPU = 4 

def make_env():
    """Utility function for multiprocessed env"""
    def _init():
        env = CallbreakEnv()
        env = Monitor(env) # Helps track stats like avg reward
        return env
    return _init

def train():
    print(f"--- STARTING A2C TRAINING ({TOTAL_TIMESTEPS} Steps) ---")
    
    # 1. Setup Parallel Environments (A2C is faster with this)
    env = SubprocVecEnv([make_env() for _ in range(NUM_CPU)])
    
    # 2. Setup Evaluation Environment
    eval_env = CallbreakEnv()
    eval_env = Monitor(eval_env)
    
    # 3. Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000 // NUM_CPU,
        save_path=CHECKPOINT_DIR,
        name_prefix="a2c_model"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./best_model_a2c/',
        log_path='./results_a2c/',
        eval_freq=50_000 // NUM_CPU,
        deterministic=True,
        render=False
    )

    # 4. Initialize Model
    # Note: We cannot easily load the PPO weights into A2C because 
    # the internal structures differ. We train A2C from scratch.
    print("Initializing A2C Policy...")
    model = A2C(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        learning_rate=0.003, # A2C usually needs higher LR than PPO
        ent_coef=0.01,
        gamma=0.99,
        gae_lambda=0.95,
        n_steps=5,            # A2C updates frequently
    )

    # 5. Train
    print(f"Training A2C for {TOTAL_TIMESTEPS} steps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[checkpoint_callback, eval_callback])
    
    # 6. Save Final
    model.save(MODEL_NAME)
    print(f"A2C Training Complete. Model saved to {MODEL_NAME}.zip")

if __name__ == "__main__":
    train()