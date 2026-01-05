import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker # <--- Added this import
from callbreak_env import CallbreakEnv

# CONFIG
MODEL_PATH = "ppo_callbreak_pro.zip"
NUM_GAMES = 10000

# --- HELPER FUNCTION ---
def mask_fn(env):
    return env.unwrapped._get_valid_mask()

def evaluate():
    print(f"Loading {MODEL_PATH}...")
    try:
        model = MaskablePPO.load(MODEL_PATH)
    except:
        print("Model not found. Train it first!")
        return

    # 1. Create Environment
    env = CallbreakEnv()
    
    # 2. WRAP IT (Crucial Fix)
    # This adds the .action_masks() method the library looks for
    env = ActionMasker(env, mask_fn)

    print(f"Simulating {NUM_GAMES} games against Random Bots...")
    
    scores = []
    
    for i in range(NUM_GAMES):
        obs, _ = env.reset()
        done = False
        tricks = 0
        
        while not done:
            # Now this works because env is wrapped
            action_masks = get_action_masks(env)
            
            # Predict best move
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            
            obs, reward, done, _, _ = env.step(action)
            
            if reward > 0:
                tricks += 1
        
        scores.append(tricks)
        
        if (i+1) % 100 == 0:
            print(f"Game {i+1} | Avg Tricks: {np.mean(scores):.2f}")

    print(f"\nFINAL RESULT over {NUM_GAMES} games:")
    print(f"Average Tricks Won: {np.mean(scores):.2f}")
    
    if np.mean(scores) > 3.25:
        print("Verdict: Stronger than Random! (Random is ~3.25)")
    else:
        print("Verdict: Playing at or below random level. Keep training.")

if __name__ == "__main__":
    evaluate()