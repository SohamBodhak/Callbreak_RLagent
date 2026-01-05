import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import os
import time
from callbreak_env import CallbreakEnv

# --- CONFIGURATION ---
TOTAL_TIMESTEPS = 30_000_000
CHECKPOINT_DIR = "phase2_checkpoints_a3c"
MODEL_NAME = "a3c_callbreak_final.pth"
NUM_WORKERS = 4  # Number of parallel threads
SAVE_INTERVAL = 100_000
LOG_INTERVAL = 1000  # Episodes

# Create dirs
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- NEURAL NETWORK (Actor-Critic) ---
class SharedAdam(torch.optim.Adam):
    """Extends Adam optimizer to share moments across processes"""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.l1 = nn.Linear(input_dim, 256)
        self.l2 = nn.Linear(256, 128)
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.actor(x), self.critic(x)

# --- WORKER PROCESS ---
def worker(global_model, optimizer, global_ep, global_steps, lock, name):
    env = CallbreakEnv()
    local_model = ActorCritic(156, 52)
    local_model.train()
    
    step_count = 0
    
    while global_steps.value < TOTAL_TIMESTEPS:
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        # Buffers
        states, actions, rewards = [], [], []
        
        while not done:
            step_count += 1
            
            # Sync with global
            local_model.load_state_dict(global_model.state_dict())
            
            # Action Selection
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            logits, _ = local_model(state_tensor)
            
            # Manual Masking (A3C advantage over standard A2C)
            mask = torch.FloatTensor(env._get_valid_mask()).unsqueeze(0)
            logits[mask == 0] = -1e9
            
            probs = F.softmax(logits, dim=1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            
            # Step
            next_state, reward, done, _, _ = env.step(action.item())
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            state = next_state
            total_reward += reward

        # Update Global Steps
        with lock:
            global_steps.value += len(rewards)
            curr_steps = global_steps.value

        # --- Calculate Loss & Backprop ---
        R = 0
        loss = 0
        optimizer.zero_grad()
        
        for i in reversed(range(len(rewards))):
            R = 0.99 * R + rewards[i]
            s_t = torch.FloatTensor(states[i]).unsqueeze(0)
            a_t = torch.tensor([actions[i]])
            
            logits, value = local_model(s_t)
            mask = torch.FloatTensor(env._get_valid_mask()).unsqueeze(0) # Re-calculate mask for history
            logits[mask == 0] = -1e9
            
            probs = F.softmax(logits, dim=1)
            dist = torch.distributions.Categorical(probs)
            log_prob = dist.log_prob(a_t)
            entropy = dist.entropy()
            
            advantage = R - value.item()
            
            # A3C Loss
            loss = loss - (log_prob * advantage) + F.smooth_l1_loss(value, torch.tensor([[R]])) - 0.01 * entropy

        loss.backward()
        
        # Gradient Sharing
        with lock:
            for lp, gp in zip(local_model.parameters(), global_model.parameters()):
                gp._grad = lp.grad
            optimizer.step()
        
        # Logging & Saving (Mimicking Callbacks)
        with lock:
            global_ep.value += 1
            if global_ep.value % LOG_INTERVAL == 0:
                print(f"Worker {name} | Ep {global_ep.value} | Steps {curr_steps} | Reward {total_reward:.2f}")
            
            if curr_steps % SAVE_INTERVAL < len(rewards): # Approx check
                save_path = f"{CHECKPOINT_DIR}/a3c_model_{curr_steps}_steps.pth"
                torch.save(global_model.state_dict(), save_path)

def train():
    print("--- STARTING A3C TRAINING (Custom PyTorch) ---")
    mp.set_start_method('spawn', force=True)
    
    global_model = ActorCritic(156, 52)
    global_model.share_memory()
    
    optimizer = SharedAdam(global_model.parameters(), lr=0.003)
    
    global_ep = mp.Value('i', 0)
    global_steps = mp.Value('i', 0)
    lock = mp.Lock()
    
    workers = [mp.Process(target=worker, args=(global_model, optimizer, global_ep, global_steps, lock, i)) for i in range(NUM_WORKERS)]
    
    print(f"Launched {NUM_WORKERS} Workers. Training...")
    [w.start() for w in workers]
    [w.join() for w in workers]
    
    torch.save(global_model.state_dict(), MODEL_NAME)
    print(f"Training Complete. Saved to {MODEL_NAME}")

if __name__ == "__main__":
    train()