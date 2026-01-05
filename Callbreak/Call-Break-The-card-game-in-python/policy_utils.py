import torch
import torch.nn.functional as F

def get_rollout_action(model, state_vector, valid_moves_mask):
    """
    Fast sampling of an action for the rollout phase.
    
    Args:
        model: The trained FastRolloutNetwork instance.
        state_vector: Tensor of shape (104,) representing Hand + Board.
        valid_moves_mask: Binary Tensor of shape (52,) where 1=Valid, 0=Invalid.
                          (You must calculate this mask using your game logic)
        
    Returns:
        int: The index (0-51) of the card to play.
    """
    # Ensure model is in eval mode (disables dropout, etc.)
    model.eval()
    
    with torch.no_grad():
        # 1. Get raw output (Logits) from the network
        logits = model(state_vector) 
        
        # 2. Mask Invalid Moves 
        # We set the probability of illegal moves to negative infinity.
        # This ensures the Softmax result for them is exactly 0.
        logits[valid_moves_mask == 0] = -float('inf')
        
        # 3. Apply Softmax to turn logits into probabilities (0.0 to 1.0)
        probs = F.softmax(logits, dim=0)
        
        # 4. Sample from the distribution
        # multinomial picks an index based on the probabilities provided.
        # This gives us the "weighted random" behavior needed for rollouts.
        action_idx = torch.multinomial(probs, num_samples=1)
        
        return action_idx.item()