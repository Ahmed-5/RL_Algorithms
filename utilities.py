import torch

def compute_td_returns(rewards, gamma, td_lambda):
    """
    Compute TD(λ) returns efficiently.
    
    Args:
        rewards: Tensor of rewards for an episode
        gamma: Discount factor
        td_lambda: TD(λ) parameter
        
    Returns:
        Tensor of TD(λ) returns
    """
    n = len(rewards)
    if n == 0:
        return torch.tensor([], device=rewards.device)
    
    # Pre-calculate discount factors
    discount_factors = (gamma * td_lambda) ** torch.arange(n, dtype=torch.float32, device=rewards.device)
    discount_factors *= (1 - td_lambda)
    discount_factors[-1] += td_lambda ** n
    
    # Create weight matrix using triu indices
    indices = torch.triu_indices(n, n, device=rewards.device)
    values = torch.ones(indices.shape[1], device=rewards.device)
    
    for i in range(indices.shape[1]):
        row, col = indices[0, i], indices[1, i]
        values[i] = discount_factors[col] / discount_factors[row]
    
    weight_matrix = torch.sparse_coo_tensor(indices, values, (n, n)).to_dense()
    returns = torch.matmul(weight_matrix, rewards)
    
    return returns