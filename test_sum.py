import numpy as np

def calculate_discounted_rewards(rewards, discount_factor, n_steps):
    discount_array = np.array([discount_factor ** i for i in range(n_steps)])
    discounted_rewards = np.sum(rewards * discount_array, axis=1)
    return discounted_rewards

# Test the function
if __name__ == "__main__":
    rewards = np.array([[1, 2, 3], [4, 5, 6]])
    discount_factor = 0.9
    n_steps = rewards.shape[-1]
    
    result = calculate_discounted_rewards(rewards, discount_factor, n_steps)
    print("Discounted Rewards:", result)

    dones = [[False, True], [False, False]]
    dones = np.array([0 if i[-1] else 1 for i in dones])
    print("Dones:", dones)

    rewards = np.array([[1, 2, 3], [4, 5, 6]])
    indecies = [1, 1]
    print("Rewards:", rewards[np.arange(len(rewards)), indecies])  