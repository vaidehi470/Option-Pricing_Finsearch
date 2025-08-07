#!/usr/bin/env python
# coding: utf-8

# In[11]:


import random

def simulate_random_walk(num_steps=10, num_simulations=10000):
    break_even_count = 0

    for _ in range(num_simulations):
        balance = 0
        for _ in range(num_steps):
            if random.random() < 0.5:
                balance += 1  # Heads → Win ₹1
            else:
                balance -= 1  # Tails → Lose ₹1

        if balance == 0:
            break_even_count += 1

    probability = break_even_count / num_simulations
    return probability

# Run simulation
steps = 100
simulations = 100000
print(f"Probability of breaking even after {steps} coin tosses: {simulate_random_walk(steps, simulations):.4f}")


# In[ ]:





# In[ ]:




