# ==========================
# LAB 10 : Q-Learning
# ==========================

import numpy as np

# Grid Size
rows = 4
cols = 4

# Q Table
Q = np.zeros((rows, cols, 4))

# Actions
actions = ["Up", "Down", "Left", "Right"]

# Learning Rate
alpha = 0.5

# Discount Factor
gamma = 0.9

# Number of Episodes
episodes = 1000

# Start, Goal and Wall
start = (3,0)
goal = (0,3)
wall = (1,1)

# --------------------------
# Step Function
# --------------------------

def move(state, action):

    r,c = state

    if action == 0:      # Up
        r = max(r-1,0)

    elif action == 1:    # Down
        r = min(r+1,3)

    elif action == 2:    # Left
        c = max(c-1,0)

    else:                # Right
        c = min(c+1,3)

    new_state = (r,c)

    if new_state == wall:
        reward = -1

    elif new_state == goal:
        reward = 10

    else:
        reward = 0

    return new_state,reward

# --------------------------
# Training
# --------------------------

for i in range(episodes):

    state = start

    while state != goal:

        action = np.random.randint(4)

        next_state,reward = move(state,action)

        r,c = state
        nr,nc = next_state

        old = Q[r,c,action]

        future = np.max(Q[nr,nc])

        Q[r,c,action] = old + alpha*(reward + gamma*future - old)

        state = next_state

# --------------------------
# Display Q Table
# --------------------------

print("Q Table\n")

print(Q)