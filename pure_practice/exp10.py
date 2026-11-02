import numpy as np
import random

random.seed(42)

r = 4
c = 4

Q = np.zeros((r,c,4))

actions = ["UP","DOWN","LEFT","RIGHT"]

goal = (0,3)
wall = (1,1)
start = (3,0)

print(Q)

a = 0.5 # learning rate
g = 0.9 # future reward importance

episodes = 1000

def move(state,action):
    r,c =  state

    if action==0:
        r = max(0,r-1)

    elif action == 1:
        r = min(3,r+1)

    elif action == 2:
        c = max(0,c-1)

    elif action == 3:
        c = min(3,c+1)

    new_state = (r,c)

    if new_state==goal:
        reward = 10
    
    elif new_state==wall:
        reward = -1

    else:
        reward = 0

    return new_state,reward

# training 

for i in range(episodes):
    state = start

    while state != goal:
        action = np.random.randint(4)

        new_state,reward = move(state,action)

        r , c = state
        nr,nc = new_state

        old = Q[r,c,action]

        future = np.max(Q[nr,nc])

        Q[r,c,action] = old+a*(reward+g*future-old)

        state = new_state

print(Q)

       
