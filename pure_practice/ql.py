import numpy as np
import random

random.seed(42)

r = 4
c = 4

Q = np.zeros((r,c,4))
episodes = 1000

a = 0.50
g = 0.90

actions = ["up","down","left","right"]

print(Q)

st = (3,0)

wall = (1,1)

end = (0,3)

def move(action , state):
    r , c = state

    if action==0:
        r = max(r-1,0)

    elif action==1:
        r = min(r+1,3)

    elif action==2:
        c = max(c-1,0)

    elif action==3:
        c = min(c+1,3)

    new_state = (r,c)

    if new_state==wall:
        reward = -1
    
    elif new_state==end:
        reward = 10

    else :
        reward = 0

    return new_state,reward

# training

for i in range(episodes):

    state = st

    while state != end:
        action = np.random.randint(4)

        new_state , reward = move(action,state)

        r,c = state
        nr , nc = new_state

        old = Q[r,c,action]

        future = np.max(Q[nr,nc])

        Q[r,c,action] = old + a *(reward+g*future-old)

        state = new_state

print(Q)