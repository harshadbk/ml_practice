import numpy as np

rows = 4
cols = 4

Q = np.zeros((rows,cols,4))

print(Q)

actions = ["UP","DOWN","LEFT","RIGHT"]

a = 0.5
g = 0.9

episodes = 1000

start = (3,0)
wall = (1,1)
goal = (0,3)

# step function

def move(state,action):
    r , c = state

    if action==0:
        r = max(r-1,0)

    if action==1:
        r = min(r+1,3)
    
    if action==2:
        c = max(c-1,0)

    if action==3:
        c = min(c+1,3)

    new_state = (r,c)

    if new_state==wall:
        reward = -1
    
    elif new_state==goal:
        reward = 10

    else :
        reward = 0

    return new_state,reward


# now learning

for i in range(episodes):
    state = start

    while state != goal:
        action = np.random.randint(4)

        next_state , reward = move(state,action)

        r ,c = state

        nr,nc = next_state

        old = Q[r,c,action]

        future = np.max(Q[nr,nc])

        Q[r,c,action] = old + a *(reward+g*future-old)

        state = next_state

print(Q)


