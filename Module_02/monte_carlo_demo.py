import numpy as np
import random
import matplotlib.pyplot as plt

GRID_SIZE = 4
ACTIONS = ["U", "D", "L", "R"]
ACTION_IDX = {a: i for i, a in enumerate(ACTIONS)}

def step(state, action):
    x, y = state
    if action == "U":
        x = max(0, x-1)
    elif action == "D":
        x = min(GRID_SIZE-1, x+1)
    elif action == "L":
        y = max(0, y-1)
    elif action == "R":
        y = min(GRID_SIZE-1, y+1)
    next_state = (x,y)
    reward = 1 if next_state == (GRID_SIZE-1, GRID_SIZE-1) else 0
    done = (next_state == (GRID_SIZE-1, GRID_SIZE-1))
    return next_state, reward, done

class MCAgent:
    def __init__(self, epsilon=0.2, alpha=0.1, gamma=0.9):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.Q = {}
    
    def get_Q(self, state, action):
        return self.Q.get((state, action), 0.0)
    
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)
        q_values = [self.get_Q(state, a) for a in ACTIONS]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(ACTIONS, q_values) if q == max_q]
        return random.choice(best_actions)
    
    def generate_episode(self):
        episode = []
        state = (0,0)
        done = False
        while not done:
            action = self.choose_action(state)
            next_state, reward, done = step(state, action)
            episode.append((state, action, reward))
            state = next_state
        return episode
    
    def update_Q(self, episode):
        G = 0
        visited = set()
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = self.gamma * G + reward
            if (state, action) not in visited:
                old_q = self.get_Q(state, action)
                self.Q[(state, action)] = old_q + self.alpha * (G - old_q)
                visited.add((state, action))

def extract_policy(agent):
    policy = {}
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            state = (x,y)
            if state == (GRID_SIZE-1, GRID_SIZE-1):  # goal
                policy[state] = "G"
                continue
            q_values = [agent.get_Q(state,a) for a in ACTIONS]
            best_a = ACTIONS[np.argmax(q_values)]
            policy[state] = best_a
    return policy

def plot_policy(policy, title="Policy"):
    fig, ax = plt.subplots()
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_xticks(np.arange(0, GRID_SIZE+1))
    ax.set_yticks(np.arange(0, GRID_SIZE+1))
    ax.grid(True)
    ax.set_title(title)

    for (x,y), action in policy.items():
        if action == "G":
            ax.text(y+0.5, GRID_SIZE-x-0.5, "G", ha="center", va="center", fontsize=16, color="green")
        else:
            arrows = {"U":"↑", "D":"↓", "L":"←", "R":"→"}
            ax.text(y+0.5, GRID_SIZE-x-0.5, arrows[action], ha="center", va="center", fontsize=14, color="red")
    plt.show()

# --- Eğitim ---
agent = MCAgent()
episodes = 50000
checkpoints = [5, 500, 2500, 50000]

for i in range(1, episodes+1):
    ep = agent.generate_episode()
    agent.update_Q(ep)
    if i in checkpoints:
        policy = extract_policy(agent)
        plot_policy(policy, title=f"Policy after {i} episodes")
