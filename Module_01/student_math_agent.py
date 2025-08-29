import numpy as np
import random
import matplotlib.pyplot as plt

class Student:
    def __init__(self):
        # Başlangıç doğruluk olasılıkları
        self.skill_add = 0.2   # toplama
        self.skill_sub = 0.1   # çıkarma
        self.min_skill = 0.0
        self.max_skill = 1.0
        self.learn_rate = 0.02
        self.forget_rate = 0.01
    
    def answer(self, problem_type):
        if problem_type == "add":
            correct = np.random.random() < self.skill_add
            # öğrenme: doğru yaparsa biraz gelişsin
            if correct:
                self.skill_add = min(self.max_skill, self.skill_add + self.learn_rate)
            else:
                self.skill_add = max(self.min_skill, self.skill_add - self.forget_rate)
            return correct
        elif problem_type == "sub":  # Explicit check for "sub"
            correct = np.random.random() < self.skill_sub
            if correct:
                self.skill_sub = min(self.max_skill, self.skill_sub + self.learn_rate)
            else:
                self.skill_sub = max(self.min_skill, self.skill_sub - self.forget_rate)
            return correct
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")

class TutorAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.95, epsilon=0.2):
        self.q_values = {a: 0.0 for a in actions}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.last_action = None
        self.actions = actions

    def choose_action(self):
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            # Handle ties randomly
            max_q = max(self.q_values.values())
            best_actions = [a for a, q in self.q_values.items() if q == max_q]
            action = random.choice(best_actions)
        self.last_action = action
        return action

    def update(self, reward):
        if self.last_action is None:
            return
        action = self.last_action
        best_q = max(self.q_values.values())
        self.q_values[action] += self.alpha * (reward + self.gamma * best_q - self.q_values[action])

def run_simulation(n_episodes=500):
    student = Student()
    agent = TutorAgent(actions=["add", "sub"])
    
    rewards = []
    try:
        for _ in range(n_episodes):
            action = agent.choose_action()
            correct = student.answer(action)
            reward = 1 if correct else -1
            agent.update(reward)
            rewards.append(reward)
            
        # Plotting
        window = min(50, len(rewards))
        avg_rewards = [np.mean(rewards[max(0, i-window):i]) for i in range(1, len(rewards)+1)]
        
        plt.figure(figsize=(10, 6))
        plt.plot(avg_rewards)
        plt.xlabel("Episode")
        plt.ylabel(f"Average Reward (window={window})")
        plt.title("AI Tutor Learning Curve")
        plt.grid(True)
        plt.show()

        print("\nFinal Results:")
        print(f"Addition skill: {student.skill_add:.3f}")
        print(f"Subtraction skill: {student.skill_sub:.3f}")
        print(f"Tutor Q-values: {dict((k, round(v, 3)) for k, v in agent.q_values.items())}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    run_simulation()
