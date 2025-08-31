import numpy as np

gamma = 0.9

# Tek state, 2 aksiyon
# R[s, a]
R = np.array([[1, 2]])  # A1=1 ödül, A2=2 ödül

# P[s, a, s'] - burada hep aynı state'e dönüyor (%100)
P = np.array([[[1.0], [1.0]]])

def policy_evaluation(policy, R, P, gamma=0.9, theta=1e-6):
    V = np.zeros(1)  # tek state
    while True:
        delta = 0
        for s in range(1):  # tek state
            a = policy[s]
            v = V[s]
            V[s] = R[s, a] + gamma * np.sum(P[s, a] * V)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V

def policy_improvement(V, R, P, gamma=0.9):
    policy = np.zeros(1, dtype=int)
    for s in range(1):
        q_values = []
        for a in range(2):
            q = R[s, a] + gamma * np.sum(P[s, a] * V)
            q_values.append(q)
        policy[s] = np.argmax(q_values)
    return policy

# Başlangıç: kötü policy (hep A1 seçiyor)
policy = np.array([0])  
V0 = policy_evaluation(policy, R, P, gamma)
print("Başlangıç policy:", policy, "Value:", V0)

# Policy Improvement
new_policy = policy_improvement(V0, R, P, gamma)
V1 = policy_evaluation(new_policy, R, P, gamma)
print("İyileştirilmiş policy:", new_policy, "Value:", V1)

# Karşılaştırma
print("Monotonic Improvement sağlandı mı?", V1[0] >= V0[0])
