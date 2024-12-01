import numpy as np

class NSFWEnv:
    def __init__(self, data):
        self.data = data  # [(text, label), ...]
        self.index = 0

    def reset(self):
        self.index = 0
        return self.data[self.index][0]

    def step(self, action):
        text, label = self.data[self.index]
        reward = 1 if action == label else -1
        done = self.index == len(self.data) - 1
        self.index += 1
        next_state = self.data[self.index][0] if not done else None
        return next_state, reward, done, {}

class QLearningAgent:
    def __init__(self):
        self.q_table = {}
        self.actions = [0, 1]  # 0: 正常, 1: NSFW

    def choose_action(self, state):
        # 简化处理，实际应用中需要将文本状态映射到特征空间
        state_key = hash(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(len(self.actions))
        return np.argmax(self.q_table[state_key])

    def learn(self, state, action, reward, next_state):
        state_key = hash(state)
        next_state_key = hash(next_state) if next_state else None
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(len(self.actions))
        q_predict = self.q_table[state_key][action]
        q_target = reward + 0.9 * np.max(self.q_table[next_state_key]) if next_state else reward
        self.q_table[state_key][action] += 0.1 * (q_target - q_predict)

# 使用示例
if __name__ == "__main__":
    # 构造训练数据
    data = [
        ("正常的文本内容。", 0),
        ("包含不良信息的文本。", 1),
        # 更多数据
    ]
    env = NSFWEnv(data)
    agent = QLearningAgent()

    for episode in range(10):
        state = env.reset()
        while True:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            if done:
                break

    # 测试
    test_text = "待检测的文本内容。"
    action = agent.choose_action(test_text)
    if action == 1:
        print("检测到NSFW内容。")
    else:
        print("内容正常。")
