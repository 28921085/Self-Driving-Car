import numpy as np


class Q_Learning:
    def __init__(self, num_actions, num_states, learning_rate=0.1, gamma=0.9, exploration_rate=1.0, exploration_decay=0.99):
        self.num_actions = num_actions
        self.num_states = num_states
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.q_table = np.zeros((num_states, num_actions))


    def get_next_Th(self, inputs): #input=[前方距離、右方距離、左方距離]
        # 將inputs轉換成狀態
        state = self.to_state(inputs)
        # 根據狀態選擇行動
        action = self.choose_action(state)
        #print(self.q_table)
        # 進行動作後得到的新狀態和獎勵
        next_state, reward = self.take_action(state, action)
        # 更新Q-table
        self.update_q_table(state, action, reward, next_state)
        # 更新探索率
        self.exploration_rate *= self.exploration_decay
        # 返回預測的方向盤角度
        return self.convert_action_to_angle(action)

    def choose_action(self, state):
        if np.random.rand() < self.exploration_rate: #隨機行動
            return np.random.randint(self.num_actions)
        else:
            return np.argmax(self.q_table[state, :])

    def update_q_table(self, state, action, reward, next_state):
        q_value = self.q_table[state][action]
        max_next_q_value = np.max(self.q_table[next_state, :])
        new_q_value = (1 - self.learning_rate) * q_value + self.learning_rate * (reward + self.gamma * max_next_q_value)
        self.q_table[state][action] = new_q_value


    def to_state(self, inputs):
        # 根據inputs計算狀態，這裡需要根據你的自走車環境實現
        # 一個簡單的示例是將inputs四捨五入到最接近的整數作為狀態
        state = np.round(inputs).astype(int)
        return state

    def take_action(self, state, action):
        # 根據狀態和動作模擬執行動作後的新狀態和獎勵，這裡需要根據你的自走車環境實現
        # 這裡提供一個假設的示例
        next_state = tuple(np.random.randint(0, 10, size=self.num_states))  # 假設下一個狀態是隨機生成的
        reward = np.random.randint(-10, 10)  # 假設獎勵是隨機生成的
        return next_state, reward

    def convert_action_to_angle(self, action):
        # 將動作轉換為方向盤角度，這裡需要根據你的自走車環境實現
        # 一個簡單的示例是將動作映射到指定的角度範圍
        angle = (action / (self.num_actions - 1)) * 80 - 40  # 假設動作空間是等分的，範圍是[-40, 40]
        return angle

# 測試
if __name__ == "__main__":
    q_learning = Q_Learning(num_actions=5, num_states=3)
    inputs = [3.2, 4.5, 2.8]  # 假設輸入的距離
    next_angle = q_learning.get_next_Th(inputs)
    print("Next Steering Angle:", next_angle)

    


