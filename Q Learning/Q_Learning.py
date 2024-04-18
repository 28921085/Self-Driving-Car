import numpy as np
from math_tool import MathTool
from self_driving_car import SelfDrivingCar

class Q_Learning:
    def __init__(self, num_actions=81, state=5, learning_rate=0.1, gamma=0.9, exploration_rate=1.0, exploration_decay=0.99):
        self.num_actions = num_actions
        self.num_state=state #一個方向的state
        self.num_3_states = state**3
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.q_table = np.zeros((self.num_3_states, num_actions))
        self.car=SelfDrivingCar()
    def train(self,epochs):
        for epoch in range(epochs):
            print("epoch:",epoch+1)
            self.car=SelfDrivingCar()
            while not(self.car.reach_goal() or self.car.check_collision()):
                Th = self.get_next_Th(self.car.get_distances())
                print(self.car)
                print(self.car.x,self.car.y)
                self.car.update_state(Th)
                print(self.car.x,self.car.y)

    def get_next_Th(self, inputs): 
        # 將inputs轉換成狀態
        state = self.direction_to_state(inputs)
        #print("direction:",inputs,"  state:",state)
        # 根據狀態選擇行動
        action = self.choose_action(state) #action = Th+40 (-40<=Th<=40  => 0<=Th<=80)
        #print("next action:",action,"next angle:",action-40)
        # 進行動作後得到的新狀態和獎勵
        next_state, reward = self.take_action(action)
        # 更新Q-table
        self.update_q_table(state, action, reward, next_state)
        # 更新探索率
        self.exploration_rate *= self.exploration_decay
        #print("next Th:",self.convert_action_to_angle(action))
        # 返回預測的方向盤角度
        return self.convert_action_to_angle(action)
    def state_to_direction(self,state): #decode hash
        res=[]
        for _ in range(3):
            res.append(state%self.num_state)
            state//=self.num_state
        res = res[::-1] #reverse
        return res

    def choose_action(self, state):
        if np.random.rand() < self.exploration_rate: #隨機行動
            #print("rand")
            return np.random.randint(self.num_actions)
        else:
            return np.argmax(self.q_table[state])
        
    def update_q_table(self, state, action, reward, next_state):
        q_value = self.q_table[state][action]
        #print(next_state)
        #print(self.state_to_direction(next_state))
        max_next_q_value = np.max(self.q_table[next_state])
        new_q_value = (1 - self.learning_rate) * q_value + self.learning_rate * (reward + self.gamma * max_next_q_value)
        self.q_table[state][action] = new_q_value


    def direction_to_state(self, inputs):
        inputs = np.round(inputs).astype(int)
        #input=[前方距離、右方距離、左方距離]
        #hash func: state = 前*state^2+右*state+左
        #inputs //= 5
        #inputs = np.minimum(inputs,4) 
        for i in range(3):
            if inputs[i]<0:
                inputs[i]=4
            else:
                inputs[i]=min(4,inputs[i]//5)
        state = inputs[0]*(self.num_state**2)+inputs[1]*self.num_state+inputs[2]
        return state

    def take_action(self, action):
        self.car.update_state(action-40)
        next_state = self.direction_to_state(self.car.get_distances())
        reward=-10
        if self.car.reach_goal():
            reward=10000
        elif self.car.check_collision():
            reward=-10000
        return next_state, reward

    def convert_action_to_angle(self, action):
        # 將動作轉換為方向盤角度，這裡需要根據你的自走車環境實現
        # 一個簡單的示例是將動作映射到指定的角度範圍
        #angle = (action / (self.num_actions - 1)) * 80 - 40  # 假設動作空間是等分的，範圍是[-40, 40]
        return action-40

# 測試
if __name__ == "__main__":
    q_learning = Q_Learning()
    #inputs = [3.2, 4.5, 2.8]  # 假設輸入的距離
    #next_angle = q_learning.get_next_Th(inputs)
    #print("Next Steering Angle:", next_angle)
    q_learning.train(5)

    


