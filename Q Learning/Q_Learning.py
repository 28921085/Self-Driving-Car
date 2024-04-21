import numpy as np
from math_tool import MathTool
from self_driving_car import SelfDrivingCar

class Q_Learning:
    def __init__(self, degree_per_actions=4, state=5, learning_rate=0.1, gamma=0.9, exploration_rate=1.0, exploration_decay=0.992):
        self.degree_per_action= degree_per_actions # 一個state代表多少角度
        self.num_actions = 40//degree_per_actions*2+1 #0
        self.num_state=state #一個方向的state
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.q_table = np.zeros((self.num_state, self.num_actions))
        print(self.q_table.shape)
        self.car=SelfDrivingCar()
    def train(self,epochs):
        for epoch in range(epochs):
            self.car=SelfDrivingCar()
            while not(self.car.reach_goal() or self.car.check_collision()):
                Th = self.get_next_Th(self.car.get_distances())
                self.car.update_state(Th)

    def get_next_Th(self, inputs): #input=[前方距離、右方距離、左方距離]
        # 將inputs轉換成狀態
        state = self.direction_to_state(inputs)
        # 根據狀態選擇行動
        action = self.choose_action(state) #action = Th+40 (-40<=Th<=40  => 0<=Th<=80)
        # 進行動作後得到的新狀態和獎勵
        next_state, reward = self.take_action(action)
        # 更新Q-table
        self.update_q_table(state, action, reward, next_state)
        # 返回預測的方向盤角度
        return self.convert_action_to_angle(action)

    def choose_action(self, state):
        if np.random.rand() < self.exploration_rate: #隨機行動
            return np.random.randint(self.num_actions)
        else:
            return np.argmax(self.q_table[state])
        
    def update_q_table(self, state, action, reward, next_state):
        q_value = self.q_table[state][action]
        max_next_q_value = np.max(self.q_table[next_state])
        new_q_value = (1 - self.learning_rate) * q_value + self.learning_rate * (reward + self.gamma * max_next_q_value)
        self.q_table[state][action] = new_q_value


    def direction_to_state(self, inputs):
        inputs = np.round(inputs).astype(int)
        #state =      0           1        2         3          4 
        #        abs(l-r)<3   3<l-r<10   l-r>10   3<r-l<10   r-l>10 
        state=0
        dif = inputs[1]-inputs[2] #r-l
        if dif < 0:
            if -dif > 3 and -dif < 10:
                state=1
            elif -dif >= 10:
                state=2
        else:
            if dif > 3 and dif < 10:
                state=3
            elif dif >= 10:
                state=4
        return state
    def rew(self,distance):
        return max(6.5-distance,0)**2*1000
    def take_action(self, action):
        self.car.update_state(self.convert_action_to_angle(action))
        next_input=self.car.get_distances()
        next_state = self.direction_to_state(next_input)
        reward=-5
        if self.car.reach_goal():
            reward+=1000000
        elif self.car.check_collision():
            reward=-200
        p=(self.rew(next_input[0])+self.rew(next_input[1])+self.rew(next_input[2]))
        reward-=p #太貼牆會大幅減少reward
        q=10*(400 - MathTool.point_to_polygon_distance(self.car.x,self.car.y,self.car.end_area)**2) #離終點越近 reward越高
        reward+= q
        #print("reward:",reward,"wall:",p,"distance:",q)
        return next_state, reward

    def convert_action_to_angle(self, action):
        #action 0  1-10  11-20
        # angle 0  -4~40  4~40
        if action == 0:
            return 0
        elif action < 11:
            return -self.degree_per_action*(action)
        #11<=action<=20
        return self.degree_per_action*(action-10)

# 測試
#if __name__ == "__main__":
#    q_learning = Q_Learning()
#    #inputs = [3.2, 4.5, 2.8]  # 假設輸入的距離
#    #next_angle = q_learning.get_next_Th(inputs)
#    #print("Next Steering Angle:", next_angle)
#    q_learning.train(100)

    


