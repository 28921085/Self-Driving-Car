import matplotlib.pyplot as plt
import numpy as np

def rew(distance):
    return -max(6.5 - distance, 0)**2 * 1000

# 生成x的值
x_values = np.linspace(0, 7, 1000)
# 计算y的值
y_values = [rew(x) for x in x_values]

# 绘制图形
plt.plot(x_values, y_values)
plt.xlabel('Distance')
plt.ylabel('Reward')
plt.title('Reward Function')
plt.grid(True)
plt.show()
#import numpy as np
#import matplotlib.pyplot as plt
#from math_tool import MathTool
#from self_driving_car import SelfDrivingCar
#
#class Q_Learning:
#    def __init__(self):
#        self.car = SelfDrivingCar()
#        self.track_points = [[-6.0, -3.0], [-6.0, 22.0], [18.0, 22.0], [18.0, 50.0], [30.0, 50.0], [30.0, 10.0], [6.0, 10.0], [6.0, -3.0], [-6.0, -3.0]]
#
#    def move_car(self, dx, dy):
#        self.car.x += dx
#        self.car.y += dy
#
#    def rotate_car(self, angle):
#        self.car.F += angle
#
#    def rew(self, distance):
#        return max(6.5 - distance, 0)**2 * 1000
#
#    def take_action(self):
#        next_input = self.car.get_distances()
#        reward = -5
#        if self.car.reach_goal():
#            reward += 1000000
#        elif self.car.check_collision():
#            reward = -200
#        p = (self.rew(next_input[0]) + self.rew(next_input[1]) + self.rew(next_input[2]))
#        reward -= p
#        q = 10 * (400 - MathTool.point_to_polygon_distance(self.car.x, self.car.y, self.car.end_area)**2)
#        reward += q
#        return reward, next_input
#
#    def generate_data(self):
#        # 隨機產生 car 的 x, y, F
#        # 生成隨機座標點，直到找到位於多邊形內部的點
#        # 找出多邊形的邊界框
#        min_x, min_y = np.min(self.track_points, axis=0)
#        max_x, max_y = np.max(self.track_points, axis=0)
#        while True:
#            random_point = [np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)]
#            if MathTool.is_inside_polygon(random_point, self.track_points):
#                break
#        self.car.F = np.random.uniform(0, 360)  # 假設範圍是 0 到 360
#
#        return self.take_action()  # 返回 reward 和 next_input
#
#    def plot_scatter(self, num_samples=100):
#        rewards = []
#        distances = []
#
#        for _ in range(num_samples):
#            reward, next_input = self.generate_data()
#            rewards.append(reward)
#            distances.append(next_input)  # 將 next_input 中的距離取平均，這只是假設
#
#        distances = np.array(distances)
#        colors = ['r', 'g', 'b']
#
#        # 繪製散點圖
#        for i in range(3):
#            plt.scatter(distances[:, i], rewards, alpha=0.5, c=colors[i], label=f'Distance {i+1}')
#            print()
#
#        plt.xlabel('Distance')
#        plt.ylabel('Reward')
#        plt.title('Reward vs. Distance')
#        plt.legend()
#        plt.grid(True)
#        plt.show()
#
## 測試
#if __name__ == "__main__":
#    q_learning = Q_Learning()
#    q_learning.plot_scatter(num_samples=100)
#