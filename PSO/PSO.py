import numpy as np
from self_driving_car import SelfDrivingCar
from MLPnework import MLPnetwork
from math_tool import MathTool

class PSO:
    def __init__(self, dim=26,particle_size=30, max_iter=100, lb=-1, ub=1):
        self.w = 0.5  # 惯性权重
        self.c1 = 1  # 自我认知学习因子
        self.c2 = 2  # 社会学习因子
        self.particle_size = particle_size  # 粒子数量
        self.dim = dim  # 参数维度
        self.max_iter = max_iter  # 最大迭代次数
        self.lb = lb  # 参数下界
        self.ub = ub  # 参数上界

        self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.particle_size, self.dim))  # 初始化粒子位置
        self.V = np.random.uniform(low=-1, high=1, size=(self.particle_size, self.dim))  # 初始化粒子速度
        self.pbest = self.X.copy()  # 初始化每个粒子的最优位置
        self.gbest = self.X[np.random.randint(0, self.particle_size)]  # 初始化全局最优位置
        self.pbest_scores = np.full(self.particle_size, float('inf'))  # 初始化每个粒子的最优得分
        self.gbest_score = float('inf')  # 初始化全局最优得分
        self.mlp=MLPnetwork()

    def optimize(self):
        for t in range(self.max_iter):
            error = float('inf')
            for i in range(self.particle_size):
                score = self.cal_score(self.X[i])
                error = min(error,score)
                if score < self.pbest_scores[i]:
                    self.pbest_scores[i] = score
                    self.pbest[i] = self.X[i]
                if score < self.gbest_score:
                    self.gbest_score = score
                    self.gbest = self.X[i]
            for i in range(self.particle_size):
                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)
                self.V[i] = self.w * self.V[i] + self.c1 * r1 * (self.pbest[i] - self.X[i]) + self.c2 * r2 * (self.gbest - self.X[i])
                self.X[i] = self.X[i] + self.V[i]
                self.X[i] = np.clip(self.X[i], self.lb, self.ub)
            

            print(f'Iteration {t}/{self.max_iter}, Error(this epoch): {error}')
        self.mlp.set_params(self.gbest)

    def cal_score(self,x):
        car=SelfDrivingCar()
        self.mlp.set_params(x)
        while not(car.check_collision() or car.reach_goal()):
            #print(self.mlp.get_next_Th(car.get_distances()))
            car.update_state(self.mlp.get_next_Th(car.get_distances()))
        if car.reach_goal():
            return 0
        return MathTool.point_to_polygon_distance(car.x,car.y,car.end_area)

    def get_next_Th(self, inputs):
        return self.mlp.get_next_Th(inputs)
    
