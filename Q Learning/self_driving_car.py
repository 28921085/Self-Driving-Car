import numpy as np
from math_tool import MathTool

class SelfDrivingCar:
    def __init__(self, initial_x=0, initial_y=0, initial_F=90, b=6):
        """
        參數：
        initial_x, initial_y: 初始座標
        initial_F: 初始角度
        b: 車輛長度
        """
        self.x = initial_x
        self.y = initial_y
        self.F = initial_F
        self.b = b

        self.distances=[0,0,0] #[前方距離,右方距離,左方距離]
        self.load_data()

    def load_data(self):
        # 讀取檔案
        with open("軌道座標點.txt", 'r') as file:
            lines = file.readlines()
            # 解析檔案內容
            self.x, self.y, self.F = map(float, lines[0].split(','))

            end_rect_left_top = list(map(float, lines[1].split(',')))
            end_rect_right_bottom = list(map(float, lines[2].split(',')))
            self.end_rect_left = end_rect_left_top[0]
            self.end_rect_right = end_rect_right_bottom[0]
            self.end_rect_top = end_rect_left_top[1]
            self.end_rect_bottom = end_rect_right_bottom[1]

            # 存儲在 self.end_area 2D 列表中
            self.end_area = [
                [self.end_rect_right,self.end_rect_top],   # 右上
                [self.end_rect_right,self.end_rect_bottom],   # 右下
                [self.end_rect_left, self.end_rect_bottom],   # 左下
                [self.end_rect_left, self.end_rect_top],   # 左上
                [self.end_rect_right,self.end_rect_top]    # 右上 (重複以閉合區域)
            ]
            self.track_points = [list(map(float, line.split(','))) for line in lines[3:]]

    def update_state(self,Th):
        """
        根據模擬方程式更新自走車的狀態
        Th: 模型車方向盤所打的角度
        """
        self.Th=Th
        
        self.F, self.x, self.y = MathTool.get_next_state(self.x,self.y,self.F,Th,self.b)
        self.calculate_distances()

    def calculate_distances(self):
        # 計算車體前、左、右方的距離
        front_distance = self.calculate_distance_in_direction(self.F)
        right_distance = self.calculate_distance_in_direction(self.F - 45)
        left_distance = self.calculate_distance_in_direction(self.F + 45)
        # 更新 distances 屬性
        self.distances = [front_distance, right_distance, left_distance]

    def calculate_distance_in_direction(self, angle):
        #根據給定的角度計算車體在該方向上的距離
        x_direction = np.cos(np.radians(angle))
        y_direction = np.sin(np.radians(angle))
        car_center = np.array([self.x, self.y])
        # 計算車體在該方向上的端點
        end_point = car_center + self.b / 2 * np.array([x_direction, y_direction])
        # 尋找與牆壁的交點，並計算距離
        min_distance = float('inf')
        for i in range(len(self.track_points)):
            line_start = np.array(self.track_points[i])
            line_end = np.array(self.track_points[(i + 1) % len(self.track_points)])
            # 檢查射線是否和線段相交
            intersection_point = MathTool.ray_segment_intersection(car_center, end_point, line_start, line_end)
            if intersection_point is not None:
                # 計算交點到車體中心的距離
                distance = np.linalg.norm(intersection_point - car_center)
                min_distance = min(min_distance, distance)

        return min_distance 
    
    def check_collision(self):
        return self.check_car_collision(self.track_points)

    def reach_goal(self):
        return self.check_car_collision(self.end_area)

    def check_car_collision(self,target):#檢查車體是否與target發生碰撞
        car_center = np.array([self.x, self.y])
        # 檢查車體與每個牆壁的碰撞
        for i in range(len(target)-1):
            line_start = np.array(target[i])
            line_end = np.array(target[i+1])

            if MathTool.line_segment_circle_intersection(car_center,self.b/2.0, line_start, line_end):
                return True
        return False
    def set_position(self,x,y):
        self.x=x
        self.y=y
    def get_distances(self):
        return self.distances
if __name__ == "__main__":
    car = SelfDrivingCar()