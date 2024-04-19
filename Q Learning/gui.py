import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle
import self_driving_car as car
import threading
import time
import Q_Learning

class GUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Perceptron Training Results")
        self.geometry("600x800")

        # 初始化行進軌跡
        self.track_trace = [[0,0]]
        self.car=car.SelfDrivingCar()
        self.Q=Q_Learning.Q_Learning()

        self.create_simulation_figure()
        self.create_sensor()
        self.create_buttons()
        self.create_collision_label()  # 新增碰撞標籤

        self.start_simulation()
        

    def draw_map(self):
        # 提取x和y座標以繪製軌道
        track_x = [point[0] for point in self.car.track_points]
        track_y = [point[1] for point in self.car.track_points]
        # 設定圖形範圍
        self.ax.set_xlim(min(track_x) - 5, max(track_x) + 5)
        self.ax.set_ylim(min(track_y) - 5, max(track_y) + 5)
        # 繪製軌道
        self.ax.plot(track_x + [track_x[0]], track_y + [track_y[0]], 'b-')
        # 繪製終點區域
        rect_patch = plt.Rectangle((self.car.end_rect_left, self.car.end_rect_top),
                                   self.car.end_rect_right - self.car.end_rect_left,
                                   self.car.end_rect_bottom - self.car.end_rect_top,
                                   color='g', alpha=0.3)
        self.ax.add_patch(rect_patch)

    def draw_car(self):
        # 車體參數
        car_diameter = 6.0

        # 繪製車子外圓
        circle_patch = Circle((self.car.x, self.car.y), car_diameter / 2, color='r')
        self.ax.add_patch(circle_patch)

        # 繪製車體朝向箭頭
        arrow_length = 2.0
        arrow_x = self.car.x + arrow_length * np.cos(np.radians(self.car.F))
        arrow_y = self.car.y + arrow_length * np.sin(np.radians(self.car.F))
        arrow_patch = plt.Arrow(self.car.x, self.car.y, arrow_x - self.car.x, arrow_y - self.car.y,
                                width=0.5, color='blue')
        self.ax.add_patch(arrow_patch)

    def draw_fig(self):
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.draw_map()
        self.draw_car()
        self.fig_setting()

    def draw_track_trace(self):
        if self.track_trace:
            track_trace_x = [point[0] for point in self.track_trace]
            track_trace_y = [point[1] for point in self.track_trace]
            self.ax.plot(track_trace_x, track_trace_y, 'k--')

    def fig_setting(self):
        # 顯示圖形
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title('self-driving car simulation')

    def create_simulation_figure(self):
        self.draw_fig()
        # Embed the matplotlib plot in the tkinter window
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack()

    def update_gui(self):
        self.car.calculate_distances()  # 計算距離
        self.update_distances_labels()  # 更新距離顯示
        self.update_collision_label()  # 更新碰撞標籤
        self.ax.clear()  # 清除前一個畫面
        self.draw_track_trace()
        self.draw_map()  # 重新繪製地圖
        self.draw_car()  # 重新繪製車子
        self.fig_setting()
        self.canvas.draw()  # 實時更新圖形


    def start_simulation(self):
        # 定義一個新的執行緒來執行模擬
        simulation_thread = threading.Thread(target=self.run_simulation)
        # 設定執行緒為背景執行，主視窗關閉時結束執行緒
        simulation_thread.daemon = True
        # 啟動執行緒
        simulation_thread.start()

    def run_simulation(self):
        # 定義每秒執行一次的循環
        epochs=1000
        current=0
        print("current epoch:",current)
        while current < epochs:
            cnt=0
            if self.car.reach_goal() or self.car.check_collision():
                print("current epoch:",current)
                #print(self.Q.exploration_rate)
                if self.car.reach_goal():
                    print("reach goal")
                #    self.after(0, self.update_gui)
                #    time.sleep(0.01)
                self.car=car.SelfDrivingCar()
                self.Q.car=car.SelfDrivingCar()
                self.track_trace = [[0,0]]
                current += 1
                self.Q.exploration_rate *= self.Q.exploration_decay
                cnt=0
                
                np.set_printoptions(suppress=True)
                #print(self.Q.q_table)
                
            else:
                cnt+=1
                self.car.update_state(self.Q.get_next_Th(self.car.distances,cnt))  # 這裡的 10 是假設的模擬方向盤角度
                self.track_trace.append((self.car.x, self.car.y))
            if current > 990:
                # 在主執行緒中執行 GUI 更新
                self.after(0, self.update_gui)
                # 休眠一秒
                time.sleep(0.2)

    def create_sensor(self):
        # 新增標籤來顯示 distances
        self.distances_labels = [ttk.Label(self, text=f"前方距離: 0.0"),ttk.Label(self, text=f"右方距離: 0.0")
                                 ,ttk.Label(self, text=f"左方距離: 0.0")]
        for i, label in enumerate(self.distances_labels):
            label.pack()
        self.car.calculate_distances()  # 計算距離
        self.update_distances_labels()  # 更新距離顯示

    def update_distances_labels(self):
        # 更新 distances 的顯示文字
        self.distances_labels[0].config(text=f"前方距離: {self.car.distances[0]:.2f}")
        self.distances_labels[1].config(text=f"右方距離: {self.car.distances[1]:.2f}")
        self.distances_labels[2].config(text=f"左方距離: {self.car.distances[2]:.2f}")

    def create_collision_label(self):
        # 新增碰撞標籤
        self.collision_label = ttk.Label(self, text="是否碰撞: 否")
        self.collision_label.pack()

    def update_collision_label(self):
        if self.car.reach_goal():
            pass

        else:
            is_collision = self.car.check_collision()
            # 更新碰撞標籤的文字
            collision_text = "是" if is_collision else "否"
            self.collision_label.config(text=f"是否碰撞: {collision_text}")

    def create_buttons(self):
        self.buttons_frame = ttk.Frame(self)
        self.buttons_frame.pack(pady=20)

        # 修改按钮事件为移动车子
        self.up_button = ttk.Button(self.buttons_frame, text="Up", command=self.move_up)
        self.down_button = ttk.Button(self.buttons_frame, text="Down", command=self.move_down)
        self.left_button = ttk.Button(self.buttons_frame, text="Left", command=self.move_left)
        self.right_button = ttk.Button(self.buttons_frame, text="Right", command=self.move_right)

        # 新增旋轉車頭的按鈕
        self.rotate_left_button = ttk.Button(self.buttons_frame, text="Rotate Left", command=self.rotate_left)
        self.rotate_right_button = ttk.Button(self.buttons_frame, text="Rotate Right", command=self.rotate_right)


        self.up_button.grid(row=0, column=1, padx=10, pady=5)
        self.down_button.grid(row=2, column=1, padx=10, pady=5)
        self.left_button.grid(row=1, column=0, padx=10, pady=5)
        self.right_button.grid(row=1, column=2, padx=10, pady=5)

        # 新增旋轉車頭的按鈕
        self.rotate_left_button.grid(row=1, column=3, padx=10, pady=5)
        self.rotate_right_button.grid(row=1, column=4, padx=10, pady=5)



    def move_car(self, dx, dy):
        self.car.x += dx
        self.car.y += dy
         # 更新行進軌跡
        self.update_gui()

    def rotate_car(self, angle):
        # 更新車子朝向角度
        self.car.F += angle
        self.update_gui()

    def move_up(self):
        self.move_car(0, 1)

    def move_down(self):
        self.move_car(0, -1)

    def move_left(self):
        self.move_car(-1, 0)

    def move_right(self):
        self.move_car(1, 0)

    def rotate_left(self):
        self.rotate_car(10)  # 旋轉角度可調整

    def rotate_right(self):
        self.rotate_car(-10)  # 旋轉角度可調整

if __name__ == "__main__":
    app = GUI()
    app.protocol("WM_DELETE_WINDOW", app.quit)
    app.mainloop()
