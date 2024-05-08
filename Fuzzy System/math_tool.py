import numpy as np
class MathTool:
    def on_segment(p, q, r):
        if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
            return True
        return False

    @staticmethod
    def segments_intersect(A, B, P, Q):
        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if val == 0:
                return 0
            return 1 if val > 0 else 2
        # Check if general orientations are different
        o1 = orientation(A, B, P)
        o2 = orientation(A, B, Q)
        o3 = orientation(P, Q, A)
        o4 = orientation(P, Q, B)
        if o1 != o2 and o3 != o4:
            return True  # Intersecting
        # Special case: collinear segments
        if o1 == 0 and MathTool.on_segment(A, P, B):
            return True
        if o2 == 0 and MathTool.on_segment(A, Q, B):
            return True
        if o3 == 0 and MathTool.on_segment(P, A, Q):
            return True
        if o4 == 0 and MathTool.on_segment(P, B, Q):
            return True
        return False  # Not intersecting

    @staticmethod
    def line_segment_circle_intersection(center, radius, point_a, point_b):
        # 向量化線段的兩個端點
        vector_a = np.array(point_a)
        vector_b = np.array(point_b)
        # 向量化圓心
        vector_center = np.array(center)
        # 計算線段的向量
        segment_vector = vector_b - vector_a
        # 計算線段中點到圓心的向量
        vector_to_center = vector_center - vector_a
        # 將向量投影到線段的方向上
        projection = np.dot(vector_to_center, segment_vector) / np.dot(segment_vector, segment_vector)
        # 計算投影點
        projection_point = vector_a + projection * segment_vector
        # 計算投影點到圓心的距離
        distance_to_center = np.linalg.norm(vector_center - projection_point)
        # 檢查投影點是否在線段範圍內
        if 0 <= projection <= 1:
            # 在線段範圍內，如果投影點到圓心的距離小於半徑，表示相交
            return distance_to_center < radius
        else:
            # 若投影在線段範圍外，則判斷線段的兩個端點是否在圓內
            return np.linalg.norm(vector_center - vector_a) < radius or np.linalg.norm(vector_center - vector_b) < radius
        
    @staticmethod
    def ray_segment_intersection(ray_start, ray_end, segment_start, segment_end):#計算射線與線段的交點
        # 計算射線的方向向量
        ray_direction = ray_end - ray_start
        # 計算線段的方向向量
        segment_direction = segment_end - segment_start
        # 使用叉積計算射線和線段的交點
        cross_product = np.cross(ray_direction, segment_direction)
        # 如果叉積為 0，表示兩線平行或重疊
        if np.abs(cross_product) < 1e-8:
            return None
        # 計算射線起點到線段起點的向量
        start_to_start = segment_start - ray_start
        # 使用叉積計算交點與射線起點的位置參數
        t = np.cross(start_to_start, segment_direction) / cross_product
        u = np.cross(start_to_start, ray_direction) / cross_product
        # 檢查交點是否在射線上且在線段上
        if t >= 0 and u >= 0 and u <= 1:
            intersection_point = ray_start + t * ray_direction
            return intersection_point
        else:
            return None
    @staticmethod
    def get_next_state(x,y,F,Th,b):
        pi=3.1415926
        #self.Th=Th
        Th = Th/180*pi
        F= F/180*pi

        F_next = F - np.arcsin(2 * np.sin(Th) / b)
        x_next = x + np.cos(F + Th) + np.sin(Th) * np.sin(F)
        y_next = y + np.sin(F + Th) - np.sin(Th) * np.cos(F)
        F_next = F_next/pi*180
        # 限制角度的範圍
        F_next = np.clip(F_next, -90, 270)
        return F_next,x_next,y_next
    @staticmethod
    def point_to_polygon_distance(x, y, polygon):
        polygon = np.array(polygon)
        point = np.array([x, y])
        if MathTool.is_inside_polygon(point, polygon):
            return 0
        distances = []
        for i in range(len(polygon) - 1):
            p1 = polygon[i]
            p2 = polygon[i + 1]
            distances.append(MathTool.point_to_line_distance(point, p1, p2))
        #print(distances)
        return min(distances)
    @staticmethod
    def is_inside_polygon(point, polygon):
        # 使用射線法判斷點是否在多邊形內部
        # 參考：https://en.wikipedia.org/wiki/Point_in_polygon#Ray_casting_algorithm
        x, y = point
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
    @staticmethod
    def point_to_line_distance(point, line_start, line_end):
        # 計算點到線段的最短距離
        # 參考：https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_two_points
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        return numerator / denominator
