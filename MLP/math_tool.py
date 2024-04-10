import numpy as np
class MathTool:
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
    def on_segment(p, q, r):
        if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
            return True
        return False

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
    def ray_segment_intersection(ray_start, ray_end, segment_start, segment_end):
        """
        計算射線與線段的交點
        """
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