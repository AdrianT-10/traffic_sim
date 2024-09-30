import math
import random
import os
import time
import numpy as np
from pygame.math import Vector2
from astar import a_star
from typing import List, Tuple, Dict, Optional

ACTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1), (0, 0)]

class Vector2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vector2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Vector2(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar):
        return Vector2(self.x / scalar, self.y / scalar)

    def length(self):
        return (self.x**2 + self.y**2)**0.5

    def length_squared(self):
        return self.x**2 + self.y**2

    def normalize(self):
        length = self.length()
        if length != 0:
            return self / length
        return Vector2(0, 0)

    def dot(self, other):
        return self.x * other.x + self.y * other.y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __str__(self):
        return f"Vector2({self.x}, {self.y})"
    
def generate_datamatrix(vehicle, obstacle_vehicles=[]):
    # 生成一个12x12x9的数据矩阵，表示地图上每个位置的状态
    data_matrix = np.zeros((12, 12, 8), dtype=np.float32)
    
    # 设置主车辆的起始位置和终点位置
    start_grid = vehicle.start
    end_grid = vehicle.end
    data_matrix[start_grid[1], start_grid[0], 0] = 1
    data_matrix[end_grid[1], end_grid[0], 0] = 1

    # 设置主车辆的当前位置
    x, y, _, _, _, _ = vehicle.get_state()
    grid_x = int(x // 50)
    grid_y = int(y // 50)
    if 0 <= grid_x < 12 and 0 <= grid_y < 12:
        data_matrix[grid_y, grid_x, 1] = 1

    # 设置主车辆距离起点和终点的距离
    start_position = vehicle.grid_to_pixel(start_grid)
    end_position = vehicle.grid_to_pixel(end_grid)
    distance_start = math.sqrt((x - start_position.x)**2 + (y - start_position.y)**2)
    distance_end = math.sqrt((x - end_position.x)**2 + (y - end_position.y)**2)
    if 0 <= grid_x < 12 and 0 <= grid_y < 12:
        data_matrix[grid_y, grid_x, 2] = 1 + distance_start / 700
        data_matrix[grid_y, grid_x, 3] = 1 + distance_end / 700

    # 设置主车辆的速度和加速度
    speed = vehicle.get_speed_kmh()
    _, _, _, _, _, acceleration = vehicle.get_state()
    #print(f"speed: {speed}, acceleration: {acceleration}")
    if 0 <= grid_x < 12 and 0 <= grid_y < 12:
        data_matrix[grid_y, grid_x, 4] = 1 + speed / VehicleInfo.MAX_SPEED
        data_matrix[grid_y, grid_x, 5] = 1 + acceleration / VehicleInfo.MAX_ACCEL

    # 设置其他车辆的信息(雷达检测到的车辆)
    radar_data = vehicle.scan_radar(obstacle_vehicles)
    for obs in radar_data:
        obs_x, obs_y, _, _, _, _ = obs.get_state()
        obs_grid_x = int(obs_x // 50)
        obs_grid_y = int(obs_y // 50)
        if 0 <= obs_grid_x < 12 and 0 <= obs_grid_y < 12:
            data_matrix[obs_grid_y, obs_grid_x, 6] = 1
            data_matrix[obs_grid_y, obs_grid_x, 7] = 1 + ACTIONS.index(obs.action)

    return data_matrix

class VehicleInfo:
    LENGTH = 40.0
    WIDTH = 25.0
    MAX_SPEED = 222  # 最大速度 (像素/秒) 80km/h
    MIN_SPEED = 27.75  # 最小速度 (像素/秒) 10km/h
    MAX_ACCEL = 49.05  # 最大加速度 (像素/秒^2) 4.905 m/s^2 0.5g    
    MAX_DECEL = 78.48  # 最大减速度 (像素/秒^2) 7.848 m/s^2 0.8g
    WHEEL_WIDTH = 8.0  # 默认车轮宽度
    WHEEL_DIAMETER = 12.0  # 默认车轮直径

class Vehicle:
    def __init__(self, start, end, vehicle_id, data_save_path=None):
        self.id = vehicle_id
        self.wheel_width = VehicleInfo.WHEEL_WIDTH
        self.wheel_diameter = VehicleInfo.WHEEL_DIAMETER
        self.grid_size = 50  # 每个网格的大小
        self.grid_position = Vector2(start[0], start[1])
        self.start = start
        self.end = end
        self.x = self.grid_position.x * self.grid_size + self.grid_size // 2
        self.y = self.grid_position.y * self.grid_size + self.grid_size // 2
        # 设置初始速度为41.625像素/秒 
        initial_speed = 83.25  # 30km/h
        
        # 计算朝向终点的单位向量，处理起点和终点相同的情况
        direction = Vector2(end[0] - start[0], end[1] - start[1])
        if direction.x == 0 and direction.y == 0:
            # 如果起点和终点相同，设置一个默认方向（例如，向右）
            direction = Vector2(1, 0)
        else:
            direction = direction.normalize()
        
        # 设置初始速度向量
        self.velocity = direction * initial_speed
        
        self.acceleration = Vector2(0, 0)
        self.target = Vector2(end[0], end[1])
        self.path = []
        self.need_replan = True
        self.reached_target = False
        self.color = (1, 0, 0)  # 车身颜色 (红色)
        self.wheel_color = (0, 0, 0)  # 轮子颜色 (黑色)
        self.radar_range = 150  # 雷达范围，单位为像素
        self.yaw = 0.0
        self.data_matrix = None

        self.last_grid_position = Vector2(start[0], start[1])
        self.data_save_path = data_save_path or "E:\\research\\code\\new_pro\\vehicle_data"
        self.data_save_count = 0

        self.use_ai_control = False
        self.ai_controller = None
        self.ai_action = None
        self.ai_acceleration = 0       
        self.generate_data = False
        self.current_action = None
        self.current_target = None
        self.action_progress = 0
        self.reached_grid_center = True
        self.is_colliding = False
        self.collision_count = 0
        if not os.path.exists(self.data_save_path) and self.generate_data:
            os.makedirs(self.data_save_path)

        self.update_obb()

    def set_ai_controller(self, ai_controller):
        self.ai_controller = ai_controller

    def move(self, dt, obstacles):
        if self.reached_target:
            return

        current_grid_x = int(self.x // self.grid_size)
        current_grid_y = int(self.y // self.grid_size)
        grid_center_x = (current_grid_x + 0.5) * self.grid_size
        grid_center_y = (current_grid_y + 0.5) * self.grid_size

        # 检查是否到达网格中心
        if abs(self.x - grid_center_x) < 1 and abs(self.y - grid_center_y) < 1:
            if not self.reached_grid_center:
                self.reached_grid_center = True
                self.grid_position = Vector2(current_grid_x, current_grid_y)
                if self.generate_data:
                    self.update_and_save_data(obstacles)

        if self.reached_grid_center:
            if self.use_ai_control and self.ai_controller:
                state = generate_datamatrix(self, obstacles)
                self.current_action, self.ai_acceleration = self.ai_controller.get_action(state)
                #print(f"Vehicle {self.id} action: {self.current_action}, acceleration: {self.ai_acceleration}")
            else:
                self.plan_path(obstacles)
            
            self.reached_grid_center = False

        # 确定目标位置
        if self.use_ai_control and self.current_action is not None:
            dx, dy = ACTIONS[self.current_action]
            target_x = (current_grid_x + dx + 0.5) * self.grid_size
            target_y = (current_grid_y + dy + 0.5) * self.grid_size
        elif not self.use_ai_control and self.path:
            next_grid = self.path[0]
            target_x, target_y = self.get_grid_center(next_grid)
        else:
            print(f"Warning: Vehicle {self.id} has no valid action or path")
            return

        # 移动逻辑
        direction = Vector2(target_x - self.x, target_y - self.y)
        distance = direction.length()

        if distance < 1:  # 如果非常接近目标点，直接到达
            self.x, self.y = target_x, target_y
            self.reached_grid_center = True
            if not self.use_ai_control and self.path:
                self.path.pop(0)
        else:
            move_distance = min(distance, self.velocity.length() * dt)
            move_direction = direction.normalize()
            self.x += move_direction.x * move_distance
            self.y += move_direction.y * move_distance

        # 更新朝向、速度和加速度
        if direction.length() > 0:
            self.yaw = math.atan2(direction.y, direction.x)

        if not self.use_ai_control:
            desired_velocity = direction.normalize() * min(VehicleInfo.MAX_SPEED, distance / dt)
            self.acceleration = (desired_velocity - self.velocity) / dt
            if self.acceleration.length() > VehicleInfo.MAX_ACCEL:
                self.acceleration = self.acceleration.normalize() * VehicleInfo.MAX_ACCEL
            self.velocity += self.acceleration * dt
            if self.velocity.length() > VehicleInfo.MAX_SPEED:
                self.velocity = self.velocity.normalize() * VehicleInfo.MAX_SPEED
        else:
            self.acceleration = direction.normalize() * (self.ai_acceleration - 1) * 10 * VehicleInfo.MAX_ACCEL
            
            # 限制加速度
            if self.acceleration.length() > VehicleInfo.MAX_ACCEL:
                self.acceleration = self.acceleration.normalize() * VehicleInfo.MAX_ACCEL

            # 更新速度
            self.velocity += self.acceleration * dt
             # 确保速度不低于最小速度
            if self.velocity.length() < VehicleInfo.MIN_SPEED:
                self.velocity = self.velocity.normalize() * VehicleInfo.MIN_SPEED
            # 限制速度不超过最大速度
            if self.velocity.length() > VehicleInfo.MAX_SPEED:
                self.velocity = self.velocity.normalize() * VehicleInfo.MAX_SPEED
                
        self.update_obb()

        new_grid_position = Vector2(int(self.x // self.grid_size), int(self.y // self.grid_size))
        if new_grid_position.x != self.grid_position.x or new_grid_position.y != self.grid_position.y:
            self.grid_position = new_grid_position
            if self.generate_data:
                self.update_and_save_data(obstacles)

        # 检查是否到达终点
        end_pos = self.grid_to_pixel(self.end)
        if abs(self.x - end_pos.x) < self.grid_size / 10 and abs(self.y - end_pos.y) < self.grid_size / 10:
            self.reached_target = True
            self.velocity = Vector2(0, 0)
            self.acceleration = Vector2(0, 0)

    def update_and_save_data(self, obstacles):
        if self.generate_data:
            self.data_matrix = generate_datamatrix(self, obstacles)
            self.save_data_matrix()

    def save_data_matrix(self):
        if self.generate_data:
            if not os.path.exists(self.data_save_path):
                os.makedirs(self.data_save_path)
            filename = f"{self.id - 1}_{self.data_save_count}.npy"
            filepath = os.path.join(self.data_save_path, filename)
            np.save(filepath, self.data_matrix)
            self.data_save_count += 1

    def plan_path(self, obstacles):
        start = (int(self.grid_position.x), int(self.grid_position.y))
        end = (int(self.target.x), int(self.target.y))
        obstacle_grid_coords = [(int(obs.grid_position.x), int(obs.grid_position.y)) for obs in obstacles]
        new_path = a_star(start, end, obstacle_grid_coords)
        
        if new_path:
            self.path = new_path[1:]  # 排除起始点

    def get_grid_center(self, grid_pos):
        return (grid_pos[0] * self.grid_size + self.grid_size / 2,
                grid_pos[1] * self.grid_size + self.grid_size / 2)

    def grid_to_pixel(self, grid_pos):
        return Vector2(grid_pos[0] * self.grid_size + self.grid_size // 2,
                       grid_pos[1] * self.grid_size + self.grid_size // 2)
    
    def get_speed_kmh(self):
        return self.velocity.length() / 3.6  # 将像素/秒转换为km/h

    def get_state(self):
        velocity_length = self.velocity.length() if isinstance(self.velocity, Vector2) else abs(self.velocity)
        acceleration_length = self.acceleration.length() if isinstance(self.acceleration, Vector2) else abs(self.acceleration)
        return self.x, self.y, self.yaw, velocity_length, 0, acceleration_length

    def update_obb(self):
        self.half_width = VehicleInfo.WIDTH / 2
        self.half_length = VehicleInfo.LENGTH / 2
        self.obb_axes = [
            Vector2(math.cos(self.yaw), math.sin(self.yaw)),
            Vector2(-math.sin(self.yaw), math.cos(self.yaw))
        ]

    def project_onto_axis(self, axis):
        center = Vector2(self.x, self.y)
        projection = [
            axis.dot(center + self.obb_axes[0] * self.half_length + self.obb_axes[1] * self.half_width),
            axis.dot(center + self.obb_axes[0] * self.half_length - self.obb_axes[1] * self.half_width),
            axis.dot(center - self.obb_axes[0] * self.half_length + self.obb_axes[1] * self.half_width),
            axis.dot(center - self.obb_axes[0] * self.half_length - self.obb_axes[1] * self.half_width)
        ]
        return min(projection), max(projection)

    def scan_radar(self, obstacle_vehicles):
        detected_obstacles = []
        for obs in obstacle_vehicles:
            distance = math.sqrt((self.x - obs.x)**2 + (self.y - obs.y)**2)
            if distance <= self.radar_range:
                detected_obstacles.append(obs)
        return detected_obstacles
    
    def check_collision(self, obstacles):
        was_colliding = self.is_colliding
        self.is_colliding = False

        for obs in self.scan_radar(obstacles):
            if self._check_collision_with(obs):
                self.is_colliding = True
                break

        if self.is_colliding and not was_colliding:
            self.collision_count += 1
            return True
        return False

    def _check_collision_with(self, other_vehicle):
        dx = abs(self.grid_position.x - other_vehicle.grid_position.x)
        dy = abs(self.grid_position.y - other_vehicle.grid_position.y)
        if dx > 1 or dy > 1:
            return False

        for vehicle in (self, other_vehicle):
            for axis in vehicle.obb_axes:
                proj1_min, proj1_max = self.project_onto_axis(axis)
                proj2_min, proj2_max = other_vehicle.project_onto_axis(axis)
                if proj1_max < proj2_min or proj2_max < proj1_min:
                    return False
        return True
    
    def calculate_reward(self, collision, reached_target):
        reward = 0

        # 碰撞惩罚
        if collision:
            reward -= 100

        # 到达目标奖励
        elif reached_target:
            reward += 1000

        else:
            # 距离奖励：越接近目标，奖励越高
            end_pos = self.grid_to_pixel(self.end)
            current_distance = math.sqrt((self.x - end_pos.x)**2 + (self.y - end_pos.y)**2)
            max_distance = math.sqrt(2) * 12 * self.grid_size  # 地图对角线长度
            distance_reward = (max_distance - current_distance) / max_distance
            reward += distance_reward * 10  # 缩放因子

            # 速度奖励：鼓励保持较高速度
            speed_kmh = self.get_speed_kmh()
            speed_reward = min(speed_kmh / 50, 1)  # 最高奖励速度为50km/h
            reward += speed_reward * 5  # 缩放因子

            # 朝向奖励：车辆朝向越接近目标方向，奖励越高
            direction_to_target = Vector2(end_pos.x - self.x, end_pos.y - self.y).normalize()
            current_direction = Vector2(math.cos(self.yaw), math.sin(self.yaw))
            alignment = direction_to_target.dot(current_direction)
            reward += alignment * 2  # 缩放因子

        return reward
    
    def stop(self):
        self.velocity = Vector2(0, 0)
        self.acceleration = Vector2(0, 0)
        self.need_replan = False
        
class ReservationSystem:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.reservations = {}  # 只使用一个字典来存储所有预约

    def reserve(self, vehicle_id, current_pos, future_pos):
        if future_pos not in self.reservations:
            self.reservations[future_pos] = vehicle_id
            if current_pos in self.reservations and self.reservations[current_pos] == vehicle_id:
                del self.reservations[current_pos]  # 释放旧位置
            return True
        return False

    def confirm_move(self, vehicle_id, new_pos):
        # confirm_move 和 reserve 的功能基本相同
        # 确保新位置被预约，并清除旧的预约
        old_pos = next((pos for pos, vid in self.reservations.items() if vid == vehicle_id), None)
        if old_pos and old_pos != new_pos:
            del self.reservations[old_pos]
        self.reservations[new_pos] = vehicle_id

    def clear_reservations(self, vehicle_id):
        self.reservations = {k: v for k, v in self.reservations.items() if v != vehicle_id}

    def is_position_available(self, pos):
        return pos not in self.reservations
       
class ObstacleVehicle(Vehicle):
    next_id = -1

    @classmethod
    def generate_valid_obstacle(cls, grid_size: int, other_vehicles_info: Dict[Tuple[int, int], Tuple[int, int]], reservation_system: ReservationSystem) -> Optional['ObstacleVehicle']:
        max_attempts = 100
        for _ in range(max_attempts):
            position = (random.randint(0, grid_size-1), random.randint(0, grid_size-1))
            if position not in other_vehicles_info and reservation_system.is_position_available(position):
                temp_obstacle = cls(position, grid_size, reservation_system)
                valid_action = temp_obstacle.choose_valid_action(other_vehicles_info)
                if valid_action != (0, 0):
                    temp_obstacle.action = valid_action
                    temp_obstacle.target = Vector2(position[0] + valid_action[0], position[1] + valid_action[1])
                    temp_obstacle.set_initial_orientation()
                    reservation_system.reserve(temp_obstacle.id, position, temp_obstacle.get_target_position())
                    return temp_obstacle
        return None

    def __init__(self, grid_position: Tuple[int, int], grid_size: int = 12, reservation_system: ReservationSystem = None):
        self.id = ObstacleVehicle.next_id
        ObstacleVehicle.next_id -= 1
        super().__init__(grid_position, grid_position, self.id)
        self.grid_size = grid_size
        self.reservation_system = reservation_system
        self.grid_position = Vector2(grid_position[0], grid_position[1])
        self.x = (grid_position[0] + 0.5) * 50
        self.y = (grid_position[1] + 0.5) * 50
        self.color = (0, 0, 0)  # 黑色
        self.velocity = Vector2(0, 0)
        self.acceleration = Vector2(0, 0)
        self.action = (0, 0)  # 初始化为静止状态
        self.target = Vector2(grid_position[0], grid_position[1])
        self.action_completed = False
        self.update_obb()

    def choose_valid_action(self, other_vehicles_info: Dict[Tuple[int, int], Tuple[int, int]]) -> Tuple[int, int]:
        valid_actions = []
        current_pos = (int(self.x // 50), int(self.y // 50))
        
        for action in ACTIONS:
            new_x = current_pos[0] + action[0]
            new_y = current_pos[1] + action[1]
            new_pos = (new_x, new_y)
            
            if self.is_valid_move(new_pos, other_vehicles_info) and self.reservation_system.is_position_available(new_pos):
                valid_actions.append(action)
        
        if valid_actions:
            chosen_action = random.choice(valid_actions)
            new_pos = (current_pos[0] + chosen_action[0], current_pos[1] + chosen_action[1])
            self.reservation_system.reserve(self.id, current_pos, new_pos)
            return chosen_action
        return (0, 0)
    
    def move(self, dt):
        if not self.action_completed:
            target_x = self.target.x * 50 + 25
            target_y = self.target.y * 50 + 25
            
            direction = Vector2(target_x - self.x, target_y - self.y)
            distance = math.sqrt((self.x - target_x)**2 + (self.y - target_y)**2)
            
            if distance < 1:  # 如果非常接近目标点，视为动作完成
                self.x, self.y = target_x, target_y
                self.action_completed = True
                self.reservation_system.confirm_move(self.id, (int(self.x // 50), int(self.y // 50)))
            else:
                speed = 100 # 100像素/秒
                move_distance = min(distance, speed * dt)
                move_direction = direction.normalize()
                self.x += move_direction.x * move_distance
                self.y += move_direction.y * move_distance

            self.grid_position = Vector2(int(self.x // 50), int(self.y // 50))
            
            # 更新朝向
            if direction.length() > 0:
                self.yaw = math.atan2(direction.y, direction.x)
 
        self.update_obb()

    def is_valid_move(self, new_pos: Tuple[int, int], other_vehicles_info: Dict[Tuple[int, int], Tuple[int, int]]) -> bool:
        if not (0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size):
            return False
        return self.reservation_system.is_position_available(new_pos)

    def is_safe_wait(self, current_pos: Tuple[int, int], other_vehicles_info: Dict[Tuple[int, int], Tuple[int, int]]) -> bool:
        # 检查当前位置是否会与其他车辆的下一个位置冲突
        for _, target in other_vehicles_info.items():
            if current_pos == target:
                return False
        return True

    def least_conflicting_action(self, current_pos: Tuple[int, int], other_vehicles_info: Dict[Tuple[int, int], Tuple[int, int]]) -> Tuple[int, int]:
        min_conflicts = float('inf')
        best_action = (0, 0)
        
        for action in ACTIONS:
            new_x = current_pos[0] + action[0]
            new_y = current_pos[1] + action[1]
            new_pos = (new_x, new_y)
            
            if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size:
                conflicts = sum(1 for _, target in other_vehicles_info.items() if new_pos == target)
                if conflicts < min_conflicts:
                    min_conflicts = conflicts
                    best_action = action
        
        return best_action
    
    def get_target_position(self) -> Tuple[int, int]:
        if self.target.x is not None and self.target.y is not None:
            return (int(self.target.x // 50), int(self.target.y // 50))
        return (int(self.x // 50), int(self.y // 50))  # 如果没有目标，返回当前位置
    
    def set_initial_orientation(self):
        self.yaw = math.atan2(self.action[1], self.action[0])

    def is_expired(self):
        return self.action_completed
