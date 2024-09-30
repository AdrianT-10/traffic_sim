import os
import math
import random
import datetime
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional

# 常量定义
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

    def length(self):
        return (self.x**2 + self.y**2)**0.5

    def normalize(self):
        length = self.length()
        if length != 0:
            return Vector2(self.x / length, self.y / length)
        return Vector2(0, 0)

    def dot(self, other):
        return self.x * other.x + self.y * other.y

class VehicleInfo:
    LENGTH = 40.0
    WIDTH = 25.0
    MAX_SPEED = 222  # 最大速度 (像素/秒) 80km/h
    MIN_SPEED = 27.75  # 最小速度 (像素/秒) 10km/h
    MAX_ACCEL = 49.05  # 最大加速度 (像素/秒^2) 4.905 m/s^2 0.5g    
    MAX_DECEL = 78.48  # 最大减速度 (像素/秒^2) 7.848 m/s^2 0.8g

class Vehicle:
    def __init__(self, start, end, vehicle_id):
        self.id = vehicle_id
        self.grid_size = 50
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
        self.reached_target = False
        self.yaw = 0.0
        self.use_ai_control = True
        self.ai_controller = None
        self.current_action = None
        self.ai_acceleration = 0
        self.half_width = VehicleInfo.WIDTH / 2
        self.half_length = VehicleInfo.LENGTH / 2
        self.update_obb()

    def move(self, dt, obstacles):
        if self.reached_target:
            return

        current_grid_x = int(self.x // self.grid_size)
        current_grid_y = int(self.y // self.grid_size)

        if self.current_action is not None:
            dx, dy = ACTIONS[self.current_action]
            target_x = (current_grid_x + dx + 0.5) * self.grid_size
            target_y = (current_grid_y + dy + 0.5) * self.grid_size
        else:
            target_x, target_y = self.x, self.y

        direction = Vector2(target_x - self.x, target_y - self.y)
        distance = direction.length()

        if distance < 1:
            self.x, self.y = target_x, target_y
        else:
            move_distance = min(distance, self.velocity.length() * dt)
            move_direction = direction.normalize()
            self.x += move_direction.x * move_distance
            self.y += move_direction.y * move_distance

        if direction.length() > 0:
            self.yaw = math.atan2(direction.y, direction.x)

        self.acceleration = direction.normalize() * (self.ai_acceleration - 1) * 10 * VehicleInfo.MAX_ACCEL
        # 限制加速度
        if self.acceleration.length() > VehicleInfo.MAX_ACCEL:
            self.acceleration = self.acceleration.normalize() * VehicleInfo.MAX_ACCEL
        self.velocity += self.acceleration * dt
        if self.velocity.length() < VehicleInfo.MIN_SPEED:
            self.velocity = self.velocity.normalize() * VehicleInfo.MIN_SPEED
        if self.velocity.length() > VehicleInfo.MAX_SPEED:
            self.velocity = self.velocity.normalize() * VehicleInfo.MAX_SPEED

        self.grid_position = Vector2(int(self.x // self.grid_size), int(self.y // self.grid_size))

        end_pos = self.grid_to_pixel(self.end)
        if abs(self.x - end_pos.x) < self.grid_size / 10 and abs(self.y - end_pos.y) < self.grid_size / 10:
            self.reached_target = True
            self.velocity = Vector2(0, 0)
            self.acceleration = Vector2(0, 0)
        
        self.update_obb()

    def grid_to_pixel(self, grid_pos):
        return Vector2(grid_pos[0] * self.grid_size + self.grid_size // 2,
                       grid_pos[1] * self.grid_size + self.grid_size // 2)

    def get_state(self):
        return self.x, self.y, self.yaw, self.velocity.length(), 0, self.acceleration.length()

    def get_speed_kmh(self):
        return self.velocity.length() / 3.6

    def scan_radar(self, obstacle_vehicles):
        radar_range = 150  # 雷达范围，单位为像素
        detected_obstacles = []
        for obs in obstacle_vehicles:
            distance = math.sqrt((self.x - obs.x)**2 + (self.y - obs.y)**2)
            if distance <= radar_range:
                detected_obstacles.append(obs)
        return detected_obstacles
    
    def update_obb(self):
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

    def check_collision(self, other_vehicle):
        for vehicle in (self, other_vehicle):
            for axis in vehicle.obb_axes:
                proj1_min, proj1_max = self.project_onto_axis(axis)
                proj2_min, proj2_max = other_vehicle.project_onto_axis(axis)
                if proj1_max < proj2_min or proj2_max < proj1_min:
                    return False
        return True

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
        super().move(dt, [])  # 调用父类的 move 方法，传入空的障碍物列表
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

def generate_datamatrix(vehicle, obstacle_vehicles=[]):
    data_matrix = np.zeros((12, 12, 8), dtype=np.float32)
    
    start_grid = vehicle.start
    end_grid = vehicle.end
    data_matrix[start_grid[1], start_grid[0], 0] = 1
    data_matrix[end_grid[1], end_grid[0], 0] = 1

    x, y, _, _, _, _ = vehicle.get_state()
    grid_x = int(x // 50)
    grid_y = int(y // 50)
    if 0 <= grid_x < 12 and 0 <= grid_y < 12:
        data_matrix[grid_y, grid_x, 1] = 1

    start_position = vehicle.grid_to_pixel(start_grid)
    end_position = vehicle.grid_to_pixel(end_grid)
    distance_start = math.sqrt((x - start_position.x)**2 + (y - start_position.y)**2)
    distance_end = math.sqrt((x - end_position.x)**2 + (y - end_position.y)**2)
    if 0 <= grid_x < 12 and 0 <= grid_y < 12:
        data_matrix[grid_y, grid_x, 2] = 1 + distance_start / 700
        data_matrix[grid_y, grid_x, 3] = 1 + distance_end / 700

    speed = vehicle.get_speed_kmh()
    _, _, _, _, _, acceleration = vehicle.get_state()
    if 0 <= grid_x < 12 and 0 <= grid_y < 12:
        data_matrix[grid_y, grid_x, 4] = 1 + speed / VehicleInfo.MAX_SPEED
        data_matrix[grid_y, grid_x, 5] = 1 + acceleration / VehicleInfo.MAX_ACCEL

    radar_data = vehicle.scan_radar(obstacle_vehicles)
    for obs in radar_data:
        obs_x, obs_y, _, _, _, _ = obs.get_state()
        obs_grid_x = int(obs_x // 50)
        obs_grid_y = int(obs_y // 50)
        if 0 <= obs_grid_x < 12 and 0 <= obs_grid_y < 12:
            data_matrix[obs_grid_y, obs_grid_x, 6] = 1
            data_matrix[obs_grid_y, obs_grid_x, 7] = 1 + ACTIONS.index(obs.action)

    return data_matrix

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, H, W = x.size()
        query = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, H * W)
        energy = torch.bmm(query, key)
        attention = F.softmax(energy, dim=2)
        value = self.value(x).view(batch_size, -1, H * W)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        return self.gamma * out + x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.attention = SelfAttention(out_channels)
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.attention(out)
        out += residual
        return F.relu(out)

class ImprovedVehicleCNN(nn.Module):
    def __init__(self, input_channels=8, num_actions=9):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.res_blocks = nn.Sequential(
            ResidualBlock(64, 128),
            ResidualBlock(128, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512)
        )
        
        self.attention = SelfAttention(512)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc_action = nn.Linear(128, num_actions)
        self.fc_accel = nn.Linear(128, 1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res_blocks(x)
        x = self.attention(x)
        x = self.global_avg_pool(x).view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        action_logits = self.fc_action(x)
        acceleration = self.fc_accel(x)
        
        return action_logits, acceleration

class PrioritizedReplayBuffer:
    def __init__(self, capacity, state_shape, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.state_buffer = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.action_buffer = np.zeros((capacity,), dtype=np.int64)
        self.reward_buffer = np.zeros((capacity,), dtype=np.float32)
        self.next_state_buffer = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.done_buffer = np.zeros((capacity,), dtype=np.float32)
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        max_priority = np.max(self.priorities) if self.size > 0 else 1.0

        self.state_buffer[self.position] = state
        self.action_buffer[self.position] = action
        self.reward_buffer[self.position] = reward
        self.next_state_buffer[self.position] = next_state
        self.done_buffer[self.position] = done
        self.priorities[self.position] = max_priority
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta=0.4):
        if self.size < batch_size:
            return None

        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(self.size, batch_size, p=probabilities)
        
        state_batch = self.state_buffer[indices]
        action_batch = self.action_buffer[indices]
        reward_batch = self.reward_buffer[indices]
        next_state_batch = self.next_state_buffer[indices]
        done_batch = self.done_buffer[indices]

        total = len(self)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch), indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return self.size

class RLAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, device):
        self.device = device
        self.policy_net = ImprovedVehicleCNN().to(device)
        self.target_net = ImprovedVehicleCNN().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)
        self.memory = PrioritizedReplayBuffer(capacity=10000, state_shape=state_dim)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.target_update_frequency = 10
        self.steps_done = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(len(ACTIONS)), random.random()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_logits, acceleration = self.policy_net(state_tensor)
            action = torch.argmax(action_logits, dim=1).item()
            acceleration = acceleration.item()
        return action, acceleration

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        samples, indices, weights = self.memory.sample(self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = samples

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.LongTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        q_values, _ = self.policy_net(state_batch)
        next_q_values, _ = self.target_net(next_state_batch)
        q_value = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward_batch + self.gamma * next_q_value * (1 - done_batch)

        loss = (q_value - expected_q_value.detach()).pow(2) * weights
        priorities = loss + 1e-5
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.memory.update_priorities(indices, priorities.detach().cpu().numpy())

        self.steps_done += 1
        if self.steps_done % self.target_update_frequency == 0:
            self.update_target_network()
        self.scheduler.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

class VehicleEnv:
    def __init__(self, grid_size=12):
        self.grid_size = grid_size
        self.reservation_system = ReservationSystem(grid_size)
        self.reset()

    def reset(self):
        self.reservation_system = ReservationSystem(self.grid_size)  # 重新初始化预约系统
        start = (0, 0)
        end = (self.grid_size-1, self.grid_size-1)
        self.ego_vehicle = Vehicle(start, end, 1)
        self.obstacle_vehicles = []
        self.obstacle_generation_timer = 0
        self.obstacle_generation_interval = 0.05
        return self._get_state()

    def step(self, action):
        self.ego_vehicle.current_action = action[0]
        self.ego_vehicle.ai_acceleration = action[1]
        self.ego_vehicle.move(0.1, self.obstacle_vehicles)
        
        for obstacle in self.obstacle_vehicles:
            obstacle.move(0.1)
        
        self.obstacle_vehicles = [obs for obs in self.obstacle_vehicles if not obs.is_expired()]

        self.obstacle_generation_timer += 0.1
        if self.obstacle_generation_timer >= self.obstacle_generation_interval:
            self.obstacle_generation_timer = 0
            self._generate_new_obstacle()

        new_state = self._get_state()
        reward = self._compute_reward()
        done = self.ego_vehicle.reached_target or self._is_collision() or self._is_out_of_bounds()
        
        return new_state, reward, done

    def _generate_new_obstacle(self):
        other_vehicles_info = self._get_vehicles_info()
        new_obstacle = ObstacleVehicle.generate_valid_obstacle(self.grid_size, other_vehicles_info, self.reservation_system)
        if new_obstacle:
            self.obstacle_vehicles.append(new_obstacle)

    def _get_state(self):
        return generate_datamatrix(self.ego_vehicle, self.obstacle_vehicles)

    def _compute_reward(self):
        if self.ego_vehicle.reached_target:
            return 100
        elif self._is_collision() or self._is_out_of_bounds():
            return -100
        else:
            end_pos = self.ego_vehicle.grid_to_pixel(self.ego_vehicle.end)
            distance = ((self.ego_vehicle.x - end_pos.x)**2 + (self.ego_vehicle.y - end_pos.y)**2)**0.5
            distance_reward = -distance / (self.grid_size * 50)
            
            speed = self.ego_vehicle.velocity.length()
            speed_reward = speed / VehicleInfo.MAX_SPEED
            
            accel = self.ego_vehicle.acceleration.length()
            accel_penalty = -abs(accel) / VehicleInfo.MAX_ACCEL
            
            min_obstacle_distance = float('inf')
            for obs in self.obstacle_vehicles:
                obs_distance = ((self.ego_vehicle.x - obs.x)**2 + (self.ego_vehicle.y - obs.y)**2)**0.5
                min_obstacle_distance = min(min_obstacle_distance, obs_distance)
            obstacle_reward = min(0, (min_obstacle_distance - 50) / 50)
            
            direction_to_target = Vector2(end_pos.x - self.ego_vehicle.x, end_pos.y - self.ego_vehicle.y).normalize()
            vehicle_direction = Vector2(math.cos(self.ego_vehicle.yaw), math.sin(self.ego_vehicle.yaw))
            direction_reward = direction_to_target.dot(vehicle_direction)
            
            return distance_reward + 0.5 * speed_reward + 0.3 * accel_penalty + 0.5 * obstacle_reward + 0.3 * direction_reward

    def _is_collision(self):
        for obs in self.obstacle_vehicles:
            if self.ego_vehicle.check_collision(obs):
                return True
        return False

    def _is_out_of_bounds(self):
        return (self.ego_vehicle.x < 0 or self.ego_vehicle.x >= self.grid_size*50 or
                self.ego_vehicle.y < 0 or self.ego_vehicle.y >= self.grid_size*50)

    def _get_vehicles_info(self):
        vehicles_info = {}
        ego_pos = (int(self.ego_vehicle.x // 50), int(self.ego_vehicle.y // 50))
        vehicles_info[ego_pos] = self.ego_vehicle.end
        for obs in self.obstacle_vehicles:
            obs_pos = (int(obs.x // 50), int(obs.y // 50))
            obs_target = (int(obs.target.x), int(obs.target.y))
            vehicles_info[obs_pos] = obs_target
        return vehicles_info

def visualize_state(state, env):
    plt.clf()
    plt.imshow(np.sum(state, axis=2))
    plt.title("环境状态")
    vehicle_pos = (env.ego_vehicle.x / 50, env.ego_vehicle.y / 50)
    target_pos = env.ego_vehicle.end
    plt.plot(vehicle_pos[0], vehicle_pos[1], 'ro', markersize=10)
    plt.plot(target_pos[0], target_pos[1], 'go', markersize=10)
    for obs in env.obstacle_vehicles:
        obs_pos = (obs.x / 50, obs.y / 50)
        plt.plot(obs_pos[0], obs_pos[1], 'bo', markersize=8)
    plt.pause(0.1)
    
def train(episodes, visualize=False):
    env = VehicleEnv()
    state_dim = (12, 12, 8)
    action_dim = len(ACTIONS)
    hidden_dim = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    agent = RLAgent(state_dim, action_dim, hidden_dim, device)
    
    if os.path.exists("policy_model.pth"):
        agent.policy_net.load_state_dict(torch.load("policy_model.pth", map_location=device, weights_only=True))
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        print("加载预训练模型成功。")
    else:
        print("未找到预训练模型。从头开始训练。")
    
    rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action, acceleration = agent.select_action(state)
            next_state, reward, done = env.step((action, acceleration))
            agent.memory.add(state, action, reward, next_state, done)
            agent.train()
            
            state = next_state
            total_reward += reward
            
            if visualize:
                visualize_state(state, env)
        
        rewards.append(total_reward)
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"时间 {now}, 回合 {episode+1}/{episodes}, 总奖励: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}, 学习率: {agent.scheduler.get_last_lr()[0]:.6f}")
    
    torch.save(agent.policy_net.state_dict(), "trained_model.pth")
    
    return rewards

if __name__ == "__main__":
    num_episodes = 100000
    visualize = True  # 设置为 True 以可视化训练过程
    
    if visualize:
        plt.ion()
    
    rewards = train(num_episodes, visualize)
    
    if visualize:
        plt.ioff()
    
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title("训练过程中的奖励")
    plt.xlabel("回合")
    plt.ylabel("总奖励")
    plt.savefig("rewards_plot.png")
    plt.show()

    print("训练完成！模型已保存为 'trained_model.pth'。")
    print(f"奖励曲线已保存为 'rewards_plot.png'。")