import torch
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np
from vehicle import generate_datamatrix
from policy import ImprovedVehicleCNN as VehicleCNN
from collections import deque

class RLAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, device):
        self.device = device
        self.policy_net = VehicleCNN().to(device)
        self.target_net = VehicleCNN().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99

    def train(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        
        # 使用numpy数组存储批数据
        state_batch = np.array([s for s, _, _, _, _ in batch])
        action_batch = np.array([a for _, a, _, _, _ in batch])
        reward_batch = np.array([r for _, _, r, _, _ in batch])
        next_state_batch = np.array([s for _, _, _, s, _ in batch])
        done_batch = np.array([d for _, _, _, _, d in batch])

        # 转换为张量
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.LongTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)

        self.optimizer.zero_grad()
        q_values, _ = self.policy_net(state_batch)
        next_q_values, _ = self.target_net(next_state_batch)
        q_value = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward_batch + self.gamma * next_q_value * (1 - done_batch)
        
        loss = nn.MSELoss()(q_value, expected_q_value.detach())
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

class VehicleEnv:
    def __init__(self, sim):
        self.sim = sim

    def reset(self):
        self.sim.start_next_run()
        return generate_datamatrix(self.sim.ego_vehicle, self.sim.obstacle_vehicles)

    def step(self, action):
        # 执行动作
        self.sim.ego_vehicle.current_action = action
        self.sim.update_simulation()

        # 获取新的状态
        new_state = generate_datamatrix(self.sim.ego_vehicle, self.sim.obstacle_vehicles)

        # 计算奖励
        reward = self.sim.compute_reward()

        # 检查是否结束
        done = self.sim.ego_vehicle.reached_target or self.sim.is_collision() or self.sim.is_out_of_bounds()

        return new_state, reward, done, {}