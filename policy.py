import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F



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

class RLVehicleCNN(nn.Module):
    def __init__(self, input_channels=8, num_actions=9, include_value_function=False):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
        
        self.include_value_function = include_value_function

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
        
        action_probs = F.softmax(self.fc_action(x), dim=-1)
        acceleration = self.fc_accel(x)
        
        if self.include_value_function:
            value = self.fc_value(x)
            return action_probs, acceleration, value
        else:
            return action_probs, acceleration

    def get_action(self, state):
        if isinstance(state, np.ndarray):
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        else:
            state_tensor = state.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self(state_tensor)
        
        action_probs = outputs[0]
        acceleration = outputs[1]
        action = torch.argmax(action_probs, dim=1).item()
        return action, acceleration.item()

    def add_value_function(self):
        self.include_value_function = True
        self.fc_value = nn.Linear(128, 1).to(next(self.parameters()).device)