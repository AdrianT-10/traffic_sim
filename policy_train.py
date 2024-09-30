import os
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt

# Define possible actions
ACTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

def plot_training_process(train_losses, train_accuracies, val_losses, val_accuracies, learning_rates):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(15, 10))

    # Plot training and validation loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot learning rate
    plt.subplot(2, 2, 3)
    plt.plot(epochs, learning_rates, 'g-')
    plt.title('Learning Rate')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')

    plt.tight_layout()
    plt.savefig('training_process.png')
    plt.close()

def plot_test_results(test_accuracy, action_distribution):
    plt.figure(figsize=(12, 5))

    # Plot test accuracy
    plt.subplot(1, 2, 1)
    plt.bar(['Test Accuracy'], [test_accuracy])
    plt.title('Test Accuracy')
    plt.ylabel('Accuracy')

    # Plot action distribution
    plt.subplot(1, 2, 2)
    actions = ['NW', 'N', 'NE', 'W', 'E', 'SW', 'S', 'SE', 'Stay']
    plt.bar(actions, action_distribution)
    plt.title('Action Distribution')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('test_results.png')
    plt.close()

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def natural_sort_key(s):
    parts = s.split('_')
    return (int(parts[0]), int(parts[1].split('.')[0]))

class VehicleDataset(Dataset):
    def __init__(self, data_dir):
        self.data_files = sorted(
            [f for f in os.listdir(data_dir) if f.endswith('.npy')],
            key=natural_sort_key
        )
        self.data_dir = data_dir
        self.vehicle_ids = [int(f.split('_')[0]) for f in self.data_files]

    def __len__(self):
        return len(self.data_files) - 1  # 减1是因为我们需要下一个状态来计算动作

    def __getitem__(self, idx):
        current_file = self.data_files[idx]
        next_file = self.data_files[idx + 1]
        
        current_data = np.load(os.path.join(self.data_dir, current_file))
        next_data = np.load(os.path.join(self.data_dir, next_file))
        
        # 直接使用加载的数组，而不是尝试访问 'data_matrix' 键
        current_input = current_data
        next_input = next_data
        
        current_pos = np.argwhere(current_input[:,:,1] == 1)[0]
        next_pos = np.argwhere(next_input[:,:,1] == 1)[0]
        
        action = tuple(next_pos - current_pos)
        action_index = ACTIONS.index(action) if action in ACTIONS else 8
        acceleration = current_input[current_pos[0], current_pos[1], 5] 

        return torch.FloatTensor(current_input), action_index, acceleration

    def get_filename(self, idx):
        return self.data_files[idx]

    def get_vehicle_id(self, idx):
        return self.vehicle_ids[idx]

def collate_fn(batch):
    inputs, actions, accelerations = zip(*batch)
    inputs = torch.stack(inputs)
    actions = torch.tensor(actions, dtype=torch.long)
    accelerations = torch.tensor(accelerations, dtype=torch.float)
    return inputs, actions, accelerations

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

class WeightedMultiTaskLoss(nn.Module):
    def __init__(self, action_weight=0.9, accel_weight=0.1, label_smoothing=0.1):
        super().__init__()
        self.action_weight = action_weight
        self.accel_weight = accel_weight
        self.action_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.accel_criterion = nn.MSELoss()

    def forward(self, action_outputs, action_targets, accel_outputs, accel_targets):
        action_loss = self.action_criterion(action_outputs, action_targets)
        accel_loss = self.accel_criterion(accel_outputs.squeeze(), accel_targets)
        return self.action_weight * action_loss + self.accel_weight * accel_loss
    
def train_epoch(model, train_loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    correct_actions = 0
    total_actions = 0

    for inputs, action_targets, accel_targets in train_loader:
        inputs = inputs.to(device)
        action_targets = action_targets.to(device)
        accel_targets = accel_targets.to(device)

        optimizer.zero_grad()
        action_outputs, accel_outputs = model(inputs)
        loss = criterion(action_outputs, action_targets, accel_outputs, accel_targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        _, predicted = action_outputs.max(1)
        correct_actions += (predicted == action_targets).sum().item()
        total_actions += action_targets.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = correct_actions / total_actions
    return avg_loss, accuracy

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct_actions = 0
    total_actions = 0

    with torch.no_grad():
        for inputs, action_targets, accel_targets in val_loader:
            inputs = inputs.to(device)
            action_targets = action_targets.to(device)
            accel_targets = accel_targets.to(device)

            action_outputs, accel_outputs = model(inputs)
            
            loss = criterion(action_outputs, action_targets, accel_outputs, accel_targets)
            total_loss += loss.item()

            _, predicted = action_outputs.max(1)
            correct_actions += (predicted == action_targets).sum().item()
            total_actions += action_targets.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = correct_actions / total_actions
    return avg_loss, accuracy

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device):
    criterion = WeightedMultiTaskLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = OneCycleLR(optimizer, max_lr=learning_rate, epochs=num_epochs, steps_per_epoch=len(train_loader))
    early_stopping = EarlyStopping(patience=50, min_delta=0.001)

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    learning_rates = []

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        learning_rates.append(optimizer.param_groups[0]['lr'])
        
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f'{now}, 回合 {epoch+1}/{num_epochs}:')
        print(f'训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}, 训练准确性: {train_accuracy:.4f}, 验证准确性: {val_accuracy:.4f}')

        # 使用改进的早停机制
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("损失未减少，提前停止训练")
            break

        # 保存最佳模型
        if epoch == 0 or val_loss < min(val_losses[:-1]):
            torch.save(model.state_dict(), 'best_model.pth')
            print("保存模型")
        else:
            print(f'已经忍耐：{early_stopping.counter}')
    return train_losses, train_accuracies, val_losses, val_accuracies, learning_rates

def create_subsets(dataset, train_id_max, val_id_max):
    train_indices = [i for i, vid in enumerate(dataset.vehicle_ids) if vid < train_id_max]
    val_indices = [i for i, vid in enumerate(dataset.vehicle_ids) if train_id_max <= vid < val_id_max]
    test_indices = [i for i, vid in enumerate(dataset.vehicle_ids) if vid >= val_id_max]
    
    return Subset(dataset, train_indices), Subset(dataset, val_indices), Subset(dataset, test_indices)

def print_dataset_info(dataset, name):
    vehicle_ids = [dataset.dataset.get_vehicle_id(i) for i in dataset.indices]
    unique_vehicles = len(set(vehicle_ids))
    max_id = max(vehicle_ids)
    print(f"{name} 数据集:")
    print(f"  总数据量: {len(dataset)}")
    print(f"  车数量: {unique_vehicles}")
    print(f"  最大车辆id: {max_id}")
    print(f"  id范围: {min(vehicle_ids)} - {max_id}")
    print(f"  前几个数据:")
    for i in range(min(5, len(dataset))):
        print(f"    {dataset.dataset.get_filename(dataset.indices[i])}")
    print()

def main():
    data_dir = "E:\\research\\code\\new_pro\\vehicle_data"  # Replace with your actual data directory
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # Fixed hyperparameters
    batch_size = 128
    learning_rate = 0.00001
    num_epochs = 100

    # Load data
    full_dataset = VehicleDataset(data_dir)
    
    # Custom split based on vehicle IDs
    train_id_max = 16800
    val_id_max = 21600
    
    train_dataset, val_dataset, test_dataset = create_subsets(full_dataset, train_id_max, val_id_max)
    
    # Print detailed dataset information
    print_dataset_info(train_dataset, "训练")
    print_dataset_info(val_dataset, "验证")
    print_dataset_info(test_dataset, "测试")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Initialize model
    model = ImprovedVehicleCNN().to(device)

    # Train model
    train_losses, train_accuracies, val_losses, val_accuracies, learning_rates = train_model(
        model, train_loader, val_loader, num_epochs, learning_rate, device)

    # Plot training process
    plot_training_process(train_losses, train_accuracies, val_losses, val_accuracies, learning_rates)

    # Load best model for testing
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Evaluate on test set
    criterion = WeightedMultiTaskLoss()
    test_loss, test_accuracy = validate(model, test_loader, criterion, device)
    print(f"测试损失: {test_loss:.4f}, 测试准确率: {test_accuracy:.4f}")

    # Calculate action distribution
    action_distribution = [0] * 9
    model.eval()
    with torch.no_grad():
        for inputs, action_targets, _ in test_loader:
            inputs = inputs.to(device)
            action_outputs, _ = model(inputs)
            _, predicted = action_outputs.max(1)
            for action in predicted.cpu().numpy():
                action_distribution[action] += 1

    # Normalize action distribution
    action_distribution = [count / sum(action_distribution) for count in action_distribution]

    # Plot test results
    plot_test_results(test_accuracy, action_distribution)

if __name__ == "__main__":
    main()