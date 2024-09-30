import sys
import torch
import os
import gc
import time
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, 
                             QSlider, QSpinBox, QGroupBox, QDoubleSpinBox, QProgressBar, QSplitter, QCheckBox, 
                             QTextEdit, QOpenGLWidget, QGridLayout, QComboBox, QFileDialog)
from PyQt5.QtCore import Qt, QTimer, QSize, QThread, pyqtSignal, QObject, pyqtSlot, QMetaObject, Q_ARG
from PyQt5.QtGui import QIcon, QIntValidator
from OpenGL.GL import *
from OpenGL.GLU import *
from vehicle import Vehicle, Vector2, ObstacleVehicle, generate_datamatrix, ReservationSystem
from renderer import RoadVehicleRenderer
from policy import RLVehicleCNN
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Tuple, List

# 定义动作空间
ACTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1), (0, 0)]

# PPO算法实现
class PPO:
    def __init__(self, model, lr=3e-4, gamma=0.99, epsilon=0.2, value_coef=0.5, entropy_coef=0.01):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.device = next(model.parameters()).device

    def update(self, states, actions, old_probs, rewards, dones, advantages, returns):
        states = torch.from_numpy(np.array(states)).float().to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        old_probs = torch.tensor(old_probs, dtype=torch.float).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float).to(self.device)

        for _ in range(10):
            new_probs, acceleration, values = self.model(states)
            new_probs = new_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            ratio = torch.exp(new_probs - old_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = (returns - values.squeeze(1)).pow(2).mean()
            
            entropy = -(new_probs * torch.log(new_probs + 1e-8)).mean()
            
            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return actor_loss.item(), critic_loss.item(), entropy.item()

class RLTrainer(QObject):
    finished = pyqtSignal()
    update_log = pyqtSignal(str)

    def __init__(self, main_window, num_episodes):
        super().__init__()
        self.main_window = main_window
        self.num_episodes = num_episodes

    def run(self):
        device = next(self.main_window.model.parameters()).device
        start_time = time.time()
        for episode in range(self.num_episodes):
            state = self.main_window.reset_simulation()
            done = False
            episode_reward = 0
            states, actions, old_probs, rewards, dones = [], [], [], [], []
            
            while not done:
                action, acceleration = self.main_window.ai_controller.get_action(state)
                
                next_state, reward, done, _ = self.main_window.step(action, acceleration)
                
                states.append(state)
                actions.append(action)
                old_probs.append(0)  # 我们现在不使用old_probs，可以后续优化
                rewards.append(reward)
                dones.append(done)
                
                state = next_state
                episode_reward += reward
            
            advantages, returns = self.compute_advantages_and_returns(rewards, dones)
            
            actor_loss, critic_loss, entropy = self.main_window.ppo.update(
                states, actions, old_probs, rewards, dones, advantages, returns)
            
            self.update_log.emit(f"Episode {episode}, Reward: {episode_reward}, Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}, Entropy: {entropy:.4f}")
            
            if episode % 100 == 0:
                torch.save(self.main_window.model.state_dict(), f"rl_model_episode_{episode}.pth")
        
        # 保存最终模型
        final_model_path = f"rl_model_final.pth"
        torch.save(self.main_window.model.state_dict(), final_model_path)
        self.update_log.emit(f"Final model saved as {final_model_path}")

        total_time = time.time() - start_time
        self.update_log.emit(f"Training completed. Total time: {total_time:.2f}s")
        self.finished.emit()

    def compute_advantages_and_returns(self, rewards, dones):
        advantages = []
        returns = []
        running_return = 0
        running_advantage = 0
        
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.main_window.ppo.gamma * running_return * (1 - dones[t])
            running_advantage = rewards[t] + self.main_window.ppo.gamma * running_advantage * (1 - dones[t]) - running_return
            advantages.insert(0, running_advantage)
            returns.insert(0, running_return)
        
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        return advantages, returns

class AsyncLogger:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1)

    def log(self, message):
        self.executor.submit(self._log, message)

    def _log(self, message):
        with open("simulation_log.txt", "a") as f:
            f.write(message + "\n")

    def shutdown(self):
        self.executor.shutdown(wait=True)

class OpenGLUpdateSignal(QObject):
    update_signal = pyqtSignal(object, object)

class AIController:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def get_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(state_tensor)
            action_probs = outputs[0]
            acceleration = outputs[1]
        action = torch.argmax(action_probs, dim=1).item()
        acceleration = acceleration.item()
        return action, acceleration

class OpenGLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super(OpenGLWidget, self).__init__(parent)
        self.renderer = RoadVehicleRenderer()
        self.setMinimumSize(600, 600)

    def initializeGL(self):
        self.renderer.initialize()

    def resizeGL(self, width, height):
        side = min(width, height)
        glViewport((width - side) // 2, (height - side) // 2, side, side)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, 12, 0, 12, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        self.renderer.update_size(side, side)

    def paintGL(self):
        self.renderer.render()

    def update_scene(self, ego_vehicle, obstacle_vehicles):
        self.renderer.update_simulation(ego_vehicle, obstacle_vehicles)
        self.update()

    def sizeHint(self):
        return QSize(600, 600)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # 将所有控件的初始化放在这里
        self.training_episodes_input = QSpinBox()
        self.training_episodes_input.setRange(1, 100000)
        self.training_episodes_input.setValue(1000)
        self.training_episodes_input.setSingleStep(100)
        self.initUI()
        self.simulation_running = False
        self.simulation_paused = False
        self.speed_multiplier = 1.0
        self.ego_vehicle = None
        self.obstacle_vehicles = []
        self.obstacle_generation_timer = 0
        self.obstacle_generation_interval = 0.2
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_simulation)
        self.timer.start(16)  # ~60 FPS
        self.current_run = 0
        self.collision_count = 0  #总碰撞计数
        self.total_runs = 1
        self.generate_data = False
        self.data_save_path = "E:\\research\\code\\new_pro\\vehicle_data"

        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ai_controller = None
        self.ppo = None
        self.rl_trainer = None  # 初始化 rl_trainer 为 None

        self.setWindowIcon(QIcon('assets/icon.ico'))
        self.simulation_log = ""
        self.async_logger = AsyncLogger()

        self.opengl_updater = OpenGLUpdateSignal()
        self.opengl_updater.update_signal.connect(self.opengl_widget.update_scene)
        self.reservation_system = ReservationSystem(12)

        self.reset_simulation()

    def initUI(self):
        self.setWindowTitle('Simulator')
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        self.opengl_widget = OpenGLWidget()
        splitter.addWidget(self.opengl_widget)
        
        self.control_panel = QWidget()
        control_layout = QVBoxLayout(self.control_panel)
        
        self.add_control_panel_components(control_layout)
        
        splitter.addWidget(self.control_panel)
        
        splitter.setSizes([600, 200])
        
        self.setMinimumSize(800, 600)
        self.ai_control_checkbox.stateChanged.connect(self.toggle_ai_control)
        

    def add_control_panel_components(self, layout):
        # 车辆和障碍物设置
        settings_group = QGroupBox("车辆和障碍物设置")
        settings_layout = QGridLayout()
        
        settings_layout.addWidget(QLabel("起始:"), 0, 0)
        self.start_x = QSpinBox()
        self.start_y = QSpinBox()
        self.start_x.setRange(0, 11)
        self.start_y.setRange(0, 11)
        settings_layout.addWidget(self.start_x, 0, 1)
        settings_layout.addWidget(self.start_y, 0, 2)
        
        settings_layout.addWidget(QLabel("终点:"), 0, 3)
        self.end_x = QSpinBox()
        self.end_y = QSpinBox()
        self.end_x.setRange(0, 11)
        self.end_y.setRange(0, 11)
        self.end_x.setValue(11)
        self.end_y.setValue(11)
        settings_layout.addWidget(self.end_x, 0, 4)
        settings_layout.addWidget(self.end_y, 0, 5)
        
        settings_layout.addWidget(QLabel("障碍物生成间隔:"), 1, 0, 1, 2)
        self.obstacle_interval = QDoubleSpinBox()
        self.obstacle_interval.setRange(0.01, 1.0)
        self.obstacle_interval.setValue(0.10)
        self.obstacle_interval.setSingleStep(0.01)
        settings_layout.addWidget(self.obstacle_interval, 1, 2, 1, 2)
        
        settings_layout.addWidget(QLabel("仿真次数:"), 2, 0)
        self.sim_runs_input = QComboBox()
        self.sim_runs_input.addItems(['1', '10', '50', '100', '500', '1000'])
        self.sim_runs_input.setEditable(True)
        self.sim_runs_input.setValidator(QIntValidator(1, 10000))
        settings_layout.addWidget(self.sim_runs_input, 2, 1, 1, 2)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # 控制选项
        control_group = QGroupBox("控制选项")
        control_layout = QGridLayout()
        
        self.ai_control_checkbox = QCheckBox("使用模型控制")
        self.ai_control_checkbox.stateChanged.connect(self.toggle_ai_control)
        control_layout.addWidget(self.ai_control_checkbox, 0, 0)
        
        # 添加训练回合数设置
        training_layout = QHBoxLayout()
        training_layout.addWidget(QLabel("训练回合数:"))
        training_layout.addWidget(self.training_episodes_input)
        layout.addLayout(training_layout)
        
        self.data_generation_checkbox = QCheckBox("生成数据")
        self.data_generation_checkbox.stateChanged.connect(self.toggle_data_generation)
        control_layout.addWidget(self.data_generation_checkbox, 0, 1)
        
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

        # 仿真控制
        sim_control_layout = QHBoxLayout()
        self.start_button = QPushButton('启动仿真', self)
        self.start_button.clicked.connect(self.toggle_start)
        sim_control_layout.addWidget(self.start_button)

        self.pause_button = QPushButton('暂停', self)
        self.pause_button.clicked.connect(self.toggle_pause)
        self.pause_button.setEnabled(False)
        sim_control_layout.addWidget(self.pause_button)

        # 添加选择保存路径的按钮
        self.select_path_button = QPushButton("选择保存路径", self)
        self.select_path_button.clicked.connect(self.select_save_path)
        layout.addWidget(self.select_path_button)

        self.load_model_button = QPushButton("加载模型", self)
        self.load_model_button.clicked.connect(self.load_model_from_file)
        sim_control_layout.addWidget(self.load_model_button)

        # 添加RL训练按钮
        self.rl_train_button = QPushButton('开始RL训练', self)
        self.rl_train_button.clicked.connect(self.start_rl_training)
        sim_control_layout.addWidget(self.rl_train_button)

        layout.addLayout(sim_control_layout)

        # 仿真速度控制
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel('仿真速度:'))
        self.speed_slider = QSlider(Qt.Horizontal, self)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(10)
        self.speed_slider.setValue(1)
        self.speed_slider.valueChanged.connect(self.update_speed)
        speed_layout.addWidget(self.speed_slider)
        self.speed_label = QLabel('1.0x', self)
        speed_layout.addWidget(self.speed_label)
        layout.addLayout(speed_layout)

        # 进度条
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # 车辆信息
        self.vehicle_info_label = QLabel('车辆信息:', self)
        layout.addWidget(self.vehicle_info_label)

        # 清除日志按钮
        log_control_layout = QHBoxLayout()
        self.clear_log_button = QPushButton("清除日志", self)
        self.clear_log_button.clicked.connect(self.clear_simulation_log)
        log_control_layout.addWidget(self.clear_log_button)
        layout.addLayout(log_control_layout)

        # 模拟日志
        self.log_text = QTextEdit(self)
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

    def select_save_path(self):
        directory = QFileDialog.getExistingDirectory(self, "选择保存路径", "", QFileDialog.ShowDirsOnly)
        if directory:
            self.data_save_path = directory
            self.update_simulation_log(f"数据保存路径已更改为: {self.data_save_path}")
            if self.ego_vehicle:
                self.ego_vehicle.data_save_path = self.data_save_path

    def load_model_from_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "PyTorch Model Files (*.pth)")
        if file_name:
            try:
                self.model = RLVehicleCNN(include_value_function=False).to(self.device)
                state_dict = torch.load(file_name, map_location=self.device, weights_only=True)
                
                # 加载可用的权重
                self.model.load_state_dict(state_dict, strict=False)
                
                self.model.eval()
                self.ai_controller = AIController(self.model, self.device)
                self.update_simulation_log(f"模型已从 {file_name} 加载")
            except Exception as e:
                self.update_simulation_log(f"加载模型时出错: {str(e)}")

    def toggle_ai_control(self, state):
        if state == Qt.Checked:
            if self.model is None:
                self.update_simulation_log("请先加载模型")
                self.ai_control_checkbox.setChecked(False)
                return
        
        self.use_ai_control = state == Qt.Checked
        if self.ego_vehicle:
            self.ego_vehicle.use_ai_control = self.use_ai_control

    def toggle_data_generation(self, state):
        self.generate_data = state == Qt.Checked
        if self.ego_vehicle:
            self.ego_vehicle.generate_data = self.generate_data
            self.update_vehicle_data_path()

    def update_vehicle_data_path(self):
        if self.ego_vehicle and self.ego_vehicle.generate_data:
            self.ego_vehicle.data_save_path = self.data_save_path
            if not os.path.exists(self.data_save_path):
                os.makedirs(self.data_save_path)
        else:
            self.ego_vehicle.data_save_path = None

    def update_speed(self, value):
        self.speed_multiplier = value
        self.speed_label.setText(f'速度倍率: {value:.1f}x')

    def clear_simulation_log(self):
        self.simulation_log = ""
        self.log_text.clear()

    def update_simulation_log(self, message):
        # 使用 QMetaObject.invokeMethod 确保在主线程中更新 UI
        QMetaObject.invokeMethod(self.log_text, "append", 
                                 Qt.QueuedConnection, 
                                 Q_ARG(str, message))
        self.async_logger.log(message)

    def toggle_start(self):
        if not self.simulation_running and not self.simulation_paused:
            try:
                self.total_runs = int(self.sim_runs_input.currentText())
            except ValueError:
                self.update_simulation_log("非法输入仿真次数，使用默认值 1")
                self.total_runs = 1
            self.collision_count = 0
            self.current_run = 0
            self.start_next_run()
        else:
            self.simulation_running = False
            self.simulation_paused = False
            self.start_button.setText('启动仿真')
            self.pause_button.setText('暂停仿真')
            self.pause_button.setEnabled(False)

    def toggle_pause(self):
        if self.simulation_running:
            self.simulation_running = False
            self.simulation_paused = True
            self.pause_button.setText('继续')
        elif self.simulation_paused:
            self.simulation_running = True
            self.simulation_paused = False
            self.pause_button.setText('暂停')

    def start_next_run(self):
        if self.current_run < self.total_runs:
            self.reset_simulation()  # 在每次运行开始时重置仿真
            self.current_run += 1
            start = (self.start_x.value(), self.start_y.value())
            end = (self.end_x.value(), self.end_y.value())
            self.ego_vehicle = Vehicle(start, end, self.current_run, self.data_save_path)
            self.ego_vehicle.use_ai_control = self.ai_control_checkbox.isChecked()
            self.ego_vehicle.generate_data = self.generate_data
            self.update_vehicle_data_path()
            self.ego_vehicle.set_ai_controller(self.ai_controller)
            self.obstacle_vehicles = []
            self.obstacle_generation_timer = 0
            self.obstacle_generation_interval = self.obstacle_interval.value()
            self.simulation_running = True
            self.simulation_paused = False
            self.start_button.setText('停止')
            self.pause_button.setText('暂停')
            self.pause_button.setEnabled(True)
            self.progress_bar.setValue(int((self.current_run - 1) / self.total_runs * 100))
        else:
            self.simulation_running = False
            self.simulation_paused = False
            self.start_button.setText('启动仿真')
            self.pause_button.setText('暂停')
            self.pause_button.setEnabled(False)
            self.progress_bar.setValue(100)
            self.update_simulation_log("所有仿真运行完成!")
            self.collision_count = 0
            self.current_run = 0
            #self.clear_simulation_log()

    @pyqtSlot(str)
    def update_log_from_trainer(self, message):
        self.update_simulation_log(message)

    def start_rl_training(self):
        if self.model is None:
            self.update_simulation_log("请先加载模型再开始RL训练")
            return
        else:
            self.update_simulation_log("开始RL训练")
            device = torch.cuda.get_device_name(0)
            self.update_simulation_log(f"使用设备: {device}")
        # 在开始训练前添加值函数层
        self.model.add_value_function()
        
        # 更新 ai_controller 以使用新的模型结构
        self.ai_controller = AIController(self.model, self.device)
        
        self.ppo = PPO(self.model)  # 只在开始训练时初始化PPO
        
        num_episodes = self.training_episodes_input.value()
    
        self.rl_training_thread = QThread()
        self.rl_trainer = RLTrainer(self, num_episodes)
        self.rl_trainer.moveToThread(self.rl_training_thread)
        self.rl_training_thread.started.connect(self.rl_trainer.run)
        self.rl_trainer.finished.connect(self.rl_training_thread.quit)
        self.rl_trainer.finished.connect(self.rl_trainer.deleteLater)
        self.rl_training_thread.finished.connect(self.rl_training_thread.deleteLater)
        self.rl_trainer.update_log.connect(self.update_log_from_trainer)
        self.rl_training_thread.start()

    def reset_simulation(self):
        self.ego_vehicle = None
        self.obstacle_vehicles = []
        self.obstacle_generation_timer = 0
        self.reservation_system = ReservationSystem(12)  # 重新初始化预约系统
        ObstacleVehicle.next_id = -1  # 重置障碍物ID计数器

        start = (self.start_x.value(), self.start_y.value())
        end = (self.end_x.value(), self.end_y.value())
        self.ego_vehicle = Vehicle(start, end, self.current_run, self.data_save_path)
        self.ego_vehicle.use_ai_control = True
        self.ego_vehicle.generate_data = self.generate_data
        self.ego_vehicle.set_ai_controller(self.ai_controller)
        
        state = generate_datamatrix(self.ego_vehicle, self.obstacle_vehicles)
        return state

    def step(self, action, acceleration):
        self.ego_vehicle.action = ACTIONS[action]
        self.ego_vehicle.ai_acceleration = acceleration  # 使用 ai_acceleration 来存储标量加速度
        
        self.update_simulation()
        
        new_state = generate_datamatrix(self.ego_vehicle, self.obstacle_vehicles)
        
        reward = self.ego_vehicle.calculate_reward(self.ego_vehicle.is_colliding, self.ego_vehicle.reached_target)
        
        done = self.ego_vehicle.is_colliding or self.ego_vehicle.reached_target or self.is_out_of_bounds()
        
        return new_state, reward, done, {}
    
    def update_simulation(self):
        if self.simulation_running and self.ego_vehicle:
            base_dt = 0.016
            dt = base_dt * self.speed_multiplier

            # 获取当前状态
            state = generate_datamatrix(self.ego_vehicle, self.obstacle_vehicles)

            # 获取 AI 控制器的动作
            if self.ego_vehicle.use_ai_control and self.ai_controller:
                action, acceleration = self.ai_controller.get_action(state)
                #self.update_simulation_log(f"AI 控制: 动作 {ACTIONS[action]}, 加速度 {acceleration:.1f}")

            # 执行动作
            self.ego_vehicle.move(dt, self.obstacle_vehicles)

            # 生成和更新障碍物
            self.update_obstacles(dt)

            # 检查碰撞
            if self.ego_vehicle.check_collision(self.obstacle_vehicles):
                self.collision_count += 1
                self.update_simulation_log(f"碰撞发生! 总碰撞次数: {self.collision_count}")
                
            # 更新 OpenGL 场景
            self.opengl_updater.update_signal.emit(self.ego_vehicle, self.obstacle_vehicles)

            # 更新车辆信息
            self.update_vehicle_info()

            # 检查是否到达目的地或失败
            if self.ego_vehicle.reached_target:
                self.update_simulation_log(f"仿真完成! 到达终点")
                self.start_next_run()
            elif self.is_out_of_bounds():
                self.update_simulation_log(f"仿真失败! 超出边界")
                self.start_next_run()


    def generate_obstacle_vehicle(self):
        other_vehicles_info = self.get_vehicles_info()
        new_obstacle = ObstacleVehicle.generate_valid_obstacle(12, other_vehicles_info, self.reservation_system)
        return new_obstacle

    def update_obstacles(self, dt):
        # 更新现有障碍物
        for obstacle in self.obstacle_vehicles:
            obstacle.move(dt)
        
        # 移除过期的障碍物
        expired_obstacles = [obs for obs in self.obstacle_vehicles if obs.is_expired()]
        for obs in expired_obstacles:
            self.reservation_system.clear_reservations(obs.id)
            self.obstacle_vehicles.remove(obs)

        # 获取所有车辆的信息
        vehicles_info = self.get_vehicles_info()

        # 为现有的障碍车辆选择新的动作（如果需要）
        for obstacle in self.obstacle_vehicles:
            if obstacle.action_completed:
                new_action = obstacle.choose_valid_action(vehicles_info)
                if new_action != (0, 0):
                    obstacle.action = new_action
                    obstacle.target = Vector2(obstacle.grid_position.x + new_action[0], obstacle.grid_position.y + new_action[1])
                    obstacle.action_completed = False
                else:
                    # 如果没有有效的动作，考虑移除这个障碍物
                    self.reservation_system.clear_reservations(obstacle.id)
                    self.obstacle_vehicles.remove(obstacle)

        # 生成新的障碍物
        self.obstacle_generation_timer += dt
        if self.obstacle_generation_timer >= self.obstacle_generation_interval / self.speed_multiplier:
            self.obstacle_generation_timer = 0
            new_obstacle = self.generate_obstacle_vehicle()
            if new_obstacle:
                self.obstacle_vehicles.append(new_obstacle)

    def update_vehicle_info(self):
        if self.ego_vehicle:
            x, y, _, speed, _, acc = self.ego_vehicle.get_state()
            info_text = f"当前运行: {self.current_run}/{self.total_runs}\n"
            info_text += f"速度: {speed/3.6:.1f} km/h\n"
            info_text += f"位置: ({x:.1f}, {y:.1f})\n"
            info_text += f"加速度: {acc / 10:.1f} m/s^2"
            self.vehicle_info_label.setText(info_text)

    def is_collision(self):
        return any(self.ego_vehicle.collides_with(obs) for obs in self.obstacle_vehicles)

    def is_out_of_bounds(self):
        x, y = self.ego_vehicle.grid_position.x, self.ego_vehicle.grid_position.y
        return x < 0 or x > 11 or y < 0 or y > 11

    def closeEvent(self, event):
        if self.rl_trainer:
            self.rl_trainer.finished.emit()  # 发送完成信号以停止训练线程
        self.unload_model()
        self.async_logger.shutdown()
        super().closeEvent(event)

    def unload_model(self):
        self.model = None
        self.ai_controller = None
        torch.cuda.empty_cache()  # 如果使用GPU，清理显存
        self.update_simulation_log("模型已清除")

    def generate_obstacle_vehicle(self):
        other_vehicles_info = self.get_vehicles_info()
        new_obstacle = ObstacleVehicle.generate_valid_obstacle(12, other_vehicles_info, self.reservation_system)
        if new_obstacle:
            return new_obstacle
        else:
            self.update_simulation_log("无法生成障碍物")
            return None

    def get_available_positions(self, other_vehicles_info: Dict[Tuple[int, int], Tuple[int, int]]) -> List[Tuple[int, int]]:
        occupied = set(other_vehicles_info.keys()) | set(other_vehicles_info.values())
        all_positions = [(x, y) for x in range(12) for y in range(12)]
        return [pos for pos in all_positions if pos not in occupied]
    
    def get_vehicles_info(self) -> Dict[Tuple[int, int], Tuple[int, int]]:
        vehicles_info = {}
        
        if self.ego_vehicle:
            ego_pos = (int(self.ego_vehicle.x // 50), int(self.ego_vehicle.y // 50))
            ego_target = self.ego_vehicle.end
            vehicles_info[ego_pos] = ego_target

        for obs in self.obstacle_vehicles:
            obs_pos = (int(obs.x // 50), int(obs.y // 50))
            obs_target = obs.get_target_position()
            vehicles_info[obs_pos] = obs_target

        return vehicles_info

    def is_valid_position(self, pos: Tuple[int, int], vehicles_info: Dict[Tuple[int, int], Tuple[int, int]]) -> bool:
        return pos not in vehicles_info and pos not in vehicles_info.values()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())