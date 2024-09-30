from OpenGL.GL import *
from OpenGL.GLU import *
import math
import numpy as np
from PyQt5.QtCore import QMutex
from vehicle import VehicleInfo
from vehicle import ObstacleVehicle

class RoadVehicleRenderer:
    def __init__(self):
        self.width = 600
        self.height = 600
        self.grid_count = 12
        
        self.ego_vehicle = None
        self.obstacle_vehicles = []
        
        self.mutex = QMutex()

    def initialize(self):
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def render(self):
        self.mutex.lock()
        try:
            self.clear()
            self.render_grid()
            
            if self.ego_vehicle:
                self.render_radar_range(self.ego_vehicle)
            
            for vehicle in self.obstacle_vehicles:
                self.render_vehicle(vehicle)
            if self.ego_vehicle:
                self.render_vehicle(self.ego_vehicle)
                self.render_path(self.ego_vehicle.path)
            
            if self.ego_vehicle:
                detected_obstacles = self.ego_vehicle.scan_radar(self.obstacle_vehicles)
                self.render_radar_points(detected_obstacles)
        finally:
            self.mutex.unlock()

    def update_simulation(self, ego_vehicle, obstacle_vehicles):
        self.mutex.lock()
        try:
            self.ego_vehicle = ego_vehicle
            self.obstacle_vehicles = obstacle_vehicles
        finally:
            self.mutex.unlock()

    def update_size(self, width, height):
        self.mutex.lock()
        try:
            self.width = width
            self.height = height
        finally:
            self.mutex.unlock()

    def render_radar_range(self, vehicle):
        x, y = vehicle.x / 50, vehicle.y / 50
        radar_range = vehicle.radar_range / 50

        glColor4f(0, 1, 0, 0.2)  # 半透明绿色
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(x, y)
        for angle in range(0, 361, 10):
            rad = math.radians(angle)
            glVertex2f(x + radar_range * math.cos(rad), y + radar_range * math.sin(rad))
        glEnd()

    def render_radar_points(self, detected_obstacles):
        if detected_obstacles:
            glColor3f(1, 0, 0)  # 红色
            glPointSize(10.0)  # 增大点的大小
            glBegin(GL_POINTS)
            for obs in detected_obstacles:
                glVertex2f(obs.x / 50, obs.y / 50)
            glEnd()

    def render_grid(self):
        glColor3f(0.5, 0.5, 0.5)
        glBegin(GL_QUADS)
        glVertex2f(0, 0)
        glVertex2f(12, 0)
        glVertex2f(12, 12)
        glVertex2f(0, 12)
        glEnd()

        glColor3f(1, 1, 1)
        glBegin(GL_LINES)
        for i in range(self.grid_count + 1):
            glVertex2f(i, 0)
            glVertex2f(i, 12)
            glVertex2f(0, i)
            glVertex2f(12, i)
        glEnd()

    def render_vehicle(self, vehicle):
        x, y, yaw, _, _, _ = vehicle.get_state()
        x, y = x / 50, y / 50  # 转换为 12x12 坐标系

        glPushMatrix()
        glTranslatef(x, y, 0)
        glRotatef(math.degrees(yaw), 0, 0, 1)
        
        if isinstance(vehicle, ObstacleVehicle):
            self.draw_obstacle_vehicle(vehicle.color)
        else:
            self.draw_vehicle(vehicle.color, vehicle.wheel_color)
        
        glPopMatrix()

    def draw_vehicle(self, body_color, wheel_color):
        # 绘制车身
        glColor3f(*body_color)
        glBegin(GL_QUADS)
        glVertex2f(-VehicleInfo.LENGTH/100, -VehicleInfo.WIDTH/100)
        glVertex2f(VehicleInfo.LENGTH/100, -VehicleInfo.WIDTH/100)
        glVertex2f(VehicleInfo.LENGTH/100, VehicleInfo.WIDTH/100)
        glVertex2f(-VehicleInfo.LENGTH/100, VehicleInfo.WIDTH/100)
        glEnd()

        # 绘制车轮
        glColor3f(*wheel_color)
        wheel_positions = [
            (VehicleInfo.LENGTH/180, VehicleInfo.WIDTH/100),
            (VehicleInfo.LENGTH/180, -VehicleInfo.WIDTH/100),
            (-VehicleInfo.LENGTH/180, VehicleInfo.WIDTH/100),
            (-VehicleInfo.LENGTH/180, -VehicleInfo.WIDTH/100)
        ]
        for wx, wy in wheel_positions:
            glPushMatrix()
            glTranslatef(wx, wy, 0)
            glBegin(GL_QUADS)
            glVertex2f(-VehicleInfo.WHEEL_WIDTH/100, -VehicleInfo.WHEEL_DIAMETER/200)
            glVertex2f(VehicleInfo.WHEEL_WIDTH/100, -VehicleInfo.WHEEL_DIAMETER/200)
            glVertex2f(VehicleInfo.WHEEL_WIDTH/100, VehicleInfo.WHEEL_DIAMETER/200)
            glVertex2f(-VehicleInfo.WHEEL_WIDTH/100, VehicleInfo.WHEEL_DIAMETER/200)
            glEnd()
            glPopMatrix()

    def draw_obstacle_vehicle(self, body_color):
        self.draw_vehicle(body_color, (0, 0, 0))

    def draw_polygon(self, vertices):
        glBegin(GL_POLYGON)
        for vertex in vertices.T:
            glVertex2f(vertex[0], vertex[1])
        glEnd()

    def render_path(self, path):
        if not path:
            return
        
        glColor3f(0, 1, 0)
        glBegin(GL_LINE_STRIP)
        for x, y in path:
            glVertex2f(x + 0.5, y + 0.5)
        glEnd()
        
    def clear(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)