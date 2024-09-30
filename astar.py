class Node:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent
        self.g = 0  # 起点到当前节点的实际代价
        self.h = 0  # 当前节点到目标节点的估计代价
        self.f = 0  # 总代价

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __lt__(self, other):
        return self.f < other.f
    
import heapq

def a_star(start, end, obstacles):
    """
    使用A*算法规划路径。

    Args:
        start: 起点坐标 (x, y)。
        end: 终点坐标 (x, y)。
        obstacles: 障碍物列表，每个障碍物是一个 (x, y) 坐标元组。

    Returns:
        路径列表，每个元素是一个 (x, y) 坐标元组，或者 None 如果没有找到路径。
    """
    
    start_node = Node(start[0], start[1])
    end_node = Node(end[0], end[1])

    open_list = []
    closed_list = []

    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)
        closed_list.append(current_node)

        if current_node == end_node:
            return reconstruct_path(current_node)

        for neighbor in get_neighbors(current_node, obstacles):
            if neighbor in closed_list:
                continue

            tentative_g_score = current_node.g + 1  # 假设每个移动的代价为1

            if neighbor not in open_list or tentative_g_score < neighbor.g:
                neighbor.parent = current_node
                neighbor.g = tentative_g_score
                neighbor.h = heuristic(neighbor, end_node)  # 使用曼哈顿距离作为启发式函数
                neighbor.f = neighbor.g + neighbor.h

                if neighbor not in open_list:
                    heapq.heappush(open_list, neighbor)

    return None  # 没有找到路径

def reconstruct_path(node):
    """从目标节点回溯到起点节点，构建路径列表。"""
    path = []
    while node is not None:
        path.append((node.x, node.y))
        node = node.parent
    return path[::-1]  # 反转路径

def get_neighbors(node, obstacles):
    """获取当前节点的邻居节点，不包括障碍物节点。"""
    neighbors = []
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]: # 八个方向
        x = node.x + dx
        y = node.y + dy
        if 0 <= x < 12 and 0 <= y < 12 and (x, y) not in obstacles:  # 检查边界和障碍物
            neighbors.append(Node(x, y))
    return neighbors

def heuristic(node, end_node):
    """计算当前节点到目标节点的曼哈顿距离。"""
    return abs(node.x - end_node.x) + abs(node.y - end_node.y)