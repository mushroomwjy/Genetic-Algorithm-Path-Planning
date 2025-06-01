import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Tuple, Set, Optional
from datetime import datetime

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

class GridMap:
    """表示网格地图"""
    def __init__(self, size: int = 25, obstacle_ratio: float = 0.2):
        self.size = size
        self.obstacle_ratio = obstacle_ratio
        self.obstacles = self._generate_obstacles()
        
    def _generate_obstacles(self) -> Set[Tuple[int, int]]:
        """生成随机障碍物"""
        num_obstacles = int(self.size * self.size * self.obstacle_ratio)
        obstacles = set()
        
        while len(obstacles) < num_obstacles:
            x = random.randint(0, self.size - 1)
            y = random.randint(0, self.size - 1)
            obstacles.add((x, y))
            
        return obstacles
    
    def is_valid(self, pos: Tuple[int, int]) -> bool:
        """检查位置是否有效（在地图内且不是障碍物）"""
        x, y = pos
        return 0 <= x < self.size and 0 <= y < self.size and (x, y) not in self.obstacles
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """获取相邻的有效位置，包括对角线方向"""
        x, y = pos
        # 增加对角线方向的移动
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        neighbors = []
        
        for dx, dy in directions:
            new_pos = (x + dx, y + dy)
            # 确保对角线移动不会穿过障碍物
            if abs(dx) + abs(dy) == 2:
                if not self.is_valid((x + dx, y)) or not self.is_valid((x, y + dy)):
                    continue
            if self.is_valid(new_pos):
                neighbors.append(new_pos)
                
        return neighbors

class GeneticAlgorithm:
    """遗传算法实现"""
    def __init__(self, grid_map: GridMap, start: Tuple[int, int], end: Tuple[int, int], 
                 pop_size: int = 200, max_gen: int = 100, 
                 p_cross: float = 0.7, p_mutation: float = 0.3,
                 w_length: float = 0.8, w_smooth: float = 0.2):  # 调整权重
        self.grid_map = grid_map
        self.start = start
        self.end = end
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.p_cross = p_cross
        self.p_mutation = p_mutation
        self.w_length = w_length
        self.w_smooth = w_smooth
        
        self.path_lengths = []  # 记录每代最优路径长度
        self.fitness_values = []  # 记录每代最大适应度
        self.avg_lengths = []  # 记录每代平均路径长度
        self.avg_fitness = []  # 记录每代平均适应度
        
    def initialize_population(self) -> List[List[Tuple[int, int]]]:
        """初始化种群"""
        population = []
        
        # 生成多样化的初始种群
        for _ in range(self.pop_size):
            if random.random() < 1:  # 使用改进的随机路径
                path = self._generate_improved_random_path()
            else:  # 使用A*启发式路径
                path = self._generate_heuristic_path()
            population.append(path)
            
        return population
    
    def _generate_random_path(self) -> List[Tuple[int, int]]:
        """生成随机路径（复用改进版随机路径生成逻辑）"""
        return self._generate_improved_random_path()
    
    def _generate_improved_random_path(self) -> List[Tuple[int, int]]:
        """生成改进的随机路径，增加到达终点的可能性"""
        path = [self.start]
        current = self.start
        max_steps = self.grid_map.size * 2  # 增加最大步数
        
        for _ in range(max_steps):
            neighbors = self.grid_map.get_neighbors(current)
            
            if not neighbors:
                break
                
            # 优先选择靠近终点的邻居
            if self.end in neighbors:
                path.append(self.end)
                return path
                
            # 计算每个邻居到终点的曼哈顿距离
            distances = [abs(n[0] - self.end[0]) + abs(n[1] - self.end[1]) for n in neighbors]
            min_dist = min(distances)
            
            # 选择距离最小的邻居的概率更高
            candidates = []
            for i, dist in enumerate(distances):
                if dist == min_dist:
                    candidates.append(neighbors[i])
            
            if candidates and random.random() < 0.7:  # 70%概率选择最优方向
                next_pos = random.choice(candidates)
            else:  # 30%概率随机选择
                next_pos = random.choice(neighbors)
                
            path.append(next_pos)
            current = next_pos
            
        # 如果没有到达终点，尝试直接连接
        if path[-1] != self.end:
            path.append(self.end)
            
        return path
    
    def _generate_heuristic_path(self) -> List[Tuple[int, int]]:
        """使用A*启发式生成路径，提高初始种群质量"""
        open_set = {self.start}
        closed_set = set()
        came_from = {}
        
        g_score = {self.start: 0}
        f_score = {self.start: self._heuristic_cost_estimate(self.start)}
        
        while open_set:
            current = min(open_set, key=lambda pos: f_score[pos])
            
            if current == self.end:
                # 重构路径
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path
                
            open_set.remove(current)
            closed_set.add(current)
            
            for neighbor in self.grid_map.get_neighbors(current):
                if neighbor in closed_set:
                    continue
                    
                # 对角线移动的代价为√2，直线移动为1
                dx = neighbor[0] - current[0]
                dy = neighbor[1] - current[1]
                move_cost = 1.414 if abs(dx) + abs(dy) == 2 else 1
                
                tentative_g_score = g_score[current] + move_cost
                
                if neighbor not in open_set:
                    open_set.add(neighbor)
                elif tentative_g_score >= g_score.get(neighbor, float('inf')):
                    continue
                    
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + self._heuristic_cost_estimate(neighbor)
                
        # 如果无法找到路径，返回随机路径
        return self._generate_random_path()
    
    def _heuristic_cost_estimate(self, pos: Tuple[int, int]) -> float:
        """启发式函数：计算到终点的对角线距离"""
        dx = abs(pos[0] - self.end[0])
        dy = abs(pos[1] - self.end[1])
        # 使用对角线距离估计（考虑对角线移动的优势）
        return min(dx, dy) * 1.414 + abs(dx - dy)
    
    def fitness(self, path: List[Tuple[int, int]]) -> float:
        """计算路径的适应度"""
        if not self._is_valid_path(path):
            return 0.001
            
        # 路径长度（越短越好）
        length = self._calculate_path_length(path)
        
        # 平滑度（方向变化越少越好）
        smoothness = 0
        for i in range(2, len(path)):
            dx1 = path[i][0] - path[i-1][0]
            dy1 = path[i][1] - path[i-1][1]
            dx2 = path[i-1][0] - path[i-2][0]
            dy2 = path[i-1][1] - path[i-2][1]
            
            # 如果方向改变，平滑度减1
            if (dx1, dy1) != (dx2, dy2):
                smoothness += 1
                
        # 综合适应度
        fitness = self.w_length * (1 / length) + self.w_smooth * (1 / (smoothness + 1))
        return fitness
    
    def _calculate_path_length(self, path: List[Tuple[int, int]]) -> float:
        """计算路径的实际长度，考虑对角线移动"""
        length = 0.0
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            length += 1.414 if abs(dx) + abs(dy) == 2 else 1
        return length
    
    def _is_valid_path(self, path: List[Tuple[int, int]]) -> bool:
        """检查路径是否有效"""
        if not path or path[0] != self.start or path[-1] != self.end:
            return False
            
        for i in range(len(path) - 1):
            if path[i+1] not in self.grid_map.get_neighbors(path[i]):
                return False
                
        return True
    
    def select(self, population: List[List[Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
        """选择操作 - 使用锦标赛选择"""
        selected = []
        
        for _ in range(self.pop_size):
            # 随机选择5个个体进行锦标赛
            tournament = random.sample(population, min(5, len(population)))
            best = max(tournament, key=lambda p: self.fitness(p))
            selected.append(best.copy())
            
        return selected
    
    def crossover(self, parent1: List[Tuple[int, int]], parent2: List[Tuple[int, int]]) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """交叉操作"""
        if len(parent1) < 3 or len(parent2) < 3 or random.random() > self.p_cross:
            return parent1.copy(), parent2.copy()
            
        # 寻找共同节点
        common_nodes = set(parent1) & set(parent2)
        common_nodes.discard(self.start)
        common_nodes.discard(self.end)
        
        if not common_nodes:
            # 如果没有共同节点，尝试在相近位置交叉
            return self._position_based_crossover(parent1, parent2)
            
        crossover_point = random.choice(list(common_nodes))
        
        # 找到交叉点在两个父路径中的位置
        idx1 = parent1.index(crossover_point)
        idx2 = parent2.index(crossover_point)
        
        # 生成子代
        child1 = parent1[:idx1] + parent2[idx2:]
        child2 = parent2[:idx2] + parent1[idx1:]
        
        # 简化路径（移除回路）
        child1 = self._simplify_path(child1)
        child2 = self._simplify_path(child2)
        
        return child1, child2
    
    def _position_based_crossover(self, parent1: List[Tuple[int, int]], parent2: List[Tuple[int, int]]) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """基于位置的交叉操作，当没有共同节点时使用"""
        if len(parent1) < 3 or len(parent2) < 3:
            return parent1.copy(), parent2.copy()
            
        # 随机选择一个位置
        idx1 = random.randint(1, len(parent1) - 2)
        idx2 = random.randint(1, len(parent2) - 2)
        
        # 生成子代
        child1 = parent1[:idx1] + parent2[idx2:]
        child2 = parent2[:idx2] + parent1[idx1:]
        
        # 确保起点和终点正确
        child1[0] = self.start
        child1[-1] = self.end
        child2[0] = self.start
        child2[-1] = self.end
        
        # 简化路径（移除回路）
        child1 = self._simplify_path(child1)
        child2 = self._simplify_path(child2)
        
        return child1, child2
    
    def mutate(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """变异操作"""
        if len(path) < 3 or random.random() > self.p_mutation:
            return path.copy()
            
        # 随机选择变异类型
        if random.random() < 0.7:  # 70%概率使用节点替换变异
            return self._node_replacement_mutation(path)
        else:  # 30%概率使用插入变异
            return self._insertion_mutation(path)
    
    def _node_replacement_mutation(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """节点替换变异"""
        # 随机选择一个中间节点
        idx = random.randint(1, len(path) - 2)
        current = path[idx]
        
        # 获取有效邻居
        neighbors = self.grid_map.get_neighbors(current)
        if not neighbors:
            return path.copy()
            
        # 随机选择一个邻居替换当前节点
        new_node = random.choice(neighbors)
        new_path = path.copy()
        new_path[idx] = new_node
        
        # 简化路径（移除回路）
        new_path = self._simplify_path(new_path)
        
        return new_path
    
    def _insertion_mutation(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """插入变异：在路径中插入一个新节点"""
        if len(path) < 3:
            return path.copy()
            
        # 随机选择一个位置
        idx = random.randint(1, len(path) - 1)
        node = path[idx]
        
        # 获取邻居
        neighbors = self.grid_map.get_neighbors(node)
        if not neighbors:
            return path.copy()
            
        # 随机选择一个邻居插入
        new_node = random.choice(neighbors)
        new_path = path[:idx] + [new_node] + path[idx:]
        
        # 简化路径（移除回路）
        new_path = self._simplify_path(new_path)
        
        return new_path
    
    def _simplify_path(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """简化路径，移除回路"""
        simplified = []
        visited = set()
        
        for node in path:
            if node in visited:
                # 移除回路
                while simplified[-1] != node:
                    visited.remove(simplified.pop())
            else:
                simplified.append(node)
                visited.add(node)
                
        return simplified
    
    def evolve(self) -> Optional[List[Tuple[int, int]]]:
        """进化过程"""
        population = self.initialize_population()
        
        # 确保初始种群中有一些有效路径
        valid_paths = [p for p in population if self._is_valid_path(p)]
        if not valid_paths:
            print("警告：初始种群中没有有效路径，算法可能难以收敛")
            # 尝试增加一些随机生成的路径
            for _ in range(min(10, self.pop_size)):
                p = self._generate_improved_random_path()
                if self._is_valid_path(p):
                    population[_] = p
        
        for gen in range(self.max_gen):
            # 计算适应度
            fitness_values = [self.fitness(path) for path in population]
            max_fitness = max(fitness_values)
            best_idx = fitness_values.index(max_fitness)
            best_path = population[best_idx]
            
            # 计算路径长度
            lengths = [self._calculate_path_length(path) if self._is_valid_path(path) else float('inf') for path in population]
            valid_paths = [l for l in lengths if l != float('inf')]
            valid_count = len(valid_paths)
            
            # 计算平均适应度
            avg_fitness = sum(fitness_values) / len(fitness_values)
            self.avg_fitness.append(avg_fitness)
            
            # 计算平均路径长度
            avg_length = sum(valid_paths) / valid_count if valid_paths else float('inf')
            self.avg_lengths.append(avg_length)
            
            # 记录统计信息
            self.fitness_values.append(max_fitness)
            self.path_lengths.append(min(lengths) if valid_paths else float('inf'))
            
            # 输出调试信息
            if valid_paths:
                print(f"第{gen + 1}代: 最大适应度={max_fitness:.4f}, 平均适应度={avg_fitness:.4f}, "
                      f"最短路径={min(valid_paths):.2f}, 平均路径={avg_length:.2f}, "
                      f"有效路径={valid_count}/{self.pop_size}")
            else:
                print(f"第{gen + 1}代: 最大适应度={max_fitness:.4f}, 平均适应度={avg_fitness:.4f}, "
                      f"无有效路径, 有效路径={valid_count}/{self.pop_size}")
            
            # 选择
            selected = self.select(population)
            
            # 交叉和变异
            new_population = [best_path]  # 精英保留
            
            while len(new_population) < self.pop_size:
                parent1, parent2 = random.sample(selected, 2)
                child1, child2 = self.crossover(parent1, parent2)
                
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.append(child1)
                if len(new_population) < self.pop_size:
                    new_population.append(child2)
                    
            population = new_population
            
            # 终止条件：连续10代没有改进则提前终止
            if gen >= 20:
                recent_fitness = self.fitness_values[-20:]
                if recent_fitness.count(recent_fitness[0]) == len(recent_fitness):
                    print(f"连续20代没有改进，提前终止进化，共{gen + 1}代")
                    break
                    
            # 如果找到有效路径，且适应度足够高，也可以提前终止
            if self._is_valid_path(best_path) and max_fitness > 0.1:
                print(f"找到高质量有效路径，提前终止进化，共{gen + 1}代")
                return best_path
                
        # 返回最佳路径
        fitness_values = [self.fitness(path) for path in population]
        best_idx = fitness_values.index(max(fitness_values))
        return population[best_idx]

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def visualize_map(grid_map: GridMap, start: Tuple[int, int], end: Tuple[int, int], 
                  path: Optional[List[Tuple[int, int]]] = None, title: str = ""):
    """可视化地图和路径"""
    plt.figure(figsize=(10, 10))
    
    # 绘制障碍物
    obstacles_x = [pos[0] for pos in grid_map.obstacles]
    obstacles_y = [pos[1] for pos in grid_map.obstacles]
    plt.scatter(obstacles_x, obstacles_y, s=660, color='black', marker='s', label='障碍物')
    
    # 绘制起点和终点
    plt.scatter(start[0], start[1], s=200, color='green', marker='o', label='起点')
    plt.scatter(end[0], end[1], s=200, color='red', marker='o', label='终点')
    
    # 绘制路径
    if path:
        path_x = [pos[0] for pos in path]
        path_y = [pos[1] for pos in path]
        
        # 绘制对角线连接线
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i+1]
            plt.plot([x1, x2], [y1, y2], 'b-', linewidth=2)
        
        # 绘制路径点
        plt.scatter(path_x, path_y, s=50, color='blue')
    
    # 设置坐标轴范围
    plt.xlim(-0.5, grid_map.size - 0.5)
    plt.ylim(-0.5, grid_map.size - 0.5)
    
    # 设置网格
    plt.grid(True, linestyle='-', alpha=0.7)
    
    # 设置标题和图例
    plt.title(title, fontsize=16)
    plt.legend(loc='upper right')
    
    # 设置坐标轴刻度
    plt.xticks(range(grid_map.size))
    plt.yticks(range(grid_map.size))
    
    plt.tight_layout()

    if path:
        timestamp = get_timestamp()
        filename = f"./data/{timestamp}_map.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"地图已保存为: {filename}")

    plt.show()

def visualize_optimization_curve(ga: GeneticAlgorithm, title: str):
    """可视化优化曲线，增加平均适应度和平均路径长度曲线"""
    plt.figure(figsize=(10, 8))  # 增加图表高度以容纳额外曲线
    
    # 绘制适应度曲线（同时显示最大和平均）
    plt.subplot(2, 1, 1)
    plt.plot(ga.fitness_values, 'r-', linewidth=2, label='最大适应度')
    plt.plot(ga.avg_fitness, 'g--', linewidth=2, label='平均适应度')
    plt.title(f'{title} - 适应度曲线', fontsize=14)
    plt.xlabel('代数')
    plt.ylabel('适应度')
    plt.grid(True)
    plt.legend()
    
    # 绘制路径长度曲线（同时显示最优和平均）
    plt.subplot(2, 1, 2)
    valid_best_lengths = [l for l in ga.path_lengths if l != float('inf')]
    valid_avg_lengths = [l for l in ga.avg_lengths if l != float('inf')]
    
    if valid_best_lengths and valid_avg_lengths:
        min_length = min(valid_best_lengths)
        max_length = max(max(valid_best_lengths), max(valid_avg_lengths))
        
        # 绘制最优路径长度
        plt.plot(
            [l if l != float('inf') else min_length * 1.1 for l in ga.path_lengths], 
            'b-', linewidth=2, label='最优路径长度'
        )
        
        # 绘制平均路径长度
        plt.plot(
            [l if l != float('inf') else min_length * 1.1 for l in ga.avg_lengths], 
            'g--', linewidth=2, label='平均路径长度'
        )
        
        plt.ylim(min_length * 0.9, max_length * 1.1)
    
    plt.title(f'{title} - 路径长度曲线', fontsize=14)
    plt.xlabel('代数')
    plt.ylabel('路径长度')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()

    timestamp = get_timestamp()
    filename = f"./data/{timestamp}_opti.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"地图已保存为: {filename}")

    plt.show()

def main():
    """主函数"""
    # 创建网格地图
    grid_map = GridMap(size=25, obstacle_ratio=0.32)
    
    # 获取用户输入的起点和终点
    while True:
        try:
            xs, ys = map(int, input("请输入起点坐标(xs,ys)，用逗号分隔: ").split(','))
            if not grid_map.is_valid((xs, ys)):
                print("起点无效，请重新输入！")
                continue
            break
        except (ValueError, TypeError):
            print("输入格式错误，请重新输入！")
    
    while True:
        try:
            xe, ye = map(int, input("请输入终点坐标(xe,ye)，用逗号分隔: ").split(','))
            if not grid_map.is_valid((xe, ye)):
                print("终点无效，请重新输入！")
                continue
            break
        except (ValueError, TypeError):
            print("输入格式错误，请重新输入！")

    start = (xs, ys)
    end = (xe, ye)
    
    # 显示地图
    visualize_map(grid_map, start, end, title="包含起点终点障碍物的地图")
    
    # 遗传算法参数
    pop_size = 30
    max_gen = 100
    p_cross = 0.7
    p_mutation = 0.2
    w_length = 0.8
    w_smooth = 0.2
    
    print("\n开始使用遗传算法规划路径...")
    print(f"参数设置: 种群大小={pop_size}, 最大代数={max_gen}, 交叉概率={p_cross}, "
          f"变异概率={p_mutation}, 路径长度权重={w_length}, 平滑度权重={w_smooth}")
    print(f"路径目标: 起始位置={start}, 终止位置={end}")

    # 运行遗传算法
    ga = GeneticAlgorithm(grid_map, start, end, pop_size, max_gen, p_cross, p_mutation, w_length, w_smooth)
    best_path = ga.evolve()
    
    # 检查路径是否有效
    if ga._is_valid_path(best_path):
        # 显示路径和优化曲线
        title = f'w_length={w_length},w_smooth={w_smooth}遗传算法规划路径'
        visualize_map(grid_map, start, end, best_path, title)
        
        title = f'w_length={w_length},w_smooth={w_smooth}遗传算法优化曲线'
        visualize_optimization_curve(ga, title)
        
        print(f"\n找到有效路径，长度为{ga._calculate_path_length(best_path):.2f}")
        print("路径:", best_path)
    else:
        print("\n未找到有效路径！尝试减少障碍物比例或调整参数")
        # 显示起点终点不可达的地图
        visualize_map(grid_map, start, end, title="该地图起点终点不可达")

if __name__ == "__main__":
    main()    