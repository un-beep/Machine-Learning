import numpy as np
import random

# 假设每个设备的电流状态和有功功率状态
device_states = {
    'dish_washer': {'state_00': {'current': 0.0, 'power': 0.0}, 
                    'state_01': {'current': 0.4, 'power': 15.02}, 
                    'state_10': {'current': 1.2, 'power': 141.98}, 
                    'state_11': {'current': 6.4, 'power': 776.38}
                },

    'tv_pvr':      {'state_0': {'current': 0.0, 'power': 0.0},
                    'state_1': {'current': 0.5, 'power': 38.72},
                },

    'furnace':     {'state_00': {'current': 0.0, 'power': 0.0},
                    'state_01': {'current': 1.3, 'power': 109.62}, 
                    'state_10': {'current': 2.2, 'power': 185.86},
                },

    'fridge':      {'state_00': {'current': 0.0, 'power': 0.0}, 
                    'state_01': {'current': 1.0, 'power': 130.74},
                },
}

# 目标电流和功率
target_current = 2.7  # 总电流
target_power = 268    # 总有功功率

# 设备数量
num_devices = len(device_states)

# 状态映射: 对应每个设备的状态编码（4选一状态）
state_mapping = {
    'tv_pvr': {0: 'state_0', 1: 'state_1'},
    'furnace': {0: 'state_00', 1: 'state_01', 2: 'state_10'},
    'fridge': {0: 'state_00', 1: 'state_01'},
    'dish_washer': {0: 'state_00', 1: 'state_01', 2: 'state_10', 3: 'state_11'}
}

# 初始化种群
def init_population(pop_size):
    population = []
    for _ in range(pop_size):
        individual = []
        for device in device_states:
            num_states = len(device_states[device])  # 获取设备的状态数量
            # 根据设备状态数选择对应的编码
            individual.append(random.randint(0, num_states - 1))
        population.append(individual)
    return population

# 计算电流和有功功率
def calculate_total_current(individual):
    total_current = 0.0
    total_power = 0.0
    for i, state_code in enumerate(individual):
        device = list(device_states.keys())[i]
        state_name = state_mapping[device][state_code]
        total_current += device_states[device][state_name]['current']
        total_power += device_states[device][state_name]['power']
    return total_current, total_power

# 适应度函数：计算目标值与实际值的差异
def fitness(individual):
    total_current, total_power = calculate_total_current(individual)
    
    # 计算两个目标：电流和功率与目标值的差异（最小化差异）
    current_diff = abs(total_current - target_current)
    power_diff = abs(total_power - target_power)
    
    # 返回电流差异和功率差异
    return current_diff, power_diff

# 选择操作：轮盘赌选择
def selection(population):
    total_fitness = sum(fitness(ind)[0] + fitness(ind)[1] for ind in population)
    selected = random.choices(population, k=2, weights=[(fitness(ind)[0] + fitness(ind)[1]) / total_fitness for ind in population])
    return selected

# 交叉操作：单点交叉
def crossover(parent1, parent2):
    crossover_point = random.randint(1, num_devices - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# 变异操作：随机选择一个位置，改变其状态
def mutate(individual, mutation_rate=0.05):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            num_states = len(device_states[list(device_states.keys())[i]])
            individual[i] = random.randint(0, num_states - 1)
    return individual

# 非支配排序
def non_dominated_sort(population):
    dominated_count = [0] * len(population)
    dominates = [[] for _ in range(len(population))]
    fronts = [[]]
    
    for i in range(len(population)):
        for j in range(len(population)):
            if i != j:
                f_i, f_j = fitness(population[i]), fitness(population[j])
                if f_i[0] <= f_j[0] and f_i[1] <= f_j[1] and (f_i[0] < f_j[0] or f_i[1] < f_j[1]):
                    dominates[i].append(j)
                elif f_j[0] <= f_i[0] and f_j[1] <= f_i[1] and (f_j[0] < f_i[0] or f_j[1] < f_i[1]):
                    dominated_count[i] += 1
        
        if dominated_count[i] == 0:
            fronts[0].append(i)
    
    return fronts

# NSGA-II主流程
def nsga2(pop_size, generations, mutation_rate):
    population = init_population(pop_size)
    
    for gen in range(generations):
        new_population = []
        
        # 选择操作：选出适应度较高的个体
        while len(new_population) < pop_size:
            parent1, parent2 = selection(population)
            
            # 交叉操作
            child1, child2 = crossover(parent1, parent2)
            
            # 变异操作
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            
            # 将子代加入新种群
            new_population.append(child1)
            new_population.append(child2)
        
        # 更新种群
        population = new_population[:pop_size]
        
        # 非支配排序
        fronts = non_dominated_sort(population)
        population = [population[i] for front in fronts for i in front]
        
        # 输出当前代的最佳个体及其适应度
        best_individual = population[0]
        print(f"Generation {gen+1}: Best Fitness = {fitness(best_individual)}")
    
    # 返回最终最优解
    best_individual = population[0]
    return best_individual

# 执行NSGA-II算法
best_individual = nsga2(pop_size=500, generations=100, mutation_rate=0.1)

# 输出最终结果
print("\nBest Individual Found:")
recognized_devices = {list(device_states.keys())[i]: list(state_mapping[list(device_states.keys())[i]].keys())[best_individual[i]] for i in range(num_devices)}
for device, state in recognized_devices.items():
    print(f"{device} is in {state}")
