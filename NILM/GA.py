import math
import random
import pandas as pd
from collections import Counter

# 假设每个设备的电流状态
device_states = {
    'dish_washer': {'state_00': {'current': 0.0}, 
                    'state_01': {'current': 0.4}, 
                    'state_10': {'current': 1.2}, 
                    'state_11': {'current': 6.4}
                },

    'tv_pvr':      {'state_0': {'current': 0.0},
                    'state_1': {'current': 0.5},
                },

    'furnace':     {'state_00': {'current': 0.0},
                    'state_01': {'current': 1.3}, 
                    'state_10': {'current': 2.2},
                },
}

# 设备数量
num_devices = len(device_states)

# 状态映射: 对应每个设备的状态编码（4选一状态）
state_mapping = {
    'tv_pvr': {0: 'state_0', 1: 'state_1'},
    'furnace': {0: 'state_00', 1: 'state_01', 2: 'state_10'},
    'dish_washer': {0: 'state_00', 1: 'state_01', 2: 'state_10', 3: 'state_11'}
}

# 初始化种群
def init_population(pop_size):
    population = []
    for _ in range(pop_size):
        individual = []
        for device in device_states:
            num_states = len(device_states[device])  # 获取设备的状态数量
            individual.append(random.randint(0, num_states - 1))
        population.append(individual)
    return population

# 计算电流适应度函数，最小化目标
def fitness(individual, target_current):
    total_current = 0.0
    err = 1e-6
    for i, state_code in enumerate(individual):
        device = list(device_states.keys())[i]
        state_name = state_mapping[device][state_code]
        total_current += device_states[device][state_name]['current']
    return 1/(abs(target_current - total_current)+err)

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

# 主遗传算法流程
def genetic_algorithm(target_current, pop_size, generations, mutation_rate):

    population = init_population(pop_size)
    for gen_no in range(generations):
        # 计算适应度
        fitness_values = [fitness(individual, target_current) for individual in population]
        
        # 选择：轮盘赌选择最小适应度（即最接近目标电流的解）
        total_fitness = sum(fitness_values)
        probabilities = [1 - (fitness_value / total_fitness) for fitness_value in fitness_values]
        selected_population = random.choices(population, probabilities, k=pop_size)
        
        # 交叉：生成新子代
        next_generation = []
        for i in range(0, pop_size, 2):
            parent1, parent2 = selected_population[i], selected_population[i + 1]
            child1, child2 = crossover(parent1, parent2)
            next_generation.append(child1)
            next_generation.append(child2)
        
        # 变异操作
        next_generation = [mutate(individual, mutation_rate) for individual in next_generation]

        # 更新种群
        population = next_generation

        # 输出当前代数和最优解
        best_individual = max(population, key=lambda ind: fitness(ind, target_current))
        print(f"Generation {gen_no}: Best solution {best_individual} with fitness {fitness(best_individual, target_current)}")

    return population

if __name__ == "__main__":
    path = r"NILM\Electricity_Data"
    My_I_Data = pd.read_csv(path + r'\I_data_with_total_and_state.csv', parse_dates=True).head(1000)
    
    My_I_Data = My_I_Data.iloc[:, -4:]
    total_current_col = My_I_Data.columns[0]
    state_col = My_I_Data.columns[1:4]

    Nr = 0
    row = len(My_I_Data)
    for i in range(row):
        # 目标电流
        target_current = My_I_Data[total_current_col].iloc[i]
        # 执行遗传算法
        best_individual_list = genetic_algorithm(target_current, pop_size=50, generations=100, mutation_rate=0.05)
        best_individual = max(best_individual_list, key=lambda ind: fitness(ind, target_current))
        
        # 打印结果（根据需要保存或进一步处理）
        print(f"Row {i}: Best solution {best_individual}")
        true_list = My_I_Data[state_col].iloc[i].tolist()
        if best_individual == true_list:
            Nr += 1

    # 计算准确率
    accuracy = Nr / len(My_I_Data)
    print(f"Accuracy = {accuracy}")
