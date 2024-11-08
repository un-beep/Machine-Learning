import math
import random
import matplotlib.pyplot as plt

# 假设每个设备的电流状态和有功功率状态
device_states = {
    'dish_washer': {'state_00': {'current': 0.0, 'power': 0.0}, 
                    'state_01': {'current': 0.4, 'power': 15.12}, 
                    'state_10': {'current': 1.2, 'power': 142.01}, 
                    'state_11': {'current': 6.4, 'power': 776.64}
                },

    'tv_pvr':      {'state_0': {'current': 0.0, 'power': 0.0},
                    'state_1': {'current': 0.5, 'power': 24.31},
                },

    'furnace':     {'state_00': {'current': 0.0, 'power': 0.0},
                    'state_01': {'current': 1.3, 'power': 110.32}, 
                    'state_10': {'current': 2.2, 'power': 182.14},
                },
}

# 目标电流和功率
target_current = 1.7  # 总电流
target_power = 146    # 总有功功率

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

# 计算电流和功率适应度函数，最小化目标
# 电流函数function1
def function1(individual):
    total_current = 0.0
    err = 1e-6
    for i, state_code in enumerate(individual):
        device = list(device_states.keys())[i]
        state_name = state_mapping[device][state_code]
        total_current += device_states[device][state_name]['current']
    return abs(round(target_current - total_current, 1))

# 功率函数function2
def function2(individual):
    total_power = 0.0
    err = 1e-6
    for i, state_code in enumerate(individual):
        device = list(device_states.keys())[i]
        state_name = state_mapping[device][state_code]
        total_power += device_states[device][state_name]['power']
    return abs(round(target_power - total_power, 1))

# 输出值对应下标
def index_of(a, list):
    try:
        return list.index(a)
    except ValueError:
        return -1

# 根据数值排序列表，从小到大排序，返回排序后的下标
def sort_by_values(list1, values):
    sorted_list = []
    values_copy = values[:]
    while len(sorted_list) != len(list1):
        min_index = index_of(min(values_copy), values_copy)
        if min_index in list1:
            sorted_list.append(min_index)
        values_copy[min_index] = math.inf
    return sorted_list

# NSGA-II的快速非支配排序，返回排序后的Pareto前沿
def fast_non_dominated_sort(values1, values2):
    S = [[] for _ in range(len(values1))]
    front = [[]]
    n = [0 for _ in range(len(values1))]
    rank = [0 for _ in range(len(values1))]

    for p in range(len(values1)):
        S[p] = []
        n[p] = 0
        for q in range(len(values1)):
            if (values1[p] < values1[q] and values2[p] < values2[q]) or \
               (values1[p] <= values1[q] and values2[p] < values2[q]) or \
               (values1[p] < values1[q] and values2[p] <= values2[q]):
                S[p].append(q)
            elif (values1[q] < values1[p] and values2[q] < values2[p]) or \
                 (values1[q] <= values1[p] and values2[q] < values2[p]) or \
                 (values1[q] < values1[p] and values2[q] <= values2[p]):
                 n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while front[i]:
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] = n[q] - 1
                if n[q] == 0:
                    rank[q] = i + 1
                    if q not in Q:
                        Q.append(q)
        i += 1
        front.append(Q)
    del front[-1]
    return front

# 计算拥挤距离
def crowding_distance(values1, values2, front):
    distance = [0 for _ in range(len(front))]
    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])
    # 边界最大化拥挤距离
    err = 1e-6
    distance[0] = distance[-1] = float('inf')
    for k in range(1, len(front) - 1):
        distance[k] += (values1[sorted1[k + 1]] - values1[sorted1[k - 1]]) / (max(values1) - min(values1) + err)
        distance[k] += (values2[sorted2[k + 1]] - values2[sorted2[k - 1]]) / (max(values2) - min(values2) + err)
    return distance



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

# 主要参数，种群数，最大遗传代数
pop_size = 100
max_gen = 100
gen_no = 0

progress = []

population = init_population(pop_size)

while gen_no < max_gen:
    function1_values = [function1(population[i]) for i in range(pop_size)]
    function2_values = [function2(population[i]) for i in range(pop_size)]
    non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:], function2_values[:])
    print(f"The best front for Generation number {gen_no} is")
    for value in non_dominated_sorted_solution[0]:
        print(population[value], end=" ")
    print("\n")
    
    # Store progress for visualization
    progress.append((function1_values, function2_values))
    
    crowding_distance_values = []
    for i in range(len(non_dominated_sorted_solution)):
        crowding_distance_values.append(crowding_distance(function1_values[:], function2_values[:], non_dominated_sorted_solution[i][:]))
    solution2 = population[:]
    
    # 生成第一代子代
    while len(solution2) != 2 * pop_size:
        a1 = random.randint(0, pop_size - 1)
        b1 = random.randint(0, pop_size - 1)
        solution2.append(mutate(crossover(population[a1], population[b1])[0]))
    
    function1_values2 = [function1(solution2[i]) for i in range(2 * pop_size)]
    function2_values2 = [function2(solution2[i]) for i in range(2 * pop_size)]
    non_dominated_sorted_solution2 = fast_non_dominated_sort(function1_values2[:], function2_values2[:])
    crowding_distance_values2 = []
    for i in range(len(non_dominated_sorted_solution2)):
        crowding_distance_values2.append(crowding_distance(function1_values2[:], function2_values2[:], non_dominated_sorted_solution2[i][:]))
    
    new_solution = []
    for i in range(len(non_dominated_sorted_solution2)):
        non_dominated_sorted_solution2_1 = [index_of(non_dominated_sorted_solution2[i][j], non_dominated_sorted_solution2[i]) for j in range(len(non_dominated_sorted_solution2[i]))]
        front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])
        front = [non_dominated_sorted_solution2[i][front22[j]] for j in range(len(non_dominated_sorted_solution2[i]))]
        front.reverse()
        for value in front:
            new_solution.append(value)
            if len(new_solution) == pop_size:
                break
        if len(new_solution) == pop_size:
            break
    population = [solution2[i] for i in new_solution]
    gen_no += 1

# Let's plot the final front
function1_values = [function1(population[i]) for i in range(pop_size)]
function2_values = [function2(population[i]) for i in range(pop_size)]

# # Visualize the final front
# plt.xlabel('Function 1', fontsize=15)
# plt.ylabel('Function 2', fontsize=15)
# plt.title('Final Front')
# plt.scatter(function1_values, function2_values)
# plt.show()

# # Visualize the progress over generations
# for gen, (f1_vals, f2_vals) in enumerate(progress):
#     print(f"电流差异：{f1_vals}，功率差异：{f2_vals}\n")
#     plt.figure(figsize=(10, 6))
#     plt.scatter(f1_vals, f2_vals)
#     plt.xlabel('Function 1', fontsize=15)
#     plt.ylabel('Function 2', fontsize=15)
#     plt.title(f'Generation {gen}')
#     plt.show()


index = index_of(min(function1_values),function1_values)
best_individual = population[index]
print(f"电流差异：{function1_values[0]}，功率差异：{function2_values[0]}\n")
# 输出最终结果
print("\nBest Individual Found:")
recognized_devices = {list(device_states.keys())[i]: list(state_mapping[list(device_states.keys())[i]].keys())[best_individual[i]] for i in range(num_devices)}
for device, state in recognized_devices.items():
    print(f"{device} is in {state}")
