import numpy as np

# 设备功率范围（选择中间值）
powers = np.array([1150, 95, 500, 200, 1200])  # 窗式空调机、家用电冰箱、电饭煲、电子计算机、电热淋浴器

# 初始化种群
def initialize_population(size, num_devices):
    return np.random.randint(2, size=(size, num_devices))

# 适应度函数
def fitness(individual, actual_load):
    estimated_load = np.dot(individual, powers)
    return 1 / (1 + abs(actual_load - estimated_load))  # 返回相似度

# 选择操作
def select(population, fitness_scores):
    probabilities = fitness_scores / np.sum(fitness_scores)
    return population[np.random.choice(range(len(population)), p=probabilities)]

# 交叉操作
def crossover(parent1, parent2):
    point = np.random.randint(1, len(parent1) - 1)
    child = np.concatenate((parent1[:point], parent2[point:]))
    return child

# 变异操作
def mutate(individual, mutation_rate=0.01):
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] = 1 - individual[i]  # 0变1，1变0
    return individual

# 遗传算法主函数
def genetic_algorithm(actual_load, population_size=100, generations=100):
    population = initialize_population(population_size, len(powers))

    for generation in range(generations):
        fitness_scores = np.array([fitness(ind, actual_load) for ind in population])
        new_population = []

        for _ in range(population_size):
            parent1 = select(population, fitness_scores)
            parent2 = select(population, fitness_scores)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = np.array(new_population)

    # 返回最佳个体
    best_index = np.argmax(fitness_scores)
    return population[best_index]

# 使用示例
if __name__ == "__main__":
    actual_load = 2200  # 示例总负荷
    best_solution = genetic_algorithm(actual_load)
    print("最佳设备状态（0/1表示）：", best_solution)
    print("对应负荷：", np.dot(best_solution, powers), "W")
