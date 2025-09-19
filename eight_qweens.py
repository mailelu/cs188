import random

# -------- 参数设置 --------
N = 8                    # 皇后数量 / 棋盘大小
POP_SIZE = 50           # 种群规模
GENERATIONS = 1000       # 最大代数
## TOURNAMENT_SIZE = 3      # 锦标赛选择规模
CROSSOVER_RATE = 0.8     # 交叉概率
MUTATION_RATE = 0.2      # 变异概率

MAX_PAIRS = N * (N - 1) // 2  # 最大不互攻击对数 (8皇后=28)

# -------- 适应度函数 --------
def fitness(ind):
    """计算个体适应度：不互相攻击的皇后对数"""
    attacks = 0
    for i in range(len(ind)):
        for j in range(i+1,len(ind)):
            if ind[i] == ind[j] or  abs(ind[i] - ind[j]) == abs(i - j):
                attacks += 1
    return 28 - attacks        
# -------- 初始化 --------
def random_individual():
    """生成一个随机棋盘"""
    chessboard = list(range(0,8))
    random.shuffle(chessboard)
    return chessboard

def init_population():
    """生成初始种群"""
    population = []
    for _ in range(POP_SIZE):
        population.append(random_individual())
    return population

# -------- 选择 --------
def selection(pop, fits):
    """选择父代个体"""
    probabilities = []
    total = sum(fits)
    if total == 0:  # 避免除零
        return random.choice(pop)
    choose = random.random()
    for i in range(len(fits)):
        if i == 0:
            probabilities.append(fits[i]/total)
        else:
            probabilities.append(fits[i]/total + probabilities[i-1])
    for i in range(1,len(probabilities)):
        if probabilities[i-1] <= choose and choose < probabilities[i]:
            return pop[i]
    return pop[-1]

# -------- 交叉 --------
def crossover(p1, p2):
    """交叉产生子代"""
    if random.random() > CROSSOVER_RATE:
        return p1[:], p2[:]
    point1 = random.randint(0, N-2)
    point2 = random.randint(point1+1, N-1)
    c1, c2 = p1[:], p2[:]
    c1[point1:point2], c2[point1:point2] = p2[point1:point2], p1[point1:point2]
    # 修复重复
    def repair(child):
        counts = [0]*N
        for i in range(N):
            counts[child[i]] += 1
        missing = [i for i in range(N) if counts[i] == 0]
        for i in range(N):
            if counts[child[i]] > 1:
                counts[child[i]] -= 1
                child[i] = missing.pop()
        return child
    return repair(c1), repair(c2)

# -------- 变异 --------
def mutate(ind):
    """随机变异"""
    for i in range(len(ind)):
        if random.random() < MUTATION_RATE:
            a = random.randint(0,len(ind)-1)
            ind[i], ind[a] = ind[a], ind[i]
    return ind

# -------- 主循环 --------
def genetic_algorithm():
    """主进化过程"""
    pop = init_population()
    fits = [fitness(ind) for ind in pop]

    for gen in range(GENERATIONS):
        # 在这里实现选择 + 交叉 + 变异 + 更新种群
        new_pop = []
        new_fits = []
        max_fits = max(range(len(fits)), key=lambda i: fits[i])
        max_ind = pop[max_fits]
        max_fit = fits[max_fits]
        new_pop.append(max_ind)
        new_fits.append(max_fit)
        while len(new_pop) < POP_SIZE:
            (child1, child2) = crossover(selection(pop, fits),selection(pop, fits))
            child1 = mutate(child1)
            child2 = mutate(child2)
            child1_fit = fitness(child1)
            child2_fit = fitness(child2)
            if child1_fit  == 28:
                return child1
            if child2_fit  == 28:
                return child2
            if len(new_pop) < POP_SIZE:
                new_pop.append(child1)
                new_fits.append(child1_fit)
            if len(new_pop) < POP_SIZE:
                new_pop.append(child2)
                new_fits.append(child2_fit)
        pop,fits = list(new_pop),list(new_fits)
    # 返回最好解
    max_fits = max(range(len(fits)), key=lambda i: fits[i])
    return pop[max_fits]

if __name__ == "__main__":
    solution = genetic_algorithm()
    print("Solution:", solution)
# 打印棋盘
def print_board(ind):
    for i in range(N):
        row = ["Q" if ind[i] == j else "." for j in range(N)]
        print(" ".join(row))
    print()

# 运行
solution = genetic_algorithm()
print("找到解:", solution)
if solution:
    print_board(solution)
