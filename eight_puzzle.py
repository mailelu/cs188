import heapq

class PuzzleState:
    def __init__(self, board, parent=None, move=None, depth=0, cost=0):
        self.board = board        # 当前棋盘
        self.parent = parent      # 父节点
        self.move = move          # 从父节点走到这一步的动作
        self.depth = depth        # g(n)
        self.cost = cost          # f(n) = g(n) + h(n)

    def __lt__(self, other):
        return self.cost < other.cost

def is_goal(state, goal):
    """判断当前状态是否为目标状态"""
    return state.board == goal

def get_neighbors(current_state: PuzzleState) -> list[PuzzleState]:
    """
    返回当前状态的所有合法下一步状态。
    
    参数:
        current_state: PuzzleState 对象，表示当前棋盘状态
        
    返回:
        neighbors: 包含 PuzzleState 对象的列表，每个对象表示从当前状态移动一步后的新状态
    """
    neighbors = []
    # 找空格位置
    zero_row, zero_col = None, None
    for i in range(3):
        for j in range(3):
            if current_state.board[i][j] == 0:
                zero_row, zero_col = i, j
                break
        if zero_row is not None:
            break
    # 四个方向的移动偏移
    moves = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
    for move, (dr, dc) in moves.items():
        new_row, new_col = zero_row + dr, zero_col + dc
        if 0 <= new_row < 3 and 0 <= new_col < 3:
            # 复制棋盘
            new_board = [row[:] for row in current_state.board]
            # 交换空格与目标块
            new_board[zero_row][zero_col], new_board[new_row][new_col] = new_board[new_row][new_col], new_board[zero_row][zero_col]
            # 创建新的 PuzzleState
            neighbors.append(PuzzleState(new_board, parent=current_state, move=move, depth=current_state.depth + 1))

    return neighbors

def manhattan_distance(state, goal):
    """计算曼哈顿距离启发式"""  #计算相对距离
    h = 0 
    for i in range(3):
    	for j in range(3):
            val = state.board[i][j]
            if  val == 0:
                continue
            for k in range(3):
                for l in range(3):
            	    if goal.board[k][l] == val:
                        h += abs(i-k) + abs(j-l)
    return h    

def misplaced_tiles(current, goal):
    """计算放错位置的方块数另一种启发式"""
    count = 0
    for i in range(3):
        for j in range(3):
            if current.board[i][j] != 0 and current.board[i][j] != goal.board[i][j]:
                count += 1
    return count

def reconstruct_path(current):
    """从终点状态回溯出完整路径"""
    path = []
    while current.parent is not None:
        path.append(current.move)
        current = current.parent
    path.reverse()
    return path

def a_star(start_state: PuzzleState, goal_board: list[list[int]], 
           heuristic=manhattan_distance) -> list:
    """
    使用 A* 算法寻找 8 数码问题的最短解路径。

    参数:
        start_state: PuzzleState 对象，表示初始棋盘状态
        goal_board: 二维列表，表示目标棋盘状态
        heuristic: 启发式函数，默认曼哈顿距离

    返回:
        path: 从初始到目标的动作序列或状态序列
        如果无解，返回 None
    """
    
    # 初始化 open_set 和 visited
    open_set = []
    start_state.cost = start_state.depth + heuristic(start_state, PuzzleState(goal_board))
    heapq.heappush(open_set, (start_state.cost, start_state))

    visited = set()
    visited.add(tuple(tuple(row) for row in start_state.board))

    while open_set:
         # 弹出 f(n) 最小的状态作为当前节点
        current_cost, current_state = heapq.heappop(open_set)

        if is_goal(current_state, goal_board):
            return reconstruct_path(current_state) # 回溯路径并返回
		# 遍历当前状态的所有邻居,更新邻居信息
        for neighbor in get_neighbors(current_state):
            neighbor_tuple = tuple(tuple(row) for row in neighbor.board)
            if neighbor_tuple not in visited:
                neighbor.depth = current_state.depth + 1
                neighbor.cost = neighbor.depth + heuristic(neighbor, PuzzleState(goal_board))
                heapq.heappush(open_set, (neighbor.cost, neighbor))
                visited.add(neighbor_tuple)

    return  None
