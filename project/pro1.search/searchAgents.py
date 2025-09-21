# searchAgents.py
# ---------------
# 许可信息：你可以自由使用或扩展这些项目用于教育目的,前提是 (1) 不分发或发布解决方案, (2) 保留此声明, (3) 清楚地标明归属 UC Berkeley, 包括链接 http://ai.berkeley.edu.
# 
# 归属信息：Pacman AI 项目由 UC Berkeley 开发.
# 核心项目和自动评分器主要由 John DeNero (denero@cs.berkeley.edu) 和 Dan Klein (klein@cs.berkeley.edu) 创建.
# 学生端自动评分由 Brad Miller, Nick Hay 和 Pieter Abbeel (pabbeel@cs.berkeley.edu) 添加.

"""
该文件包含所有可以用来控制 Pacman 的智能体. 要选择智能体, 请在运行 pacman.py 时使用 -p 选项.
可以通过 -a 向智能体传递参数.
例如, 要加载一个使用深度优先搜索 (dfs) 的 SearchAgent, 运行如下命令:

> python pacman.py -p SearchAgent -a fn=depthFirstSearch

其他搜索策略的命令可以在项目说明中找到.

请仅修改文件中要求修改的部分. 查找包含 "*** YOUR CODE HERE ***" 的行.

需要填写的部分大约从文件的四分之三处开始. 具体细节请参考项目说明.

祝你好运, 搜索顺利!
"""
from typing import List, Tuple, Any
from game import Directions
from game import Agent
from game import Actions
import util
import time
import search
import pacman

class GoWestAgent(Agent):
    "一个一直向西移动直到无法移动的智能体."

    def getAction(self, state):
        "智能体接收一个 GameState 对象 (定义在 pacman.py 中)."
        if Directions.WEST in state.getLegalPacmanActions():
            return Directions.WEST
        else:
            return Directions.STOP

#######################################################
# 此部分已为你编写, 但只有在填充 search.py 的部分后才会工作 #
#######################################################
class SearchAgent(Agent):
    """
    这个通用搜索智能体会使用提供的搜索算法来解决给定的搜索问题, 并找到一条路径, 然后返回沿该路径前进的动作.

    默认情况下, 该智能体会在 PositionSearchProblem 上运行深度优先搜索 (DFS), 以找到位置 (1,1).

    fn 参数可选值包括:
      depthFirstSearch 或 dfs
      breadthFirstSearch 或 bfs

    注意: 你不应修改 SearchAgent 中的任何代码.
    """

    def __init__(self, fn='depthFirstSearch', prob='PositionSearchProblem', heuristic='nullHeuristic'):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError(fn + ' is not a search function in search.py.')
        func = getattr(search, fn)
        if 'heuristic' not in func.__code__.co_varnames:
            print('[SearchAgent] using function ' + fn)
            self.searchFunction = func
        else:
            if heuristic in globals().keys():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError(heuristic + ' is not a function in searchAgents.py or search.py.')
            print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic))
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.searchFunction = lambda x: func(x, heuristic=heur)

        # Get the search problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError(prob + ' is not a search problem type in SearchAgents.py.')
        self.searchType = globals()[prob]
        print('[SearchAgent] using problem type ' + prob)

    def registerInitialState(self, state):
        """
        这是智能体第一次看到游戏棋盘的布局.Here, 我们选择一条通往目标的路径.
        在这个阶段,智能体应计算出到目标的路径,并将其存储在局部变量中.所有的工作都在此方法中完成！

        state: 一个 GameState 对象 (pacman.py)

        """
        if self.searchFunction == None: raise Exception("No search function provided for SearchAgent")
        starttime = time.time()
        problem = self.searchType(state) # Makes a new search problem
        self.actions  = self.searchFunction(problem) # Find a path
        if self.actions == None:
            self.actions = []
        totalCost = problem.getCostOfActions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)

    def getAction(self, state):
        """
        返回先前选择路径中的下一个动作（在 registerInitialState 中）.
        如果没有可执行的后续动作,则返回 Directions.STOP.

        state: 一个 GameState 对象 (pacman.py)

        """
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP

class PositionSearchProblem(search.SearchProblem):
    """

    一个搜索问题定义了状态空间、起始状态、目标测试、后继函数和代价函数.
    该搜索问题可用于在 Pacman 棋盘上寻找到特定点的路径.

    状态空间由 Pacman 游戏中的 (x,y) 位置组成.

    注意：该搜索问题已完全指定;你不应修改它.

    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        存储起始状态和目标.

        gameState: 一个 GameState 对象 (pacman.py)
        costFn: 一个从搜索状态 (tuple) 到非负数的函数
        goal: gameState 中的一个位置

        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print('Warning: this does not look like a regular search maze')

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        返回后继状态、所需动作以及代价 1.

        如 search.py 所述：
        对于给定状态,该方法应返回一个三元组列表 (successor, action, stepCost),
        其中 'successor' 是当前状态的后继状态,'action' 是到达该后继状态所需的动作,
        而 'stepCost' 是扩展到该后继状态的增量代价.

        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        返回特定动作序列的代价.如果这些动作包含非法移动,则返回 999999.

        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

class StayEastSearchAgent(SearchAgent):
    """
    一个用于位置搜索的智能体,其代价函数会对位于棋盘西侧的位置进行惩罚.

    进入位置 (x,y) 的代价函数为 1/2^x.

    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: .5 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn, (1, 1), None, False)

class StayWestSearchAgent(SearchAgent):
    """
    一个用于位置搜索的智能体,其代价函数会对位于棋盘东侧的位置进行惩罚.

    进入位置 (x,y) 的代价函数为 2^x.

    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: 2 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn)

def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def euclideanHeuristic(position, problem, info={}):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    "PositionSearchProblem 的欧几里得距离启发式."
    xy1 = position
    xy2 = problem.goal
    return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5

#####################################################
# 下面部分尚未完成, 需要编写代码!                      #
#####################################################

class CornersProblem(search.SearchProblem):
    """
    这个搜索问题会找到穿过布局中所有四个角的路径.

    你必须选择合适的状态空间和后继函数.
    """

    def __init__(self, startingGameState: pacman.GameState):
        """
        Stores the walls, pacman's starting position and corners.
        """
        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top, right = self.walls.height-2, self.walls.width-2
        self.corners = ((1,1), (1,top), (right, 1), (right, top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                print('Warning: no food in corner ' + str(corner))
        self._expanded = 0 # DO NOT CHANGE; Number of search nodes expanded

    def getStartState(self):
        """
        返回起始状态 (在你的状态空间中, 而不是完整的 Pacman 状态空间)
        """
        "*** YOUR CODE HERE ***"

        util.raiseNotDefined()

    def isGoalState(self, state: Any):
        """
        返回该搜索状态是否为目标状态
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def getSuccessors(self, state: Any):
        """
        返回后继状态, 所需动作, 以及代价 1.

        如 search.py 所述:
        对于给定状态, 应返回一个三元组列表 (successor, action, stepCost),
        其中 'successor' 是当前状态的后继状态,
        'action' 是到达该后继状态所需动作,
        'stepCost' 是扩展到该状态的增量代价.
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            # 如果动作合法, 将后继状态添加到 successors 列表
            # 以下代码片段用于判断新位置是否碰到墙:
            #   x,y = currentPosition
            #   dx, dy = Actions.directionToVector(action)
            #   nextx, nexty = int(x + dx), int(y + dy)
            #   hitsWall = self.walls[nextx][nexty]

            "*** YOUR CODE HERE ***"

        self._expanded += 1 # DO NOT CHANGE
        return successors

    def getCostOfActions(self, actions):
        """
        返回特定动作序列的代价. 如果包含非法动作, 返回 999999. 这需要你实现.
        """
        if actions == None: return 999999
        x,y= self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
        return len(actions)


def cornersHeuristic(state: Any, problem: CornersProblem):
    """
    CornersProblem 的启发式函数.

    state: 当前搜索状态
           (在你的搜索问题中选择的数据结构)

    problem: 该布局的 CornersProblem 实例

    该函数应始终返回一个数字, 作为从当前状态到目标状态的最短路径的下界;
    即它应是可接受的 (admissible) 并且一致 (consistent).
    """
    corners = problem.corners # 四个角的坐标
    walls = problem.walls # 迷宫的墙信息, Grid 类型 (game.py)

    "*** YOUR CODE HERE ***"
    return 0 # 默认返回 0, 即最简单的解

class AStarCornersAgent(SearchAgent):
    "用于 CornersProblem 的 A* 搜索智能体, 使用 cornersHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, cornersHeuristic)
        self.searchType = CornersProblem

class FoodSearchProblem:
    """
    与寻找 Pacman 游戏中所有食物 (dots) 的路径相关的搜索问题.

    搜索状态是一个元组 (pacmanPosition, foodGrid)
      pacmanPosition: 一个元组 (x,y) 表示 Pacman 的位置
      foodGrid:       一个 Grid (见 game.py), True 表示食物存在, False 表示不存在
    """
    def __init__(self, startingGameState: pacman.GameState):
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self.startingGameState = startingGameState
        self._expanded = 0 # DO NOT CHANGE
        self.heuristicInfo = {}  # 启发式函数可存储信息的字典

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state[1].count() == 0

    def getSuccessors(self, state):
        "返回后继状态、所需动作以及代价 1."
        successors = []
        self._expanded += 1 # DO NOT CHANGE
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextFood = state[1].copy()
                nextFood[nextx][nexty] = False
                successors.append( ( ((nextx, nexty), nextFood), direction, 1) )
        return successors

    def getCostOfActions(self, actions):
        """
        返回特定动作序列的代价. 如果包含非法动作, 返回 999999.
        """
        x,y= self.getStartState()[0]
        cost = 0
        for action in actions:
            "用于 FoodSearchProblem 的 A* 搜索智能体, 使用 foodHeuristic"
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost

class AStarFoodSearchAgent(SearchAgent):
    "用于 FoodSearchProblem 的 A* 搜索智能体, 使用 foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, foodHeuristic)
        self.searchType = FoodSearchProblem

def foodHeuristic(state: Tuple[Tuple, List[List]], problem: FoodSearchProblem):
    """
    这是你为 FoodSearchProblem 定义的启发式函数.

    这个启发式函数必须是一致的以保证正确性.首先,尝试设计一个可接受的启发式函数;
    几乎所有可接受的启发式函数也都是一致的.

    如果使用 A* 时找到的解比统一代价搜索 (uniform cost search) 的解还差,
    说明你的启发式函数 *不* 一致,并且很可能不可接受！另一方面,
    不可接受或不一致的启发式函数可能仍能找到最优解,所以要小心.

    状态是一个元组 (pacmanPosition, foodGrid),其中 foodGrid 是一个 Grid
    (见 game.py),元素为 True 或 False.你可以调用 foodGrid.asList() 获取
    食物坐标列表.

    如果你想访问墙壁、胶囊等信息,可以查询 problem.例如,problem.walls
    会返回一个 Grid,表示墙的位置.

    如果你想在启发式函数的多次调用中 *存储* 信息以便重用,可以使用
    字典 problem.heuristicInfo.例如,如果你只想统计墙的数量一次并保存,
    可以写: problem.heuristicInfo['wallCount'] = problem.walls.count()
    之后多次调用该启发式函数时,可以直接访问
    problem.heuristicInfo['wallCount'].
    """

    position, foodGrid = state
    "*** YOUR CODE HERE ***"
    return 0

class ClosestDotSearchAgent(SearchAgent):
    "使用一系列搜索找到所有食物的智能体"
    def registerInitialState(self, state):
        self.actions = []
        currentState = state
        while(currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState) # The missing piece
            self.actions += nextPathSegment
            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    t = (str(action), str(currentState))
                    raise Exception('findPathToClosestDot returned an illegal move: %s!\n%s' % t)
                currentState = currentState.generateSuccessor(0, action)
        self.actionIndex = 0
        print('Path found with cost %d.' % len(self.actions))

    def findPathToClosestDot(self, gameState: pacman.GameState):
        """
        返回从 gameState 开始到最近食物的一条路径 (动作列表).
        """
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition()
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState)

        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class AnyFoodSearchProblem(PositionSearchProblem):
    """
    用于寻找任意食物路径的搜索问题.

    这个搜索问题和 PositionSearchProblem 类似, 但目标测试不同,
    需要你在下面实现. 状态空间和后继函数不需要修改.

    上面定义的 AnyFoodSearchProblem(PositionSearchProblem) 继承了
    PositionSearchProblem 的方法.

    你可以使用这个搜索问题帮助实现 findPathToClosestDot 方法.
    """


    def __init__(self, gameState):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def isGoalState(self, state: Tuple[int, int]):
        """
        状态是 Pacman 的位置. 在这里实现目标测试以完成问题定义.
        """
        x,y = state

        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def mazeDistance(point1: Tuple[int, int], point2: Tuple[int, int], gameState: pacman.GameState) -> int:
    """
    返回任意两点之间的迷宫距离, 使用你已实现的搜索函数.
    gameState 可以是任意游戏状态 -- 该状态下 Pacman 的位置会被忽略.

    使用示例: mazeDistance((2,4), (5,6), gameState)

    这个函数可能对你的 ApproximateSearchAgent 有用.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(search.bfs(prob))
