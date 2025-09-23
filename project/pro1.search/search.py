# search.py
# ---------
# 许可信息：您可以自由使用或扩展这些项目用于教育目的,前提是 (1) 不分发或发布解决方案,(2) 保留此声明,(3) 清楚地标明归属 UC Berkeley,包括链接 http://ai.berkeley.edu.
# 
# 归属信息：Pacman AI 项目由 UC Berkeley 开发.
# 核心项目和自动评分器主要由 John DeNero (denero@cs.berkeley.edu) 和 Dan Klein (klein@cs.berkeley.edu) 创建.
# 学生端自动评分由 Brad Miller, Nick Hay 和 Pieter Abbeel (pabbeel@cs.berkeley.edu) 添加.

"""
在 search.py 中,您将实现通用搜索算法,这些算法由
Pacman 代理（在 searchAgents.py 中）调用.
"""

import util

class SearchProblem:
    """
    该类概述了搜索问题的结构,但不实现任何方法
    （面向对象术语：抽象类）.

    您不需要修改此类中的任何内容.
    """

    def getStartState(self):
        """
        返回搜索问题的起始状态
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
        state: 搜索状态

        如果且仅当状态是有效目标状态时返回 True
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
        state: 搜索状态

        对于给定状态,这应返回一个三元组列表 (successor,
        action, stepCost),其中 'successor' 是当前状态的后继状态,
        'action' 是达到该状态所需的动作,'stepCost' 是扩展到该
        后继状态的增量代价.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
        actions: 要采取的动作列表

        此方法返回特定动作序列的总成本.
        动作序列必须由合法移动组成.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    返回解决 tinyMaze 的动作序列.对于其他迷宫,
    动作序列将不正确,因此仅用于 tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    优先搜索搜索树中最深的节点.

    您的搜索算法需要返回一个动作列表,以达到目标.
    确保实现图搜索算法.

    为了入门,您可以尝试以下简单命令来理解传入的搜索问题：

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    Start: (5, 5)
    Is the start a goal? False
    Start's successors: [((5, 4), 'South', 1), ((4, 5), 'West', 1)]
    """
    "*** YOUR CODE HERE ***"
    path = []
    visited = set()
    startState  = problem.getStartState()
    frontier= util.Stack()
    frontier.push((startState,path))
    while not frontier.isEmpty():
        frontier_item = frontier.pop()   #弹栈
        if problem.isGoalState(frontier_item[0]):
            return frontier_item[1]
        visited.add(frontier_item[0])  #visited加入弹出的点
        for success in problem.getSuccessors(frontier_item[0]):  # 入栈可选路径
            if success[0] not in visited:  #如果没有被访问,则加入
                frontier.push((success[0], frontier_item[1] + [success[1]]))    

def breadthFirstSearch(problem: SearchProblem):
    """优先搜索搜索树中最浅的节点."""
    "*** YOUR CODE HERE ***"
    path = []
    visited = set()
    startState  = problem.getStartState()
    frontier= util.Queue()    #使用队列
    frontier.push((startState,path))
    while not frontier.isEmpty():
        frontier_item = frontier.pop()   #弹栈
        if frontier_item[0] not in visited:
            if problem.isGoalState(frontier_item[0]):
                return frontier_item[1]
            visited.add(frontier_item[0])  #visited加入弹出的点
            for success in problem.getSuccessors(frontier_item[0]):  # 入栈可选路径
                if success[0] not in visited:  #如果没有被访问,则加入
                    frontier.push((success[0], frontier_item[1] + [success[1]]))    

def uniformCostSearch(problem: SearchProblem):
    """优先搜索总代价最小的节点."""
    "*** YOUR CODE HERE ***"
    path = []
    visited = set()
    startState  = problem.getStartState()
    frontier= util.PriorityQueue()    #使用优先队列
    startCost = 0
    frontier.push((startState,path,startCost), startCost)
    while not frontier.isEmpty():
        frontier_item = frontier.pop()   #弹栈
        if frontier_item[0] not in visited:
            if problem.isGoalState(frontier_item[0]):
                return frontier_item[1]
            visited.add(frontier_item[0])  #visited加入弹出的点
            for success in problem.getSuccessors(frontier_item[0]):  # 入栈可选路径
                if success[0] not in visited:  #如果没有被访问,则加入
                    frontier.update((success[0], frontier_item[1] + [success[1]],frontier_item[2] + success[2]), frontier_item[2] + success[2])    

def nullHeuristic(state, problem=None):
    """
    启发式函数用于估计从当前状态到提供的 SearchProblem 中最近目标的代价.
    该启发式函数很简单.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """优先搜索总代价与启发式估价之和最小的节点."""
    "*** YOUR CODE HERE ***"
    """
        state = ( (x,y), visited_corner)
        frontier = ( state, path, cost)
        cost =  pre_cost + g(n) + h(n)
    """
    path = []
    visited = set()
    startState  = problem.getStartState()
    frontier= util.PriorityQueue()    #使用优先队列
    startCost = 0 
    frontier.push((startState,path,startCost), startCost + heuristic(startState, problem))
    while not frontier.isEmpty():
        frontier_item = frontier.pop()   #弹栈
        if frontier_item[0] not in visited:
            if problem.isGoalState(frontier_item[0]):
                return frontier_item[1]
            visited.add(frontier_item[0])  #visited加入弹出的点
            """
                success = ( ((x,y),visited_corner) , action cost )
            """
            for success in problem.getSuccessors(frontier_item[0]):  # 入栈可选路径
                if success[0] not in visited:  #如果没有被访问,则加入
                    frontier.update((success[0], frontier_item[1] + [success[1]],frontier_item[2] + success[2]), frontier_item[2] + success[2] + heuristic(success[0], problem))    


# 缩写
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
