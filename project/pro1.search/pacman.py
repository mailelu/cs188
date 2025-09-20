# pacman.py
# ---------
# 许可信息：你可以自由使用或扩展这些项目用于教育目的，前提是 (1) 你不分发或发布解决方案，(2) 保留此声明，(3) 明确归属加州大学伯克利分校，包括链接 http://ai.berkeley.edu。
# 
# 归属信息：Pacman AI 项目由加州大学伯克利分校开发。
# 核心项目和自动评分器主要由 John DeNero (denero@cs.berkeley.edu) 和 Dan Klein (klein@cs.berkeley.edu) 创建。
# 学生端自动评分由 Brad Miller、Nick Hay 和 Pieter Abbeel (pabbeel@cs.berkeley.edu) 添加。


"""
Pacman.py 文件包含经典吃豆人游戏的逻辑以及运行游戏的主程序。该文件可分为三个部分：

  (i) 吃豆人世界的接口：
          吃豆人是一个复杂的环境。你可能不想阅读我们为保证游戏正常运行而编写的所有代码。
          这一部分包含了你需要理解的、用于完成项目的关键代码。此外，game.py 中也有一些你需要了解的内容。

  (ii) 吃豆人的隐藏逻辑：
          这一部分包含了吃豆人环境中所有的逻辑代码，比如决定谁可以移动、碰撞时谁会死亡等。
          通常你不需要阅读这部分代码，但如果有兴趣，可以自行研究。

  (iii) 游戏启动框架：
          最后一部分包含了读取命令、设置游戏、启动新游戏的代码，以及如何链接所有外部模块（代理函数、图形界面等）。
          查看这一部分可以了解所有可用选项。

要开始你的第一局游戏，在命令行输入 'python pacman.py'。
移动按键为 'a'、's'、'd'、'w'（或使用方向键）。祝你玩得愉快！
"""

from game import GameStateData
from game import Game
from game import Directions
from game import Actions
from util import nearestPoint
from util import manhattanDistance
import util, layout
import sys, types, time, random, os

###################################################
# 你的吃豆人世界接口：一个游戏状态 #
###################################################

class GameState:
    """
    一个 GameState 指定完整的游戏状态，包括食物、胶囊、
    代理配置和分数变化。

    GameStates 由 Game 对象使用来捕获游戏的实际状态，
    并可被代理用来推理游戏情况。

    大部分 GameState 信息存储在 GameStateData 对象中。
    我们强烈建议通过下面的访问方法访问数据，而不是直接引用 GameStateData 对象。

    注意，在经典吃豆人中，Pacman 总是代理 0。
    """

    ####################################################
    # 访问方法：使用这些方法来获取状态数据 #
    ####################################################

    # 静态变量用于跟踪哪些状态已经调用过 getLegalActions
    explored = set()
    def getAndResetExplored():
        tmp = GameState.explored.copy()
        GameState.explored = set()
        return tmp
    getAndResetExplored = staticmethod(getAndResetExplored)

    def getLegalActions( self, agentIndex=0 ):
        """
        返回指定代理的合法动作
        """
#        GameState.explored.add(self)
        if self.isWin() or self.isLose(): return []

        if agentIndex == 0:  # Pacman 正在移动
            return PacmanRules.getLegalActions( self )
        else:
            return GhostRules.getLegalActions( self, agentIndex )

    def generateSuccessor( self, agentIndex, action):
        """
        返回指定代理执行动作后的后继状态
        """
        # 检查后继状态是否存在
        if self.isWin() or self.isLose(): raise Exception('无法生成终结状态的后继状态。')

        # 复制当前状态
        state = GameState(self)

        # 让代理的逻辑处理其动作对棋盘的影响
        if agentIndex == 0:  # Pacman 正在移动
            state.data._eaten = [False for i in range(state.getNumAgents())]
            PacmanRules.applyAction( state, action )
        else:                # 幽灵正在移动
            GhostRules.applyAction( state, action, agentIndex )

        # 时间推进
        if agentIndex == 0:
            state.data.scoreChange += -TIME_PENALTY # 等待的惩罚
        else:
            GhostRules.decrementTimer( state.data.agentStates[agentIndex] )

        # 处理多代理效果
        GhostRules.checkDeath( state, agentIndex )

        # 记录信息
        state.data._agentMoved = agentIndex
        state.data.score += state.data.scoreChange
        GameState.explored.add(self)
        GameState.explored.add(state)
        return state

    def getLegalPacmanActions( self ):
        return self.getLegalActions( 0 )

    def generatePacmanSuccessor( self, action ):
        """
        生成指定 Pacman 动作后的后继状态
        """
        return self.generateSuccessor( 0, action )

    def getPacmanState( self ):
        """
        返回 Pacman 的 AgentState 对象 (在 game.py 中)

        state.pos 给出当前位置
        state.direction 给出移动方向
        """
        return self.data.agentStates[0].copy()

    def getPacmanPosition( self ):
        return self.data.agentStates[0].getPosition()

    def getGhostStates( self ):
        return self.data.agentStates[1:]

    def getGhostState( self, agentIndex ):
        if agentIndex == 0 or agentIndex >= self.getNumAgents():
            raise Exception("传递给 getGhostState 的索引无效")
        return self.data.agentStates[agentIndex]

    def getGhostPosition( self, agentIndex ):
        if agentIndex == 0:
            raise Exception("Pacman 的索引传递给 getGhostPosition")
        return self.data.agentStates[agentIndex].getPosition()

    def getGhostPositions(self):
        return [s.getPosition() for s in self.getGhostStates()]

    def getNumAgents( self ):
        return len( self.data.agentStates )

    def getScore( self ):
        return float(self.data.score)

    def getCapsules(self):
        """
        返回剩余胶囊的位置列表 (x,y)
        """
        return self.data.capsules

    def getNumFood( self ):
        return self.data.food.count()

    def getFood(self):
        """
        返回布尔型食物指示变量的 Grid。

        Grid 可通过列表访问，因此要检查 (x,y) 是否有食物，只需调用：
        currentFood = state.getFood()
        if currentFood[x][y] == True: ...
        """
        return self.data.food

    def getWalls(self):
        """
        返回布尔型墙壁指示变量的 Grid。

        Grid 可通过列表访问，因此要检查 (x,y) 是否有墙壁，只需调用：
        walls = state.getWalls()
        if walls[x][y] == True: ...
        """
        return self.data.layout.walls

    def hasFood(self, x, y):
        return self.data.food[x][y]

    def hasWall(self, x, y):
        return self.data.layout.walls[x][y]

    def isLose( self ):
        return self.data._lose

    def isWin( self ):
        return self.data._win

    #############################################
    #             辅助方法：                   #
    # 通常不需要直接调用这些方法              #
    #############################################

    def __init__( self, prevState = None ):
        """
        通过复制前一状态生成新状态
        """
        if prevState != None: # 初始状态
            self.data = GameStateData(prevState.data)
        else:
            self.data = GameStateData()

    def deepCopy( self ):
        state = GameState( self )
        state.data = self.data.deepCopy()
        return state

    def __eq__( self, other ):
        """
        允许比较两个状态
        """
        return hasattr(other, 'data') and self.data == other.data

    def __hash__( self ):
        """
        允许状态作为字典的键
        """
        return hash( self.data )

    def __str__( self ):
        return str(self.data)

    def initialize( self, layout, numGhostAgents=1000 ):
        """
        根据布局数组创建初始游戏状态 (参见 layout.py)
        """
        self.data.initialize(layout, numGhostAgents)

############################################################################
#                     吃豆人的隐藏逻辑                                      #
#                                                                          #
#                      通常不需要查看本节的代码                              #
############################################################################

SCARED_TIME = 40    # 幽灵被吓住的移动回合数
COLLISION_TOLERANCE = 0.7 # 幽灵与 Pacman 接触以致死亡的距离阈值
TIME_PENALTY = 1 # 每回合损失的分数

class ClassicGameRules:
    """
    这些游戏规则管理游戏的控制流程，决定游戏何时以及如何开始和结束
    """
    def __init__(self, timeout=30):
        self.timeout = timeout

    def newGame( self, layout, pacmanAgent, ghostAgents, display, quiet = False, catchExceptions=False):
        agents = [pacmanAgent] + ghostAgents[:layout.getNumGhosts()]
        initState = GameState()
        initState.initialize( layout, len(ghostAgents) )
        game = Game(agents, display, self, catchExceptions=catchExceptions)
        game.state = initState
        self.initialState = initState.deepCopy()
        self.quiet = quiet
        return game

    def process(self, state, game):
        """
        检查是否该结束游戏
        """
        if state.isWin(): self.win(state, game)
        if state.isLose(): self.lose(state, game)

    def win( self, state, game ):
        if not self.quiet: print("Pacman 胜利！得分: %d" % state.data.score)
        game.gameOver = True

    def lose( self, state, game ):
        if not self.quiet: print("Pacman 死亡！得分: %d" % state.data.score)
        game.gameOver = True

    def getProgress(self, game):
        return float(game.state.getNumFood()) / self.initialState.getNumFood()

    def agentCrash(self, game, agentIndex):
        if agentIndex == 0:
            print("Pacman 崩溃")
        else:
            print("一个幽灵崩溃")

    def getMaxTotalTime(self, agentIndex):
        return self.timeout

    def getMaxStartupTime(self, agentIndex):
        return self.timeout

    def getMoveWarningTime(self, agentIndex):
        return self.timeout

    def getMoveTimeout(self, agentIndex):
        return self.timeout

    def getMaxTimeWarnings(self, agentIndex):
        return 0

class PacmanRules:
    """
    这些函数管理 Pacman 在经典游戏规则下如何与环境互动
    """
    PACMAN_SPEED=1

    def getLegalActions( state ):
        """
        返回可能动作的列表
        """
        return Actions.getPossibleActions( state.getPacmanState().configuration, state.data.layout.walls )
    getLegalActions = staticmethod( getLegalActions )

    def applyAction( state, action ):
        """
        编辑状态以反映动作的结果
        """
        legal = PacmanRules.getLegalActions( state )
        if action not in legal:
            raise Exception("非法动作 " + str(action))

        pacmanState = state.data.agentStates[0]

        # 更新配置
        vector = Actions.directionToVector( action, PacmanRules.PACMAN_SPEED )
        pacmanState.configuration = pacmanState.configuration.generateSuccessor( vector )

        # 吃食物
        next = pacmanState.configuration.getPosition()
        nearest = nearestPoint( next )
        if manhattanDistance( nearest, next ) <= 0.5 :
            # 移除食物
            PacmanRules.consume( nearest, state )
    applyAction = staticmethod( applyAction )

    def consume( position, state ):
        x,y = position
        # 吃食物
        if state.data.food[x][y]:
            state.data.scoreChange += 10
            state.data.food = state.data.food.copy()
            state.data.food[x][y] = False
            state.data._foodEaten = position
            # TODO: 缓存 numFood?
            numFood = state.getNumFood()
            if numFood == 0 and not state.data._lose:
                state.data.scoreChange += 500
                state.data._win = True
        # 吃胶囊
        if( position in state.getCapsules() ):
            state.data.capsules.remove( position )
            state.data._capsuleEaten = position
            # 重置所有幽灵的害怕计时器
            for index in range( 1, len( state.data.agentStates ) ):
                state.data.agentStates[index].scaredTimer = SCARED_TIME
    consume = staticmethod( consume )

class GhostRules:
    """
    这些函数管理幽灵如何与环境互动
    """
    GHOST_SPEED=1.0
    def getLegalActions( state, ghostIndex ):
        """
        幽灵不能停下，除非到达死胡同才能掉头，
        但可以在交叉口转 90 度
        """
        conf = state.getGhostState( ghostIndex ).configuration
        possibleActions = Actions.getPossibleActions( conf, state.data.layout.walls )
        reverse = Actions.reverseDirection( conf.direction )
        if Directions.STOP in possibleActions:
            possibleActions.remove( Directions.STOP )
        if reverse in possibleActions and len( possibleActions ) > 1:
            possibleActions.remove( reverse )
        return possibleActions
    getLegalActions = staticmethod( getLegalActions )

    def applyAction( state, action, ghostIndex):
        legal = GhostRules.getLegalActions( state, ghostIndex )
        if action not in legal:
            raise Exception("非法幽灵动作 " + str(action))

        ghostState = state.data.agentStates[ghostIndex]
        speed = GhostRules.GHOST_SPEED
        if ghostState.scaredTimer > 0: speed /= 2.0
        vector = Actions.directionToVector( action, speed )
        ghostState.configuration = ghostState.configuration.generateSuccessor( vector )
    applyAction = staticmethod( applyAction )

    def decrementTimer( ghostState):
        timer = ghostState.scaredTimer
        if timer == 1:
            ghostState.configuration.pos = nearestPoint( ghostState.configuration.pos )
        ghostState.scaredTimer = max( 0, timer - 1 )
    decrementTimer = staticmethod( decrementTimer )

    def checkDeath( state, agentIndex):
        pacmanPosition = state.getPacmanPosition()
        if agentIndex == 0: # Pacman 刚移动；任何幽灵都可以杀死他
            for index in range( 1, len( state.data.agentStates ) ):
                ghostState = state.data.agentStates[index]
                ghostPosition = ghostState.configuration.getPosition()
                if GhostRules.canKill( pacmanPosition, ghostPosition ):
                    GhostRules.collide( state, ghostState, index )
        else:
            ghostState = state.data.agentStates[agentIndex]
            ghostPosition = ghostState.configuration.getPosition()
            if GhostRules.canKill( pacmanPosition, ghostPosition ):
                GhostRules.collide( state, ghostState, agentIndex )
    checkDeath = staticmethod( checkDeath )

    def collide( state, ghostState, agentIndex):
        if ghostState.scaredTimer > 0:
            state.data.scoreChange += 200
            GhostRules.placeGhost(state, ghostState)
            ghostState.scaredTimer = 0
            # 为第一人称模式添加
            state.data._eaten[agentIndex] = True
        else:
            if not state.data._win:
                state.data.scoreChange -= 500
                state.data._lose = True
    collide = staticmethod( collide )

    def canKill( pacmanPosition, ghostPosition ):
        return manhattanDistance( ghostPosition, pacmanPosition ) <= COLLISION_TOLERANCE
    canKill = staticmethod( canKill )

    def placeGhost(state, ghostState):
        ghostState.configuration = ghostState.start
    placeGhost = staticmethod( placeGhost )

#############################
# 游戏启动框架               #
#############################
def default(str):
    return str + ' [默认值: %default]'

def parseAgentArgs(str):
    if str == None: return {}
    pieces = str.split(',')
    opts = {}
    for p in pieces:
        if '=' in p:
            key, val = p.split('=')
        else:
            key,val = p, 1
        opts[key] = val
    return opts

def readCommand( argv ):
    """
    处理用于从命令行运行 pacman 的命令
    """
    from optparse import OptionParser
    usageStr = """
    用法:      python pacman.py <选项>
    示例:      (1) python pacman.py
                    - 启动交互式游戏
                (2) python pacman.py --layout smallClassic --zoom 2
                或  python pacman.py -l smallClassic -z 2
                    - 在较小的棋盘上启动交互式游戏，并放大显示
    """
    parser = OptionParser(usageStr)

    parser.add_option('-n', '--numGames', dest='numGames', type='int',
                      help=default('要玩的游戏数量'), metavar='GAMES', default=1)
    parser.add_option('-l', '--layout', dest='layout',
                      help=default('用于加载地图布局的 LAYOUT_FILE'),
                      metavar='LAYOUT_FILE', default='mediumClassic')
    parser.add_option('-p', '--pacman', dest='pacman',
                      help=default('pacmanAgents 模块中要使用的代理类型'),
                      metavar='TYPE', default='KeyboardAgent')
    parser.add_option('-t', '--textGraphics', action='store_true', dest='textGraphics',
                      help='仅显示文本输出', default=False)
    parser.add_option('-q', '--quietTextGraphics', action='store_true', dest='quietGraphics',
                      help='生成最小输出且不显示图形', default=False)
    parser.add_option('-g', '--ghosts', dest='ghost',
                      help=default('ghostAgents 模块中要使用的幽灵代理类型'),
                      metavar = 'TYPE', default='RandomGhost')
    parser.add_option('-k', '--numghosts', type='int', dest='numGhosts',
                      help=default('使用的幽灵最大数量'), default=4)
    parser.add_option('-z', '--zoom', type='float', dest='zoom',
                      help=default('缩放图形窗口大小'), default=1.0)
    parser.add_option('-f', '--fixRandomSeed', action='store_true', dest='fixRandomSeed',
                      help='固定随机种子以始终玩相同游戏', default=False)
    parser.add_option('-r', '--recordActions', action='store_true', dest='record',
                      help='将游戏记录写入文件（按游戏时间命名）', default=False)
    parser.add_option('--replay', dest='gameToReplay',
                      help='重放已记录的游戏文件（pickle）', default=None)
    parser.add_option('-a','--agentArgs',dest='agentArgs',
                      help='发送给代理的逗号分隔值，例如 "opt1=val1,opt2,opt3=val3"')
    parser.add_option('-x', '--numTraining', dest='numTraining', type='int',
                      help=default('训练回合数（抑制输出）'), default=0)
    parser.add_option('--frameTime', dest='frameTime', type='float',
                      help=default('帧之间的延迟时间；<0 表示键盘控制'), default=0.1)
    parser.add_option('-c', '--catchExceptions', action='store_true', dest='catchExceptions',
                      help='开启游戏中的异常处理和超时', default=False)
    parser.add_option('--timeout', dest='timeout', type='int',
                      help=default('单个游戏中代理计算的最长时间'), default=30)

    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0:
        raise Exception('无法理解的命令行输入: ' + str(otherjunk))
    args = dict()

    # 固定随机种子
    if options.fixRandomSeed: random.seed('cs188')

    # 选择布局
    args['layout'] = layout.getLayout( options.layout )
    if args['layout'] == None: raise Exception("找不到布局 " + options.layout)

    # 选择 Pacman 代理
    noKeyboard = options.gameToReplay == None and (options.textGraphics or options.quietGraphics)
    pacmanType = loadAgent(options.pacman, noKeyboard)
    agentOpts = parseAgentArgs(options.agentArgs)
    if options.numTraining > 0:
        args['numTraining'] = options.numTraining
        if 'numTraining' not in agentOpts: agentOpts['numTraining'] = options.numTraining
    pacman = pacmanType(**agentOpts) # 使用 agentArgs 实例化 Pacman
    args['pacman'] = pacman

    # 不显示训练游戏
    if 'numTrain' in agentOpts:
        options.numQuiet = int(agentOpts['numTrain'])
        options.numIgnore = int(agentOpts['numTrain'])

    # 选择幽灵代理
    ghostType = loadAgent(options.ghost, noKeyboard)
    args['ghosts'] = [ghostType( i+1 ) for i in range( options.numGhosts )]

    # 选择显示方式
    if options.quietGraphics:
        import textDisplay
        args['display'] = textDisplay.NullGraphics()
    elif options.textGraphics:
        import textDisplay
        textDisplay.SLEEP_TIME = options.frameTime
        args['display'] = textDisplay.PacmanGraphics()
    else:
        import graphicsDisplay
        args['display'] = graphicsDisplay.PacmanGraphics(options.zoom, frameTime = options.frameTime)
    args['numGames'] = options.numGames
    args['record'] = options.record
    args['catchExceptions'] = options.catchExceptions
    args['timeout'] = options.timeout

    # 特殊情况：已记录的游戏不使用 runGames 方法或 args 结构
    if options.gameToReplay != None:
        print('正在重放记录的游戏 %s.' % options.gameToReplay)
        import pickle
        f = open(options.gameToReplay, 'rb')
        try: recorded = pickle.load(f)
        finally: f.close()
        recorded['display'] = args['display']
        replayGame(**recorded)
        sys.exit(0)

    return args

def loadAgent(pacman, nographics):
    # 遍历所有 pythonPath 目录寻找对应模块
    pythonPathStr = os.path.expandvars("$PYTHONPATH")
    if pythonPathStr.find(';') == -1:
        pythonPathDirs = pythonPathStr.split(':')
    else:
        pythonPathDirs = pythonPathStr.split(';')
    pythonPathDirs.append('.')

    for moduleDir in pythonPathDirs:
        if not os.path.isdir(moduleDir): continue
        moduleNames = [f for f in os.listdir(moduleDir) if f.endswith('gents.py')]
        for modulename in moduleNames:
            try:
                module = __import__(modulename[:-3])
            except ImportError:
                continue
            if pacman in dir(module):
                if nographics and modulename == 'keyboardAgents.py':
                    raise Exception('使用键盘需要图形界面（不能使用文本显示）')
                return getattr(module, pacman)
    raise Exception('代理 ' + pacman + ' 未在任何 *Agents.py 文件中指定。')

def replayGame( layout, actions, display ):
    import pacmanAgents, ghostAgents
    rules = ClassicGameRules()
    agents = [pacmanAgents.GreedyAgent()] + [ghostAgents.RandomGhost(i+1) for i in range(layout.getNumGhosts())]
    game = rules.newGame( layout, agents[0], agents[1:], display )
    state = game.state
    display.initialize(state.data)

    for action in actions:
        # 执行动作
        state = state.generateSuccessor( *action )
        # 更新显示
        display.update( state.data )
        # 考虑游戏特定条件（胜利、失败等）
        rules.process(state, game)

    display.finish()

def runGames( layout, pacman, ghosts, display, numGames, record, numTraining = 0, catchExceptions=False, timeout=30 ):
    import __main__
    __main__.__dict__['_display'] = display

    rules = ClassicGameRules(timeout)
    games = []

    for i in range( numGames ):
        beQuiet = i < numTraining
        if beQuiet:
            # 抑制输出和图形
            import textDisplay
            gameDisplay = textDisplay.NullGraphics()
            rules.quiet = True
        else:
            gameDisplay = display
            rules.quiet = False
        game = rules.newGame( layout, pacman, ghosts, gameDisplay, beQuiet, catchExceptions)
        game.run()
        if not beQuiet: games.append(game)

        if record:
            import time, pickle
            fname = ('recorded-game-%d' % (i + 1)) +  '-'.join([str(t) for t in time.localtime()[1:6]])
            f = open(fname, 'wb')
            components = {'layout': layout, 'actions': game.moveHistory}
            pickle.dump(components, f)
            f.close()

    if (numGames-numTraining) > 0:
        scores = [game.state.getScore() for game in games]
        wins = [game.state.isWin() for game in games]
        winRate = wins.count(True)/ float(len(wins))
        print('平均分数:', sum(scores) / float(len(scores)))
        print('分数:       ', ', '.join([str(score) for score in scores]))
        print('胜率:       %d/%d (%.2f)' % (wins.count(True), len(wins), winRate))
        print('记录:       ', ', '.join([ ['失败', '胜利'][int(w)] for w in wins]))

    return games

if __name__ == '__main__':
    """
    当从命令行运行 pacman.py 时调用的主函数：

    > python pacman.py

    查看用法说明了解更多细节。

    > python pacman.py --help
    """
    args = readCommand( sys.argv[1:] ) # 根据输入获取游戏组件
    runGames( **args )

    # import cProfile
    # cProfile.run("run
    pass