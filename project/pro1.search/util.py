# util.py
# -------
# 许可信息：你可以自由地使用或扩展这些项目用于教学目的,但必须满足以下条件：
# (1) 你不得分发或发布解决方案,
# (2) 你必须保留此声明,
# (3) 你必须清晰地注明加州大学伯克利分校的来源,包括链接 http://ai.berkeley.edu.
#
# 来源信息：Pacman AI 项目由加州大学伯克利分校开发.
# 核心项目和自动评分器主要由 John DeNero (denero@cs.berkeley.edu) 和 Dan Klein (klein@cs.berkeley.edu) 创建.
# 学生端的自动评分功能由 Brad Miller、Nick Hay 和 Pieter Abbeel (pabbeel@cs.berkeley.edu) 添加.


import sys
import inspect
import heapq, random


class FixedRandom:
    def __init__(self):
        fixedState = (3, (2147483648, 507801126, 683453281, 310439348, 2597246090, \
            2209084787, 2267831527, 979920060, 3098657677, 37650879, 807947081, 3974896263, \
            881243242, 3100634921, 1334775171, 3965168385, 746264660, 4074750168, 500078808, \
            776561771, 702988163, 1636311725, 2559226045, 157578202, 2498342920, 2794591496, \
            4130598723, 496985844, 2944563015, 3731321600, 3514814613, 3362575829, 3038768745, \
            2206497038, 1108748846, 1317460727, 3134077628, 988312410, 1674063516, 746456451, \
            3958482413, 1857117812, 708750586, 1583423339, 3466495450, 1536929345, 1137240525, \
            3875025632, 2466137587, 1235845595, 4214575620, 3792516855, 657994358, 1241843248, \
            1695651859, 3678946666, 1929922113, 2351044952, 2317810202, 2039319015, 460787996, \
            3654096216, 4068721415, 1814163703, 2904112444, 1386111013, 574629867, 2654529343, \
            3833135042, 2725328455, 552431551, 4006991378, 1331562057, 3710134542, 303171486, \
            1203231078, 2670768975, 54570816, 2679609001, 578983064, 1271454725, 3230871056, \
            2496832891, 2944938195, 1608828728, 367886575, 2544708204, 103775539, 1912402393, \
            1098482180, 2738577070, 3091646463, 1505274463, 2079416566, 659100352, 839995305, \
            1696257633, 274389836, 3973303017, 671127655, 1061109122, 517486945, 1379749962, \
            3421383928, 3116950429, 2165882425, 2346928266, 2892678711, 2936066049, 1316407868, \
            2873411858, 4279682888, 2744351923, 3290373816, 1014377279, 955200944, 4220990860, \
            2386098930, 1772997650, 3757346974, 1621616438, 2877097197, 442116595, 2010480266, \
            2867861469, 2955352695, 605335967, 2222936009, 2067554933, 4129906358, 1519608541, \
            1195006590, 1942991038, 2736562236, 279162408, 1415982909, 4099901426, 1732201505, \
            2934657937, 860563237, 2479235483, 3081651097, 2244720867, 3112631622, 1636991639, \
            3860393305, 2312061927, 48780114, 1149090394, 2643246550, 1764050647, 3836789087, \
            3474859076, 4237194338, 1735191073, 2150369208, 92164394, 756974036, 2314453957, \
            323969533, 4267621035, 283649842, 810004843, 727855536, 1757827251, 3334960421, \
            3261035106, 38417393, 2660980472, 1256633965, 2184045390, 811213141, 2857482069, \
            2237770878, 3891003138, 2787806886, 2435192790, 2249324662, 3507764896, 995388363, \
            856944153, 619213904, 3233967826, 3703465555, 3286531781, 3863193356, 2992340714, \
            413696855, 3865185632, 1704163171, 3043634452, 2225424707, 2199018022, 3506117517, \
            3311559776, 3374443561, 1207829628, 668793165, 1822020716, 2082656160, 1160606415, \
            3034757648, 741703672, 3094328738, 459332691, 2702383376, 1610239915, 4162939394, \
            557861574, 3805706338, 3832520705, 1248934879, 3250424034, 892335058, 74323433, \
            3209751608, 3213220797, 3444035873, 3743886725, 1783837251, 610968664, 580745246, \
            4041979504, 201684874, 2673219253, 1377283008, 3497299167, 2344209394, 2304982920, \
            3081403782, 2599256854, 3184475235, 3373055826, 695186388, 2423332338, 222864327, \
            1258227992, 3627871647, 3487724980, 4027953808, 3053320360, 533627073, 3026232514, \
            2340271949, 867277230, 868513116, 2158535651, 2487822909, 3428235761, 3067196046, \
            3435119657, 1908441839, 788668797, 3367703138, 3317763187, 908264443, 2252100381, \
            764223334, 4127108988, 384641349, 3377374722, 1263833251, 1958694944, 3847832657, \
            1253909612, 1096494446, 555725445, 2277045895, 3340096504, 1383318686, 4234428127, \
            1072582179, 94169494, 1064509968, 2681151917, 2681864920, 734708852, 1338914021, \
            1270409500, 1789469116, 4191988204, 1716329784, 2213764829, 3712538840, 919910444, \
            1318414447, 3383806712, 3054941722, 3378649942, 1205735655, 1268136494, 2214009444, \
            2532395133, 3232230447, 230294038, 342599089, 772808141, 4096882234, 3146662953, \
            2784264306, 1860954704, 2675279609, 2984212876, 2466966981, 2627986059, 2985545332, \
            2578042598, 1458940786, 2944243755, 3959506256, 1509151382, 325761900, 942251521, \
            4184289782, 2756231555, 3297811774, 1169708099, 3280524138, 3805245319, 3227360276, \
            3199632491, 2235795585, 2865407118, 36763651, 2441503575, 3314890374, 1755526087, \
            17915536, 1196948233, 949343045, 3815841867, 489007833, 2654997597, 2834744136, \
            417688687, 2843220846, 85621843, 747339336, 2043645709, 3520444394, 1825470818, \
            647778910, 275904777, 1249389189, 3640887431, 4200779599, 323384601, 3446088641, \
            4049835786, 1718989062, 3563787136, 44099190, 3281263107, 22910812, 1826109246, \
            745118154, 3392171319, 1571490704, 354891067, 815955642, 1453450421, 940015623, \
            796817754, 1260148619, 3898237757, 176670141, 1870249326, 3317738680, 448918002, \
            4059166594, 2003827551, 987091377, 224855998, 3520570137, 789522610, 2604445123, \
            454472869, 475688926, 2990723466, 523362238, 3897608102, 806637149, 2642229586, \
            2928614432, 1564415411, 1691381054, 3816907227, 4082581003, 1895544448, 3728217394, \
            3214813157, 4054301607, 1882632454, 2873728645, 3694943071, 1297991732, 2101682438, \
            3952579552, 678650400, 1391722293, 478833748, 2976468591, 158586606, 2576499787, \
            662690848, 3799889765, 3328894692, 2474578497, 2383901391, 1718193504, 3003184595, \
            3630561213, 1929441113, 3848238627, 1594310094, 3040359840, 3051803867, 2462788790, \
            954409915, 802581771, 681703307, 545982392, 2738993819, 8025358, 2827719383, \
            770471093, 3484895980, 3111306320, 3900000891, 2116916652, 397746721, 2087689510, \
            721433935, 1396088885, 2751612384, 1998988613, 2135074843, 2521131298, 707009172, \
            2398321482, 688041159, 2264560137, 482388305, 207864885, 3735036991, 3490348331, \
            1963642811, 3260224305, 3493564223, 1939428454, 1128799656, 1366012432, 2858822447, \
            1428147157, 2261125391, 1611208390, 1134826333, 2374102525, 3833625209, 2266397263, \
            3189115077, 770080230, 2674657172, 4280146640, 3604531615, 4235071805, 3436987249, \
            509704467, 2582695198, 4256268040, 3391197562, 1460642842, 1617931012, 457825497, \
            1031452907, 1330422862, 4125947620, 2280712485, 431892090, 2387410588, 2061126784, \
            896457479, 3480499461, 2488196663, 4021103792, 1877063114, 2744470201, 1046140599, \
            2129952955, 3583049218, 4217723693, 2720341743, 820661843, 1079873609, 3360954200, \
            3652304997, 3335838575, 2178810636, 1908053374, 4026721976, 1793145418, 476541615, \
            973420250, 515553040, 919292001, 2601786155, 1685119450, 3030170809, 1590676150, \
            1665099167, 651151584, 2077190587, 957892642, 646336572, 2743719258, 866169074, \
            851118829, 4225766285, 963748226, 799549420, 1955032629, 799460000, 2425744063, \
            2441291571, 1928963772, 528930629, 2591962884, 3495142819, 1896021824, 901320159, \
            3181820243, 843061941, 3338628510, 3782438992, 9515330, 1705797226, 953535929, \
            764833876, 3202464965, 2970244591, 519154982, 3390617541, 566616744, 3438031503, \
            1853838297, 170608755, 1393728434, 676900116, 3184965776, 1843100290, 78995357, \
            2227939888, 3460264600, 1745705055, 1474086965, 572796246, 4081303004, 882828851, \
            1295445825, 137639900, 3304579600, 2722437017, 4093422709, 273203373, 2666507854, \
            3998836510, 493829981, 1623949669, 3482036755, 3390023939, 833233937, 1639668730, \
            1499455075, 249728260, 1210694006, 3836497489, 1551488720, 3253074267, 3388238003, \
            2372035079, 3945715164, 2029501215, 3362012634, 2007375355, 4074709820, 631485888, \
            3135015769, 4273087084, 3648076204, 2739943601, 1374020358, 1760722448, 3773939706, \
            1313027823, 1895251226, 4224465911, 421382535, 1141067370, 3660034846, 3393185650, \
            1850995280, 1451917312, 3841455409, 3926840308, 1397397252, 2572864479, 2500171350, \
            3119920613, 531400869, 1626487579, 1099320497, 407414753, 2438623324, 99073255, \
            3175491512, 656431560, 1153671785, 236307875, 2824738046, 2320621382, 892174056, \
            230984053, 719791226, 2718891946, 624), None)
        self.random = random.Random()
        self.random.setstate(fixedState)

"""
  对于实现 SearchAgents 有用的数据结构
"""

class Stack:
    "A container with a last-in-first-out (LIFO) queuing policy."
    def __init__(self):
        self.list = []

    def push(self,item):
        "Push 'item' onto the stack"
        self.list.append(item)

    def pop(self):
        "Pop the most recently pushed item from the stack"
        return self.list.pop()

    def isEmpty(self):
        "Returns true if the stack is empty"
        return len(self.list) == 0

class Queue:
    "A container with a first-in-first-out (FIFO) queuing policy."
    def __init__(self):
        self.list = []

    def push(self,item):
        "Enqueue the 'item' into the queue"
        self.list.insert(0,item)

    def pop(self):
        """
          Dequeue the earliest enqueued item still in the queue. This
          operation removes the item from the queue.
        """
        return self.list.pop()

    def isEmpty(self):
        "Returns true if the queue is empty"
        return len(self.list) == 0

class PriorityQueue:
    """
      实现优先队列数据结构.
      每个插入的元素都带有一个优先级,调用者通常希望能快速获取队列中最低优先级的元素.
      该数据结构允许 O(1) 时间访问最低优先级元素.
    """
    def  __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # 如果元素已在优先队列中，且其优先级更高，则更新它的优先级并重建堆。
        # 如果元素已在优先队列中，且其优先级相同或更低，则不做任何操作。
        # 如果元素不在优先队列中，则执行与 self.push 相同的操作。
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)

class PriorityQueueWithFunction(PriorityQueue):
    """
    实现了一个与 Queue 和 Stack 类具有相同 push/pop 接口的优先队列.
    该设计便于与这两个类直接替换使用.
    调用者必须提供一个优先级函数,用于提取每个元素的优先级.
    """
    def  __init__(self, priorityFunction):
        "priorityFunction (item) -> priority"
        self.priorityFunction = priorityFunction      # store the priority function
        PriorityQueue.__init__(self)        # super-class initializer

    def push(self, item):
        "Adds an item to the queue with priority from the priority function"
        PriorityQueue.push(self, item, self.priorityFunction(item))


def manhattanDistance( xy1, xy2 ):
    "返回点 xy1 和 xy2 之间的曼哈顿距离"
    return abs( xy1[0] - xy2[0] ) + abs( xy1[1] - xy2[1] )

"""
  各种课程项目中有用的数据结构和函数

  搜索项目不需要使用本行以下的内容.
"""

class Counter(dict):
    """
    Counter 用于跟踪一组键的计数.

    Counter 类是 Python 标准字典类型的扩展.
    它专门用来存储数值(整数或浮点数),并包含了一些额外函数以便于统计数据.
    特别地,所有键的默认值为 0.

    使用普通字典：
    a = {}
    print(a['test'])
    会报错,而 Counter 类的对应用法：

    >>> a = Counter()
    >>> print(a['test'])
    0

    会返回默认值 0.

    注意：如果你确定一个键在 Counter 中已存在,你仍然可以使用字典语法来引用它.

    >>> a = Counter()
    >>> a['test'] = 2
    >>> print(a['test'])
    2

    这对于在不初始化计数的情况下统计事物非常有用,例如：

    >>> a['blah'] += 1
    >>> print(a['blah'])
    1

    Counter 还包含一些额外功能,在实现分类器时非常有用.
    两个 Counter 可以相加、相减或相乘(点积).
    它们还可以被归一化,并可以提取总计数以及最大值对应的键.
    """
    def __getitem__(self, idx):
        self.setdefault(idx, 0)
        return dict.__getitem__(self, idx)

    def incrementAll(self, keys, count):
        """
        将 keys 中的所有元素的值增加相同的 count.

        >>> a = Counter()
        >>> a.incrementAll(['one','two', 'three'], 1)
        >>> a['one']
        1
        >>> a['two']
        1
        """
        for key in keys:
            self[key] += count

    def argMax(self):
        """
        返回具有最大值的键.
        """
        if len(self.keys()) == 0: return None
        all = self.items()
        values = [x[1] for x in all]
        maxIndex = values.index(max(values))
        return all[maxIndex][0]

    def sortedKeys(self):
        """
        返回一个按值排序的键列表.
        值最高的键会排在最前面.

        >>> a = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> a['third'] = 1
        >>> a.sortedKeys()
        ['second', 'third', 'first']
        """
        sortedItems = self.items()
        compare = lambda x, y:  sign(y[1] - x[1])
        sortedItems.sort(cmp=compare)
        return [x[0] for x in sortedItems]

    def totalCount(self):
        """
        返回所有键的值的总和.
        """
        return sum(self.values())

    def normalize(self):
        """
        修改计数器,使所有键的值之和为 1.
        各键之间的比例保持不变.
        注意：对空 Counter 进行归一化会产生错误.
        """
        total = float(self.totalCount())
        if total == 0: return
        for key in self.keys():
            self[key] = self[key] / total

    def divideAll(self, divisor):
        """
        将所有值除以 divisor.
        """
        divisor = float(divisor)
        for key in self:
            self[key] /= divisor

    def copy(self):
        """
        返回当前 Counter 的一个副本
        """
        return Counter(dict.copy(self))

    def __mul__(self, y ):
        """
        两个 Counter 相乘等价于它们向量的点积,
        每个唯一键对应向量中的一个元素.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['second'] = 5
        >>> a['third'] = 1.5
        >>> a['fourth'] = 2.5
        >>> a * b
        14
        """
        sum = 0
        x = self
        if len(x) > len(y):
            x,y = y,x
        for key in x:
            if key not in y:
                continue
            sum += x[key] * y[key]
        return sum

    def __radd__(self, y):
        """
        将另一个 Counter 加到当前 Counter 上,
        相当于把第二个 Counter 的值累加到当前 Counter.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['third'] = 1
        >>> a += b
        >>> a['first']
        1
        """
        for key, value in y.items():
            self[key] += value

    def __add__( self, y ):
        """
        两个 Counter 相加会得到一个新的 Counter,
        其键是两者的并集,值是两者对应值的和.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['third'] = 1
        >>> (a + b)['first']
        1
        """
        addend = Counter()
        for key in self:
            if key in y:
                addend[key] = self[key] + y[key]
            else:
                addend[key] = self[key]
        for key in y:
            if key in self:
                continue
            addend[key] = y[key]
        return addend

    def __sub__( self, y ):
        """
        两个 Counter 相减会得到一个新的 Counter,
        其键是两者的并集,值是第一个减去第二个的结果.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['third'] = 1
        >>> (a - b)['first']
        -5
        """
        addend = Counter()
        for key in self:
            if key in y:
                addend[key] = self[key] - y[key]
            else:
                addend[key] = self[key]
        for key in y:
            if key in self:
                continue
            addend[key] = -1 * y[key]
        return addend

def raiseNotDefined():
    fileName = inspect.stack()[1][1]
    line = inspect.stack()[1][2]
    method = inspect.stack()[1][3]

    print("*** 方法尚未实现: %s at line %s of %s" % (method, line, fileName))
    sys.exit(1)

def normalize(vectorOrCounter):
    """
    通过将每个值除以总和来归一化一个向量或 Counter.
    """
    normalizedCounter = Counter()
    if type(vectorOrCounter) == type(normalizedCounter):
        counter = vectorOrCounter
        total = float(counter.totalCount())
        if total == 0: return counter
        for key in counter.keys():
            value = counter[key]
            normalizedCounter[key] = value / total
        return normalizedCounter
    else:
        vector = vectorOrCounter
        s = float(sum(vector))
        if s == 0: return vector
        return [el / s for el in vector]

def nSample(distribution, values, n):
    if sum(distribution) != 1:
        distribution = normalize(distribution)
    rand = [random.random() for i in range(n)]
    rand.sort()
    samples = []
    samplePos, distPos, cdf = 0,0, distribution[0]
    while samplePos < n:
        if rand[samplePos] < cdf:
            samplePos += 1
            samples.append(values[distPos])
        else:
            distPos += 1
            cdf += distribution[distPos]
    return samples

def sample(distribution, values = None):
    if type(distribution) == Counter:
        items = sorted(distribution.items())
        distribution = [i[1] for i in items]
        values = [i[0] for i in items]
    if sum(distribution) != 1:
        distribution = normalize(distribution)
    choice = random.random()
    i, total= 0, distribution[0]
    while choice > total:
        i += 1
        total += distribution[i]
    return values[i]

def sampleFromCounter(ctr):
    items = sorted(ctr.items())
    return sample([v for k,v in items], [k for k,v in items])

def getProbability(value, distribution, values):
    """
      给出在由 (distribution, values) 定义的离散分布下某个值的概率.
    """
    total = 0.0
    for prob, val in zip(distribution, values):
        if val == value:
            total += prob
    return total

def flipCoin( p ):
    r = random.random()
    return r < p

def chooseFromDistribution( distribution ):
    "接收一个 Counter 或 (概率, 键) 对的列表,并进行采样."
    if type(distribution) == dict or type(distribution) == Counter:
        return sample(distribution)
    r = random.random()
    base = 0.0
    for prob, element in distribution:
        base += prob
        if r <= base: return element

def nearestPoint( pos ):
    """
    找到最接近某个位置的网格点(离散化).
    """
    ( current_row, current_col ) = pos

    grid_row = int( current_row + 0.5 )
    grid_col = int( current_col + 0.5 )
    return ( grid_row, grid_col )

def sign( x ):
    """
    根据 x 的符号返回 1 或 -1.
    """
    if( x >= 0 ):
        return 1
    else:
        return -1

def arrayInvert(array):
    """
    将一个以列表嵌套列表形式存储的矩阵进行转置.
    """
    result = [[] for i in array]
    for outer in array:
        for inner in range(len(outer)):
            result[inner].append(outer[inner])
    return result

def matrixAsList( matrix, value = True ):
    """
    将矩阵转换为与指定值匹配的坐标列表.
    """
    rows, cols = len( matrix ), len( matrix[0] )
    cells = []
    for row in range( rows ):
        for col in range( cols ):
            if matrix[row][col] == value:
                cells.append( ( row, col ) )
    return cells

def lookup(name, namespace):
    """
    根据名字从已导入的模块中获取方法或类.
    用法: lookup(functionName, globals())
    """
    dots = name.count('.')
    if dots > 0:
        moduleName, objName = '.'.join(name.split('.')[:-1]), name.split('.')[-1]
        module = __import__(moduleName)
        return getattr(module, objName)
    else:
        modules = [obj for obj in namespace.values() if str(type(obj)) == "<type 'module'>"]
        options = [getattr(module, name) for module in modules if name in dir(module)]
        options += [obj[1] for obj in namespace.items() if obj[0] == name ]
        if len(options) == 1: return options[0]
        if len(options) > 1: raise Exception('名称冲突: %s')
        raise Exception('%s 未找到(既不是方法也不是类)' % name)

def pause():
    """
    暂停输出流,等待用户反馈.
    """
    input("<按回车键继续>")


# 处理超时的代码
#
# FIXME
# 注意：TimeoutFunction 不是可重入的.
# 后设置的超时会悄悄地覆盖之前的超时.
# 可以通过维护一个全局的超时列表来解决.
# 当前,所有调用此功能的测试用例中的学生代码都会被统一包装.
#
import signal
import time
class TimeoutFunctionException(Exception):
    """在超时时抛出的异常"""
    pass


class TimeoutFunction:
    def __init__(self, function, timeout):
        self.timeout = timeout
        self.function = function

    def handle_timeout(self, signum, frame):
        raise TimeoutFunctionException()

    def __call__(self, *args, **keyArgs):
        # 如果系统支持 SIGALRM 信号,则用它来在函数运行太久时触发异常.
        # 否则就在函数返回后检查耗时,如果超时则抛出异常.
        if hasattr(signal, 'SIGALRM'):
            old = signal.signal(signal.SIGALRM, self.handle_timeout)
            signal.alarm(self.timeout)
            try:
                result = self.function(*args, **keyArgs)
            finally:
                signal.signal(signal.SIGALRM, old)
            signal.alarm(0)
        else:
            startTime = time.time()
            result = self.function(*args, **keyArgs)
            timeElapsed = time.time() - startTime
            if timeElapsed >= self.timeout:
                self.handle_timeout(None, None)
        return result



_ORIGINAL_STDOUT = None
_ORIGINAL_STDERR = None
_MUTED = False

class WritableNull:
    def write(self, string):
        pass

def mutePrint():
    global _ORIGINAL_STDOUT, _ORIGINAL_STDERR, _MUTED
    if _MUTED:
        return
    _MUTED = True

    _ORIGINAL_STDOUT = sys.stdout
    #_ORIGINAL_STDERR = sys.stderr
    sys.stdout = WritableNull()
    #sys.stderr = WritableNull()

def unmutePrint():
    global _ORIGINAL_STDOUT, _ORIGINAL_STDERR, _MUTED
    if not _MUTED:
        return
    _MUTED = False

    sys.stdout = _ORIGINAL_STDOUT
    #sys.stderr = _ORIGINAL_STDERR
