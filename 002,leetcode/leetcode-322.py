# leetcode-322: BFS广度优先搜索，怎么保证路径最短的？
class Solution(object):
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        if amount == 0:
            return 0
        value1 = [0]
        value2 = []
        nc =  0
        visited = [False]*(amount+1)  # 记录是否来过这里
        visited[0] = True  # 已经来过amount=0这个数额的money
        while value1:
            nc += 1  # 记录步数
            for v in value1:
                for coin in coins:   # 广度搜索：遍历一遍cions中的元素
                    newval = v + coin
                    if newval == amount:
                        return nc  # 出口一：存在路径
                    elif newval > amount:
                        continue   # 跳出for coin循环，但是，仍然在外层的for循环中，继续遍历下一个value1中的元素v
                    elif not visited[newval]:
                        visited[newval] = True
                        value2.append(newval)
            value1, value2 = value2, []   # 当value2=[]时，跳出while循环，执行return -1 语句

        return -1   # 出口二：不存在路径