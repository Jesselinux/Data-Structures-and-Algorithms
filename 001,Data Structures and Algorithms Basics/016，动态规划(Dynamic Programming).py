# 1,零钱组合排序问题：
def coin(n):
    dp = [0] * (n + 1)
    dp[0] = dp[1] = dp[2] = 1
    dp[3] = 2
    for i in range(4, n + 1):
        dp[i] = dp[i - 1] + dp[i - 3] + dp[i - 4]
    
    return dp[n]

if __name__ == '__main__':
	print(coin(10))

# 2,入室抢劫：
def rob(nums):
    n = len(nums)
    dp = [ [0] * 2 for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = max(dp[i - 1][0], dp[i - 1][1]) # forget it
        dp[i][1] = nums[i - 1] + dp[i - 1][0]       # let's do it
    return max(dp[n][0], dp[n][1])  

def rob2(nums):
    yes, no = 0, 0
    for i in nums: 
        no, yes = max(no, yes), i + no
    return max(no, yes)

# 3,入室抢劫(2)：环形街道,第一家与最后一家是邻居
def rob(nums):
    def rob(nums):
        yes, no = 0, 0
        for i in nums: 
            no, yes = max(no, yes), i + no
        return max(no, yes)
    return max(rob(nums[len(nums) != 1:]), rob(nums[:-1]))

def rob2(nums):
    if len(nums) == 0:
        return 0

    if len(nums) == 1:
        return nums[0]

    return max(robRange(nums, 0, len(nums) - 1),\
               robRange(nums, 1, len(nums)))

def robRange(nums, start, end):
    yes, no = nums[start], 0
    for i in range(start + 1, end): 
        no, yes = max(no, yes), i + no
    return max(no, yes)

if __name__ == '__main__':
	nums = [2,7,9,3,1]
	rob(nums)

# 4,铺地板(1):
def minCostClimbingStairs(cost):
    n = len(cost) + 1
    dp = [0] * n
    for i in range(2, n):
        dp[i] = min(dp[i - 2] + cost[i - 2], dp[i - 1] + cost[i - 1])
    return dp[n - 1]

def minCostClimbingStairs2(cost):
    dp0, dp1, dp2 = 0, 0, 0
    for i in range(2, len(cost) + 1):
        dp2 = min(dp0 + cost[i - 2], dp1 + cost[i - 1])
        dp0, dp1 = dp1, dp2
    return dp2

if __name__ == '__main__':
	cost = [1, 100, 1, 1, 1, 100, 1, 1, 100, 1]
	minCostClimbingStairs2(cost)

# 5,A-Z解码：
def numDecodings(s):
    if s=="" or s[0]=='0': return 0
    dp=[1,1]
    for i in range(2,len(s)+1):
        # if it is 0, then dp[i] = 0
        result = 0
        if 10 <=int(s[i-2:i]) <=26:
            result = dp[i-2]
        if s[i-1] != '0':
            result += dp[i-1]
        dp.append(result)
    return dp[len(s)]

if __name__ == '__main__':
	numDecodings("226")

# 6,二叉树储存1-n的结构数量：
def numTress(n):
    if n <= 2:
        return n
    sol = [0] * (n + 1)
    sol[0] = sol[1] = 1
    for i in range(2, n + 1):
        for left in range (0, i):
            sol[i] += sol[left] * sol[i - 1 - left]
    
    return sol[n]   

if __name__ == '__main__':
	l = [numTress(i) for i in range(1, 6)]
	print(l)

# 7,最大子序列乘积：
def maxProduct(nums):
    if len(nums) == 0:
        return 0
    maximum = minimum = result = nums[0]
    for i in range(1, len(nums)):
        maximum, minimum = max(maximum * nums[i], minimum * nums[i], nums[i]), \
                           min(maximum * nums[i], minimum * nums[i], nums[i])
        result = max(result, maximum)
    return result

if __name__ == '__main__':
	nums = [2,3,-2,4]
	maxProduct(nums)

# 8,股票买卖(每天交易一次)：
def maxProfit(prices):
    if len(prices) < 2:
        return 0
    min_price = prices[0]
    max_profit = 0
    for price in prices:
        if price < min_price:
            min_price = price
        if price - min_price > max_profit:
            max_profit = price - min_price
    return max_profit

def maxProfit2(prices):
    if len(prices) < 2:
        return 0
    minPrice = prices[0]
    dp = [0] * len(prices)
    for i in range(len(prices)):
        dp[i] = max(dp[i-1], prices[i] - minPrice)
        minPrice = min(minPrice, prices[i])
    return dp[-1]

if __name__ == '__main__':
	prices = [7,1,5,3,6,4]
	maxProfit2(prices)

# 9,股票买卖(任意多次交易)：
def maxProfit1(prices):
    max_profit = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            max_profit += prices[i] - prices[i - 1]
    return max_profit

def maxProfit2(prices):
    max_profit = 0
    for i in range(1, len(prices)):
        max_profit += max(0, prices[i] - prices[i - 1])
    return max_profit

if __name__ == '__main__':
	prices = [7,1,5,3,6,4]
	maxProfit2(prices)

# 10,股票买卖(任意多次交易，手续费k元)：
def maxProfit3(prices, fee):
    cash, hold = 0, -prices[0]
    for i in range(1, len(prices)):
        cash, hold = max(cash, hold + prices[i] - fee), max(hold, cash - prices[i])
    return cash

if __name__ == '__main__':
	prices = [1, 3, 2, 8, 4, 9]
	fee = 2
	maxProfit3(prices, fee)

# 11,股票买卖(一天可以先卖后买，交易两次)：
def maxProfit4(prices):
    total_max_profit = 0
    n = len(prices)
    left_profits = [0] * n
    min_price = float('inf')

    for i in range(n):
        min_price = min(min_price, prices[i])
        total_max_profit = max(total_max_profit, prices[i] - min_price)
        left_profits[i] = total_max_profit

    max_profit = 0
    max_price = float('-inf')
    for i in range(n - 1, 0, -1):
        max_price = max(max_price, prices[i])
        max_profit = max(max_profit, max_price - prices[i])
        total_max_profit = max(total_max_profit, max_profit + left_profits[i - 1])
    return total_max_profit

if __name__ == '__main__':
	prices = [3,3,5,0,0,3,1,4]
	maxProfit4(prices)

# 12,股票买卖(T+0交易)：
def maxProfit5(prices, k):
    if len(prices) < 2:
        return 0

    if len(prices) <= k / 2:
        maxProfit(prices)
        
    local = [0] * (k+1)
    globl = [0] * (k+1)
    
    for i in range(1, len(prices)):
        diff = prices[i] - prices[i - 1]
        j = k
        while j > 0:
            local[j] = max(globl[j - 1], local[j] + diff)
            globl[j] = max(globl[j], local[j])
            j -= 1
            
    return globl[k]

if __name__ == '__main__':
	prices = [2,5,7,1,4,3,1,3]
	k = 3
	maxProfit5(prices, k)

# 13,股票买卖(卖掉股票之后的一天不允许买入)：
def maxProfit6(prices):
    if len(prices) < 2:
        return 0
    n = len(prices)
    sell = [0] * n
    buy  = [0] * n
    sell[0] = 0;
    buy[0] = -prices[0]
    for i in range(1, n):
        sell[i] = max(sell[i - 1], buy[i - 1] + prices[i])
        buy[i] = max(buy[i - 1], (sell[i - 2] if i > 1 else 0) - prices[i])
            
    return sell[-1]

if __name__ == '__main__':
	prices = [1,2,3,0,2]
	maxProfit6(prices)

# 14,路径条数：
def uniquePaths(m, n):
    aux = [[1 for x in range(n)] for x in range(m)]
    for i in range(1, m):
        for j in range(1, n):
            aux[i][j] = aux[i][j-1]+aux[i-1][j]
    return aux[-1][-1]

def uniquePaths(m, n):
    aux = [1 for x in range(n)]
    for i in range(1, m):
        for j in range(1, n):
            aux[j] = aux[j]+aux[j-1]
    return aux[-1]

if __name__ == '__main__':
	uniquePaths(3, 4)

# 15,棋盘上移动路径：
def uniquePathsWithObstacles(obstacleGrid):
    M, N = len(obstacleGrid), len(obstacleGrid[0])
    dp = [1] + [0] * (N-1)
    for i in range(M):
        for j in range(N):
            if obstacleGrid[i][j] == 1:
                dp[j] = 0
            elif j > 0:
                dp[j] += dp[j-1]
    return dp[N-1]

if __name__ == '__main__':
	grid = [
	    [0,0,0,0,0,0,0],
	    [0,0,1,0,0,0,0],
	    [0,0,0,0,1,0,0]
	]
	uniquePathsWithObstacles(grid)

# 16,路径条数(3):
def movingBoard(board):
    result = board
    m = len(board)
    n = len(board[0])
    for i in range(1, m):
        for j in range (0, n):
            result[i][j] = max(0 if j == 0 else result[i-1][j-1], \
                               result[i-1][j], \
                               0 if j == n-1 else result[i-1][j+1] ) \
                            + board[i][j]
    return max(result[-1])

def movingBoard2(board):
    result = board[0]
    m = len(board)
    n = len(board[0])
    for i in range(1, m):
        for j in range (0, n):
            result[j] = max(0 if j == 0 else result[j-1], \
                            result[j], \
                            0 if j == n-1 else result[j+1] ) \
                        + board[j]
    return max(result)

if __name__ == '__main__':
	board = [
	    [3,-2, 6,-3, 4, 1, 2],
	    [0, 4, 1, 3,-1, 4, 3],
	    [2, 2,-1, 3, 2, 0, 2]
	]
	movingBoard(board)

# 17,最大方形：
def maximalSquare(matrix):
    if matrix == []:
        return 0
    m, n = len(matrix), len(matrix[0])
    dp = [[0] * n for x in range(m)]
    ans = 0
    for x in range(m):
        for y in range(n):
            dp[x][y] = int(matrix[x][y])
            if x and y and dp[x][y]:
                dp[x][y] = min(dp[x - 1][y - 1], dp[x][y - 1], dp[x - 1][y]) + 1
            ans = max(ans, dp[x][y])
    return ans * ans

def maximalSquare2(matrix):
    if matrix == []:
        return 0
    m, n = len(matrix), len(matrix[0])
    dp = matrix[0]
    ans = 0
    for x in range(0, m):
        for y in range(0, n):
            dp[y] = int(matrix[x][y])
            if matrix[x][y]:
                dp[y] = min(dp[y - 1], dp[y - 1], dp[y]) + 1
            ans = max(ans, dp[y])
    return ans * ans

if __name__ == '__main__':
	matrix = [
	    [1,0,1,0,0],
	    [1,0,1,1,1],
	    [1,1,1,1,1],
	    [1,0,0,1,0]
	]
	maximalSquare2(matrix)

# 18,0/1背包问题：
def knapSack(W, wt, val, n):
    K = [[0 for x in range(W+1)] for x in range(n+1)]
 
    # Build table K[][] in bottom up manner
    for i in range(n+1):
        for w in range(W+1):
            if i==0 or w==0:
                K[i][w] = 0
            elif wt[i-1] <= w:
                K[i][w] = max(val[i-1] + K[i-1][w-wt[i-1]],  K[i-1][w])
            else:
                K[i][w] = K[i-1][w]
 
    return K[n][W]

if __name__ == '__main__':
	val = [5,7,10,13,3,11]
	wt = [2,3,4,6,1,5]
	W = 14
	n = len(val)
	print(knapSack(W, wt, val, n))

# 19,最大公共子序列：
def LCS(X, Y, m, n):
     
    matrix = [[0 for k in range(n+1)] for l in range(m+1)]
     
    result = 0
 
    for i in range(m + 1):
        for j in range(n + 1):
            if (i == 0 or j == 0):
                matrix[i][j] = 0
            elif (X[i-1] == Y[j-1]):
                matrix[i][j] = matrix[i-1][j-1] + 1
                result = max(result, matrix[i][j])
            else:
                matrix[i][j] = max(matrix[i-1][j], matrix[i][j-1])
    return result

if __name__ == '__main__':
	X = 'AGGTAB'
	Y = 'GXTXAYB'
	 
	m = len(X)
	n = len(Y)
	LCS(X, Y, m, n)

# 20,最大递增子序列：
def lengthOfLIS1(nums):
    sortNums = sorted(nums)
    n = len(nums)
    return LCS(nums, sortNums, n, n)

def lengthOfLIS2(nums):
    if not nums:
        return 0
    dp = [1]*len(nums)
    for i in range (1, len(nums)):
        for j in range(i):
            if nums[i] >nums[j]:
                dp[i] = max(dp[i], dp[j]+1)
    return max(dp)

#using binary search
def lengthOfLIS3(nums):
    def search(temp, left, right, target):
        if left == right:
            return left
        mid = left+(right-left)//2
        return search(temp, mid+1, right, target) if temp[mid]<target else search(temp, left, mid, target)
    temp = []
    for num in nums:
        pos = search(temp, 0, len(temp), num)
        if pos >=len(temp):
            temp.append(num)
        else:
            temp[pos]=num
    return len(temp)

from bisect import bisect 
#using binary search
def lengthOfLIS4(nums):

    temp = []
    for num in nums:
        pos = bisect(temp, num) 
        if pos >=len(temp):
            temp.append(num)
        else:
            temp[pos]=num
    return len(temp)

if __name__ == '__main__':
	nums = [10, 9, 2, 5, 3, 7, 101, 18]
	lengthOfLIS4(nums)