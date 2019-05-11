# leetcode-64: O(m*n),逐步更新grid元素为最短路径值，是一种时间复杂度对高的解法
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        m = len(grid)  # 行数
        n = len(grid[0])  # 列数
        # 更新第一列元素这为路径值
        for i in range(1,m):
            grid[i][0] += grid[i-1][0]
        # 更新第一行值为路径值
        for j in range(1, n):
            grid[0][j] += grid[0][j-1]
        # 从第二行第二列开始更新[i][j]数值为路径值
        for i in range(1, m):
            for j in range(1, n):
                grid[i][j] += min(grid[i-1][j], grid[i][j-1])
        return grid[m-1][n-1]

# leetcode-64：最小路径和，beats 68.36%
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        r = len(grid)  # 行数
        c = len(grid[0])  # 列数
        dp = [0] * c
        for i in range(r):
            for j in  range(c):
                if j == 0:
                    dp[j] = dp[j]
                elif i == 0:
                    dp[j] = dp[j-1]
                else:
                    dp[j] = min(dp[j-1], dp[j])
                dp[j] += grid[i][j]
        return dp[c-1]