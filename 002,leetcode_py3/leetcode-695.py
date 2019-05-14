# leetcode-695:广度优先搜索(DFS),beats 44.88%
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        if not grid:
            return 0
        self.r, self.c, max_area = len(grid), len(grid[0]), 0
        for i in range(self.r):
            for j in range(self.c):
                max_area = max(max_area, self.dfs(grid, i, j))
        return max_area
    
    def dfs(self, grid: List[List[int]], n: int, m: int) -> int:
        if n < 0 or n >= self.r or m<0 or m>=self.c or grid[n][m] == 0:
            return 0
        area, grid[n][m] = 1, 0
        area += self.dfs(grid, n-1, m)
        area += self.dfs(grid, n+1, m)
        area += self.dfs(grid, n, m+1)
        area += self.dfs(grid, n, m-1)
        return area