# leetcode-132：
# 动态规划：战胜了81.88%
class Solution:
    def minCut(self, s: str) -> int:
        n = len(s)
        dp = [i - 1 for i in range(0, n + 1)] # dp[i]: the minimum cuts needed for s[:i]
        for i in range(0, n + 1):
            j = 0
            while i - j >= 0 and i + j < n and s[i + j] == s[i - j]:
                dp[i + j + 1] = min(dp[i + j + 1], dp[i - j] + 1)
                j += 1
            j = 1
            while i - j + 1 >= 0 and i + j < n and s[i - j + 1] == s[i + j]:
                dp[i + j + 1] = min(dp[i + j + 1], dp[i - j + 1] + 1)
                j += 1
        return dp[n] 

# 战胜了43.38%        
class Solution(object):
    def minCut(self, s):
        size = len(s)
        ans = [i for i in range(size)]
        p = [[False for i in range(size)] for j in range(size)]
        j = 1
        while j < size:
            i,ans[j] = j - 1,min(ans[j],ans[j - 1] + 1) 
            p[j][j] = True
            while i >= 0:
                if s[i] == s[j] and ((j - i) < 2 or  p[i+1][j-1]):
                    p[i][j] = True
                    if i == 0:
                        ans[j] = 0
                    else:
                        ans[j] = min(ans[j],ans[i - 1] + 1)
                i -= 1
            j += 1
        return ans[size - 1]