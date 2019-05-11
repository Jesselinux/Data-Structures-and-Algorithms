# leetcode-343:动态规划,beats 57.79%
class Solution:
    def integerBreak(self, n: int) -> int:
        sums = [0, 1]
        for i in range(2, n+1):
            tmp = 0
            for j in range(1, i):
                tmp = max(tmp, j*sums[i-j], j*(i-j))
            sums.append(tmp)
        return sums[n]