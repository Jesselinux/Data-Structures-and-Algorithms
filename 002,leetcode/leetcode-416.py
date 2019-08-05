# leetcode-416：动态规划， beats 56.04%
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        c = sum(nums)
        if c % 2 != 0:
            return False
        c = c//2  # 取整
        w = [False] * (c+1)
        w[0] = True
        for num in nums:
            for i in range(c, num-1, -1):
                w[i] = w[i] or w[i-num]
        return w[c]