# leetcode-198:
# 动态规划一：时间复杂度为O(n),空间复杂度为O(1), 战胜了80.6%
class Solution:
    def rob(self, nums: List[int]) -> int:
        yes, no = 0, 0
        for i in nums:
            no, yes = max(no, yes), no+i
        return max(no, yes)

# 动态规划二：战胜了18.44%，与上面的差异仅在于语句：last, now = 0, 0
class Solution:
    def rob(self, nums: List[int]) -> int:
        last = now = 0  # 将这句改为：last, now = 0, 0后，战胜了80.6%
        for i in nums:
            last, now = now, max(last+i, now)
        return now