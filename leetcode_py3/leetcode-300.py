# leetcode-300：动态规划，beats 53.08%
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        if len(nums) == 0:
            return 0
        LIS = [1] * len(nums)
        for i in range(1, len(nums)):
            LIS[i] = max([1+LIS[j] for j in range(i) if nums[j] < nums[i]] + [1])
        return max(LIS)