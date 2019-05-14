# leetcode-462：双指针，快排
class Solution(object):  # O(NlogN),战胜了48.89%
    def minMoves2(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums = sorted(nums)  # python内置函数sort使用的是快排，时间复杂度是O(NlogN)
        res = 0
        left, right = 0, len(nums)-1
        while left < right:  # 双指针，这里的时间复杂度是O(N)
            res += nums[right] - nums[left]
            left += 1
            right -= 1
        return res