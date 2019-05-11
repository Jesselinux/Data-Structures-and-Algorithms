# leetcode-152：动态优化，与leetcode-121极其相似
class Solution:  # O(n)
    def maxProduct(self, nums: List[int]) -> int:
        max_p = min_p = nums[0]
        res = nums[0]
        for i in range(1, len(nums)):
            lastmax = max_p
            max_p = max(min_p*nums[i], lastmax*nums[i], nums[i])
            min_p = min(min_p*nums[i], lastmax*nums[i], nums[i])
            res = max(res, max_p)
        return res

# 暴力解法：O(n^2)
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        if not nums:
            return 0
        if len(nums) == 1:
            return nums[0]
        
        l = len(nums)
        res = []
        element = 1
        for i in range(l):
            for j in range(i+1, l):
                for r in range(i, j+1):
                    element = element * nums[r]
                res.append(element)
                element = 1
        if not res:
            return 0
        else:
            return max(max(res), max(nums))