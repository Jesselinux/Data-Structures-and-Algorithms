# leetcode-169：
# O(n):多数投票问题
class Solution(object):  # 只战胜了26.62%,还不如python内置函数效率高
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        cnt = 0
        majority = nums[0]
        for num in nums:
            if cnt == 0:
                majority = num
            if majority == num:  # 众数的cnt一定大于等于零，因为the element that appears more than ⌊ n/2 ⌋ times
                cnt += 1
            else:
                cnt -= 1
        return majority

# 使用python内置函数，战胜了 45.13% 
class Solution(object):
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums = sorted(nums)
        return nums[len(nums) // 2]
        # return sorted(nums)[int(len(nums)/2)]