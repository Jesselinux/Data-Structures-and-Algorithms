# keetcode-1：哈希表
class Solution:   # 战胜了87.27%
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dict = {}
        for i in range(len(nums)):
            if nums[i] in dict:
                return [dict[nums[i]], i]
            dict[target - nums[i]] = i

# 和上面阶梯思想一致，细节不同而已
class Solution:   # 战胜了87.27%
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        my_dict = {}
        for i in range(len(nums)):
            if target - nums[i] in my_dict:
                return [my_dict[target - nums[i]], i]
            my_dict[nums[i]] = i