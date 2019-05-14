# leetcode-46:
# 递归(回溯法）：战胜了72.70%
class Solution:
    def __init__(self):
        self.res = []  # 定义类中的全局变量res，
    
    def permute(self, nums: List[int]) -> List[List[int]]:
        self.permutation_recursion([], nums)
        return self.res
       
    def permutation_recursion(self, result, nums):  # 真正的功能实现函数，在此函数中定义res不容易在外部调用
        if (len(nums)==0):
            self.res.append(result)
        for i in range(len(nums)):
            self.permutation_recursion(result+[nums[i]], nums[0:i]+nums[i+1:])  # nums[0:i]+nums[i+1:] = nums - nums[i]

# 非递归方式：战胜了92.28%
class Solution:
	def permute(self, nums: List[int]) -> List[List[int]]:
	    perms = [[]]   
	    for n in nums:
	        new_perms = []
	        for perm in perms:
	            for i in range(len(perm)+1):   
	                new_perms.append(perm[:i] + [n] + perm[i:])   # insert n
	        perms = new_perms
	    return perms