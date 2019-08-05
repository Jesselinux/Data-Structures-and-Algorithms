# 暴力解法：
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        nums.sort()

# 三向切分快速排序思想：
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        head, now, tail = 0, 0, len(nums) - 1
        while now <= tail:
            if nums[now] == 0:
                nums[now], nums[head] = nums[head], nums[now]
                now += 1
                head += 1
            elif nums[now] == 2:
                nums[now], nums[tail] = nums[tail], nums[now]
                tail -= 1
            else:
                now += 1

class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        zero, two = -1, len(nums)
        i = 0
        while i < two:
        	if nums[i] == 1:
        		i += 1
        	elif nums[i] == 2:
        		two -= 1
        		nums[two], nums[i] = nums[i], nums[two]
        	else:
        		zero += 1
        		nums[zero], nums[i] = nums[i], nums[zero]
        		i += 1