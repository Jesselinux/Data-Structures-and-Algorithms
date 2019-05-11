# 暴力解法：
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        return sorted(nums)[-k]
        

# 快排：时间复杂度-O(N), 空间复杂度-O(1).
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        return self.quick_sort(nums, k)
    
    def quick_sort(self, nums, k):
        k = len(nums) -k
        left = 0
        right = len(nums) - 1
        while left < right:
            j = self.partition(nums, left, right)
            if j == k:
                break
            elif j < k:
                left = j + 1
            else:
                right = j -1
        return nums[k]
    
    def partition(self, nums, left, right):
        while True:
            while nums[left] < nums[right]:
                right -= 1
            else:
                nums[left], nums[right] = nums[right], nums[left]
                if left >= right:
                    break
                left += 1
                
            while nums[left] < nums[right]:
                left += 1
            else:
                nums[left], nums[right] = nums[right], nums[left]
                if left >= right:
                    break
                right -= 1
        return left


# 堆排：时间复杂度-O(NlogK), 空间复杂度-O(K).
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        return self.heap_sort(nums, k)
    
    def heap_sort(self, nums, k):
        for i in range(len(nums)//2 - 1, -1, -1):
            self.heap_adjust(nums, i, len(nums))
            
        cnt = 0
        for i in range(len(nums) - 1, 0, -1):
            self.heap_swap(nums, 0, i)
            cnt += 1
            if cnt == k:
                return nums[i]
            self.heap_adjust(nums, 0, i)
        return nums[0]
        
    def heap_adjust(self, nums, start, length):
        tmp = nums[start]
        k = start * 2 + 1
        while k < length:
            left = start * 2 + 1
            right = left + 1
            
            if right < length and nums[right] > nums[left]:
                k = right
                
            if nums[k] > tmp:
                nums[start] = nums[k]
                start = k
            else:
                break
            k = k*2 + 1
        nums[start] = tmp
    
    def heap_swap(self, nums, i, j):
        nums[i], nums[j] = nums[j], nums[i]
        return nums

# 递归解法：
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
    	if len(nums) == 0:
    		return
    	nums = self.quicksort(nums)
    	return nums[-k]

    def quicksort(self, nums):
    	if len(nums) == 0:
    		return []
    	pivot = nums[0]
    	left = self.quicksort([x for x in nums[1:] if x < pivot])
    	right = self.quicksort([x for x in nums[1:] if x >= pivot])
    	return left + [pivot] + right