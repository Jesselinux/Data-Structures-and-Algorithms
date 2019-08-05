class Solution:
	def maxSlidingWindow(self, nums, k):
		ans = []
		if k > len(nums):
			ans.append(max(nums))
		else:
			n = len(nums) - k + 1
			for i in range(n):
				slidingWindow = nums[i:i+k]
				ans.append(max(slidingWindow))
		return ans

