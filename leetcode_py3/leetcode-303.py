# leetcode-303:
class NumArray:  # 动态规划， beats 83.14%

    def __init__(self, nums: List[int]):
        self.nums = nums
        self.sums = [-1] * (len(self.nums) + 1)
        for i in range(1, len(self.nums) + 1):
            self.sums[i] = self.sums[i-1] + self.nums[i-1]

    def sumRange(self, i: int, j: int) -> int:
        return self.sums[j+1] - self.sums[i]

# Your NumArray object will be instantiated and called as such:
# obj = NumArray(nums)
# param_1 = obj.sumRange(i,j)

class NumArray:  # 记忆性搜索法，beats 7.05%

    def __init__(self, nums: List[int]):
        self.nums = nums
        self.sums = [-1] * (len(self.nums) + 1)
        
    def sumRange(self, i: int, j: int) -> int:
        if self.sums[i] == -1:
            self.sums[i] = self.sum_x(i)
        if self.sums[j+1] == -1:
            self.sums[j+1] = self.sum_x(j+1)
        return self.sums[j+1] - self.sums[i]
    
    def sum_x(self, x):
        sum = 0
        for i in range(x):
            sum += self.nums[i]
        return sum