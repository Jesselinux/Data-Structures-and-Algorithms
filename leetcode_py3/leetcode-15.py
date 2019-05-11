class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        if len(nums) > 2:
            nums.sort()
            L, res = len(nums), []
            for i in range(L-2):
                if i > 0 and nums[i] == nums[i-1]:
                    continue
                target = -1 * nums[i]
                j, k = i+1, L-1
                while j<k:
                    if nums[j] + nums[k] == target:
                        res.append([nums[i], nums[j], nums[k]])
                        j = j + 1
                        while j<k and nums[j] == nums[j-1]:
                            j = j+1
                    elif nums[j] + nums[k] < target:
                        j = j+1
                    else:
                        k = k-1
        else:
            print('the length of the list is small than "3"')
               # leetcode不会通过这个if...else,测试时可以去掉这个外层循环。
    
    return res