# 1,反转列表：
def reverse(nums):
    n = len(nums)
    for i in range(len(nums) // 2):
        nums[i], nums[n-1-i] = nums[n-1-i], nums[i]
    print(nums)

def reverse2(nums):
    i, j = 0, len(nums) - 1
    while (i < j):
        nums[i], nums[j] = nums[j], nums[i]
        i += 1
        j -= 1
    print(nums)

if __name__ == '__main__':
	nums = [1,2,3]
	reverse(nums)

# 2,求两个数的和：
def twoSum(nums, target):
    dic = {}
    for i, num in enumerate(nums):
        if num in dic:
            return [dic[num], i]
        else:
            dic[target - num] = i
            
def twoSum2(num, target):
    index = []
    numtosort = num[:]; numtosort.sort()
    i = 0; j = len(numtosort) - 1
    while i < j:
        if numtosort[i] + numtosort[j] == target:
            for k in range(0,len(num)):
                if num[k] == numtosort[i]:
                    index.append(k)
                    break
            for k in range(len(num)-1,-1,-1):
                if num[k] == numtosort[j]:
                    index.append(k)
                    break
            index.sort()
            break
        elif numtosort[i] + numtosort[j] < target:
            i = i + 1
        elif numtosort[i] + numtosort[j] > target:
            j = j - 1

    return (index[0]+1,index[1]+1)

# 3，三数之和为0的组合：
def threeSum(nums):
    res = []
    nums.sort()
    for i in range(len(nums)-2):
        if i > 0 and nums[i] == nums[i-1]:
            continue
        l, r = i+1, len(nums)-1
        while l < r:
            s = nums[i] + nums[l] + nums[r]
            if s < 0:
                l +=1 
            elif s > 0:
                r -= 1
            else:
                res.append((nums[i], nums[l], nums[r]))
                while l < r and nums[l] == nums[l+1]:
                    l += 1
                while l < r and nums[r] == nums[r-1]:
                    r -= 1
                l += 1; r -= 1
    return res

if __name__ == '__main__':
	nums = [-1, 0, 1, 2, -1, -4, 2, -1, 2]
	print(threeSum(nums))

# 4,四数之和：
def fourSum(num, target):
    num.sort(); res = []
    for i in range(len(num)):
        if i > 0 and num[i] == num[i-1]: 
            continue 
        for j in range(i + 1 ,len (num)):
            if j > i + 1 and num[j] == num[j-1]: 
                continue 
            l = j + 1
            r = len(num) - 1
            while l < r:
                sum = num[i] + num[j] + num[l] + num[r]
                if sum > target:
                    r -= 1
                elif sum < target:
                    l += 1
                elif l > j + 1 and num[l] == num[l-1]:
                    l += 1
                elif r < len(num) - 1 and num[r] == num[r+1]:
                    r -= 1
                else :
                    res.append([num[i],num[j],num[l],num[r]])
                    l += 1
                    r -= 1
    return res

def fourSum2(num, target):
    numLen, res, dict = len(num), set(), {}
    if numLen < 4: 
        return []
    num.sort()
    for p in range(numLen):
        for q in range(p+1 , numLen): 
            if num[p] + num[q] not in dict:
                dict[num[p] + num[q]] = [(p,q)]
            else :
                dict[num[p] + num[q]].append((p,q))
    for i in range(numLen):
        for j in range(i+1, numLen-2 ):
            T = target-num[i]- num[j]
            if T in dict:
                for k in dict[T]:
                    if k[0] > j: res.add((num[i],num[j],num [k[0]],num[k[1 ]]))
    return [list(i) for i in res]

if __name__ == '__main__':
	nums = [-1, 0, 1, 2, -1, -4, 2, -1, 2]
	print(fourSum2(nums, 0))
# 5,合并两个有序数组：
def merge(nums1, m, nums2, n):
    while m > 0 and n > 0:
        if nums1[m-1] >= nums2[n-1]:
            nums1[m+n-1] = nums1[m-1]
            m = m - 1
        else:
            nums1[m+n-1] = nums2[n-1]
            n = n - 1
    if n > 0:
        nums1[:n] = nums2[:n]

if __name__ == '__main__':
	nums1 = [1,2,3,0,0,0]
	m = 3
	nums2 = [2,5,6]
	n = 3
	merge(nums1, m, nums2, n)

# 6,有序数组最小元素差：
import sys
def printClosest(ar1, ar2):
    m = len(ar1)
    n = len(ar2)

    diff = sys.maxsize
    
    p1 = 0
    p2 = 0
    
    while (p1 < m and p2 < n):
        if abs(ar1[p1] - ar2[p2]) < diff:
            diff = abs(ar1[p1] - ar2[p2])
        
        if (ar1[p1] > ar2[p2]):
            p2 += 1
        else:
            p1 += 1

    return diff

# 7,连续子串的最大值：
from itertools import accumulate

def max_subarray(numbers, ceiling):
    
    cum_sum = [0]
    cum_sum = cum_sum + numbers
    cum_sum = list(accumulate(cum_sum))

    l = 0
    r = 1 # two pointers start at tip of the array.
    maximum = 0
    while l < len(cum_sum):
        while r < len(cum_sum) and cum_sum[r] - cum_sum[l] <= ceiling:
            r += 1
        if cum_sum[r - 1] - cum_sum[l] > maximum: # since cum_sum[0] = 0, thus r always > 0.
            maximum = cum_sum[r - 1] - cum_sum[l]
            pos = (l, r - 2)
        l += 1
    return pos

if __name__ == '__main__':
	a = [4, 7, 12, 1, 2, 3, 6]
	m = 15
	result = max_subarray(a, m)
	print(result)

# 8,查找主元素：主元素是出现次数超过数组长度一般的元素
# Boyer-Moore Voting Algorithm
def majority(alist):
    result = count = 0
    for i in alist:
        if count == 0:
            result = i
            count = 1
        elif result == i:
            count += 1
        else:
            count -= 1
    return result

# 9,查找主元素(1/3):
def majority2(alist):
    n1 = n2 = None
    c1 = c2 = 0
    for num in alist:
        if n1 == num:
            c1 += 1
        elif n2 == num:
            c2 += 1
        elif c1 == 0:
            n1, c1 = num, 1
        elif c2 == 0:
            n2, c2 = num, 1
        else:
            c1, c2 = c1 - 1, c2 - 1
    size = len(alist)
    return [n for n in (n1, n2) 
               if n is not None and alist.count(n) > size / 3]  

# 10,颜色排序：
def sortColors(nums):

    count = [0] * 3
    for num in nums:
        count[num] += 1
    i = 0
    for j in range(3):
        for _ in range(count[j]):
            nums[i] = j
            i += 1

def sortColors2(nums):
    i, l, r = 0, 0, len(nums) - 1
    while i <= r:
        if nums[i] == 0:
            nums[i], nums[l] = nums[l], nums[i]
            i, l = i + 1, l + 1
        elif nums[i] == 2:
            nums[i], nums[r] = nums[r], nums[i]
            r -= 1
        else:
            i += 1

if __name__ == '__main__':
	nums = [2,0,2,1,1,0]
	sortColors(nums)
	print(nums)

# 11,k个最近元素：给定一个有序数组，以及两个整数变量k 和 x，请找出数组中离x最 近的k个元素，并且返回的k个元素需按升序排列。如果两个数字距离x相等，要 求取较小的那个
def findClosestElements(alist, k, x):
    left = right = bisect.bisect_left(alist, x)
    while right - left < k:
        if left == 0: return alist[:k]
        if right == len(alist): return alist[-k:]
        if x - alist[left - 1] <= alist[right] - x: left -= 1
        else: right += 1
    return alist[left:right]

def findClosestElements2(self, arr, k, x):
    diffTuples = sorted((abs(x - num), num) for num in arr)
    return sorted(map(lambda x: x[1], diffTuples[:k])) #prefer the smaller number for same diff.

# 12,数组的'山峰'高度：
def longestMountain(A):
    N = len(A)
    ans = base = 0

    while base < N:
        end = base
        if end + 1 < N and A[end] < A[end + 1]: #if base is a left-boundary
            #set end to the peak of this potential mountain
            while end+1 < N and A[end] < A[end+1]:
                end += 1

            if end + 1 < N and A[end] > A[end + 1]: #if end is really a peak..
                #set 'end' to right-boundary of mountain
                while end+1 < N and A[end] > A[end+1]:
                    end += 1
                #record candidate answer
                ans = max(ans, end - base + 1)

        base = max(end, base + 1)

    return ans

if __name__ == '__main__':
	A = [2,1,4,7,3,2,5]
	print(longestMountain(A))

# 13,最大的定积分：
def maxArea1(height):
    res = 0
    for i in range(len(height)):
        for j in range(i+1, len(height)):
            res = max(res, min(height[i], height[j]) * (j - i))
    return res 

def maxArea2(height):
    left = 0; right = len(height)-1
    res = 0
    while left < right:
        water = min(height[left], height[right]) * (right-left)
        res = max(res, water)
        if height[left] < height[right]: 
            left += 1
        else:
            right -= 1
    return res 

if __name__ == '__main__':
	height = [3, 1, 2, 4, 5]
	print(maxArea2(height))

# 14,蓄水量：
# Brute Force
# Time complexity: O(n^2)
# Space complexity: O(1)O(1)
def trap1(height):
    if not height or len(height) < 3:
        return 0    
    ans, size = 0, len(height)
    for i in range (1, size-1):
        max_left = max_right = 0
        for j in range(i-1, -1, -1):
            max_left = max(max_left, height[j])
        for j in range(i+1, size):
            max_right = max(max_right, height[j])
        ans +=  max(0, min(max_left, max_right) - height[i])
    
    return ans

# Dynamic Programming
# Time complexity: O(n)
# Space complexity: O(n)
def trap2(height):
    if not height or len(height) < 3:
        return 0
    ans, size = 0, len(height)
    left_max, right_max, anss = [0] * size, [0] * size, [0] * size
    left_max[0] = height[0]
    for i in range (1, size):
        left_max[i] = max(height[i], left_max[i-1])
    right_max[-1] = height[-1]
    for i in range (size-2, -1, -1):
        right_max[i] = max(height[i], right_max[i+1])
    for i in range (1, size-1):
        anss[i] =  min(left_max[i], right_max[i]) - height[i]
        ans += min(left_max[i], right_max[i]) - height[i]

    return ans

# Two Pointers
# Time complexity: O(n)
# Space complexity: O(1)
def trap3(height):
    if not height or len(height) < 3:
        return 0
    left, right = 0, len(height) - 1
    left_max, right_max = 0, 0
    ans = 0
    while (left < right):
        if (height[left] < height[right]):
            if height[left] >= left_max:
                left_max = height[left]  
            else:
                ans += (left_max - height[left])
            left += 1
        
        else:
            if height[right] >= right_max:
                right_max = height[right] 
            else:
                ans += (right_max - height[right])
            right -= 1
    return ans;

def trap4(height): 
    ans, current = 0, 0
    st = []
    while (current < len(height)):
        while (len(st) != 0 and height[current] > height[st[-1]]):
            top = st[-1]
            print("current: ", current, "   top: ", top)
            print("before: ", st)
            st.pop()
            if len(st) == 0:
                break
            distance = current - st[-1] - 1
            bounded_height = min(height[current], height[st[-1]]) - height[top]
            ans += distance * bounded_height
            print("after: ", st)
        st.append(current)
        current += 1
    return ans

if __name__ == '__main__':
	height = [0,1,0,2,1,0,1,3,2,1,2,1]
	print(trap4(height))