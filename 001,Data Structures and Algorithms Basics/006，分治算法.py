# 1，查找第K大元素：
import time
# O(nlgn)：
def findKthLargest1(nums, k):
    start = time.time()
    rst = sorted(nums, reverse=True)
    t = time.time() - start
    return rst[k-1], len(rst), t

# O(nk) time, bubble sort idea, TLE
def findKthLargest2(nums, k):
    start = time.time()
    for i in range(k):
        for j in range(len(nums)-i-1):
            if nums[j] > nums[j+1]:
                # exchange elements, time consuming
                nums[j], nums[j+1] = nums[j+1], nums[j]
    t = time.time() - start
    return nums[len(nums)-k], len(nums), t

# O(n) time, quick selection
def findKthLargest(nums, k):
    # convert the kth largest to smallest
    start = time.time()
    rst = findKthSmallest(nums, len(nums)+1-k)
    t = time.time() - start
    return rst, len(nums), t
    
def findKthSmallest(nums, k):
    if nums:
        pos = partition(nums, 0, len(nums)-1)
        if k > pos+1:
            return findKthSmallest(nums[pos+1:], k-pos-1)
        elif k < pos+1:
            return findKthSmallest(nums[:pos], k)
        else:
            return nums[pos]
 
# choose the right-most element as pivot   
def partition(nums, l, r):
    low = l
    while l < r:
        if nums[l] < nums[r]:
            nums[l], nums[low] = nums[low], nums[l]
            low += 1
        l += 1
    nums[low], nums[r] = nums[r], nums[low]
    return low

if __name__ == '__main__':
	import random
	from random import randint
	import sys
	import matplotlib.pyplot as plt
	def generate_random_array(n):
	    return [randint(1, 3 * n) for e in range(n)]

	# 单元检测   
	l = generate_random_array(1000000)
	r = findKthLargest(l, len(l)//2)
	print(r)

	# 画图：时间复杂度
	# random_lists = [generate_random_array(1000 * n) for n in range(1, 21)]
	# rst = [findKthLargest(l, len(random_lists)//2) for l in random_lists]
	# x = list(zip(*rst))[1]
	# y = list(zip(*rst))[2]
	# plt.plot(x, y)
	# plt.show()

# 2，快速指数：
def fast_power1_flaw(x, n):
    if n <= 0:
        return 1
    elif n == 1:
        return x
    elif n % 2:
        return fast_power1_flaw(x * x, n // 2) * x
    else:
        return fast_power1_flaw(x * x, n // 2) 

def fast_power2(x, n):
    if n == 0:
        return 1.0
    elif n < 0:
        return 1 / fast_power2(x, -n)
    elif n % 2:
        return fast_power2(x * x, n // 2) * x
    else:
        return fast_power2(x * x, n // 2)

if __name__ == '__main__':
	print(fast_power1_flaw(5, 3))
	print(fast_power2(5, 3))

# 3, 搜索峰值:数组没有重复值，但可能存在多个峰值，返回任意一个峰值的index
def search_peak(alist):
    return peak_helper(alist, 0, len(alist) - 1)

def peak_helper(alist, start, end):
    if start == end:
        return start
    
    if (start + 1 == end):
        if alist[start] > alist[end]:
            return start
        return end
    
    mid = (start + end) // 2
    if alist[mid] > alist[mid - 1] and alist[mid] > alist[mid + 1]:
        return mid
    if alist[mid - 1] > alist[mid] and alist[mid] > alist[mid + 1]:
        return peak_helper(alist, start, mid - 1)
    return peak_helper(alist, mid + 1, end)

# 4, 给定两个排好序的数组, 只有一个不同的地方：在第一个数组某个位置上多一个元素,查找该元素index
# Input : {3, 5, 7, 9, 11, 13} {3, 5, 7, 11, 13}，Output : 3
def find_extra(arr1, arr2):
    for i in range(len(arr2)):
        if (arr1[i] != arr2[i]):
            return i 
    return len(arr1)-1

def find_extra_fast(arr1, arr2):
    index = len(arr2)
    # left and right are end points denoting
    # the current range.
    left, right = 0, len(arr2) - 1
    while (left <= right):
        mid = (left + right) // 2;
 
        # If middle element is same of both
        # arrays, it means that extra element
        # is after mid so we update left to mid+1
        if (arr2[mid] == arr1[mid]):
            left = mid + 1
 
        # If middle element is different of the
        # arrays, it means that the index we are
        # searching for is either mid, or before
        # mid. Hence we update right to mid-1.
        else:
            index = mid
            right = mid - 1;
 
    # when right is greater than left our
    # search is complete.
    return index

if __name__ == '__main__':
	ar1 = [3, 5, 7, 9, 11, 13]
	ar2 = [3, 5, 7, 11, 13]
	print(find_extra(ar1, ar2))
	print(find_extra_fast(ar1, ar2))

# 5, 加和值最大的子序列:
# O(n^2)
def subarray1(alist):
    result = -sys.maxsize
    for i in range(0, len(alist)):
        sum = 0
        for j in range (i, len(alist)):
            sum += alist[j]
            if sum > result:
                result = sum
    return result

# O(n lgn)
def subarray2(alist):
    return subarray2_helper(alist, 0, len(alist)-1)

def subarray2_helper(alist, left, right):
    if (left == right):
        return alist[left]
    mid = (left + right) // 2
    return max(subarray2_helper(alist, left, mid), 
               subarray2_helper(alist, mid+1, right), 
               maxcrossing(alist, left, mid, right))

def maxcrossing(alist, left, mid, right):
    sum = 0
    left_sum = -sys.maxsize
    for i in range (mid, left-1, -1):
        sum += alist[i]
        if (sum > left_sum):
            left_sum = sum
            
    sum = 0
    right_sum = -sys.maxsize
    for i in range (mid+1, right+1):
        sum += alist[i]
        if (sum > right_sum):
            right_sum = sum        

    return left_sum + right_sum

# O(n)
def subarray3(alist):
    result = -sys.maxsize
    local = 0
    for i in alist:
        local = max(local + i, i)
        result = max(result, local)
    return result

if __name__ == '__main__':
	alist = [-2,-3,4,-1,-2,1,5,-3]
	print(subarray1(alist))
	print(subarray2(alist))
	print(subarray3(alist))

# 6, 查找逆序对: 如果有两个元素a[i], a[j],如果a[i] > a[j] 且 i < j，那么a[i], a[j]构成一个逆序对
# 法1： O(n^2)
def countInv(arr):
    n = len(arr)
    inv_count = 0
    for i in range(n):
        for j in range(i+1, n):
            if (arr[i] > arr[j]):
                inv_count += 1
 
    return inv_count

# 法2：
def merge(left,right):
    result = list()
    i,j = 0,0
    inv_count = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        elif right[j] < left[i]:
            result.append(right[j])
            j += 1
            inv_count += (len(left)-i)
    result += left[i:]
    result += right[j:]
    return result,inv_count

# O(nlgn)
def countInvFast(array):
    if len(array) < 2:
        return array, 0
    middle = len(array) // 2
    left,inv_left = countInvFast(array[:middle])
    right,inv_right = countInvFast(array[middle:])
    merged, count = merge(left,right)
    count += (inv_left + inv_right)
    return merged, count

if __name__ == '__main__': 
	arr = [1, 20, 6, 4, 5]
	print("Number of inversions are", countInv(arr))
	print("Number of inversions are", countInvFast(arr))

# 7，奇-偶数换序 ：input:{ a1, a2, a3, a4, ….., an, b1, b2, b3, b4, …., bn },output:{a1, b1, a2, b2, a3, b3, ……, an, bn }
# O(n^2):
def shuffleArray1(a, n): 
    # Rotate the element to the left
    i, q, k = 0, 1, n
    while(i < n):             
        j = k 
        while(j > i + q):
            # print(i, j, q, k)
            a[j - 1], a[j] = a[j], a[j - 1]
            j -= 1
        # for ii in range(0, 2 * n): 
        #     print(a[ii], end = " ")
        # print()
        i += 1
        k += 1
        q += 1
    return a

# O(n log n)：
def shuffleArray2(a, left, right): 
    # If only 2 element, return
    if (right - left == 1):
        return 
    # Finding mid to divide the array
    mid = (left + right) // 2
 
    # Using temp for swapping first
    # half of second array
    temp = mid + 1
 
    # Mid is use for swapping second
    # half for first array
    mmid = (left + mid) // 2
 
    # Swapping the element
    for i in range(mmid + 1, mid + 1):
        (a[i], a[temp]) = (a[temp], a[i])
        temp += 1
 
    # Recursively doing for 
    # first half and second half
    shuffleArray2(a, left, mid)
    shuffleArray2(a, mid + 1, right)
    return a

def shuffleArray3(a):
    n = len(a) // 2
    start = n + 1
    j = n + 1
    done = 0
    
    while (done < 2 * n - 2):
        #print(done, start, j)
        if (start == j):
            start = start - 1
            j = j - 1
        done += 1
        
        i = j - n if j > n else j
        j = 2 * i if j > n else 2 * i - 1
        
        a[start], a[j] = a[j], a[start]
    return a

if __name__ == '__main__':
	a = [1, 3, 5, 7, 2, 4, 6, 8] 
	print(shuffleArray1(a, len(a) // 2))
	print(shuffleArray2(a, 0, len(a) - 1))

	b = [-1, 1, 3, 5, 7, 2, 4, 6, 8 ]
	shuffleArray3(b)
	for i in range(1, len(b)):
	    print(b[i], end = " ")

# 8, 元素右边最小的元素：input: [9, 5, 3, 1, 26], output: [1, 1, 1, 26, 0]
# O(n^2):
def countSmaller1(nums):   # 法1
    n = len(nums)
    count = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if nums[i] > nums[j]:
                count[i] += 1
                
    return count

# O(n^2):
def countSmaller2(nums):      # 法2
    snums = []
    ans = [0] * len(nums)

    for i in range(len(nums) - 1, -1, -1):
        index = findIndex(snums, nums[i])
        ans[i] = index
        snums.insert(index, nums[i]) 
    return ans

def findIndex(snums, target):
    start = 0
    end = len(snums) - 1

    if len(snums) == 0: 
        return 0

    while start <= end:
        mid = start + (end - start) // 2
        if snums[mid] < target:
            start=mid + 1
        else:
            end = mid - 1
    return start

# O(nlgn):
def countSmaller3(nums):
    def sort(enum):
        half = len(enum) // 2
        if half:
            left, right = sort(enum[:half]), sort(enum[half:])
            m, n = len(left), len(right)
            i = j = 0
            while i < m or j < n:
                if j == n or i < m and left[i][1] <= right[j][1]:
                    smaller[left[i][0]] += j
                    enum[i+j] = left[i]
                    i += 1
                else:
                    enum[i+j] = right[j]
                    j += 1
        #     print("left: ", left)
        #     print("right: ", right)
        #     print("smaller: ", smaller)
        # print("enum: ", enum)
        return enum
    smaller = [0] * len(nums)
    sort(list(enumerate(nums)))
    return smaller
if __name__ == '__main__':
	nums1 = [5, 2, 6, 1]
	#nums2 = [9, 5, 3, 1, 26]
	print(countSmaller1(nums1))
	print(countSmaller2(nums1))
	print(countSmaller3(nums1))
	print(list(enumerate(nums1)))  # 遍历列表元素

# 9, 查找两个元素合并后的中位数：要求时间复杂度为O(log (m+n))：
def findMedianSortedArrays1(A, B):
    l = len(A) + len(B)
    if l % 2 == 1:
        return kth1(A, B, l // 2)
    else:
        return (kth1(A, B, l // 2) + kth1(A, B, l // 2 - 1)) / 2.   
    
def kth1(a, b, k):
    if not a:
        return b[k]
    if not b:
        return a[k]
    ia, ib = len(a) // 2 , len(b) // 2
    ma, mb = a[ia], b[ib]
    
    # when k is bigger than the sum of a and b's median indices 
    if ia + ib < k:
        # if a's median is bigger than b's, b's first half doesn't include k
        if ma > mb:
            return kth1(a, b[ib + 1:], k - ib - 1)
        else:
            return kth1(a[ia + 1:], b, k - ia - 1)
    # when k is smaller than the sum of a and b's indices
    else:
        # if a's median is bigger than b's, a's second half doesn't include k
        if ma > mb:
            return kth1(a[:ia], b, k)
        else:
            return kth1(a, b[:ib], k)

def find2(nums1, s1, e1, nums2, s2, e2, k):
    if e1 < s1:
        return nums2[k + s2]
    if e2 < s2:
        return nums1[k + s1]
    
    if k < 1:
        return min(nums1[k + s1], nums2[k + s2])
    
    ia, ib = (s1 + e1) // 2 , (s2 + e2) // 2
    ma, mb = nums1[ia], nums2[ib]
    if (ia - s1) + (ib - s2) < k:
        if ma > mb:
            return find2(nums1, s1, e1, nums2, ib + 1, e2, k - (ib - s2) - 1)
        else:
            return find2(nums1, ia + 1, e1, nums2, s2, e2, k - (ia - s1) - 1)
    else:
        if ma > mb:
            return find2(nums1, s1, ia - 1, nums2, s2, e2, k)
        else:
            return find2(nums1, s1, e1, nums2, s2, ib - 1, k)

def findMedianSortedArrays2(nums1, nums2):
    l = len(nums1) + len(nums2)
    if l % 2 == 1:
        return find2(nums1, 0, len(nums1) - 1, nums2, 0, len(nums2) - 1, l // 2)
    else:
        return (find2(nums1, 0, len(nums1) - 1, nums2, 0, len(nums2) - 1, l // 2) 
                + find2(nums1, 0, len(nums1) - 1, nums2, 0, len(nums2) - 1, l // 2 - 1)) / 2.0

if __name__ == '__main__':
	A = [1, 12, 15, 26, 38]
	B = [2, 13, 17]

	print(findMedianSortedArrays1(A, B))
	print(findMedianSortedArrays2(A, B))

# 10，快速整数乘法：
# 法一：
def karatsuba(x,y):
    """Function to multiply 2 numbers in a more efficient manner than the grade school algorithm"""
    if len(str(x)) == 1 or len(str(y)) == 1:
        return x*y
    else:
        n = max(len(str(x)),len(str(y)))
        nby2 = n // 2

        a = x // 10**(nby2)
        b = x % 10**(nby2)
        c = y // 10**(nby2)
        d = y % 10**(nby2)

        ac = karatsuba(a,c)
        bd = karatsuba(b,d)
        ad_plus_bc = karatsuba(a+b,c+d) - ac - bd

            # this little trick, writing n as 2*nby2 takes care of both even and odd n
        prod = ac * 10**(2*nby2) + (ad_plus_bc * 10**nby2) + bd

        return prod
 
 # 法二：
 # third_grade_algorithm.py
import functools
def prod(x, y):
    # x, y are strings --> returns a string of x*y
    return str(eval("%s * %s" % (x, y)))

def plus(x, y):
    # x, y are strings --> returns a string of x+y
    return str(eval("%s + %s" % (x, y)))

def one_to_n_product(d, x):
    """d is a single digit, x is n-digit --> returns a string of d*x
    """
    #print(d, x)
    result = ""
    carry = "0"
    for i, digit in enumerate(reversed(x)):
        #print("d: ", d, "  digit: ", digit)
        r = plus(prod(d, digit), carry)
        #print("r: ", r)
        if (len(r) == 1):
            carry = '0'
        else:
            carry = r[:-1]
        digit = r[-1]
        #print("   c: ", carry, "  d: ", digit)
        result = digit + result
    
    
    return carry + result

def sum_middle_products(middle_products):
    # middle_products is a list of strings --> returns a string
    max_length = max([len(md) for md in middle_products])
    for i, md in enumerate(middle_products):
        middle_products[i] = "0" * (max_length - len(md)) + md
 
    #print(middle_products)
    carry = "0"
    result = ""
    for i in range(1, max_length + 1):
        row = [carry] + [md[-i] for md in middle_products]
        r = functools.reduce(plus, row)
        carry, digit = r[:-1], r[-1]
        result = digit + result
    return carry + result


def algorithm(x, y):
    x, y = str(x), str(y)
    middle_products = []
    for i, digit in enumerate(reversed(y)):
        middle_products.append(one_to_n_product(digit, x) + "0" * i)
    #print(middle_products)
    return int(sum_middle_products(middle_products))

if __name__ == '__main__':
    print(karatsuba(1090, 324098))
    print(algorithm(1090, 324098))

# 11, 多项式乘法的快速傅里叶变换：
def mults(A, B):
    m, n = len(A), len(B)
    result = [0] * (m + n - 1)
    for i in range (m):
        for j in range(n):
            result[i + j] += A[i] * B[j]
    return result

def printPoly(poly):
    n = len(poly)
    show = ""
    for i in range(n-1, -1, -1):
        show += str(poly[i])
        if (i != 0):
            show = show + "x^" + str(i)
        if (i != 0):
            show = show + " + "
    print(show)

if __name__ == '__main__':
    A = [5, 0, 10, 6]
    B = [1, 2, 4]
    r = mults(A, B)
    printPoly(r)

    from numpy import convolve
    A = [5, 0, 10, 6]
    B = [1, 2, 4]
    print(convolve(A, B))

# 12, 水槽问题:
# Utility method to get
# sum of first n numbers
def getCumulateSum(n):
    return (n * (n + 1)) // 2
 
 
# Method returns minimum number of days
# after  which tank will become empty
def minDaysToEmpty(C, l):
 
    # if water filling is more than 
    # capacity then after C days only
    # tank will become empty
    if (C <= l) : return C 
 
    # initialize binary search variable
    lo, hi = 0, 1e4
 
    # loop until low is less than high
    while (lo < hi): 
        mid = int((lo + hi) / 2)
 
        # if cumulate sum is greater than (C - l) 
        # then search on left side
        if (getCumulateSum(mid) >= (C - l)): 
            hi = mid
         
        # if (C - l) is more then 
        # search on right side
        else:
            lo = mid + 1   
     
    # Final answer will be obtained by 
    # adding l to binary search result
    return (l + lo)

import math  # 法二：
def solve(a, b, c):
    r = pow(b, 2) - 4 * a * c
    if (r < 0):
        raise ValueError("No Solution") 
    return (-b + math.sqrt(r)) / (2 * a)

def minDaysToEmpty2(C, l):
    co = -2 * (C - l)
    return  math.ceil(solve(1, 1, co)) + l

if __name__ == '__main__':
    C, l = 5, 2
    print(minDaysToEmpty(C, l))

    C, l = 5, 2
    print(minDaysToEmpty2(C, l))


# 13，用最少步数收集所有硬币：
def minSteps(height):
    
    def minStepHelper(height, left, right, h):
        if left >= right:
            return 0
        
        m = left
        for i in range(left, right):
            if height[i] < height[m]:
                m = i
         
        return min(right - left, 
                   minStepHelper(height, left, m, height[m]) +
                   minStepHelper(height, m + 1, right, height[m]) +
                   height[m] - h)
    
    return minStepHelper(height, 0, len(height), 0)    

if __name__ == '__main__':
    height = [3, 1, 2, 5, 1]
    print(minSteps(height))