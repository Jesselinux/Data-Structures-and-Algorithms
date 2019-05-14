# 1，在旋转有序数列中查找最小值 ：例如：[3,4,5,6,1,2]
def searchlazy(alist):
    alist.sort()
    return alist[0]

def searchslow(alist):
    mmin = alist[0]
    for i in alist:
        mmin = min(mmin, i)
    return mmin 
        
def search(alist):
    if len(alist) == 0:
        return -1    
    left, right = 0, len(alist) - 1
    while left + 1 < right: 
        if (alist[left] < alist[right]):
            return alist[left];
        mid = left + (right - left) // 2
        if (alist[mid] >= alist[left]):
            left = mid + 1
        else:
            right = mid
    return alist[left] if alist[left] < alist[right] else alist[right]

if __name__ == '__main__':
    list1 = [3,4,5,6,1,2]
    print(search(list1))

# 2, 旋转数组中查找指定数字
def search(alist, target):
    if len(alist) == 0:
        return -1    
    left, right = 0, len(alist) - 1
    while left + 1 < right: 
        mid = left + (right - left) // 2
        if alist[mid] == target:
            return mid
        
        if (alist[left] < alist[mid]):
            if alist[left] <= target and target <= alist[mid]:
                right = mid
            else:
                left = mid
        else:
            if alist[mid] <= target and target <= alist[right]:
                left = mid
            else: 
                right = mid
                            
    if alist[left] == target:
        return left
    if alist[right] == target:
        return right
        
    return -1

if __name__ == '__main__':
    num_list = [5,7,8,9,1,2,3]
    print(search(num_list, 2))

# 3, 搜索插入位置: 假设数组中不存在重复数。若数组中找到此目标值则返回目标值的index，否则返回目标值按顺序应该被插入的位置index
def search_insert_position(alist, target):
    if len(alist) == 0:
        return 0  
    left, right = 0, len(alist) - 1
    while left + 1 < right: 
        mid = left + (right - left) // 2
        if alist[mid] == target:
            return mid
        
        if (alist[mid] < target):
            left = mid
        else:
            right = mid
            
    if alist[left] >= target:
        return left
    if alist[right] >= target:
        return right
        
    return right + 1

if __name__ == '__main__':
    num_list = [5,7,8,9,1,2,3]
    print(search_insert_position(num_list, 6))

# 4, 查找给定目标值的开始和结束位置：含有重复数字
def search_range(alist, target):
    if len(alist) == 0:
        return (-1, -1)  
    
    lbound, rbound = -1, -1

    # search for left bound 
    left, right = 0, len(alist) - 1
    while left + 1 < right: 
        mid = left + (right - left) // 2
        if alist[mid] == target:
            right = mid
        elif (alist[mid] < target):
            left = mid
        else:
            right = mid
            
    if alist[left] == target:
        lbound = left
    elif alist[right] == target:
        lbound = right
    else:
        return (-1, -1)

    # search for right bound 
    left, right = 0, len(alist) - 1        
    while left + 1 < right: 
        mid = left + (right - left) // 2
        if alist[mid] == target:
            left = mid
        elif (alist[mid] < target):
            left = mid
        else:
            right = mid
            
    if alist[right] == target:
        rbound = right
    elif alist[left] == target:
        rbound = left
    else:
        return (-1, -1)        
        
    return (lbound, rbound)

if __name__ == '__main__':
    list_test = [1,1,2,2,2,2,2,3,3,6,6,6]
    print(search_range(list_test, 2))

# 5, 给定一个有序的字符串序列，这个序列中的字符串用空字符隔开，找到给定字符串位置:
def search_empty(alist, target):
    if len(alist) == 0:
        return -1
      
    left, right = 0, len(alist) - 1
    
    while left + 1 < right:
        while left + 1 < right and alist[right] == "":
            right -= 1
        if alist[right] == "":
            right -= 1
        if right < left:
            return -1
        
        mid = left + (right - left) // 2
        while alist[mid] == "":
            mid += 1
            
        if alist[mid] == target:
            return mid
        if alist[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
            
    if alist[left] == target:
        return left
    if alist[right] == target:
        return right    
        
    return -1 

if __name__ == '__main__':
    str_test = '   a   b      c   d     e          f'
    print(search_empty(str_test, 'c'))

# 6, 查找数据流中某个元素第一次出现的位置，数据流长度未知：
def search_first(alist):
    left, right = 0, 1
    
    while alist[right] == 0:
        left = right
        right *= 2
        
        if (right > len(alist)):
            right = len(alist) - 1
            break
    
    return left + search_range(alist[left:right+1], 1)[0]

if __name__ == '__main__':
    alist = [0, 0, 0, 0, 0, 1]
    r = search_first(alist)
    print(r)

# 7, 供暖设备在同一水平线上的位置分布，请找到能给所有房 屋供暖的供暖设备的最小供暖半径:
from bisect import bisect

def findRadius(houses, heaters):
    heaters.sort()
    ans = 0

    for h in houses:
        hi = bisect(heaters, h)
        left = heaters[hi-1] if hi - 1 >= 0 else float('-inf')
        right = heaters[hi] if hi < len(heaters) else float('inf')
        ans = max(ans, min(h - left, right - h))
    return ans

if __name__ == '__main__':
    houses = [1,12,23,34]
    heaters = [12,24]
    print(findRadius(houses, heaters))

# 8, 计算平方根：
def sqrt(x):
    if x == 0:
        return 0
    left, right = 1, x
    while left <= right:
        mid = left + (right - left) // 2
        if (mid == x // mid):
            return mid
        if (mid < x // mid):
            left = mid + 1
        else:
            right = mid - 1
    return right

def sqrtNewton(x):
    r = x
    while r*r > x:
        r = (r + x//r) // 2
    return r

if __name__ == '__main__':
    print(sqrt(68))
    print(sqrtNewton(89))

# 9, 给定一个包含n+1个整数的数组，其中每个元素为1到n闭区间的整数值，请证明至少 存在一个重复数。假设只有一个重复数，查找该重复数
def findDuplicate1(nums):

    low = 1
    high = len(nums)-1

    while low < high:
        mid = low + (high - low) // 2
        count = 0
        for i in nums:
            if i <= mid:
                count+=1
        if count <= mid:
            low = mid+1
        else:
            high = mid
    return low

def findDuplicate2(nums):  # Detect Cycle
    # The "tortoise and hare" step.  We start at the end of the array and try
    # to find an intersection point in the cycle.
    slow = 0
    fast = 0

    # Keep advancing 'slow' by one step and 'fast' by two steps until they
    # meet inside the loop.
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]

        if slow == fast:
            break

    # Start up another pointer from the end of the array and march it forward
    # until it hits the pointer inside the array.
    finder = 0
    while True:
        slow   = nums[slow]
        finder = nums[finder]

        # If the two hit, the intersection index is the duplicate element.
        if slow == finder:
            return slow

if __name__ == '__main__':
    nums = [3,5,6,3,1,4,2]
    print(findDuplicate2(nums))

# 10, 查找矩阵中第k小的元素，该矩阵的行、列向量均为有序向量：
# 法1：
from bisect import bisect
def kthSmallest(matrix, k):
    lo, hi = matrix[0][0], matrix[-1][-1]
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if sum(bisect(row, mid) for row in matrix) < k:
            lo = mid+1
        else:
            hi = mid
    return lo

# 法2：
def kthSmallest(matrix, k):
    n = len(matrix)
    L, R = matrix[0][0], matrix[n - 1][n - 1]
    while L < R:
        mid = L + (R - L) // 2
        temp = search_lower_than_mid(matrix, n, mid)
        if temp < k:
            L = mid + 1
        else:
            R = mid
    return L

def search_lower_than_mid(matrix, n, x):
    i, j = n - 1, 0
    cnt = 0
    while i >= 0 and j < n:
        if matrix[i][j] <= x:
            j += 1
            cnt += i + 1
        else:
            i -= 1
    return cnt

if __name__ == '__main__':
    matrix = [
        [1, 4, 8, 10,15],
        [3, 5, 6, 7, 20],
        [9, 20,22,24,29],
        [11,22,23,29,39]
    ]
    
    print(kthSmallest(matrix, 7))

# 11，合并区间：
class Interval:
    def __init__(self, s=0, e=0):
        self.start = s
        self.end = e
    
    def __str__(self):
        return "[" + self.start + "," + self.end + "]"
    
    def __repr__(self):
        return "[%s, %s]" % (self.start, self.end)

def merge(intervals):
    intervals.sort(key=lambda x: x.start)

    merged = []
    for interval in intervals:
        # if the list of merged intervals is empty or if the current
        # interval does not overlap with the previous, simply append it.
        if not merged or merged[-1].end < interval.start:
            merged.append(interval)
        else:
        # otherwise, there is overlap, so we merge the current and previous
        # intervals.
            merged[-1].end = max(merged[-1].end, interval.end)

    return merged

if __name__ == '__main__':
    i1 = Interval(1,9)
    i2 = Interval(2,5)
    i3 = Interval(19,20)
    i4 = Interval(10,11)
    i5 = Interval(12,20)
    i6 = Interval(0,3)
    i7 = Interval(0,1)
    i8 = Interval(0,2)
    intervals = [i1,i2,i3,i4,i5,i6,i7,i8]
    print(merge(intervals))

# 12, 插入区间： 这个集合中插入一个新的区间, 如需要，并合并区间：
def insert1(intervals, newInterval):
    merged = []
    for i in intervals:
        if newInterval is None or i.end < newInterval.start:
            merged += i,
        elif i.start > newInterval.end:
            merged += newInterval,
            merged += i,
            newInterval = None
        else:
            newInterval.start = min(newInterval.start, i.start)
            newInterval.end = max(newInterval.end, i.end)
    if newInterval is not None:
        merged += newInterval,
    return merged

def insert2(intervals, newInterval):
    left, right = [], []
    for i in intervals:
        if i.end < newInterval.start:
            left += i,
        elif i.start > newInterval.end:
            right += i,
        else:
            newInterval.start = min(newInterval.start, i.start)
            newInterval.end = max(newInterval.end, i.end)
    return left + [Interval(newInterval.start, newInterval.end)] + right

def insert3(intervals, newInterval):
    if len(intervals) == 0:
        intervals += newInterval,
    
    startPos = searchPosition(intervals, newInterval.start)
    endPos = searchPosition(intervals, newInterval.end)
    
    newStart = 0
    if (startPos >= 0 and intervals[startPos].end >= newInterval.start):
        newStart = intervals[startPos].start
    else:
        newStart = newInterval.start
        startPos += 1
        
    newEnd = 0
    if (endPos >= 0):
        newEnd = max(newInterval.end, intervals[endPos].end)
    else:
        newEnd = newInterval.end
        
    for i in range(startPos, endPos+1):
        intervals.pop(startPos)  # note: NOT i, but startPos, since one element is removed.

    intervals.insert(startPos, Interval(newStart, newEnd))
    return intervals

# return (actual insertion position - 1)
def searchPosition(intervals, x):
    start = 0
    end = len(intervals) - 1
    while (start <= end):
        mid = start + (end - start) // 2
        if (intervals[mid].start == x):
            return mid
        if (intervals[mid].start < x):
            start = mid + 1
        else:
            end = mid - 1

    return end

if __name__ == '__main__':
    i1 = Interval(1,2)
    i2 = Interval(3,5)
    i3 = Interval(6,7)
    i4 = Interval(8,10)
    i5 = Interval(12,16)
    intervals = [i1,i2,i3,i4,i5]
    new = Interval(4,8)
    print(insert3(intervals, new))