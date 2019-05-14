# 1，冒泡排序：
import time
from random import randint
import matplotlib.pyplot as plt

def _bubble_sort(nums: list, reverse=False):
    start = time.time()
    for i in range(len(nums)):
        # Get (i+1) largest in the correct position
        for j in range(len(nums) - i - 1):
            if nums[j] > nums[j + 1]:
                nums[j], nums[j + 1] = nums[j + 1], nums[j]
    if reverse:
        #nums = nums[::-1]  # why this is not working?
        nums.reverse()
    t = time.time() - start
    return len(nums), t

def bubble_sort_mod(array):
    import time
    start = time.time()    
    for i in range(len(array)): # n pass
        is_sorted = True  # initialize is_sorted
        for j in range(1, len(array) - i):
            if (array[j] < array[j - 1]):
                # swap
                array[j], array[j - 1] = array[j - 1], array[j]
                is_sorted = False
        
        if (is_sorted): break
    t = time.time() - start
    return len(array), t         

def bubble_sorted(nums: list, reverse=False) -> list:
    """Bubble Sort"""
    nums_copy = list(nums)
    _bubble_sort(nums_copy, reverse=reverse)
    return nums_copy

if __name__ == '__main__':
	l = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]
	l = bubble_sorted(l, reverse=True)
	print(l)

	# 画图：时间复杂度
	def generate_random_array(n):
		return [randint(0, n) for e in range(n)]
	random_lists = [generate_random_array(100 * n) for n in range(1, 20)]
	rst = [_bubble_sort(l) for l in random_lists]
	x = list(zip(*rst))[0]
	y = list(zip(*rst))[1]
	plt.plot(x, y)
	plt.show()

# 2, 选择排序：
def selection_sort(items):
    start = time.time()
    for i in range(len(items)):   # n
        pos_min = i   #idx
        for j in range(i + 1, len(items)):  # n
            if (items[j] < items[pos_min]):
                pos_min = j

        items[i], items[pos_min] = items[pos_min], items[i]
    t = time.time() - start
    return len(items), t  

if __name__ == '__main__':
	l = [1, 3, 5, 7, 9, 2, 4, 6, 8, 99]
	selection_sort(l)
	print(l)

# 3，插入排序：
def insert_sort(items):
    start = time.time()
    for sort_inx in range(1,len(items)):
        unsort_inx = sort_inx
        while unsort_inx > 0 and items[unsort_inx-1] > items[unsort_inx]:
            items[unsort_inx-1], items[unsort_inx] = items[unsort_inx], items[unsort_inx-1]
            unsort_inx = unsort_inx-1
    t = time.time() - start
    return len(items), t            

if __name__ == '__main__':
	l = [1, 3, 5, 7, 9, 2, 4, 6, 8, 99]
	selection_sort(l)
	print(l)

# 4，希尔排序：
def shell_sort(nums):
    start = time.time()

    gap = len(nums)
    length = len(nums)

    while (gap > 0):
        for i in range(gap, length):
            for j in range(i, gap - 1, -gap):
                if (nums[j - gap] > nums[j]):
                    nums[j], nums[j - gap] = nums[j - gap], nums[j]

        if (gap == 2): 
            gap = 1
        else:
            gap = gap // 2

    t = time.time() - start
    return len(nums), t

# 5，计数排序：
def count_sort(items):
    start = time.time()    
    mmax, mmin = items[0], items[0]
    for i in range(1, len(items)):
        if (items[i] > mmax): mmax = items[i]
        elif (items[i] < mmin): mmin = items[i]
    print(mmax)
    nums = mmax - mmin + 1
    counts = [0] * nums
    for i in range (len(items)):
        counts[items[i] - mmin] = counts[items[i] - mmin] + 1

    pos = 0
    for i in range(nums):
        for j in range(counts[i]):
            items[pos] = i + mmin
            pos += 1
            
    t = time.time() - start
    return len(items), t


# 6，基数排序：
def radix_sort(array, base=10):
    start = time.time()    
    def list_to_buckets(array, base, iteration):
        buckets = [[] for _ in range(base)]
        for number in array:
            # Isolate the base-digit from the number
            digit = (number // (base ** iteration)) % base
            # Drop the number into the correct bucket
            buckets[digit].append(number)
        return buckets

    def buckets_to_list(buckets):
        numbers = []
        for bucket in buckets:
            # append the numbers in a bucket
            # sequentially to the returned array
            for number in bucket:
                numbers.append(number)
        return numbers

    maxval = max(array)

    it = 0
    # Iterate, sorting the array by each base-digit
    while base ** it <= maxval:
        array = buckets_to_list(list_to_buckets(array, base, it))
        it += 1

    t = time.time() - start
    return len(array), t

# 7，归并排序：
def _merge(a: list, b: list) -> list:
    """Merge two sorted list"""
    c = []
    while len(a) > 0 and len(b) > 0:
        if a[0] < b[0]:
            c.append(a[0])
            a.remove(a[0])
        else:
            c.append(b[0])
            b.remove(b[0])

    if len(a) == 0:
        c += b
    else:
        c += a
    return c


def _merge_sorted(nums: list) -> list:
    # Won't sort in place
    if len(nums) <= 1:
        return nums

    m = len(nums) // 2
    a = _merge_sorted(nums[:m])
    b = _merge_sorted(nums[m:])
    return _merge(a, b)


# Wrapper
def merge_sorted(nums: list, reverse=False) -> list:
    import time
    start = time.time()
    """Merge Sort"""
    nums = _merge_sorted(nums)
    if reverse:
        nums = nums[::-1]

    t = time.time() - start
    return nums, len(nums), t

if __name__ == '__main__':
	l = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]
	l = merge_sorted(l, reverse=True)[0]
	print(l)

# 8，快速排序：
def _quick_sorted(nums: list) -> list:
    if len(nums) <= 1:
        return nums

    pivot = nums[0]
    left_nums = _quick_sorted([x for x in nums[1:] if x < pivot])
    right_nums = _quick_sorted([x for x in nums[1:] if x >= pivot])
    return left_nums + [pivot] + right_nums


def quick_sorted(nums: list, reverse=False) -> list:
    """Quick Sort"""
    start = time.time()
    nums = _quick_sorted(nums)
    if reverse:
        nums = nums[::-1]
    t = time.time() - start
    return nums, len(nums), t

if __name__ == '__main__':
	l = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]
	l = quick_sorted(l, reverse=True)
	print(l)

	# 画图：时间复杂度
	# rst = [quick_sorted(l) for l in random_lists]
	# r = [(l[1],l[2]) for l in rst]
	# x = list(zip(*r))[0]
	# y = list(zip(*r))[1]
	# plt.plot(x, y)
	# plt.show()

# 9, 堆排序：
def head_sort(list):
    length_list = len(list)
    first=int(length_list/2-1)
    for start in range(first,-1,-1):
        max_heapify(list,start,length_list-1)
    for end in range(length_list-1,0,-1):
        list[end],list[0]=list[0],list[end]
        max_heapify(list,0,end-1)
    return list

def max_heapify(ary,start,end):
    root = start
    while True:
        child = root*2 + 1
        if child > end:
            break
        if child + 1 <= end and ary[child]<ary[child+1]:
            child = child + 1
        if ary[root]<ary[child]:
            ary[root],ary[child]=ary[child],ary[root]
            root=child
        else:
            break

if __name__ == '__main__':
    list=[10,23,1,53,654,54,16,646,65,3155,546,31]
    print(head_sort(list))