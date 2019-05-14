# 第一部分、创建堆
# 1,python自建堆：
class PriorityQueueBase:
    """Abstract base class for a priority queue."""

    class Item: 
        """Lightweight composite to store priority queue items."""
        __slots__ = '_key' , '_value'

        def __init__ (self, k, v):
            self._key = k
            self._value = v

        def __lt__ (self, other):                                        
            return self._key < other._key

        def is_empty(self):
            return len(self) == 0   

        def __str__(self):
            return str(self._key)
        

class HeapPriorityQueue(PriorityQueueBase):

    def __init__ (self):
        self._data = [ ]         

    def __len__ (self):
        return len(self._data)
    
    def is_empty(self):
        return len(self) == 0  

    def add(self, key, value): 
        self._data.append(self.Item(key, value)) 
        self._upheap(len(self._data) - 1)
        
    def min(self): 
        if self.is_empty():
            raise ValueError( "Priority queue is empty." )
        item = self._data[0]
        return (item._key, item._value)
    
    def remove_min(self):
        if self.is_empty():
            raise ValueError( "Priority queue is empty." )
        self._swap(0, len(self._data) - 1)
        item = self._data.pop( )
        self._downheap(0)
        return (item._key, item._value)

    def _parent(self, j): 
        return (j - 1) // 2
    
    def _left(self, j):
        return 2 * j + 1
    
    def _right(self, j):
        return 2 * j + 2

    def _has_left(self, j):
        return self._left(j) < len(self._data)
    
    def _has_right(self, j):
        return self._right(j) < len(self._data)      
    
    def _swap(self, i, j):
        self._data[i], self._data[j] = self._data[j], self._data[i]
        
    def _upheap(self, j):
        parent = self._parent(j) 
        if j > 0 and self._data[j] < self._data[parent]: 
            self._swap(j, parent) 
            self._upheap(parent) 
    
    def _downheap(self, j):
        if self._has_left(j):
            left = self._left(j)
            small_child = left
            if self._has_right(j):
                right = self._right(j) 
                if self._data[right] < self._data[left]:
                    small_child = right 
            if self._data[small_child] < self._data[j]:
                self._swap(j, small_child) 
                self._downheap(small_child)

if __name__ == '__main__':
	heap = HeapPriorityQueue()
	heap.add(4, "D")
	heap.add(3, "C")
	heap.add(1, "A")
	heap.add(5, "E")
	heap.add(2, "B")
	heap.add(7, "G")
	heap.add(6, "F")
	heap.add(26, "Z")

	for item in heap._data:
	    print(item)

	print("min is: ")
	print(heap.min())
	print()

	print("remove min: ")
	print(heap.remove_min())
	print("Now min is: ")
	print(heap.min())
	print()

	print("remove min: ")
	print(heap.remove_min())
	print("Now min is: ")
	print(heap.min())
	print()

	heap.add(1, "A")
	print("Now min is: ")
	print(heap.min())
	print()

# 2,修改python的相关函数后，堆还可以储存类对象，就像储存数字，字典等其他类型进堆一样
# Override __lt__ in Python 3, __cmp__ only in Python 2
from heapq import heappush, heappop
import heapq

class Skill(object):
    def __init__(self, priority, description):
        self.priority = priority
        self.description = description
        print('New Level:', description)
        return
    def __cmp__(self, other):
        return cmp(self.priority, other.priority)
    def __lt__(self, other):
        return self.priority < other.priority
    def __repr__(self):
        return str(self.priority) + ": " + self.description
    
if __name__ == '__main__':
	s1 = Skill(5, 'Proficient')
	s2 = Skill(10, 'Expert')
	s3 = Skill(1, 'Novice')

	l = [s1, s2, s3]

	heapq.heapify(l)
	print("The 3 largest numbers in list are : ",end="")
	print(heapq.nlargest(3, l))

	while l:
	    item = heappop(l) 
	    print(item)


# 第二部分：相关练习题
# 1，第k大元素：
import heapq  
# O(k+(n-k)lgk) time, min-heap
def findKthLargest(nums, k):
    heap = []
    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)
    
    return heapq.heappop(heap)

# O(k+(n-k)lgk) time, min-heap        
def findKthLargest1(nums, k):
    return heapq.nlargest(k, nums)[k-1]

if __name__ == '__main__':
	nums = [5,11,3,6,12,9,8,10,14,1,4,2,7,15]
	k = 5
	print(findKthLargest(nums, k))

# 2,前k高频词汇：
import collections
import heapq
import functools

@functools.total_ordering
class Element:
    def __init__(self, count, word):
        self.count = count
        self.word = word
        
    def __lt__(self, other):
        if self.count == other.count:
            return self.word > other.word
        return self.count < other.count
    
    def __eq__(self, other):
        return self.count == other.count and self.word == other.word

def topKFrequent(words, k):
    counts = collections.Counter(words)   

    freqs = []
    heapq.heapify(freqs)
    for word, count in counts.items():
        heapq.heappush(freqs, (Element(count, word), word))
        if len(freqs) > k:
            heapq.heappop(freqs)

    res = []
    for _ in range(k):
        res.append(heapq.heappop(freqs)[1])
    return res[::-1]

def topKFrequent2(nums, k):
    from collections import Counter as ct
    return [k for (k,v) in ct(nums).most_common(k)]

if __name__ == '__main__':
	words = ["i", "love", "you", "i", "love", "coding","i","like","sports","i","love","travel","coding","is","fun"]
	k = 4
	topKFrequent(words, k)

# 3,丑数：
def uglyNumber(num):
    for p in 2, 3, 5:
        while num % p == 0 < num:
            num /= p
    return num == 1

# 4,第k个丑数：
def nthUglyNumber(n):
    q2, q3, q5 = [2], [3], [5]
    ugly = 1
    for u in heapq.merge(q2, q3, q5):
        if n == 1:
            return ugly
        if u > ugly:
            ugly = u
            n -= 1
            q2 += 2 * u,
            q3 += 3 * u,
            q5 += 5 * u,

if __name__ == '__main__':
	nthUglyNumber(10)

# 5,小于k的数对
# O(kLogk) 
def kSmallestPairs(nums1, nums2, k):
    queue = []
    def push(i, j):
        if i < len(nums1) and j < len(nums2):
            heapq.heappush(queue, [nums1[i] + nums2[j], i, j])
    push(0, 0)
    pairs = []
    while queue and len(pairs) < k:
        _, i, j = heapq.heappop(queue)
        pairs.append([nums1[i], nums2[j]])
        push(i, j + 1)
        if j == 0:
            push(i + 1, 0)
    return pairs

def kSmallestPairs2(nums1, nums2, k):
    queue = []
    def push(i, j):
        if i < len(nums1) and j < len(nums2):
            heapq.heappush(queue, [nums1[i] + nums2[j], i, j])
    for i in range(0, k):
        push(i, 0)
    pairs = []
    while queue and len(pairs) < k:
        _, i, j = heapq.heappop(queue)
        pairs.append([nums1[i], nums2[j]])
        push(i, j + 1)
    return pairs

if __name__ == '__main__':
	nums1 = [1,7,11]
	nums2 = [2,4,6]
	k = 20
	print(kSmallestPairs2(nums1, nums2, k))

	nums1 = [1,1,2]
	nums2 = [1,2,3]
	k = 2
	print(kSmallestPairs(nums1, nums2, k))

# 6,合并k个有序列表：
from tree_J import LinkedList
from tree_J import Node_ll

def mergeKLists(lists):
    dummy = Node_ll(None)
    curr = dummy
    q = HeapPriorityQueue()
    for node in lists:
        if node: 
            q.put((node.value, node))
    while q.qsize()>0:
        curr.next = q.get()[1]
        curr = curr.next
        if curr.next: q.put((curr.next.value, curr.next))
    return dummy.next

# if __name__ == '__main__':
# 	lst1 = LinkedList()
# 	lst1.add_last(1)
# 	lst1.add_last(4)
# 	lst1.add_last(5)

# 	lst2 = LinkedList()
# 	lst2.add_last(1)
# 	lst2.add_last(3)
# 	lst2.add_last(4)

# 	lst3 = LinkedList()
# 	lst3.add_last(2)
# 	lst3.add_last(6)

# 	lists = [lst1.head.next, lst2.head.next, lst3.head.next]
# 	node = mergeKLists(lists)
# 	result = LinkedList()

# 	result.head.next = node
# 	result.printlist()

# 7,数据流中找中位数：
from heapq import *

class MedianFinder:

    def __init__(self):
        self.heaps = [], []

    def addNum(self, num):
        small, large = self.heaps
        heappush(small, -heappushpop(large, num))
        if len(large) < len(small):
            heappush(large, -heappop(small))

    def findMedian(self):
        small, large = self.heaps
        if len(large) > len(small):
            return float(large[0])
        return (large[0] - small[0]) / 2.0

# 8,项目管理(IPO):投资k个项目，利润最大化
import heapq
def findMaximizedCapital(k, W, Profits, Capital):
    pqCap = []
    pqPro = []
    
    for i in range(len(Profits)):
        heapq.heappush(pqCap, (Capital[i], Profits[i]))
        
    for i in range(k):
        while len(pqCap) != 0 and pqCap[0][0] <= W:
            heapq.heappush(pqPro, -heapq.heappop(pqCap)[1])
            
        if len(pqPro) == 0:
            break
        
        W -= heapq.heappop(pqPro)
    
    return W

def findMaximizedCapital2(k, W, Profits, Capital):
    current = []
    future = sorted(zip(Capital, Profits))[::-1]
    for _ in range(k):
        while future and future[-1][0] <= W:  # afford
            heapq.heappush(current, -future.pop()[1])
        if current:
            W -= heapq.heappop(current)
    return W

if __name__ == '__main__':
	k=2
	W=0
	Profits=[1,2,3]
	Capital=[0,1,1]

	print(findMaximizedCapital2(k, W, Profits, Capital))