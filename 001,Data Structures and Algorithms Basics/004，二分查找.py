# 1，顺序查找：
def search(num_list, val):
    # If empty
    if num_list == None:
        return -1
    
    for i in range(0, len(num_list)):
        if (num_list[i] == val):
            return i
    return -1

if __name__ == '__main__':
	num_list = [1,2,3,5,7,8,9]
	print(search(num_list, 5))

# 2, 二分查找：递归
def bi_search_re(num_list, val):
    def bi_search(l, h):
        # Not found
        if l > h:
            return -1
        
        # Check mid
        mid = (l + h) // 2
        if (num_list[mid] == val):
            return mid;
        elif (num_list[mid] < val):
            return bi_search(mid + 1, h)
        else:
            return bi_search(l, mid - 1)
        
    return bi_search(0, len(num_list))

if __name__ == '__main__':
	num_list = [1,2,3,5,7,8,9]
	print(search(num_list, 8))

# 3, 二分查找：while循环
def bi_search_iter(alist, item):
    left, right = 0, len(alist) - 1
    while left <= right:
        mid = (left + right) // 2
        if alist[mid] < item:
            left = mid + 1
        elif alist[mid] > item:
            right = mid - 1
        else: # alist[mid] = item
            return mid
    return -1

if __name__ == '__main__':
	num_list = [1,2,3,5,7,8,9]
	print(search(num_list, 2))

# 4，二分查找：模板（可套用的框架）
def binarysearch(alist, item):
    if len(alist) == 0:
        return -1
    
    left, right = 0, len(alist) - 1
    while left + 1 < right:
        mid = left + (right - left) // 2
        if alist[mid] == item:
            right = mid
        elif alist[mid] < item:
            left = mid
        elif alist[mid] > item:
            right = mid
    
    if alist[left] == item:
        return left
    if alist[right] == item:
        return right
    
    return -1

if __name__ == '__main__':
	num_list = [1,2,3,5,7,8,9]
	print(search(num_list, 3))

# 5，单元测试：
import unittest
class TestBinarySearch1(unittest.TestCase):
    def setUp(self):
        self._f = bi_search_iter
    
    def test_empty(self):
        alist = []
        r = self._f(alist, 5)
        self.assertEqual(-1, r)

    def test_one(self):
        alist = [1]
        r = self._f(alist, 0)
        self.assertEqual(-1, r)
        r = self._f(alist, 1)
        self.assertEqual(0, r)

    def test_two(self):
        alist = [1,10]
        r = self._f(alist, 0)
        self.assertEqual(-1, r)
        r = self._f(alist, 1)
        self.assertEqual(0, r)
        r = self._f(alist, 2)
        self.assertEqual(-1, r)
        r = self._f(alist, 10)
        self.assertEqual(1, r)
        r = self._f(alist, 11)
        self.assertEqual(-1, r)
        
    def test_multiple(self):
        alist = [1,2,3,4,5]
        r = self._f(alist, 5)
        self.assertEqual(4, r)
        r = self._f(alist, 4)
        self.assertEqual(3, r)
        r = self._f(alist, 2)
        self.assertEqual(1, r)
        r = self._f(alist, 1)
        self.assertEqual(0, r)
        r = self._f(alist, 6)
        self.assertEqual(-1, r)
        r = self._f(alist, 0)
        self.assertEqual(-1, r)
        
    def test_duplicate(self):
        alist = [1,1,1,2,3,3,3,3,3,3,4,5,5,5]
        r = self._f(alist, 5)
        self.assertEqual(5, alist[r])
        r = self._f(alist, 4)
        self.assertEqual(4, alist[r])
        r = self._f(alist, 2)
        self.assertEqual(2, alist[r])
        r = self._f(alist, 3)
        self.assertEqual(3, alist[r])
        r = self._f(alist, 1)
        self.assertEqual(1, alist[r])
        r = self._f(alist, 6)
        self.assertEqual(-1, -1)
        r = self._f(alist, 0)
        self.assertEqual(-1, -1)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False) 