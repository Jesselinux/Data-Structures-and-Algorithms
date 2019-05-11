# leetcode-378：堆
class Solution(object):  # 超过了39.93%
    def kthSmallest(self, matrix, k):
        """
        :type matrix: List[List[int]]
        :type k: int
        :rtype: int
        """
        m, n = len(matrix), len(matrix[0])
        q = [(matrix[0][0], 0, 0)]  # 最初，只是往堆里面推一个元素(矩阵的第一行第一列元素)
        ans = 0
        for _ in range(k):   # 逐步将'最可能是最小值的点'进堆
            ans, i, j = heapq.heappop(q)
            if j == 0 and i + 1 < m:
                heapq.heappush(q, (matrix[i + 1][j], i + 1, j))  # 该元素的下面那个元素进堆
            if j + 1 < n:
                heapq.heappush(q, (matrix[i][j + 1], i, j + 1))  # 该元素的右侧那个元素进堆
        return ans

class Solution(object):  # 超过了20%
    def kthSmallest(self, matrix, k):
        """
        :type matrix: List[List[int]]
        :type k: int
        :rtype: int
        """
        nums = []
        for line in matrix:
            nums.extend(line)  # 一次性将矩阵的全部元素推进最小堆里
        heapq.heapify(nums)
        res = 0
        for i in range(k):
            res = heapq.heappop(nums)  # 一次从最小堆里pop元素，第k个就是kth smallest
        return res

class Solution(object):  # 逐步将'最可能是最小值的点'进堆
    def kthSmallest(self, matrix, k):
        """
        :type matrix: List[List[int]]
        :type k: int
        :rtype: int
        """
        from heapq import * 
        m, n = len(matrix), len(matrix[0])
        h = []
        for i in range(n):
            heapq.heappush(h, (matrix[0][i], 0, i))  # 往堆里放入矩阵的第一行所有元素
        for i in range(k-1):
            item = heapq.heappop(h)  # 第一个最小值一定在矩阵的第一行第一列元素，然后逐步进行
            if item[1] + 1 < m:
                heapq.heappush(h, (matrix[item[1] + 1][item[2]], item[1] + 1, item[2]))  # 将item在矩阵位置的下面那个元素推进堆里面

# 暴力解法：战胜了41.12%
class Solution(object):
    def kthSmallest(self, matrix, k):
        """
        :type matrix: List[List[int]]
        :type k: int
        :rtype: int
        """
        l = []
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                l.append(matrix[i][j])
        l = sorted(l)
        return l[k-1]