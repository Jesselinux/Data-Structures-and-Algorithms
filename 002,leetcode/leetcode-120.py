# leetcode-120: O(n(n-1)/2):时间复杂度仍为O(n^2),逐个修改矩阵元素值为最短路径值，时间复杂度最高的一种解法，但让然战胜了82.66%
class Solution:  # 自顶向下更新矩阵
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        l = len(triangle)
        if l == 0:
            return 0
        if l == 1:
            return triangle[0][0]
        for i in range(1, l):
            for j in range(1, i):
                triangle[i][j] += min(triangle[i-1][j-1], triangle[i-1][j])
            triangle[i][i] += triangle[i-1][i-1]  # 需要注意矩阵行向量的index=0，-1 的元素，要单独更新
            triangle[i][0] += triangle[i-1][0]
        return min(triangle[l-1])

# 法二：自底向上更新矩阵, 时间复杂度与上面一样，但是，之战胜了24.81%
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        l = len(triangle)
        if l == 0:
            return 0
        if l == 1:
            return triangle[0][0]
        for i in range(l-2, -1, -1):
            for j in range(0, i+1):
                triangle[i][j] += min(triangle[i+1][j], triangle[i+1][j+1])
            
        return triangle[0][0]