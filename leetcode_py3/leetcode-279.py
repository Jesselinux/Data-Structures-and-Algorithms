# leetcode-279:广度优先搜索(DFS),beats  51.82%
class Solution:
    def numSquares(self, n: int) -> int:
        squares = []
        square, interval = 1, 3
        while square <= n:
            squares.append(square)
            square += interval
            interval += 2
        list.reverse(squares)
        q = [(0, n)]
        used = [False for i in range(n+1)]
        while q:
            temp = q.pop(0)
            if temp[1] == 0:
                return temp[0]
            for s in squares:
                if temp[1] - s >= 0 and not used[temp[1] - s]:
                    q.append((temp[0] + 1, temp[1] - s))
                    used[temp[1] - s] = True
        return -1