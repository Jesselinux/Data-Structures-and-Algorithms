# leetcode-69: 二分查找, O(lgn), 战胜了 84.44%
class Solution:
    def mySqrt(self, x: int) -> int:
        if x == 0:
            return 0
        
        l = 1
        r = x
        
        while l <= x:
            res = (l + r) // 2
            s = res ** 2
            
            if s <= x < (res + 1)**2:
                return res
            if s < x:
                l = res
            if s > x:
                r = res

class Solution(object):  # 同样是二分查找，这种写法超时
    def mySqrt(self, x):
        if x <= 1:
            return x  # 考虑到0,1的情况
        l = 0
        h = x
        while l <= h:
            m = l + (h - 1) / 2  # 中值：防止溢出，等于（l + h） / 2,后者容易溢出
            if x > m**2:
                l = m+1
            elif x < m**2:
                h = m-1
            elif x == m**2:
                return int(m)
        return int(h)