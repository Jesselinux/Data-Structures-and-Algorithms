# 哈希表：
class Solution:
    def isHappy(self, n: int) -> bool:
        dict = {}
        while n != 1:
            n = sum(int(i) ** 2 for i in str(n))
            if n in dict:
                return False
            dict[n] = 1
        return True