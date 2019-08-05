# leetcode-70:
''' 斐波那契数列问题：一共n阶台阶，倒数第一步时，无视前面怎么走，有两种走法：
    1.走一步
    2.走两步
两种走法的走法种数相加就是n阶台阶的情况下的所有种数，即：f(n)=f(n-1)+f(n-2)'''
class Solution:  # 战胜78.16%
    def climbStairs(self, n: int) -> int:
        count = [1, 2]
        for i in range(2, n):
            count.append(count[i-1] + count[i-2])
        return count[n-1]

class Solution:  # 战胜78.16%
    def climbStairs(self, n: int) -> int:
        count = [1, 1]
        for i in range(2, n+1):  # 与上面解法一致，区别仅在于列表index不同
            count.append(count[i-1] + count[i-2])
        return count[n]

# leetcode-70：斐波那契数列
class Solution:  # beats 17.26%
    def climbStairs(self, n: int) -> int:
        if n < 3:
            return n
        pre2, pre1 = 1, 2
        for i in range(2, n):
            pre1, pre2 = pre1+pre2, pre1
        return pre1

# 递归解法：不出意料超时了
class Solution:
    def climbStairs(self, n: int) -> int:
        if n == 1:
            return 1
        elif n == 2:
            return 2
        else:
            return self.climbStairs(n-1) + self.climbStairs(n-2)