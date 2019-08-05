# leetcode-504：进制转换，七进制
class Solution(object):
    def convertToBase7(self, num):
        """
        :type num: int
        :rtype: str
        """
        if num == 0:
            return '0'
        
        res = ''
        is_negative = (num < 0)
        if is_negative:
            num = -num
        while num > 0:
            res += str(num % 7)  # 这里取余后，第一个是'个位',第二个是'十位',依此类推，所以后面要加上倒序操作
            num //= 7
        res = res[::-1]  # 倒置字符串
        if is_negative:
            res = '-' + res
        return res