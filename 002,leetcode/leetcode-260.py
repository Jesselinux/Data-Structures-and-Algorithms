# leetcode-260:位操作
class Solution(object):  # 战胜了30.88%
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        diff = 0
        for num in nums:
            diff = diff ^ num
        diff = diff & (-diff)  # -diff与~(diff - 1)的二进制形式相同
        ret = [0, 0]
        for num in nums:
            if (diff & num == 0):
                ret[0] = ret[0] ^ num
            else:
                ret[1] = ret[1] ^ num
        return ret

class Solution(object):  # 战胜了30.93%
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        diff = 0
        for num in nums:
            diff = diff ^ num
        diff = diff & (~(diff - 1))
        ret = [0, 0]
        for num in nums:
            if (diff & num == 0):
                ret[0] = ret[0] ^ num
            else:
                ret[1] = ret[1] ^ num
        return ret


# reference:https://blog.csdn.net/qq_38595487/article/details/81163737