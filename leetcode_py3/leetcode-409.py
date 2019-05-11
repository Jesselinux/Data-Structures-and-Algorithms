#leetcode-409：回文
class Solution(object):  # 战胜了84.78%
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: int
        """
        cnts = {}
        for c in s:
            if c in cnts:
                cnts[c] += 1
            else:
                cnts[c] = 1
        ret = 0
        for (key, value) in cnts.items():
            ret += (value // 2) * 2  # 出现偶数次，就可以分列回文的两侧
        if ret < len(s):
            ret += 1  # 回文的中间元素可以是出现一次的元素
        return ret