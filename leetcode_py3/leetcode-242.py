# leetcode-242：哈希表
class Solution:  # 利用数组实现，战胜了57.81%
    def isAnagram(self, s: str, t: str) -> bool:
        arr = [0] * 26
        for c in s:
            arr[ord(c) - ord('a')] += 1  # 给对应字母计数；ord()函数：返回对应的ASCII码数值(是一个整数)
        for c in t:
            arr[ord(c) - ord('a')] -= 1  # 给对应字母减数
        for i in arr:
            if i != 0:
                return False  # 不为零，证明不是有效字谜
        return True

class Solution:  # 和上面的解题思想完全一致，只是利用字典实现，战胜了57.81% 
    def isAnagram(self, s: str, t: str) -> bool:
        dic = {}
        for c in s:
            dic[c] = dic.get(c, 0) + 1  # dict.get(key, default=None):返回指定键的值，如果值不在字典中返回默认值
        for c in t:
            dic[c] = dic.get(c, 0) - 1
        for key in dic:
            if dic[key] != 0:
                return False
        return True

class Solution:  # 和上面两个解题思路完全一致，细节不同而已，战胜了57.81%
    def isAnagram(self, s: str, t: str) -> bool:
        dic1, dic2 = {}, {}
        for item in s:
            dic1[item] = dic1.get(item, 0) + 1
        for item in t:
            dic2[item] = dic2.get(item, 0) + 1
        return dic1 == dic2
​
class Solution:  # 和上面三个解题思路完全一致，细节与上一个一致，战胜了57.81%
    def isAnagram(self, s: str, t: str) -> bool:
	    dic1, dic2 = [0]*26, [0]*26
	    for item in s:
	        dic1[ord(item)-ord('a')] += 1
	    for item in t:
	        dic2[ord(item)-ord('a')] += 1
	    return dic1 == dic2
	​
class Solution:  # 使用python内置函数，一行代码搞定，战胜了50.56%
    def isAnagram(self, s: str, t: str) -> bool:
    	return sorted(s) == sorted(t)