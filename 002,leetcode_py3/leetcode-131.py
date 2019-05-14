# leetcode-131：
# 递归(回溯法): 战胜了55.15%
class Solution(object):
    def partition(self, s):
        self.isPalindrome = lambda s : s == s[::-1]
        res = []
        self.helper(s, res, [])
        return res
        
    def helper(self, s, res, path):
        if not s:
            res.append(path)
            return
        for i in range(1, len(s) + 1):
            if self.isPalindrome(s[:i]):
                self.helper(s[i:], res, path + [s[:i]])

# 递归：战胜了26.86%
def partition(self, s: str) -> List[List[str]]:
        if len(s)==1:return [[s]]
        array=[]
        for i in range(1,len(s)+1):
            s_sub,item = s[0:i],[]
            if s_sub == s_sub[::-1]:
                item.append(s_sub)
                if s_sub != s:
                    item2 = self.partition(s[i:len(s)+1])
                    if len(item2):
                        for j in item2:
                            if len(''.join(item + j)) == len(s):
                                array.append(item + j)
                else:
                    array.append(item)
        return array