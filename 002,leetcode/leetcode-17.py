# leetcode-17: 递归
# 列表形式: O(3^n)的时间复杂度，n=len(digits)
class Solution:  # 战胜了81.75%
    def letterCombinations(self, digits: str) -> list:
        l = len(digits)
        letters = [' ', '*', 'abc', 'def', 'ghi','jkl', 'mno', 'pqrs', 'tuv', 'wxyz']

        if l == 1:  # 当递归到digits只有一个数字时，这是一个递归的终结者
            return list(letters[int(digits)])
        elif l == 0:
            return []
        res = []
        for i in list(letters[int(digits[0])]):
            # for j in self.letterCombinations(digits[1:]):                
            #     res.append(i + j)
            res.extend([i+j for j in self.letterCombinations(digits[1:])])
        return res

class Solution:  # 字典形式：算法与上面一致，战胜了81.75%
    def letterCombinations(self, digits: str) -> List[str]:
        num_str = {"2":"abc","3":"def","4":"ghi","5":"jkl","6":"mno","7":"pqrs","8":"tuv","9":"wxyz"}
        if len(digits) == 0:
            return []
        if len(digits) == 1:  # 当递归到digits只有一个数字时，这是一个递归的终结者
            return list(num_str[digits[0]])
        res = []
        res_after = self.letterCombinations(digits[1:])
        for i in num_str[digits[0]]:
        	res.extend([i+j for j in res_after])
        return res

# 非递归形式：时间复杂度与递归形式一样,战胜了81.75%
class Solution:
    def letterCombinations(self, digits: str) -> list:
        if not digits:
            return []
        num_str = {"2":"abc","3":"def","4":"ghi","5":"jkl",
                   "6":"mno","7":"pqrs","8":"tuv","9":"wxyz"}
        res = [i for i in num_str[digits[0]]]
        for i in digits[1:]:
            res = [m+n for m in res for n in num_str[i]]
        return res

# leetcode-17：回溯法,beats 83.25%
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        self.keys = ["","","abc","def","ghi","jkl","mno","pqrs","tuv","wxyz"]
        combinations, prefix = [], ''
        if not digits:
            return combinations
        self.doCombinations(prefix, combinations, digits)
        return combinations
    
    def doCombinations(self, prefix:str, combinations: List[str], digits: str):
        if len(prefix) == len(digits):
            combinations.append(prefix)
            return
        for i in self.keys[int(digits[len(prefix)])]:
            prefix += i
            self.doCombinations(prefix, combinations, digits)
            prefix = prefix[:-1]