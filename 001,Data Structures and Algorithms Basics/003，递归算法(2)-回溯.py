# 1,子集：回溯法
class subsets1_BackTracking(object):

    def subsets_recursive(self, nums):
        lst = []
        result = []
        self.subsets_recursive_helper(result, lst, nums, 0);
        return result;

    def subsets_recursive_helper(self, result, lst, nums, pos):
        result.append(lst[:])
        for i in range(pos, len(nums)):
            lst.append(nums[i])
            self.subsets_recursive_helper(result, lst, nums, i+1)
            lst.pop()

def subsets2_recursion(nums):
    res = [[]]
    for num in nums:
        res += [i + [num] for i in res]
    return res

def subsets3_for(nums):
    result = [[]]
    for num in nums:
        for element in result[:]:
            x=element[:]
            x.append(num)
            result.append(x)       
    return result

if __name__ == '__main__':
    nums = [1, 2, 3]
    sub = subsets1_BackTracking()

    print(sub.subsets_recursive(nums))


# 2，子集：包含重复元素的集合，求所有可能的子集组合。注意：子集个数比不重复的集合要少。
def subsets2(nums):
    res = [[]]
    for num in nums: 
        res += [ i + [num] for i in res if i + [num] not in res]
    return res

class subsets2_BackTracking(object):
    def subsets_recursive2(self, nums):
        lst = []
        result = []
        nums.sort()
        #print(nums)
        self.subsets2_recursive_helper(result, lst, nums, 0);
        return result;

    def subsets2_recursive_helper(self, result, lst, nums, pos):
        result.append(lst[:])
        for i in range(pos, len(nums)):
            if (i != pos and nums[i] == nums[i-1]):
                continue;
            lst.append(nums[i])
            self.subsets2_recursive_helper(result, lst, nums, i+1)
            lst.pop()

if __name__ == '__main__':
    nums = [1,2,2]
    print(subsets2(nums))

    sub = subsets2_BackTracking()
    print(sub.subsets_recursive2(nums))


# 3, 排列组合：
def permutation1_recursion(result, nums):
    if (len(nums)==0):
        print(result)
    for i in range(len(nums)):
        permutation1_recursion(result+str(nums[i]), nums[0:i]+nums[i+1:])  #

def permutation2(nums):
    perms = [[]]   
    for n in nums:
        new_perms = []
        for perm in perms:
            for i in range(len(perm)+1):   
                new_perms.append(perm[:i] + [n] + perm[i:])   # insert n
        perms = new_perms
    return perms 

if __name__ == '__main__':
    nums = [1, 2, 3]
    print(permutation1_recursion('', nums))
    print(permutation2(nums))


# 4, 排列组合：包含重复元素的集合，求所有可能的排列组合。注意：组合个数要少于重复的集合。
def permUnique1_recursion(result, nums):
    nums.sort()
    if (len(nums)==0):
        print(result)
    for i in range(len(nums)):
        if (i != 0 and nums[i] == nums[i-1]):
            continue;
        permUnique1_recursion(result+str(nums[i]), nums[0:i]+nums[i+1:])

def permuteUnique2_for(nums):
    ans = [[]]
    for n in nums:
        new_ans = []
        for l in ans:
            for i in range(len(l)+1):
                new_ans.append(l[:i]+[n]+l[i:])
                if i<len(l) and l[i]==n: break              #handles duplication
        ans = new_ans
    return ans

if __name__ == '__main__':
    nums = [6, 8, 6]
    permUnique1_recursion('', nums)
    print(permuteUnique2_for(nums))


# 5, 排列组合：从n个元素中有序选择k个元素。
def permSizeK_recursion(result, nums, k):
    if k == 0:
        print(result)
    for i in range(len(nums)):
        permSizeK_recursion(result+str(nums[i]), nums[0:i] + nums[i+1:], k - 1)

if __name__ == '__main__':
    nums = [1, 2, 3, 4]
    permSizeK_recursion('', nums, 2)


# 6, 排列组合：word = “medium-one”，Rule = “io”，solutions = [“medium-one”, “medIum-one”, “medium-One”, “medIum-One”]
class permutation_letter(object):
    def __init__(self):
        self.results = set()
        self.keys = set()

    def permLetter(self, word, rule):
        rule = rule.lower()
        for c in rule:
            self.keys.add(c)
        self.permHelper(word, rule, 0, "")
        
    def permHelper(self, word, rule, index, prefix):
        length = len(word)
        
        for i in range(index, length):
            c = word[i]
            if (c in self.keys):
                self.permHelper(word, rule, i + 1, prefix + c)
                
                c = c.upper()
                self.permHelper(word, rule, i + 1, prefix + c)
            else:
                prefix += c
        
        if (len(prefix) == len(word)):
            self.results.add(prefix)

if __name__ == '__main__':
    word = "helloworld"

    per_letter = permutation_letter()

    per_letter.permLetter(word, 'hd')
    print(per_letter.results)


# 7, 查找总和等于指定数字的子集:
class subsets_sum(object):
    def combination(self, nums, t):
        result = []
        tmp = []
        self.combHelper(result, tmp, nums, t, 0)
        return result
            
    def combHelper(self, result, tmp, nums, remains, start):
        if remains < 0: return
        if remains == 0:
            result.append(tmp[:])
        else:
            for i in range(start, len(nums)):
                tmp.append(nums[i])
                self.combHelper(result, tmp, nums, remains - nums[i], i)
                tmp.pop()

if __name__ == '__main__':
    candidates = [2,3,5]

    sub_sum = subsets_sum()
    print(sub_sum.combination(candidates, 8))


# 8, 查找总和等于指定数字的子集: 原集合含有重复元素。
class subsets_sum2(object):
    def combination2(self, nums, t):
        result = []
        tmp = []
        nums.sort()
        self.combHelper2(result, tmp, nums, t, 0)
        return result
            
    def combHelper2(self, result, tmp, nums, remains, start):
        if remains < 0: return
        if remains == 0:
            result.append(tmp[:])
        else:
            for i in range(start, len(nums)):
                if(i > start and nums[i] == nums[i-1]): continue; # skip duplicates
                tmp.append(nums[i])
                self.combHelper2(result, tmp, nums, remains - nums[i], i + 1)
                tmp.pop()
if __name__ == '__main__':
    candidates = [2,5,2,1,2]

    sub_sum2 = subsets_sum2()
    print(sub_sum2.combination2(candidates, 5))


# 9, 寻找所有符合语法的n组括号的组合：
def generateParenthesis(n):
    def generate(prefix, left, right, parens=[]):
        if right == 0:   parens.append(prefix)
        if left > 0:     generate(prefix + '(', left-1, right)
        if right > left: generate(prefix + ')', left, right-1)
        return parens
    return generate('', n, n)

if __name__ == '__main__':
    print(generateParenthesis(5))


# 10, 八皇后问题:
class eight_Queens(object):

    def solveNQueens(self, n):
        res = []
        self.dfs([-1]*n, 0, [], res)
        return res
     
    # nums is a one-dimension array, like [1, 3, 0, 2] means
    # first queen is placed in column 1, second queen is placed
    # in column 3, etc.
    def dfs(self, nums, index, path, res):
        if index == len(nums):
            res.append(path)
            return  # backtracking
        for i in range(len(nums)):
            nums[index] = i
            if self.valid(nums, index):  # pruning
                tmp = "." * len(nums)
                self.dfs(nums, index+1, path + [tmp[:i]+"Q"+tmp[i+1:]], res)
                
    # check whether nth queen can be placed in that column
    def valid(self, nums, n):
        for i in range(n):
            if abs(nums[i]-nums[n]) == n - i or nums[i] == nums[n]:
                return False
        return True

if __name__ == '__main__':
    game = eight_Queens()
    print(game.solveNQueens(8))