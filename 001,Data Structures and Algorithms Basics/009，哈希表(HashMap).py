# 第一部分：HashMap练习题
# 1,统计字母数：
def letterCount(s):
    freq = {}
    for piece in s:
        # only consider alphabetic characters within this piece
        word = ''.join(c for c in piece if c.isalpha())
        if word:
            freq[word] = 1 + freq.get(word, 0) #default 0

    max_word = ''
    max_count = 0
    for (w,c) in freq.items():    # (key, value) tuples represent (word, count)
        if c > max_count:
            max_word = w
            max_count = c
    print('The most frequent word is', max_word)
    print('Its number of occurrences is', max_count)    

from collections import Counter
def letterCount2(s):
    c = Counter(x for x in s if x != " ")

    for letter, count in c.most_common(4):
        print('%s: %7d' % (letter, count))

if __name__ == '__main__':
	s = "Hello World How are you I am fine thank you and you"
	letterCount(s)

# 2,统计单词数：
from collections import Counter
def wordCount(s):
    wordcount = Counter(s.split())
    print(wordcount)

if __name__ == '__main__':
	s = "Hello World How are you I am fine thank you and you"
	wordCount(s)

# 3,第一个没有重复的字符：不存在则返回-1
def firstUniqChar(s):
    letters='abcdefghijklmnopqrstuvwxyz'
    index=[s.index(l) for l in letters if s.count(l) == 1]
    return min(index) if len(index) > 0 else -1

if __name__ == '__main__':
	s = "givenastring"
	print(firstUniqChar(s))

# 4,求交集：结果中无重复值
def intersection(nums1, nums2):
    return list(set(nums1) & set(nums2))

# 5,求交集：结果中可以有重复值
def intersect(nums1, nums2):
    dict1 = dict()
    for i in nums1:
        if i not in dict1:
            dict1[i] = 1
        else:
            dict1[i] += 1
    ret = []
    for i in nums2:
        if i in dict1 and dict1[i]>0:
            ret.append(i)
            dict1[i] -= 1
    return ret

if __name__ == '__main__':
	nums1 = [1, 2, 2, 1]
	nums2 = [2, 2]
	print(intersect(nums1, nums2))

# 6,统计钻石数：
def numJewelsInStones(J, S):  # set
    setJ = set(J)
    return sum(s in setJ for s in S)

def numJewelsInStones_bf(J, S):  # 暴力解法
    count=0
    for c in S:
        if c in J:
            count += 1
    return count

if __name__ == '__main__':
	J = "aA"
	S = "aAAbbbb"
	print(numJewelsInStones(J, S))

# 7,判断是否包含重复元素：
def containsDuplicate(nums):
    return len(nums) > len(set(nums))

if __name__ == '__main__':
	nums = [1,2,3,4,5,3]
	print(containsDuplicate(nums))

# 8,判断是否包含重复元素：指定距离内
def containsNearbyDuplicate(nums, k):
    dic = {}
    for i, v in enumerate(nums):
        if v in dic and i - dic[v] <= k:
            return True
        dic[v] = i
    return False

if __name__ == '__main__':
	nums = [1,2,3,4,5,3]
	print(containsNearbyDuplicate(nums, 1))
	print(containsNearbyDuplicate(nums, 3))

# 9,网站域名访问计数：
import collections 
def subdomainVisits(cpdomains):
    ans = collections.Counter()
    for domain in cpdomains:
        count, domain = domain.split()
        count = int(count)
        frags = domain.split('.')
        for i in range(len(frags)):
            ans[".".join(frags[i:])] += count
    return ["{} {}".format(ct, dom) for dom, ct in ans.items()]

if __name__ == '__main__':
	cp = ["900 google.mail.com", "50 yahoo.com", "1 intel.mail.com", "5 wiki.org"]
	print(subdomainVisits(cp))

# 10,判断可以用一行键盘字母输出的字符串：
def findWords(words):
    line1, line2, line3 = set('qwertyuiop'), set('asdfghjkl'), set('zxcvbnm')
    ret = []
    for word in words:
        w = set(word.lower())
        if w.issubset(line1) or w.issubset(line2) or w.issubset(line3):
            ret.append(word)
    return ret

if __name__ == '__main__':
	words = ["Hello", "Alaska", "Dad", "Peace"]
	print(findWords(words))

# 11,字符串模式规则判断：pattern = "abba", str = "dog cat cat dog" 则返回True
def wordPattern(pattern, str):
    s = pattern
    t = str.split()
    return len(set(zip(s, t))) == len(set(s)) == len(set(t)) and len(s) == len(t)

if __name__ == '__main__':
	pattern = "aaaa"
	str1 = "dog cat cat dog" 
	print(wordPattern(pattern, str1))

# 12,排序之和最小的元素：
def findRestaurant(A, B):
    Aindex = {u: i for i, u in enumerate(A)}
    best, ans = 1e9, []

    for j, v in enumerate(B):
        i = Aindex.get(v, 1e9)
        if i + j < best:
            best = i + j
            ans = [v]
        elif i + j == best:
            ans.append(v)
    return ans

if __name__ == '__main__':
	A = ["Shogun", "Burger King", "Tapioca Express", "KFC"]
	B = ["KFC", "Burger King", "Shogun"]
	print(findRestaurant(A, B))

# 13,查找最长的单词：该单词由数组中其它单词逐步添加一个字母组成
def longestWord(words):
    words, resword, res = sorted(words), '', set()
    for word in words:
        if len(word) == 1 or word[:-1] in res:
            res.add(word)
            resword = word if resword == '' else word if len(word) > len(resword) else resword
    return resword

if __name__ == '__main__':
	words = ["a", "banana", "app", "appl", "ap", "apply", "apple"]
	print(longestWord(words))

# 14,快乐数字：
def isHappy(n):
    seen = set()
    while n not in seen:
        seen.add(n)
        n = sum([int(x) **2 for x in str(n)])
    return n == 1

if __name__ == '__main__':
	n = 19
	print(isHappy(n))

# 15,有效字谜：判断两个字符串元素是否一致，顺序可以不同
def isAnagram1(s, t):
    dic1, dic2 = {}, {}
    for item in s:
        dic1[item] = dic1.get(item, 0) + 1
    for item in t:
        dic2[item] = dic2.get(item, 0) + 1
    return dic1 == dic2

def isAnagram2(s, t):
    dic1, dic2 = [0]*26, [0]*26
    for item in s:
        dic1[ord(item)-ord('a')] += 1
    for item in t:
        dic2[ord(item)-ord('a')] += 1
    return dic1 == dic2

def isAnagram3(s, t):
    return sorted(s) == sorted(t)

if __name__ == '__main__':
	s = "anagram"
	t = "nagaram"
	print(isAnagram3(s, t))
	s = "rat"
	t = "car"
	print(isAnagram3(s, t))

# 16,查找所有有效字谜：
def findAnagrams(s, p):
    res = []
    n, m = len(s), len(p)
    if n < m: return res
    phash, shash = [0]*123, [0]*123
    for x in p:
        phash[ord(x)] += 1
    for x in s[:m-1]:
        shash[ord(x)] += 1
    for i in range(m-1, n):
        shash[ord(s[i])] += 1
        if i-m >= 0:
            shash[ord(s[i-m])] -= 1
        if shash == phash:
            res.append(i - m + 1)
    return res

if __name__ == '__main__':
	s = "cbaebabacd"
	p = "abc"
	print(findAnagrams(s, p))

# 17,有效字谜组：将数组中的字符串按有效字谜分组
def groupAnagrams1(strs):
    ans = collections.defaultdict(list)
    for s in strs:
        ans[tuple(sorted(s))].append(s)
    return ans.values()

def groupAnagrams2(strs):
    ans = collections.defaultdict(list)
    for s in strs:
        count = [0] * 26
        for c in s:
            count[ord(c) - ord('a')] += 1
        ans[tuple(count)].append(s)
    return ans.values()

if __name__ == '__main__':
	l =  ["eat", "tea", "tan", "ate", "nat", "bat"]
	print(groupAnagrams2(l))

# 18,按词频对字符串排序：
def frequencySort1(s):
    import collections
    if not s:
        return ""
    count_s = collections.Counter(s)
    counter = count_s.most_common()
    rs = ''
    for i in counter:
        rs += i[0] * i[1]
    return rs

def frequencySort2(s):
    import operator
    if not s:
        return ""
    counter = {}; rs = ''
    for i in s:
        counter[i] = 1 if i not in counter else counter[i]+1
    sorted_counter = sorted(counter.items(), key=operator.itemgetter(1))
    sorted_counter.reverse()
    for i in sorted_counter:
        rs += i[0] * i[1]
    return rs

if __name__ == '__main__':
	s = "Aabb"
	print(frequencySort2(s))

# 19,最少兔子的数量：数组中存储了森林中一些兔子告诉你的森林中有多少和他们一样颜色的兔子，返回森林中可能有的最少的兔子的数量
from collections import Counter
from math import ceil
def numRabbits(answers):
    C=Counter(answers)
    res=0
    #print(C)
    for key,cnt in C.items():
        key+=1
        res+=ceil(1.0*cnt/key)*key
    return int(res)

if __name__ == '__main__':
	answers = [1, 1, 2]
	print(numRabbits(answers))
	answers = [8, 8, 8]
	print(numRabbits(answers))

# 20,实现一个魔法字典：
class MagicDictionary(object):
    def _candidates(self, word):
        for i in xrange(len(word)):
            yield word[:i] + '*' + word[i+1:]
            
    def buildDict(self, words):
        self.words = set(words)
        self.near = collections.Counter(cand for word in words
                                        for cand in self._candidates(word))

    def search(self, word):  # 将word中的一个字母换成另一个字母，使得新单词存在于你构建的字典中
        return any(self.near[cand] > 1 or 
                   self.near[cand] == 1 and word not in self.words
                   for cand in self._candidates(word))

# 21,砌墙：求自顶向下一条直线，其路径穿过的brick最少
def leastBricks(wall):
    d = collections.defaultdict(int)
    for line in wall:
        i = 0
        for brick in line[:-1]:
            i += brick
            d[i] += 1
    # print len(wall), d
    return len(wall) - max(d.values() + [0])

if __name__ == '__main__':
	# wall = [[1,2,2,1],[3,1,2],[1,3,2],[2,4],[3,1,2],[1,3,1,1]]  # [i,j]为砖的长度，[i]为每层的砖
	# print(leastBricks(wall))

# 第二部分： HashMap的n种创建方式
# 1,MapBase：
class MapBase():
    
    class _Item:
        __slots__ = '_key' , '_value'
        
        def __init__ (self, k, v):
            self._key = k
            self._value = v
            
        def __eq__ (self, other):
            return self._key == other._key
        
        def __ne__ (self, other):
            return not (self == other)
        
        def __lt__ (self, other):
            return self._key < other._key
        
        def __print__ (self):
            print(str(self._key) + ":" + str(self._value),  end = ", ")

# 2,HashMapBase：
from random import randrange

class HashMapBase(MapBase):
    def __init__ (self, cap=11, p=109345121):
        self._table = cap * [ None ]
        self._n=0
        self._prime = p
        self._scale = 1 + randrange(p-1)
        self._shift = randrange(p)
        
    def _hash_function(self, k):
        return (hash(k) * self._scale + self._shift) % self._prime % len(self._table)  # index
    
    def __len__ (self):
        return self._n
    
    # O(1)
    def __getitem__ (self, k):
        j = self._hash_function(k) #index
        return self._bucket_getitem(j, k)
    
    # O(1)
    def __setitem__ (self, k, v):
        j = self._hash_function(k) #index
        print("hash for", k, "is", j)
        self._bucket_setitem(j, k, v)
        if self._n > len(self._table) // 2:  # keep load factor <= 0.5
            self.resize(2 * len(self._table) - 1)
           
    # O(1) 
    def __delitem__ (self, k):
        j = self._hash_function(k)
        self._bucket_delitem(j, k)
        self._table[j] = None
        self._n -= 1
    
    def resize(self, c):
        old = list(self.items( ))
        self._table = c * [None]
        self._n = 0
        for (k,v) in old:
            self[k] = v

# 3,UnsortedTableMap：
class UnsortedTableMap(MapBase):
    
    def __init__ (self):
        self._table = []
        
    def __getitem__ (self, k):
        for item in self._table:
            if k == item._key:
                return item._value
        raise ValueError( 'Key Error: ' + repr(k))
    
    def __setitem__ (self, k, v):
        for item in self._table:
            if k == item._key:
                item._value = v
                return
        self._table.append(self._Item(k,v))
    
    def __delitem__ (self, k):
        for j in range(len(self._table)):
            if k == self._table[j]._key:
                self._table.pop(j)
                return
        raise ValueError( 'Key Error: ' + repr(k))
    
    def __print__(self):
        for item in self._table:
            item.__print__()
        print('')

    def __len__(self):
        return len(self._table)
    
    def __iter__ (self):
        for item in self._table:
            yield item._key

# 4,SortedTableMap：
class SortedTableMap(MapBase):
    """Map implementation using a sorted table."""

    #----------------------------- nonpublic behaviors -----------------------------
    def _find_index(self, k, low, high):
        """Return index of the leftmost item with key greater than or equal to k.
        Return high + 1 if no such item qualifies.
        That is, j will be returned such that:
           all items of slice table[low:j] have key < k
           all items of slice table[j:high+1] have key >= k
        """
        if high < low:
            return high + 1                               # no element qualifies
        else:
            mid = (low + high) // 2 
            if k == self._table[mid]._key:
                return mid                                  # found exact match
            elif k < self._table[mid]._key:
                return self._find_index(k, low, mid - 1)    # Note: may return mid
            else:
                return self._find_index(k, mid + 1, high)   # answer is right of mid

    #----------------------------- public behaviors -----------------------------
    def __init__(self):
        """Create an empty map."""
        self._table = []

    def __len__(self):
        """Return number of items in the map."""
        return len(self._table)

    def __getitem__(self, k):
        """Return value associated with key k (raise KeyError if not found)."""
        j = self._find_index(k, 0, len(self._table) - 1)
        if j == len(self._table) or self._table[j]._key != k:
            raise KeyError('Key Error: ' + repr(k))
        return self._table[j]._value
  
    def __setitem__(self, k, v):
        """Assign value v to key k, overwriting existing value if present."""
        j = self._find_index(k, 0, len(self._table) - 1)
        if j < len(self._table) and self._table[j]._key == k:
            self._table[j]._value = v                         # reassign value
        else:
            self._table.insert(j, self._Item(k,v))            # adds new item
  
    def __delitem__(self, k):
        """Remove item associated with key k (raise KeyError if not found)."""
        j = self._find_index(k, 0, len(self._table) - 1)
        if j == len(self._table) or self._table[j]._key != k:
            raise KeyError('Key Error: ' + repr(k))
        self._table.pop(j)                                  # delete item
  
    def __iter__(self):
        """Generate keys of the map ordered from minimum to maximum."""
        for item in self._table:
            yield item._key

    def __reversed__(self):
        """Generate keys of the map ordered from maximum to minimum."""
        for item in reversed(self._table):
            yield item._key

    def find_min(self):
        """Return (key,value) pair with minimum key (or None if empty)."""
        if len(self._table) > 0:
            return (self._table[0]._key, self._table[0]._value)
        else:
            return None

    def find_max(self):
        """Return (key,value) pair with maximum key (or None if empty)."""
        if len(self._table) > 0:
            return (self._table[-1]._key, self._table[-1]._value)
        else:
            return None

    def find_le(self, k):
        """Return (key,value) pair with greatest key less than or equal to k.
        Return None if there does not exist such a key.
        """
        j = self._find_index(k, 0, len(self._table) - 1)      # j's key >= k
        if j < len(self._table) and self._table[j]._key == k:
            return (self._table[j]._key, self._table[j]._value)      # exact match
        elif j > 0:
            return (self._table[j-1]._key, self._table[j-1]._value)  # Note use of j-1
        else:
            return None

    def find_ge(self, k):
        """Return (key,value) pair with least key greater than or equal to k.
        Return None if there does not exist such a key.
        """
        j = self._find_index(k, 0, len(self._table) - 1)      # j's key >= k
        if j < len(self._table):
            return (self._table[j]._key, self._table[j]._value)
        else:
            return None

    def find_lt(self, k):
        """Return (key,value) pair with greatest key strictly less than k.
        Return None if there does not exist such a key.
        """
        j = self._find_index(k, 0, len(self._table) - 1)      # j's key >= k
        if j > 0:
            return (self._table[j-1]._key, self._table[j-1]._value)  # Note use of j-1
        else:
            return None

    def find_gt(self, k):
        """Return (key,value) pair with least key strictly greater than k.
        Return None if there does not exist such a key.
        """
        j = self._find_index(k, 0, len(self._table) - 1)      # j's key >= k
        if j < len(self._table) and self._table[j]._key == k:
            j += 1                                       # advanced past match
        if j < len(self._table):
            return (self._table[j]._key, self._table[j]._value)
        else:
            return None

    def find_range(self, start, stop):
        """Iterate all (key,value) pairs such that start <= key < stop.
        If start is None, iteration begins with minimum key of map.
        If stop is None, iteration continues through the maximum key of map.
        """
        if start is None:
            j = 0
        else:
            j = self._find_index(start, 0, len(self._table)-1)   # find first result
        while j < len(self._table) and (stop is None or self._table[j]._key < stop):
            yield (self._table[j]._key, self._table[j]._value)
            j += 1
            
    def _print_(self):
        for item in self._table:
            item.__print__()
        print('')

# 5,ProbeHashMap：
class ProbeHashMap(HashMapBase):
    """Hash map implemented with linear probing for collision resolution."""
    _AVAIL = object()       # sentinal marks locations of previous deletions

    def _is_available(self, j):
        """Return True if index j is available in table."""
        return self._table[j] is None or self._table[j] is ProbeHashMap._AVAIL

    def _find_slot(self, j, k):
        """Search for key k in bucket at index j.
        Return (success, index) tuple, described as follows:
        If match was found, success is True and index denotes its location.
        If no match found, success is False and index denotes first available slot.
        """
        firstAvail = None
        while True:                               
            if self._is_available(j):
                if firstAvail is None:
                    firstAvail = j                      # mark this as first avail
                if self._table[j] is None:
                    return (False, firstAvail)          # search has failed
            elif k == self._table[j]._key:
                return (True, j)                      # found a match
            j = (j + 1) % len(self._table)          # keep looking (cyclically)

    def _bucket_getitem(self, j, k):
        found, s = self._find_slot(j, k)
        if not found:
            raise KeyError('Key Error: ' + repr(k))        # no match found
        return self._table[s]._value

    def _bucket_setitem(self, j, k, v):
        found, s = self._find_slot(j, k)
        if not found:
            self._table[s] = self._Item(k,v)               # insert new item
            self._n += 1                                   # size has increased
        else:
            self._table[s]._value = v                      # overwrite existing

    def _bucket_delitem(self, j, k):
        print(j, k)
        found, s = self._find_slot(j, k)
        print(found, s)
        if not found:
            raise KeyError('Key Error: ' + repr(k))        # no match found
        self._table[s] = ProbeHashMap._AVAIL             # mark as vacated

    def __iter__(self):
        for j in range(len(self._table)):                # scan entire table
            if not self._is_available(j):
                yield self._table[j]._key
                
    def _print_ (self):
        for bucket in self._table:
            if bucket is not None: # a nonempty slot
                bucket.__print__()

# 6,ChainHashMap：
class ChainHashMap(HashMapBase):
 
    def _bucket_getitem(self, j, k):
        bucket = self._table[j]
        if bucket is None:
            raise ValueError( 'Key Error: ' + repr(k)) # no match found
        return bucket[k] # may raise KeyError

    def _bucket_setitem(self, j, k, v):
        if self._table[j] is None:
            self._table[j] = UnsortedTableMap( ) # bucket is new to the table
        oldsize = len(self._table[j])
        self._table[j][k] = v
        if len(self._table[j]) > oldsize: # key was new to the table
            self._n += 1 # increase overall map size

    def _bucket_delitem(self, j, k):
        bucket = self._table[j]
        if bucket is None:
            raise KeyError( 'Key Error: ' + repr(k)) # no match found
        del bucket[k] # may raise KeyError

    def __iter__ (self):
        for bucket in self._table:
            if bucket is not None: # a nonempty slot
                for key in bucket:
                    yield key
                    
    def _print_ (self):
        for bucket in self._table:
            if bucket is not None: # a nonempty slot
                bucket.__print__()