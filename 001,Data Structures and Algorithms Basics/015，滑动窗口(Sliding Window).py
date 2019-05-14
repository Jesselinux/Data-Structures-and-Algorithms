# 1,删除重复元素：
def removeDuplicates(alist):
    if not alist:
        return 0

    tail = 0

    for j in range(1, len(alist)):
        if alist[j] != alist[tail]:
            tail += 1
            alist[tail] = alist[j]

    return tail + 1

def removeDuplicates(alist):
    i = 0
    for n in alist:
        if i == 0 or n > alist[i - 1]:
            alist[i] = n
            i += 1
    return i

if __name__ == '__main__':
	alist = [0,0,1,1,1,2,2,3,3,4]
	print(removeDuplicates(alist))

# 2,删除后，重复值不超过两个：
def removeDuplicates(nums):
    if len(nums) <= 2: return len(nums)
    prev, curr = 1, 2
    while curr < len(nums):
        if nums[curr] == nums[prev] and  nums[curr] == nums[prev - 1]:
            curr += 1
        else:
            prev += 1
            nums[prev] = nums[curr]
            curr += 1
    return prev + 1

def removeDuplicates2(nums):
    count = 0
    for i in range(len(nums)):
        if count < 2 or nums[count - 2] != nums[i]:
            nums[count] = nums[i]
            count += 1
    return count

# 3,删除元素：给定一个数组nums和一个值val, 就地（in-palce）删除这个val的所有实 例，并返回新的数组的长度。
def removeElement(nums, val):
    i = 0
    for j in range(len(nums)):
        if nums[j] != val:
            nums[i] = nums[j]
            i += 1
    return i        

def removeElement2(nums, val):
    start, end = 0, len(nums) - 1
    while start <= end:
        if nums[start] == val:
            nums[start], end = nums[end], end - 1
        else:
            start +=1
    return start

if __name__ == '__main__':
	nums = [ 0, 1, 2, 2, 3, 0, 4, 2 ]
	removeElement(nums, 2)

# 4,最大均值子数组： 给定一个包含n个整数的数组，找到长度为k的平均值最大的连续子数组，返回最大平均值 
def findMaxnumsverage(nums, K):
    P = [0]
    for x in nums:
        P.append(P[-1] + x)

    moving_sum = max(P[i+K] - P[i] 
             for i in range(len(nums) - K + 1))
    return moving_sum / float(K)

def findMaxnumsverage2(nums, K):
    moving_sum = 0.0
    for i in range(K):
        moving_sum += nums[i]
    res = moving_sum
    for i in range(K, len(nums)):
        moving_sum += nums[i] - nums[i - K]
        res = max(res, moving_sum)
    return res / K

if __name__ == '__main__':
	nums = [ 1, 12, -5, -6, 50, 3 ]
	findMaxnumsverage2(nums, 4)

# 5,最长连续递增子序列：给定一个没排序的整数数组，找到最长的连续递增的子序列子数组的长度
def findLengthOfLCIS(nums):
    result, count = 0, 0
    for i in range(len(nums)):
        if i == 0 or nums[i-1] < nums[i]:
            count += 1
            result = max(result, count)
        else:
            count = 1
    return result

if __name__ == '__main__':
	nums = [1,3,5,4,7]
	findLengthOfLCIS(nums)

# 6,最短子数组之和: 给定一个包含n个正整数的数组和一个正整数s，找到一个长度最小的连续子数组，这个子数组的元素和大于等于s
def minsubarray(alist, target):
    if len(alist) == 0:
        return 0
    
    i = j = sums = 0
    minimum = sys.maxsize
    
    while j < len(alist):
        sums += alist[j]
        j += 1
        while sums >= target:
            minimum = min(minimum, j - i)
            sums -= alist[i]
            i += 1
    return 0 if min == sys.maxsize else minimum

# 7,实现strStr()函数: 返回子字符串needle在字符串haystack中第一次出现的位置，没有则返回-1 
def strStr(haystack, needle):
    if len(haystack) < len(needle): 
        return None
    i = 0
    while i < len(haystack) - len(needle) + 1:
        j = 0
        k = i
        while j < len(needle):
            if haystack[k] == needle[j]:
                j+=1; k+=1
            else:
                break
        if j == len(needle):
            break
        else:
            i+=1
    if i == len(haystack)-len(needle)+1:
        return None
    else:
        return haystack[i:]

def strStr2(haystack, needle):
    if len(haystack) < len(needle): 
        return None
    l1 = len(haystack)
    l2 = len(needle)
    for i in range(l1 - l2 + 1):
        count = 0
        while count < l2 and haystack[i + count] == needle[count]:
            count += 1
        if count == l2:
            return i
    return -1

# 8,子数组乘积小于K:
def bruteforce(nums, k):
    count = 0
    for i in range(len(nums)):
        product = 1
        for j in range(i, len(nums)):
            product *= nums[j]
            if (product >= k): break
            count += 1
    return count

def numSubarrayProductLessThanK(nums, k):
    product = 1
    i = 0
    ans = 0
    for j, num in enumerate(nums):
        product *= num
        while product >= k:
            product /= nums[i]
            i += 1
        ans += (j - i + 1)
    
    return ans


if __name__ == '__main__':
	nums = [1,5,4,3,6,2,7]
	k = bruteforce(nums, 100)
	print(k)
	nums = [10, 5, 2, 6]
	k = numSubarrayProductLessThanK(nums, 100)
	print(k)

# 9,不含重复字符的最长子串: 
def lengthOfLongestSubstring(s):
    usedChar = set()
    maxLength = 0
    i = j = 0
    n = len(s)
    while (i < n and j < n):
        if s[j] not in usedChar:
            usedChar.add(s[j])
            j += 1
            maxLength = max(maxLength, j - i)
        else:
            usedChar.remove(s[i])
            i += 1
    return maxLength

def lengthOfLongestSubstring2(s):
    start = maxLength = 0
    usedChar = {}

    for i, c in enumerate(s):
        if c in usedChar and start <= usedChar[c]:
            start = usedChar[c] + 1
        else:
            maxLength = max(maxLength, i - start + 1)

        usedChar[c] = i

    return maxLength

if __name__ == '__main__':
	s = 'pwwkew'
	lengthOfLongestSubstring(s)

# 10,查找重组子串：给定一个字符串s和一个非空字符串p，找到所有p的重组字符串在s中出现的初始位置 
import collections
def findAnagrams(s, p):
    begin, end = 0, 0
    count = len(p)
    ans = []
    d = collections.Counter(p)

    while end < len(s):
        # map[char]>=1, meaning the new char is in p, then count--
        if d[s[end]] > 0:
            count -= 1
        d[s[end]] -= 1
        end += 1

        # find an anagram
        if count == 0:
            ans.append(begin)

        # find a window, then advance begin to shrink the window
        if end - begin == len(p):
            # advance begin
            d[s[begin]] += 1
            begin += 1
            # # map[char]>=1, meaning the exit char is in p, then count++
            if d[s[begin-1]] >= 1:
                count += 1

    return ans

from collections import Counter

def findAnagrams2(s, p):
    res = []
    pCounter = Counter(p)
    sCounter = Counter(s[:len(p)-1])
    for i in range(len(p)-1,len(s)):
        sCounter[s[i]] += 1   # include a new char in the window
        if sCounter == pCounter:    # This step is O(1), since there are at most 26 English letters 
            res.append(i-len(p)+1)   # append the starting index
        sCounter[s[i-len(p)+1]] -= 1   # decrease the count of oldest char in the window
        if sCounter[s[i-len(p)+1]] == 0:
            del sCounter[s[i-len(p)+1]]   # remove the count if it is 0
    return res

if __name__ == '__main__':
	s = "cbaebabacd"
	p = "abc"
	findAnagrams(s, p)

# 11,最小窗口子串：
import sys
def minWindow(s, t):
    if len(t) > len(s):
        return ""
    lt = len(t)
    count = lt
    ct = collections.Counter(t)
    left = 0
    right = 0
    minLength = sys.maxsize
    notfound = 1
    ansleft = 0
    ansright = 0
    print(ct)
    for i in range(len(s)):
        # found in t
        if ct[s[i]] > 0:
            count -= 1
        ct[s[i]] -= 1
        #print(s[i], ct)
        # found a window, containing all chars from t
        while count == 0:
            right = i
            notfound = 0
            if right - left < minLength:
                minLength = right-left
                ansleft = left
                ansright = right
            # when map[left] is 0, meaning the exit char is in t, then count++
            if ct[s[left]] == 0:
                count += 1
            ct[s[left]] += 1
            #print("left: ", s[left], ct)
            left += 1
    if notfound == 1:
        return ""
    return s[ansleft:ansright+1]

if __name__ == '__main__':
	s = "ADOBECODEBANC"
	t = "ABC"
	minWindow(s, t)

# 12,最多有K个不同字符的最长子串：
def lengthOfLongestSubstringKDistinct(s, k):
    longest, start, distinct_count, visited = 0, 0, 0, [0 for _ in range(256)]
    for i, char in enumerate(s):
        if visited[ord(char)] == 0:
            distinct_count += 1
        visited[ord(char)] += 1

        while distinct_count > k:
            visited[ord(s[start])] -= 1
            if visited[ord(s[start])] == 0:
                distinct_count -= 1
            start += 1

        longest = max(longest, i - start + 1)
    return longest

def lengthOfLongestSubstringKDistinct2(s, k):
    start = 0
    longest = 0
    char_dict = {}


    for index in range(len(s)):
        char = s[index]
        char_dict[char] = char_dict.get(char, 0) + 1  # track count of chars

        # decrease the size of sliding window until you have k unique chars in sliding window
        while len(char_dict) > k: 
            char_dict[s[start]] -= 1
            if char_dict[s[start]] == 0:
                del char_dict[s[start]]
            start += 1

        longest = max(longest, index+1-start)

    return longest

if __name__ == '__main__':
	nums = 'eceba'
	k = 2
	lengthOfLongestSubstringKDistinct(nums, k)

# 13,滑动窗口最大值：
def maxSlidingWindow(nums, k):
    d = collections.deque()
    out = []
    for i, n in enumerate(nums):
        while d and nums[d[-1]] < n:
            d.pop()
        d += i,
        if d[0] == i - k:
            d.popleft()
        if i >= k - 1:
            out += nums[d[0]],
    return out

if __name__ == '__main__':
	nums = [1,3,-1,-3,5,3,6,7]
	k = 3
	maxSlidingWindow(nums, k)