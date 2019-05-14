# 1,偶数子串的数量：
def evenNum(s):
    count = 0
    for i in range(len(s)):
        if int(s[i]) % 2 == 0:
            count += i + 1
    return count

if __name__ == '__main__':
	evenNum("1234")

# 2,出勤记录：
def checkRecord(self, s):
    return not (s.count('A') > 1 or 'LLL' in s)


# 3,对具有相同首尾字符的子字符进行计数：
def countSub(s):
    result = 0
    for i in range(len(s)):
        for j in range(i, len(s)):
            if (s[i] == s[j]):
                result += 1
    return result

from collections import Counter

def countSub(s):
    counter = Counter(s)
    result = 0
    for x in counter:
        result += counter[x] * (counter[x] + 1) // 2
        
    return result

if __name__ == '__main__':
	s = "abcab"
	countSub(s)

# 4,字符串中大连续重复字符:
def maxRepeating(s):
    n = len(s)
    count = 0
    result = s[0]
    local = 1
    
    for i in range(n):
        if (i < n - 1 and s[i] == s[i+1]):
            local += 1
        else:
            if (local > count):
                count = local
                result = s[i]
            local = 1
    return result

def removeDuplicates(A):
    if not A:
        return 0

    newTail = 0
    for i in range(1, len(A)):
        if A[i] != A[newTail]:
            newTail += 1
            A[newTail] = A[i]
            
    for j in range(newTail + 1, len(A)):
        A[j] = 'X'

    print(A)
    return newTail + 1

if __name__ == '__main__':
	s = "aaaabbaacccccde"
	maxRepeating(s)
	A = [0,0,1,1,1,2,2,3,3,4]
	removeDuplicates(A)

# 5,字谜：字符串包含相同的元素
def areAnagram(str1, str2):
    if len(str1) != len(str2):
        return False
    return sorted(str1) == sorted(str2)

def areAnagram2(str1, str2):
    NO_OF_CHARS = 256
    # Create two count arrays and initialize all values as 0
    count1 = [0] * NO_OF_CHARS
    count2 = [0] * NO_OF_CHARS
 
    # For each character in input strings, increment count
    # in the corresponding count array
    for i in str1:
        count1[ord(i)]+=1
 
    for i in str2:
        count2[ord(i)]+=1
 
    # If both strings are of different length. Removing this
    # condition will make the program fail for strings like
    # "aaca" and "aca"
    if len(str1) != len(str2):
        return False
 
    # Compare count arrays
    for i in range(NO_OF_CHARS):
        if count1[i] != count2[i]:
            return False
 
    return True

from collections import Counter
def areAnagram3(str1, str2):
    return Counter(str1) == Counter(str2)

if __name__ == '__main__':
	s1 = "listen"
	s2 = "silent"
	areAnagram(s1, s2)

# 6,查找字谜：
from collections import Counter

def findAnagrams(s, p):
    res = []
    pCounter = Counter(p)
    sCounter = Counter(s[:len(p) - 1])
    for i in range(len(p) - 1,len(s)):
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

# 7,字谜匹配：
def anagramMappings1(A, B):
    answer = []
    for a in A:
        for i,b in enumerate(B):
            if a == b:
                answer.append(i)
                break
    return answer

def anagramMappings2(A, B):
    return [B.index(a) for a in A]

def anagramMappings3(A, B):
    d = {}
    for i,b in enumerate(B):
        d[b] = i
    return [d[a] for a in A]

if __name__ == '__main__':
	A = [12, 28, 46, 32, 50]
	B = [50, 12, 32, 46, 28]
	anagramMappings3(A, B)

# 8,旋转字符串：
def areRotations(string1, string2):
    size1 = len(string1)
    size2 = len(string2)

    if size1 != size2:
        return 0
 
    temp = string1 + string1
 
    return temp.count(string2) > 0

if __name__ == '__main__':
	string1 = "AACD"
	string2 = "ACDA"
	areRotations(string1, string2)

# 9,旋转字符串(2)：
def reverse(arr, start, end):
    while start < end:
        arr[start], arr[end] = arr[end], arr[start]
        start += 1
        end -= 1

def rotate(arr, d):
    n = len(arr)
    reverse(arr, 0, d - 1)
    reverse(arr, d, n - 1)
    reverse(arr, 0, n - 1)

if __name__ == '__main__':
	arr = [1, 2, 3, 4, 5, 6, 7]
	rotate(arr, 2)
	
# 10,回文：
def reverse(s):
    return s[::-1]
 
def isPalindrome(s):
    # Calling reverse function
    rev = reverse(s)
 
    # Checking if both string are equal or not
    if (s == rev):
        return True
    return False

def isPalindrome2(s):
    return s == s[::-1]

def isPalindrome3(s):
    for i in range(len(s) // 2):
        if s[i] != s[- 1 - i]:
            return False

    return True

if __name__ == '__main__':
	s = "malayalam"
	print(isPalindrome(s))

# 11,回文数字：
def intPalindrome(n):
    return str(n) == str(n)[::-1]

def isPalindrome2(x):
    if x < 0:
        return False

    ranger = 1
    while x // ranger >= 10:
        ranger *= 10
    print(ranger)
    while x:
        left = x // ranger
        right = x % 10
        if left != right:
            return False

        x = (x % ranger) // 10
        ranger //= 100

    return True

if __name__ == '__main__':
	print(isPalindrome2(1221))

# 12,旋转回文：
def isRotationOfPalindrome(s):
 
    # If string itself is palindrome
    if isPalindrome(s):
        return True
 
    # Now try all rotations one by one
    n = len(s)
    for i in range(len(s) - 1):
        s1 = s[i+1:n]
        s2 = s[0:i+1]
 
        # Check if this rotation is palindrome
        s1 += s2
        if isPalindrome(s1):
            return True
 
    return False

def isRotationOfPalindrome2(s):
    n = len(s)
    s = s + s
    for i in range(n):
        if isPalindrome(s[i : i + n]):
            return True
    return False

if __name__ == '__main__':
	print(isRotationOfPalindrome("aaaad"))

# 13,可以重置成回文：
from collections import Counter
def canRearrage(s):
    odd = 0
    counter = Counter(s)
    for key in counter.keys():
        if counter[key] % 2 == 1:
            odd += 1
        if odd > 1:
            return False
    return True

if __name__ == '__main__':
	print(canRearrage("ababcbaab"))
	print(canRearrage("ababcbaa"))

# 14,可以组成回文的最大长度：
from collections import Counter
def longestPalindrome(s):
    ans = 0
    counter = Counter(s)
    for key in counter.keys():
        v = counter[key]
        ans += v // 2 * 2
        if ans % 2 == 0 and v % 2 == 1:
            ans += 1
    return ans

# 15,数据流中的回文：
# d is the number of characters in input alphabet
d = 256
 
# q is a prime number used for evaluating Rabin Karp's
# Rolling hash
q = 103
 
def checkPalindromes(string):
 
    # Length of input string
    N = len(string)
 
    # A single character is always a palindrome
    print(string[0] + " Yes")
 
    # Return if string has only one character
    if N == 1:
        return
 
    # Initialize first half reverse and second half for
    # as firstr and second characters
    firstr = ord(string[0]) % q
    second = ord(string[1]) % q
 
    h = 1
    i = 0
    j = 0
 
    # Now check for palindromes from second character
    # onward
    for i in range(1,N):
 
        # If the hash values of 'firstr' and 'second'
        # match, then only check individual characters
        if firstr == second:
 
            # Check if str[0..i] is palindrome using
            # simple character by character match
            for j in range(0,i//2):
                if string[j] != string[i-j]:
                    break
            j += 1
            if j == i//2:
                print(string[i] + " Yes")
            else:
                print(string[i] + " No")
        else:
            print(string[i] + " No")
 
        # Calculate hash values for next iteration.
        # Don't calculate hash for next characters if
        # this is the last character of string
        if i != N-1:
 
            # If i is even (next i is odd)
            if i % 2 == 0:
 
                # Add next character after first half at
                # beginning of 'firstr'
                h = (h*d) % q
                firstr = (firstr + h*ord(string[i//2]))%q
 
                # Add next character after second half at
                # the end of second half.
                second = (second*d + ord(string[i+1]))%q
            else:
                # If next i is odd (next i is even) then we
                # need not to change firstr, we need to remove
                # first character of second and append a
                # character to it.
                second = (d*(second + q - ord(string[(i+1)//2])*h)%q
                            + ord(string[i+1]))%q

if __name__ == '__main__':
	checkPalindromes("aabaacaabaa")

# 16,最长子序列：
import collections

def longestSub(s, k):
    result = list()
    c = collections.Counter(s)
    for i in s:
        if (c[i] >= k):
            result.append(i)
    return "".join(result)

if __name__ == '__main__':
	s = "baaabaacba"
	k = 3
	longestSub(s, k)

# 17,检查子序列：
def isSubSequence(string1, string2, m, n):
    # Base Cases
    if m == 0:    return True
    if n == 0:    return False
 
    # If last characters of two strings are matching
    if string1[m-1] == string2[n-1]:
        return isSubSequence(string1, string2, m-1, n-1)
 
    # If last characters are not matching
    return isSubSequence(string1, string2, m, n-1)

def isSubSequence2(str1, str2):
    m = len(str1)
    n = len(str2)
    j = 0   # Index of str1
    i = 0   # Index of str2
    while j < m and i < n:
        if str1[j] == str2[i]:  
            j = j + 1
        i = i + 1
         
    return j == m

if __name__ == '__main__':
	str1 = "AXY"
	str2 = "ADXCPY"
	isSubSequence2(str1, str2)

# 18,字典中最长的单词：
def findLongestString(words, s):
    result = ""
    length = 0
    for w in words:
        if length < len(w) and isSubSequence2(w, s):
            result = w
            length = len(w)
    return result

if __name__ == '__main__':
	words = ["ale", "apple", "monkey", "plea"]
	s = "abpcplea"
	findLongestString(words, s)

# 19,所有之列元素和的加和：
def sumSub(arr):
    ans = sum(arr)
    return ans * pow(2, len(arr) - 1)

if __name__ == '__main__':
	arr = [5, 6, 8]
	print(sumSub(arr))

# 20,搜索类型：
def strStr(text, pattern):
    for i in range(len(text) - len(pattern)+1):
        if text[i:i+len(pattern)] == pattern:
            return i
    return -1

def strStr2(text, pattern):
    """ 
    Brute force algorithm.
    Time complexity: O(n * m). Space complexity: O(1),
    where m, n are the lengths of text and pattern.
    """
    n, m = len(text), len(pattern)
    for i in range(n - m + 1):
        start = i
        for j in range(m):
            if text[i] != pattern[j]:
                break
            i += 1
        else:  # no break occured, i.e. match was found
            return start
    return -1

# Rolling Hash

def strStr3(text, pattern):
    base = 29
    patternHash = 0
    tempBase = 1
    hayHash = 0
    
    for i in range(len(pattern) - 1, -1, -1):
        patternHash += ord(pattern[i]) * tempBase
        tempBase *= base
        
    tempBase = 1
    for i in range(len(pattern) - 1, -1, -1):
        hayHash += ord(text[i]) * tempBase
        tempBase *= base    

    if patternHash == hayHash and text[0:len(pattern)] == pattern:
        return 0
    
    tempBase /= base
    for i in range(len(pattern), len(text)):
        hayHash = (hayHash - ord(text[i - len(pattern)]) * tempBase) * base + ord(text[i])
        if hayHash == patternHash and text[i-len(pattern)+1:i+1] == pattern:
            return i - len(pattern) + 1
                                  
    return -1
                                  
if __name__ == '__main__':
	text = "THIS IS A TEST TEXT"
	pattern = "TEST"
	strStr(text, pattern)

# 21,敏感词替换：
def censor(text, word):
    word_list = text.split()
 
    result = ''
 
    stars = '*' * len(word)
 
    count = 0
 
    index = 0;
    for i in word_list:
        if i == word:
            word_list[index] = stars
        index += 1
 
    # join the words
    result =' '.join(word_list)
 
    return result

if __name__ == '__main__':
	word = "Barcelona"
	text = "It wasn't any place close to a vintage performance but Barcelona eventually overcame \
	Deportivo La Coruna and secured the La Liga title for the 25th time. It is the 9th La \
	Liga win for Barcelona in the last 14 seasons and the 7th in the last 10. In the last \
	ten, only Real Madrid twice and Atletico Madrid have broken Barcelona run of Liga success."

	censor(text, word)

# 22,字符串替换：
def translate(st) :
    l = len(st)
     
    if (l < 2) :
        return
 
    i = 0 # Index in modified string
    j = 0 # Index in original string
 
    while (j < l - 1) :
        # Replace occurrence of "AB" with "C"
        if (st[j] == 'A' and st[j + 1] == 'B') :
             
            # Increment j by 2
            j += 2
            st[i] = 'C'
            i += 1
            continue
         
        st[i] = st[j]
        i += 1
        j += 1
 
    if (j == l - 1) :
        st[i] = st[j]
        i += 1
 
    # add a null character to
    # terminate string
    return i

if __name__ == '__main__':
	st = list("helloABworldABGfGAAAB")
	length = translate(st)
	for i in range(length):
	    print(st[i])

# 23,模式计数：
def patternCount(s):
    last = s[0]
    i = 1
    counter = 0
    while (i < len(s)):
         
        # We found 0 and last character was '1',
        # state change
        if (s[i] == '0' and last == '1'):
            while (s[i] == '0' and i < len(s)):
                i += 1
                if (i == len(s)):
                    return counter
                # After the stream of 0's, we got a '1',
                # counter incremented
                if (s[i] == '1'): 
                    counter += 1
         
        # Last character stored 
        last = s[i]
        i += 1
     
    return counter

if __name__ == '__main__':
	s = "100001abc1010100"
	print(patternCount(s))

# 24,匹配字符串：
def match(first, second):
    if len(first) == 0 and len(second) == 0:
        return True
 
    # Make sure that the characters after '*' are present
    # in second string. This function assumes that the first
    # string will not contain two consecutive '*'
    if len(first) > 1 and first[0] == '*' and len(second) == 0:
        return False
 
    # If the first string contains '?', or current characters
    # of both strings match
    if (len(first) > 1 and first[0] == '?')  \
        or (len(first) != 0  \
        and len(second) !=0 and first[0] == second[0]):
        return match(first[1:],second[1:]);
 
    # If there is *, then there are two possibilities
    # a) We consider current character of second string
    # b) We ignore current character of second string.
    if len(first) !=0 and first[0] == '*':
        return match(first[1:],second) or match(first,second[1:])

if __name__ == '__main__':
	print(match("abc*c?d", "abcd")) # No because second must have 2 instances of 'c'
	print(match("*c*d", "abcd")) # Yes
	print(match("*?c*d", "abcd")) # Yes