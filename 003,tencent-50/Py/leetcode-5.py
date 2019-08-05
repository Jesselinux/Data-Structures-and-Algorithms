class Solution:
    def isPalindrome(self, s):
        i = 0
        j = len(s) - 1
        while i < j:
            if s[i] != s[j]:
                return False
            i += 1
            j -= 1
        return True
    
    def getSubPalindrome(self, c, palin):
        p = palin
        i = p.find(c)
        while i != -1:
            if self.isPalindrome(p[i:] + c):
                return p[i:]
            p = p[(i+1):]
            i = p.find(c)
        return ""
            
    def longestPalindrome(self, s: str) -> str:    
        """
        :type s: str
        :rtype: str
        """
        if s == "":
            return ""
        L = [s[0]]
        maxLen = len(s[0])
        mIndex = 0
        for i in range(1, len(s)):
            prevLen = len(L[i-1])
            if prevLen > maxLen:
                maxLen = prevLen
                mIndex = i-1
            if (i - (prevLen+1)) >= 0 and s[(i - (prevLen+1))] == s[i]:
                L.append(s[i] + L[i-1] + s[i])
            else:
                L.append(self.getSubPalindrome(s[i], L[i-1]) + s[i])
        
        if len(L[-1]) > maxLen:
            mIndex = len(L) - 1

        return L[mIndex]