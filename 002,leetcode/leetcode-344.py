class Solution:
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        k = len(s) - 1
        for j in range(k+1):
            if j == k or j > k:
                break
            s[j], s[k] = s[k], s[j]
            j += 1
            k -= 1

class Solution:
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        k = len(s)
        for i in range(k // 2):
            s[i], s[k-1-i] = s[k-1-i], s[i]

class Solution:
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        s[:] = s[::-1]
        