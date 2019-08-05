class Solution:
    def reverseWords(self, s: str) -> str:
        rs = s[::-1]
        l = rs.split()
        ls = [j[::-1] for j in l]
        return ' '.join(ls)