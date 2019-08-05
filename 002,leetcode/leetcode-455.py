class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        g = sorted(g)
        s = sorted(s)
        cnt_g = 0
        cnt_s = 0
        while cnt_g < len(g) and cnt_s < len(s):
            if g[cnt_g] <= s[cnt_s]:
                cnt_g += 1
            cnt_s += 1
        return cnt_g

class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        g.sort()
        s.sort()
        child = 0
        cookies = 0

        while len(g) > child and len(s) > cookies:
        	if g[child] <= s[cookies]:
        		child += 1
        	cookies += 1
        return child