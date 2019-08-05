# leetcode-121:
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if len(prices) < 2:
            return 0
        prof = 0
        min_p = prices[0]
        for i in prices:
            min_p = min(i, min_p)
            prof = max(i - min_p, prof)
        return prof
        
# 暴力解法（超时）：
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
	    if not prices:
	    	return 0   # 如果prices为空[]，则返回0.

	    l = len(prices)
	    res = []
	    
	    for i in range(l):
	        for j in range(i+1, l):
	            res.append(prices[j] - prices[i])
	    
	    if not res:
	    	return 0   # 如果res为空[]，则返回0.
	    else：
	    	return max(max(res), 0)