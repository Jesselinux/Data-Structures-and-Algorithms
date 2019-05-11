# leetcode-309:动态规划，beats 95.44%
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if len(prices) <= 1:
            return 0
        buy, pre_buy, sell, pre_sell = -max(prices), 0, 0, 0
        for p in prices:
            pre_buy = buy
            buy = max(pre_sell - p, pre_buy)
            pre_sell = sell
            sell = max(pre_buy + p, pre_sell)
        return sell