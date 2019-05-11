class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        frequent_of_number = {}
        for num in nums:
            frequent_of_number[num] = frequent_of_number.get(num, 0) + 1
        buckets = [[] for i in range(len(nums) + 1)]
        for key, value in frequent_of_number.items():
            buckets[value].append(key)
        print(buckets)
        result = []
        for x in range(len(nums), -1, -1):
            if k > 0 and buckets[x]:
                result += buckets[x]
                k -= len(buckets[x])
            if k == 0:
                return result

class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
    	di = {}
    	for num in nums:
    		if num not in di:
    			di[num] = 1
    		else:
    			di[num] += 1
    	bucket = [[] for _ in range(len(nums) + 1)]
    	for key, value in di.items():
    		bucket[value].append(key)
    	ans = []
    	for i in range(len(nums), -1, -1):
    		if bucket[i]:
    			ans.extend(bucket[i])
    		if len(ans) >= k:
    			break
    	return ans[:k]