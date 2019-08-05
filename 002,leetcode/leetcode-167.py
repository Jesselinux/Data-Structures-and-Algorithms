class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        i = 0
        j = len(numbers) - 1
        while i < j:
            if numbers[i] + numbers[j] > target:
                j -= 1
            elif numbers[i] + numbers[j] < target:
                i += 1
            else:
                return [i+1, j+1]

class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        a = 0
        b = len(numbers) -1

        while a < b:
            if numbers[a] + numbers[b] < target:
                a += 1
            elif numbers[a] + numbers[b] > target:
                b -= 1
            elif numbers[a] + numbers[b] == target:
                return [a+1, b+1]
        else:
            return None

class Solution:
	def twoSum(self, numbers: List[int], target: int) -> List[int]:
		length = len(numbers)
		for i in range(0, length-1):
			for j in range(i+1, length):
				a = numbers[i]
				b = numbers[j]
				sum = a +b
				if sum == target:
					break
			if sum == target:
				break
		return i+1, j+1