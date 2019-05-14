class Solution:
	def isValid(self, s):
		pre_dict = {')':'(', ']':'[', '}':'{'}
		stack = []
		for i in s:
			if i in pre_dict.values():
				stack.append(i)
			elif not stack or pre_dict[i] != stack.pop():
				return False
		return not stack
