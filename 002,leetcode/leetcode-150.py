class Solution:
	def evalRPN(self, tokens):
		stack = list()
		oper = ['+', '-', '*', '/']
		for char in tokens:
			if char not in oper:
				stack.append(int(char))
			else:
				top1 = stack.pop()
				top2 = stack.pop()
				if char == '+':
					stack.append(top2 + top1)
				elif char == '-':
					stack.append(top2 - top1)
				elif char == '*':
					stack.append(top2 * top1)
				elif char =='/':
					stack.append(int(top2 / top1))
		return stack.pop()

		