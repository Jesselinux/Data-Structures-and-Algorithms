# leetcode-241:分治思想
class Solution:  # 使用递归实现分治思想，战胜了 63.59% 
    def diffWaysToCompute(self, input: str) -> List[int]:
        return_list = []
        for i in range(len(input)):
            c = input[i]
            if c in ['+','-','*']:
                left = self.diffWaysToCompute(input[:i])
                right = self.diffWaysToCompute(input[i+1:])
                for l in left:
                    for r in right:
                        if c == '+':
                            return_list.append(l+r)
                        elif c == '-':
                            return_list.append(l-r)
                        elif c == '*':
                            return_list.append(l*r)
        if not return_list:
            return_list.append(int(input))

        return return_list