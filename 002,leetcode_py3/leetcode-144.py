# leetcode-144：二叉树前序遍历

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:  # 递归写法，beats 12.29%
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        result = []
        if not root:
            return result
        result.append(root.val)
        result += self.preorderTraversal(root.left)
        result += self.preorderTraversal(root.right)
        return result


class Solution:  # 非递归写法，beats 84.21%
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        result = []
        s = []
        if not root:
            return result
        s.append(root)
        while s:
            root = s.pop()
            result.append(root.val)
            if root.right:
                s.append(root.right)
            if root.left: 
                s.append(root.left) 
        return result