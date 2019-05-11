# leetcode-104:
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        else:
            l = 1 + self.maxDepth(root.left)   # 递归
            r = 1 + self.maxDepth(root.right)
            return max(l, r)