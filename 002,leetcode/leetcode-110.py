# leetcode-110：平衡二叉树
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:  # beats 19.84%
    def isBalanced(self, root: TreeNode) -> bool:
        if root is None:
            return True
        left_height = self.getHeight(root.left)
        right_height = self.getHeight(root.right)
        if abs(left_height - right_height) > 1:
            return False
        return self.isBalanced(root.left) and self.isBalanced(root.right)
    
    def getHeight(self, root: TreeNode) -> int:
        if root is None:
            return 0
        left_height = self.getHeight(root.left)
        right_height = self.getHeight(root.right)
        return max(left_height, right_height) +1


class Solution:  # 这个解决方案的逻辑错误点在哪？
    def _maxDepth(self, node: TreeNode) -> int:
        if (not node):
            return 0
        left_depth = self._maxDepth(node.left)
        right_depth = self._maxDepth(node.right)
        return max(left_depth, right_depth) + 1

    def _minDepth(self, node: TreeNode) -> int:
        if (not node):
            return 0
        left_depth = self._minDepth(node.left)
        right_depth = self._minDepth(node.right)
        return min(left_depth, right_depth) + 1
    
    def isBalanced(self, root: TreeNode) -> bool:
        return (self._maxDepth(root) - self._minDepth(root)) <= 1