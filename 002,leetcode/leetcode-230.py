# leetcode-230:二叉搜索树，中序遍历，beats 56.14%

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        self.cnt, self.val = 0, 0
        self.inOrder(root, k)
        return self.val
    
    def inOrder(self, root: TreeNode, k: int):
        if not root:
            return
        self.inOrder(root.left, k)
        self.cnt += 1
        if self.cnt == k:
            self.val = root.val
            return
        self.inOrder(root.right, k)