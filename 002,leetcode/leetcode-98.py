# leetcode-98:
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        if not root:
            return True
        
        pre = None
        stack = list()
        while root or stack:
            if root:
                stack.append(root)
                root = root.left
            else:
                root = stack.pop()
                if pre and root.val <= pre.val:
                    return False
                
                pre = root
                root = root.right
        return True

# reference:https://blog.csdn.net/qq_17550379/article/details/82315830