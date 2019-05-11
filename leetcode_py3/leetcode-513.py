# leetcode-513：查找树的左下角值，beats  88.92%
class Solution:
    def findBottomLeftValue(self, root: TreeNode) -> int:
	 	q = [root]
	    while q:
		    root = q.pop(0)
		    if root.right:
		    	q.append(root.right)
		    if root.left:
		    	q.append(root.left)
	    return root.val