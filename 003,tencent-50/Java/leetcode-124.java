/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    private int max = Integer.MIN_VALUE;
    
    public int maxPathSum(TreeNode root) {
        
        max_Path(root);
        return max;
    }
    
    private int max_Path(TreeNode root){
        if (root == null){
            return 0;  // 递归终止条件
        }
        
        int left = max_Path(root.left);
        int right = max_Path(root.right);
        max = Math.max(root.val + left + right, max);
        
        int tmp = Math.max(left, right) + root.val;
        return tmp > 0 ? tmp : 0;
    }
}
