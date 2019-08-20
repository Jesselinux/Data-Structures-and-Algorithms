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
    TreeNode res = null;
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        dfs(root, p, q);
        return res;
    }
    private int dfs(TreeNode root, TreeNode p, TreeNode q){
        if(root == null) return 0; // 递归的终止条件
        
        int left = dfs(root.left, p, q);
        int right = dfs(root.right, p, q);
        
        int mid = 0;
        if(root == p || root == q){
            mid = 1;
        }
        
        if(left + right + mid > 1) res = root;
        
        return left + right + mid > 0 ? 1 : 0;
    }
}
