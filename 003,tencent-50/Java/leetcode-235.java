* Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(p==null||q==null||root==null){
            return null;  // 排除无意义的情形
        }
        
        //如果左边大于等于，右边小于等于，直接返回root
        if(p.val<=root.val&&q.val>=root.val){
            return root;  // 递归的终止条件
        }

        //遍历左子树
        if(p.val<root.val&&q.val<root.val){
            return lowestCommonAncestor(root.left,p,q);
        }
        //遍历右子树
        if(p.val>root.val&&q.val>root.val){
            return lowestCommonAncestor(root.right,p,q);
        }
        
        return root;
    }
}
