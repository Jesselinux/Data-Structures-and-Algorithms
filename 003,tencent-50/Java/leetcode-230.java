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
    private int count = 0;
    private int res = 0;

    private void dfs(TreeNode node) {  // 中序遍历
        if (node == null) {
            return; // 什么都不做
        }
        
        dfs(node.left);
        count--;
        
        if (count == 0) {
            this.res = node.val;
        }
        
        dfs(node.right);
    }

    // k 如果在方法传递的过程中是值传递，所以把它设置为成员变量，这样就是引用传递
    // 因为我们要用到 k 全局的值，去数出，我是第几个中序遍历到的值
    public int kthSmallest(TreeNode root, int k) {
        count = k;
        dfs(root);
        return res;
    }

    public static void main(String[] args) {
        TreeNode treeNode1 = new TreeNode(10);
        TreeNode treeNode2 = new TreeNode(15);
        TreeNode treeNode3 = new TreeNode(20);
        treeNode2.left = treeNode1;
        treeNode2.right = treeNode3;
        Solution solution = new Solution();
        int kthSmallest = solution.kthSmallest(treeNode2, 2);
        System.out.println(kthSmallest);
    }

}
