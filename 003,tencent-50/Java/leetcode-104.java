/** 二叉树：DFS
  * 找出终止条件：当前节点为空；
  * 找出返回值：节点为空时说明高度为0，所以返回0；节点不为空时则分别求左右子树的高度的最大值，同时加1表示当前节点的高度，返回该数值；
  * 某层的执行过程：在返回值部分基本已经描述清楚；
  * 时间复杂度：O(n)
  */
class Solution {
    public int maxDepth(TreeNode root) {
        if(root == null) {  // 递归的终止条件
            return 0;
        } else {
            int left = maxDepth(root.left);  // 递归假设子问题已经解决了
            int right = maxDepth(root.right);
            return Math.max(left, right) + 1;
        }
    }
}
