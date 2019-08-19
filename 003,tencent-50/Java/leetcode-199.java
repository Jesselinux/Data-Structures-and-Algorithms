// 层序遍历：
class Solution {
    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        Queue<TreeNode> temp = new LinkedList<>();
        temp.add(root);
        while(!temp.isEmpty()){
            TreeNode face=null;
            Queue<TreeNode> queue = new LinkedList<>();
            while(!temp.isEmpty()){
                TreeNode node = temp.poll();
                if(node==null) continue;
                face = node;
                queue.add(face.left);
                queue.add(face.right);
            }
            if(face!=null) result.add(face.val);
            temp.addAll(queue);
        }
        return result;
    }
}

// 法二：
class Solution{
    List<Integer> ret=new ArrayList();
    public List<Integer> rightSideView(TreeNode root) {
        if(root==null)
            return ret;
        List<TreeNode> list=new LinkedList();
        list.add(root);
        while(!list.isEmpty()){
            int len=list.size();
            for(int i=0;i<len;i++){
                TreeNode tem=list.remove(0);
                if(tem.left!=null)
                    list.add(tem.left);
                if(tem.right!=null)
                    list.add(tem.right);
                if(i==len-1)
                    ret.add(tem.val);
            }
        }
        return ret;
    }
}

// 深度优先：
class Solution {
    public List<Integer> rightSideView(TreeNode root) {
        Map<Integer, Integer> rightmostValueAtDepth = new HashMap<Integer, Integer>();
        int max_depth = -1;

        /* These two stacks are always synchronized, providing an implicit
         * association values with the same offset on each stack. */
        Stack<TreeNode> nodeStack = new Stack<TreeNode>();
        Stack<Integer> depthStack = new Stack<Integer>();
        nodeStack.push(root);
        depthStack.push(0);

        while (!nodeStack.isEmpty()) {
            TreeNode node = nodeStack.pop();
            int depth = depthStack.pop();

            if (node != null) {
                max_depth = Math.max(max_depth, depth);

                /* The first node that we encounter at a particular depth contains
                * the correct value. */
                if (!rightmostValueAtDepth.containsKey(depth)) {
                    rightmostValueAtDepth.put(depth, node.val);
                }

                nodeStack.push(node.left);
                nodeStack.push(node.right);
                depthStack.push(depth+1);
                depthStack.push(depth+1);
            }
        }

        /* Construct the solution based on the values that we end up with at the
         * end. */
        List<Integer> rightView = new ArrayList<Integer>();
        for (int depth = 0; depth <= max_depth; depth++) {
            rightView.add(rightmostValueAtDepth.get(depth));
        }

        return rightView;
    }
}

// 广度优先：
class Solution {
    public List<Integer> rightSideView(TreeNode root) {
        Map<Integer, Integer> rightmostValueAtDepth = new HashMap<Integer, Integer>();
        int max_depth = -1;

        /* These two Queues are always synchronized, providing an implicit
         * association values with the same offset on each Queue. */
        Queue<TreeNode> nodeQueue = new LinkedList<TreeNode>();
        Queue<Integer> depthQueue = new LinkedList<Integer>();
        nodeQueue.add(root);
        depthQueue.add(0);

        while (!nodeQueue.isEmpty()) {
            TreeNode node = nodeQueue.remove();
            int depth = depthQueue.remove();

            if (node != null) {
                max_depth = Math.max(max_depth, depth);

                /* The last node that we encounter at a particular depth contains
                * the correct value, so the correct value is never overwritten. */
                rightmostValueAtDepth.put(depth, node.val);

                nodeQueue.add(node.left);
                nodeQueue.add(node.right);
                depthQueue.add(depth+1);
                depthQueue.add(depth+1);
            }
        }

        /* Construct the solution based on the values that we end up with at the
         * end. */
        List<Integer> rightView = new ArrayList<Integer>();
        for (int depth = 0; depth <= max_depth; depth++) {
            rightView.add(rightmostValueAtDepth.get(depth));
        }

        return rightView;
    }
}
