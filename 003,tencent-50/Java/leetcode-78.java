// 回溯算法：递归
class Solution {
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        backtrack(0, nums, res, new ArrayList<Integer>());
        return res;

    }

    private void backtrack(int i, int[] nums, List<List<Integer>> res, ArrayList<Integer> tmp) {
        res.add(new ArrayList<>(tmp));
        for (int j = i; j < nums.length; j++) {
            tmp.add(nums[j]);
            backtrack(j + 1, nums, res, tmp);
            tmp.remove(tmp.size() - 1);
        }
    }
}

/**
 * 循环枚举
 */

class Solution {
    public List<List<Integer>> subsets(int[] nums) {
        
        List<List<Integer>> ans = new ArrayList<>();
        ans.add(new ArrayList<>());//初始化空数组
        
        for(int i = 0;i<nums.length;i++){
            List<List<Integer>> ans_tmp = new ArrayList<>();
            //遍历之前的所有结果
            for(List<Integer> list : ans){
                List<Integer> tmp = new ArrayList<>(list);
                tmp.add(nums[i]); //加入新增数字
                ans_tmp.add(tmp);
            }
            ans.addAll(ans_tmp);
        }
        
        return ans;
    }
}


/**
 * 递归枚举
 */
public static void recursion(int[] nums, int i, List<List<Integer>> res) {
    if (i >= nums.length) return;
    int size = res.size();
    for (int j = 0; j < size; j++) {
        List<Integer> newSub = new ArrayList<Integer>(res.get(j));
        newSub.add(nums[i]);
        res.add(newSub);
    }
    recursion(nums, i + 1, res);
}
