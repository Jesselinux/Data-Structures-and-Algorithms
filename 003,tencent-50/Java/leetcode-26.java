class Solution {
    public int removeDuplicates(int[] nums) {
        if (nums.length == 0) return 0;
        int i = 0;
        for (int j = 1; j < nums.length; j++) {
            if (nums[j] != nums[i]) {
                i++;
                nums[i] = nums[j];
            }
        }
        return i + 1;
    }
}

// 错误解法：
class Solution {
    public int removeDuplicates(int[] nums) {
        int j=1;
        int ret = 0;
        Arrays.sort(nums);
        for(int i=1; i<nums.length-1; ++i){
            if (nums[i] != nums[i-1]){
                nums[j++] = nums[i];
                ret += 1;
            }
        }
        return ret;
    }
}
