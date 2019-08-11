// Bit Manipulation:
class Solution {
    public int singleNumber(int[] nums) {
        int ans = nums[0];
        if (nums.length > 1) {
           for (int i = 1; i < nums.length; i++) {
              ans = ans ^ nums[i]; // 异或操作满足交换律和结合律；且对于任何数x，都有x^x=0，x^0=x。
           }
         }
        return ans;
    }
}
