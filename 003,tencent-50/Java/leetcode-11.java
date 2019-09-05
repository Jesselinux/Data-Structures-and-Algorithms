//暴力解法:
public class Solution {
    public int maxArea(int[] height) {
        int maxarea = 0;
        for (int i = 0; i < height.length; i++)
            for (int j = i + 1; j < height.length; j++)
                maxarea = Math.max(maxarea, Math.min(height[i], height[j]) * (j - i));
        return maxarea;
    }
}

//双指针:
/**
双指针解法的思路在于，两线段之间形成的区域总是会受到其中较短那条长度的限制。
两线段距离越远，得到的面积就越大。
我们在由线段长度构成的数组中使用两个指针，一个放在开始，一个置于末尾。 
此外，我们会使用变量 maxarea，maxarea 来持续存储到目前为止所获得的最大面积。
在每一步中，我们会找出指针所指向的两条线段形成的区域，更新 maxareamaxarea，并
将指向较短线段的指针向较长线段那端移动一步。
*/
class Solution {
    public int maxArea(int[] height) {
        int i = 0, j = height.length - 1, res = 0;
        while(i < j){
            res = height[i] < height[j] ? 
                Math.max(res, (j - i) * height[i++]): 
                Math.max(res, (j - i) * height[j--]); 
        }
        return res;
    }
}
