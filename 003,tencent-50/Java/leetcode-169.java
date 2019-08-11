/**哈希表：
 * Author:Jesse;
 * Date: 2019.8；
 */
class Solution {
    public int majorityElement(int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int num:nums){
            if (map.containsKey(num)){
                map.put(num, map.get(num) + 1);
            }else{
                map.put(num, 1);
            }
        }
        
        Iterator<Map.Entry<Integer, Integer>> it = map.entrySet().iterator();
        while (it.hasNext()) {
            Map.Entry<Integer, Integer> entry = it.next();
            if(entry.getValue() > nums.length/2){
                return entry.getKey();
            }
        }
       throw new IllegalArgumentException("something is wrong");
    }
}
