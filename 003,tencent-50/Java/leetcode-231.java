/** 位运算：
  * 二进制表示下，n最高位为1，其余所有位为0；
  * 二进制表示下，n - 1最高位为0，其余所有位为1（除了n == 1的情况下，n - 1 == 0，即末位为最高位）；
  * n <= 0时一定不是2的次方。
*/
class Solution {
    public boolean isPowerOfTwo(int n) {
        return n > 0 && (n & (n - 1)) == 0;
    }
}

// 除2法：
class Solution {
    public boolean isPowerOfTwo(int n) {
        if(n == 0){
            return false;
        }
        if(n == 1){
            return true;
        }
        while(n%2 == 0){
            n = n/2;
            if(n == 1){
                return true;
            }
        }
        return false;
    }
}

