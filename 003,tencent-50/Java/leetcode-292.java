/**
  * 要想获胜，在你的回合中，必须避免石头堆中的石子数为 4 的情况。
  * 因为不管你从一堆石头中挑出一块、两块还是三块，你的对手都可以选择三块、两块或一块，以确保在再一次轮到你的时候，你会面对四块石头。
  */
class Solution {
    public boolean canWinNim(int n) {
        return (n % 4 != 0);
    }
}
