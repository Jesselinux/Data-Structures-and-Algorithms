class Solution {
    public boolean isValid(String s) {
        
        Map<Character,Character> map = new HashMap<>();
        map.put('(',')');
        map.put('[',']');
        map.put('{','}');
        
        Stack<Character> stack = new Stack<Character>();  // 第二个的Character可以不写
        char[] charArray = s.toCharArray();
        
        for (Character c:charArray){
            if (map.keySet().contains(c)){
                stack.push(c);
            }else{
                if (!stack.isEmpty() && map.get(stack.peek()) == c){
                    stack.pop();
                }else{
                    return false;
                }
            }
        }
        
        return stack.isEmpty();
        
    }
}


