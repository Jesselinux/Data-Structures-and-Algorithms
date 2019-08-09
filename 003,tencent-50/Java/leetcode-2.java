/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
/** 小技巧：
 * 对于链表问题，返回结果为头结点时，通常需要先初始化一个预先指针 pre，该指针的下一个节点指向真正的头结点head。
 * 使用预先指针的目的在于链表初始化时无可用节点值，而且链表构造过程需要指针移动，进而会导致头指针丢失，无法返回结果。
*/
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode pre = new ListNode(0);
        ListNode cur = pre;
        int carry = 0;// 是否进位，1
        
        int x;
        int y;
        
        while(l1 != null || l2 != null){
            if (l1 == null){
                x = 0;
            }else{
                x = l1.val;
            }
            
            if (l2 == null){
                y = 0;
            }else{
                y = l2.val;
            }
            
            int sum = x + y + carry;
            
            carry = sum / 10;
            sum = sum % 10;
            cur.next = new ListNode(sum);
            
            cur = cur.next;
            
            if (l1 != null){
                l1 = l1.next;
            }
            if (l2 != null){
                l2 = l2.next;
            }
            
        }
        if (carry == 1){
            cur.next = new ListNode(carry);
        }
        return pre.next;
    }
}
