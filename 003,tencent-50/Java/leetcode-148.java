/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
/**
     * 链表归并排序
     */
    public ListNode sortList(ListNode head) {

        if (head == null || head.next == null) {
            return head;
        }

        ListNode mid = findMiddle(head);
        ListNode right = sortList(mid.next);
        mid.next = null;

        ListNode left = sortList(head);

        return merge(left, right);
    }

    /**
     * 找出中间的节点
     */
    public ListNode findMiddle(ListNode node) {
        ListNode fast = node.next;
        ListNode slow = node;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }

        return slow;
    }

    /**
     * 对两组链表进行归并排序
     */
    public ListNode merge(ListNode left, ListNode right) {
        ListNode a = left;
        ListNode b = right;

        ListNode result = new ListNode(0);
        ListNode tmp = result;

        while (a != null && b != null) {
            while (a != null && a.val <= b.val) {
                tmp.next = new ListNode(a.val);
                tmp = tmp.next;
                a = a.next;
            }

            while (a != null && b != null && b.val <= a.val) {
                tmp.next = new ListNode(b.val);
                tmp = tmp.next;
                b = b.next;
            }
        }

        if (a != null) {
            tmp.next = a;
        } else if (b != null) {
            tmp.next = b;
        }

        return result.next;
    }
}
