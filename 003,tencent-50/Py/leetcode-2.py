# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        carry = 0
        l4 = None  # We use this variable to keep track of head of our result LinkedList
        while(True): # Python's makeshift Do while loop -- see if condition below which breaks it
            sumTotal = l1.val + l2.val + carry
            tempNode = ListNode(sumTotal%10) 
            carry = sumTotal//10
            try:
                l3.next = tempNode
                l3 = l3.next
            except(NameError):
                l3 = tempNode
                l4 = l3
            if(l2.next == None or l1.next == None):
                break
            else:
                l1,l2 = l1.next,l2.next
        if(l2.next == None and l1.next == None and carry ==0):
            return l4
        elif(l2.next == None and l1.next == None and carry ==1):
            l3.next = ListNode(1)
            return l4
        l = l1.next if (l2.next == None and l1.next !=None) else l2.next #select list which still has elements remaining
        # Repeat of previous while loop but only single list and carry -- there should be way to optimize this as it looks redundant
        while(True):
            sumTotal = l.val  + carry
            tempNode = ListNode(sumTotal%10)
            carry = sumTotal//10
            l3.next = ListNode(sumTotal%10)
            l3 = l3.next
            if(l.next == None and carry ==0):
                return l4
            elif(l.next == None and carry == 1):
                l3.next = ListNode(1)
                return l4
            else:
                l = l.next