# 第一部分：创建链表
# 1，python实现一个链表：
class Empty(Exception):
    pass

class Outbound(Exception):
    pass

class Node:
    def __init__ (self, value = None, next = None):
        self.value = value
        self.next = next

class LinkedList:  # 定义一个链表类
    def __init__(self):
        self.head = Node()
        self.tail = None
        self.length = 0

    def peek(self):
        if not self.head.next:
            raise Empty( 'LinkedList is empty' )
        return self.head.next

    def get_first(self):
        if not self.head.next:
            raise Empty( 'LinkedList is empty' )
        return self.head.next
        
    def get_last(self):
        if not self.head.next:
            raise Empty( 'LinkedList is empty' )
        node = self.head
        while node.next != None:
            node = node.next
        return node
    
    def get(self, index):
        if (index < 0 or index >= self.length):
            raise Outbound( 'index is out of bound' );
        if not self.head.next:
            raise Empty( 'LinkedList is empty' )
        node = self.head.next
        for i in range(index):
            node = node.next
        return node
                
    def add_first(self, value):
        node = Node(value, None)
        node.next = self.head.next
        self.head.next = node
        self.length += 1   
        
    def add_last(self, value):
        new_node = Node(value)
        node = self.head
        while node.next != None:
            node = node.next
        node.next = new_node
        self.length += 1

    def add(self, index, value):
        if (index < 0 or index > self.length):
            raise Outbound( 'index is out of bound' )
        if not self.head.next:
            raise Empty( 'LinkedList is empty' )
        new_node = Node(value)
        node = self.head
        for i in range(index):
            node = node.next
        new_node.next = node.next;
        node.next = new_node;
        self.length += 1     
        
    def remove_first(self):
        if not self.head.next:
            raise Empty( 'LinkedList is empty' )
        value = self.head.next
        self.head.next = self.head.next.next
        self.length -= 1
        return value    
        
    def remove_last(self):
        if not self.head.next:
            raise Empty( 'LinkedList is empty' )
        node = self.head.next
        prev = self.head
        while node.next != None:
            prev = node
            node = node.next
        prev.next = None
        return node.value

    def remove(self, index):
        if (index < 0 or index >= self.length):
            raise Outbound( 'index is out of bound' );
        if not self.head.next:
            raise Empty( 'LinkedList is empty' )
        node = self.head
        for i in range(index):
            node = node.next
        result = node.next;
        node.next = node.next.next;
        self.length += 1     
        return result;      
        
    def printlist(self):
        node = self.head.next
        count = 0
        while node and count<20:
            print(node.value, end = " ")
            node = node.next
            count = count + 1
        print('')

if __name__ == '__main__':
	ll = LinkedList()
	for i in range(1, 10):
	    ll.add_last(i)
	ll.printlist()

	ll.remove_first()
	print('After remove_first, Linked List is: ')
	ll.printlist()

	ll.remove_last()
	print('After remove_last, Linked List is: ')
	ll.printlist()

	# for i in range(7):   # 逐个删除链表所有元素
	# 	ll.remove_first()
	# ll.remove_first()   # 链表为空时，会使用自定义的Empty类报错

# 第二部分：链表练习题
# 2，删除节点：
def delete_node(node):
    node.value = node.next.value
    node.next = node.next.next

if __name__ == '__main__':
	ls1 = LinkedList()
	for i in range(7):
		ls1.add_last(i)
	ls1.printlist()
	delete_node(ls1.head.next.next)
	ls1.printlist()

# 3,查找中间元素：
def find_middle(lst):
    assert lst.head is not None and lst.head.next is not None
    
    head = lst.head
    fast = head
    slow = head
    
    while fast is not None and fast.next is not None:
        fast = fast.next.next
        slow = slow.next    
    return slow.value

if __name__ == '__main__':
	ls2 = LinkedList()
	for i in range(15):
		ls2.add_last(i)
	ls2.printlist()
	print(find_middle(ls2))

# 4,是否有环：
def has_cycle(lst):
    return has_cycle_helper(lst.head)

def has_cycle_helper(head):
    if head is None:
        return False
    
    slow = head 
    fast = head    
    while fast is not None and fast.next is not None:
        fast = fast.next.next
        slow = slow.next        
        if slow==fast:
            return True        
    return False

if __name__ == '__main__':
	node1 = Node(1)
	node2 = Node(2)
	node3 = Node(3)
	node1.next = node2
	node2.next = node3
	print(has_cycle_helper(node1))
	node3.next = node1
	print(has_cycle_helper(node1))

# 5,给定一个循环链表，查找环的开始节点：
def find_beginning(head):
    if head is None:
        return None
    
    slow = head
    fast = head
    
    while fast is not None and fast.next is not None:
        fast = fast.next.next
        slow = slow.next
        
        if slow==fast:
            fast = head
            break
        
    if fast is None or fast.next is None:
        return None
    
    while fast != slow:
        fast = fast.next
        slow = slow.next

    return slow

if __name__ == '__main__':
	node1 = Node(1)
	node2 = Node(2)
	node3 = Node(3)
	node4 = Node(4)
	node5 = Node(5)
	node1.next = node2
	node2.next = node3	
	node3.next = node4
	node4.next = node5
	node5.next = node3  # 可以再草稿纸上画出此链表结构，环开始节点就会一目了然
	print(find_beginning(node1).value)

# 6,删除链表的倒数第N个节点：
def remove_nth(lst, n):
    assert n<=lst.length and n > 0    
    fast = lst.head
    while n>0:
        fast = fast.next
        n = n - 1
        
    slow = lst.head
    while fast.next is not None:
        fast = fast.next
        slow = slow.next        
    result = slow.next
    slow.next = slow.next.next    
    lst.length = lst.length - 1        
    return result

if __name__ == '__main__':
	ls3 = LinkedList()
	for i in range(16):
		ls3.add_last(i)
	print(remove_nth(ls3, 3).value)
	ls3.printlist()

# 7,分裂成两个链表：对半分
def split(head):
    if (head is None):
        return
    slow = head
    fast = head
    front_last_node = slow
    while (fast is not None):
        front_last_node = slow
        slow = slow.next
        fast = fast.next.next if fast.next is not None else None
    front_last_node.next = None
    front = head
    back = slow
    return (front, back)
    
if __name__ == '__main__':
	node1 = Node(1)
	node2 = Node(2)
	node3 = Node(3)
	node4 = Node(4)
	node5 = Node(5)
	node1.next = node2
	node2.next = node3
	node3.next = node4
	node4.next = node5

	front_node = Node()
	back_node = Node()
	front_node, back_node = split(node1)
	front = LinkedList()
	front.head.next = front_node
	front.printlist()

	back = LinkedList()
	back.head.next = back_node
	back.printlist()

# 8,合并两个有序链表：返回一个新的有序列表
# O(m + n)
def mergeTwoLists1(l1, l2):  # iteratively
    dummy = cur = Node(0)
    while l1 and l2:
        if l1.value < l2.value:
            cur.next = l1
            l1 = l1.next
        else:
            cur.next = l2
            l2 = l2.next
        cur = cur.next
    cur.next = l1 or l2
    return dummy.next

# recursively    
def mergeTwoLists2(l1, l2):
    if not l1 or not l2:
        return l1 or l2
    if l1.value < l2.value:
        l1.next = mergeTwoLists2(l1.next, l2)
        return l1
    else:
        l2.next = mergeTwoLists2(l1, l2.next)
        return l2

if __name__ == '__main__':
	node11 = Node(1)
	node12 = Node(2)
	node14 = Node(4)
	node11.next = node12
	node12.next = node14

	node21 = Node(1)
	node23 = Node(3)
	node24 = Node(4)
	node21.next = node23
	node23.next = node24

	node = mergeTwoLists2(node11, node21)
	ls5 = LinkedList()
	ls5.head.next = node
	ls5.printlist()

# 9,查找链表交集开始的节点：类似两条路最后汇聚成一条路了
def getIntersectionNode1(headA, headB):
    curA, curB = headA, headB
    lenA, lenB = 0, 0
    while curA is not None:
        lenA += 1
        curA = curA.next
    while curB is not None:
        lenB += 1
        curB = curB.next
    curA, curB = headA, headB
    if lenA > lenB:
        for i in range(lenA-lenB):
            curA = curA.next
    elif lenB > lenA:
        for i in range(lenB-lenA):
            curB = curB.next
    while curB != curA:
        curB = curB.next
        curA = curA.next
    return curA

def getIntersectionNode2(headA, headB):
    if headA and headB:
        A, B = headA, headB
        while A!=B:
            A = A.next if A else headB
            B = B.next if B else headA
        return A

# 10,链表的插入排序：
def insertionSortList(head):
    dummy = Node(0)
    cur = head
    # pre is the sorted part
    # when see a new node, start from dummy
    # cur is the unsorted part
    while cur is not None:
        pre = dummy
        while pre.next is not None and pre.next.value < cur.value:
            pre = pre.next
        temp = cur.next
        cur.next = pre.next
        pre.next = cur
        cur = temp
    return dummy.next

if __name__ == '__main__':
	node1 = Node(-9)
	node2 = Node(1)
	node3 = Node(-13)
	node4 = Node(6)
	node5 = Node(5)
	node1.next = node2
	node2.next = node3
	node3.next = node4
	node4.next = node5
	lst = LinkedList()
	lst.head.next = node1
	lst.printlist()

	node = insertionSortList(node1)
	lst.head.next = node
	lst.printlist()

# 11,链表排序(O(n*lgn)):
def sortList(head):
    if head is None or head.next is None:
        return head
    mid = getMiddle(head)
    rHead = mid.next
    mid.next = None
    return merge(sortList(head), sortList(rHead))

def merge(lHead, rHead):
    dummyNode = dummyHead = Node(0)
    while lHead and rHead:
        if lHead.value < rHead.value:
            dummyHead.next = lHead
            lHead = lHead.next
        else:
            dummyHead.next = rHead
            rHead = rHead.next
        dummyHead = dummyHead.next
    if lHead:
        dummyHead.next = lHead
    elif rHead:
        dummyHead.next = rHead
    return dummyNode.next

def getMiddle(head):
    if head is None:
        return head
    slow = head
    fast = head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
    return slow

if __name__ == '__main__':
	node1 = Node(-9)
	node2 = Node(1)
	node3 = Node(-13)
	node4 = Node(6)
	node5 = Node(5)
	node1.next = node2
	node2.next = node3
	node3.next = node4
	node4.next = node5
	lst = LinkedList()
	lst.head.next = node1
	lst.printlist()

	node = sortList(node1)
	lst.head.next = node
	lst.printlist()

# 12,对链表分区：使得所有小于x的节点排在所有大于或等于x的节点之前
def partition(head, x):
    left_head = Node(None)  # head of the list with nodes values < x
    right_head = Node(None)  # head of the list with nodes values >= x
    left = left_head  # attach here nodes with values < x
    right = right_head  # attach here nodes with values >= x
    # traverse the list and attach current node to left or right nodes
    while head:
        if head.value < x:
            left.next = head
            left = left.next
        else:  # head.val >= x
            right.next = head
            right = right.next
        head = head.next
    right.next = None  # set tail of the right list to None
    left.next = right_head.next  # attach left list to the right
    return left_head.next  # head of a new partitioned list

if __name__ == '__main__':
	node1 = Node(1)
	node2 = Node(4)
	node3 = Node(3)
	node4 = Node(2)
	node5 = Node(5)
	node6 = Node(2)
	node1.next = node2
	node2.next = node3
	node3.next = node4
	node4.next = node5
	node5.next = node6
	node = partition(node1, 3)
	lst = LinkedList()
	lst.head.next = node
	lst.printlist()

# 13,反转链表(1):
def reverse1(lst):
    head = lst.head
    result = None
    current = head.next
    nxt = None    
    while current is not None:
        nxt = current.next
        current.next = result
        result = current
        current = nxt        
    head.next = result

def reverse2Recursion(node):
    if (node is None or node.next is None):
        return node
    p = reverse2Recursion(node.next)
    node.next.next = node
    node.next = None
    return p

if __name__ == '__main__':
	ls6 = LinkedList()
	for i in range(16):
		ls6.add_last(i)
	ls6.printlist()
	ls6.head.next = reverse2Recursion(ls6.head.next)
	ls6.printlist()

# 14,反转链表(2):index区间[m, n]内反转
def reverseBetween(head, m, n):
    if m == n:
        return head

    dummyNode = Node(0)
    dummyNode.next = head
    pre = dummyNode
    for i in range(m - 1):
        pre = pre.next
    # reverse the [m, n] nodes
    result = None
    current = pre.next
    for i in range(n - m + 1):
        nxt = current.next
        current.next = result
        result = current
        current = nxt
    pre.next.next = current
    pre.next = result
    return dummyNode.next

if __name__ == '__main__':
	ls7 = LinkedList()
	for i in range(16):
		ls7.add_last(i)
	ls7.printlist()
	ls7.head.next = reverseBetween(ls7.head.next, 3, 15)
	ls7.printlist()

# 15,反转链表(3):两两交换节点
def swapPairs(head):
    dummy = cur = Node(0)
    dummy.next = head    
    while cur.next and cur.next.next:
        p1 = cur.next
        p2 = cur.next.next
        cur.next  = p2
        p1.next = p2.next
        p2.next = p1
        cur = cur.next.next
    return dummy.next

if __name__ == '__main__':
	ls7 = LinkedList()
	for i in range(17):
		ls7.add_last(i)
	ls7.printlist()

	ls7.head.next = swapPairs(ls7.head.next)
	ls7.printlist()

# 16,反转链表(4):以k个为单位，反转链表
def reverseKGroup(head, k):
    if head is None or k < 2:
        return head
    
    next_head = head
    for i in range(k - 1):
        next_head = next_head.next
        if next_head is None:
            return head
    ret = next_head
    
    current = head
    while next_head:
        tail = current
        prev = None
        for i in range(k):
            if next_head:
                next_head = next_head.next
            nxt = current.next
            current.next = prev
            prev = current
            current = nxt
        tail.next = next_head or current
    return ret

if __name__ == '__main__':
	ls8 = LinkedList()
	for i in range(26):
		ls8.add_last(i)
	ls8.printlist()

	ls8.head.next = reverseKGroup(ls8.head.next, 5)
	ls8.printlist()

# 17,判断是否为回文联表：
def isPalindrome(head):
    rev = None
    slow = fast = head
    while fast and fast.next:
        fast = fast.next.next
        rev, rev.next, slow = slow, rev, slow.next
    if fast:
        slow = slow.next
    while rev and rev.value == slow.value:
        slow = slow.next
        rev = rev.next
    return not rev

if __name__ == '__main__':
	lst = LinkedList()
	lst.add_last(1)
	lst.add_last(3)
	lst.add_last(5)
	lst.add_last(3)
	lst.add_last(1)
	lst.printlist()

	print(isPalindrome(lst.head.next))
	lst.printlist()

# 18,有序链表中删除重复元素：保留一个重复值
def deleteDuplicates(head):
    if head == None:
        return head

    node = head
    while node.next:
        if node.value == node.next.value:
            node.next = node.next.next
        else:
            node = node.next
    return head

if __name__ == '__main__':
	lst = LinkedList()
	lst.add_last(1)
	lst.add_last(3)
	lst.add_last(3)
	lst.add_last(3)
	lst.add_last(5)
	lst.add_last(7)
	lst.add_last(7)
	lst.add_last(9)
	lst.head.next = deleteDuplicates(lst.head.next)
	lst.printlist()

# 19,有序链表中删除重复元素：不保留任何一个重复值
def deleteDuplicates2(head):
    dummy = pre = Node(0)
    dummy.next = head
    while head and head.next:
        if head.value == head.next.value:
            while head and head.next and head.value == head.next.value:
                head = head.next
            head = head.next
            pre.next = head
        else:
            pre = pre.next
            head = head.next
    return dummy.next

if __name__ == '__main__':
	lst = LinkedList()
	lst.add_last(1)
	lst.add_last(3)
	lst.add_last(3)
	lst.add_last(3)
	lst.add_last(5)
	lst.add_last(7)
	lst.add_last(7)
	lst.add_last(9)
	lst.head.next = deleteDuplicates2(lst.head.next)
	lst.printlist()