# 第一部分：创建Stack，使用到了上篇中的类LinkedList和Node
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

    def size(self):
        return self.length

# 1,使用array创建stack:
class ArrayStack(object):
    def __init__ (self):
        self._data = []
        
    def __len__ (self):
        return len(self._data)
    
    def is_empty(self):
        return len(self._data) == 0
    
    # O(1)
    def push(self, e):
        self._data.append(e)
        
    # O(1)
    def top(self):
        if self.is_empty( ):
            raise ValueError( 'Stack is empty' )
        return self._data[-1]
    
    # O(1)
    def pop(self):
        if self.is_empty( ):
            raise ValueError( 'Stack is empty' )
        return self._data.pop( )  
        
    def printstack(self):
        for i in range(len(self._data)):
            print(self._data[i], end = ' ')
        print()

if __name__ == '__main__':
	mystack = ArrayStack()
	print ('size was: ', str(len(mystack)))
	mystack.push(1)
	mystack.push(2)
	mystack.push(3)
	mystack.push(4)
	mystack.push(5)
	print ('size was: ', str(len(mystack)))
	mystack.printstack()
	mystack.pop()
	mystack.pop()
	print ('size was: ', str(len(mystack)))
	mystack.printstack()
	print(mystack.top())
	mystack.pop()
	mystack.pop()
	mystack.pop()
	#mystack.pop()

# 2,使用linklist实现stack: 使用到了上篇中的类LinkedList和Node
class LinkedStack(object):
    def __init__ (self):
        self._list = LinkedList()
        
    def __len__ (self):
        return self._list.length
    
    def is_empty(self):
        return self._list.length == 0
    
    # O(1)
    def push(self, e):
        self._list.add_first(e);
        
    # O(1)
    def top(self):
        return self._list.get_first().value;
    
    # O(1)
    def pop(self):
        return self._list.remove_first().value;
        
    def printstack(self):
        self._list.printlist()

if __name__ == '__main__':
	mystack = LinkedStack()
	print ('size was: ', str(len(mystack)))
	mystack.push(1)
	mystack.push(2)
	mystack.push(3)
	mystack.push(4)
	mystack.push(5)
	print ('size was: ', str(len(mystack)))
	mystack.printstack()
	mystack.pop()
	mystack.pop()
	print ('size was: ', str(len(mystack)))
	mystack.printstack()
	print(mystack.top())
	mystack.pop()
	mystack.pop()
	mystack.pop()
	#mystack.pop()

# 第二部分：创建Queue
# 1,使用array创建queue：
class ArrayQueue:
    DEFAULT_CAPACITY = 10
    def __init__(self):
        self._data = [None] * ArrayQueue.DEFAULT_CAPACITY
        self._size = 0
        self._front = 0
        
    def __len__(self):
        return self._size
    
    def is_empty(self):
        return self._size == 0
    
    def first(self):
        if self.is_empty( ):
            raise ValueError( 'Queue is empty' )
        return self._data[self._front]
    
    def dequeue(self):
        if self.is_empty( ):
            raise ValueError( 'Queue is empty' )
        answer = self._data[self._front]
        self._data[self._front] = None
        self._front = (self._front + 1) % len(self._data)
        self._size -= 1
        return answer
    
    def enqueue(self, e):
        if self._size == len(self._data):
            self._resize(2 * len(self._data))
        pos = (self._front + self._size) % len(self._data)
        self._data[pos] = e
        self._size += 1
        
    def resize(self, cap):
        old = self._data
        self._data = [None] * cap
        walk = self._front
        for k in range(self._size):
            self._data[k] = old[walk]
            walk = (1 + walk) % len(old)
        self._front = 0
        
    def printqueue(self):
        for i in range(self._size):
            pos = (self._front + self._size - 1 - i) % len(self._data)
            #print(str(i), ": ", str(pos))
            print(self._data[pos], end = " ")  
        print()

if __name__ == '__main__':
	myqueue = ArrayQueue()
	print ('size was: ', str(len(myqueue)))
	myqueue.enqueue(1)
	myqueue.enqueue(2)
	myqueue.enqueue(3)
	myqueue.enqueue(4)
	myqueue.enqueue(5)
	print ('size was: ', str(len(myqueue)))
	myqueue.printqueue()
	myqueue.dequeue()
	myqueue.dequeue()
	print ('size was: ', str(len(myqueue)))
	myqueue.printqueue()
	myqueue.enqueue(6)
	myqueue.enqueue(7)
	myqueue.printqueue()
	myqueue.dequeue()
	myqueue.dequeue()
	print ('size was: ', str(len(myqueue)))
	myqueue.printqueue()
	myqueue.dequeue()
	myqueue.dequeue()
	myqueue.dequeue()
	print ('size was: ', str(len(myqueue)))
	myqueue.printqueue()
	#myqueue.dequeue()

# 2,使用linklist创建queue： 使用到了上篇中的类LinkedList和Node
class LinkedQueue(object):
    def __init__(self):
        self.head = None
        self.tail = None
        self.count = 0  
        
    def enqueue(self, value):
        new_node = Node(value)

        if self.tail is not None:
            self.tail.next = new_node

        else:
            self.head = new_node

        self.tail = new_node
        self.count += 1

    def dequeue(self):
        if not self.is_empty():
            # point head to next node
            tmp = self.head
            self.head = self.head.next
            print("dequeue sucess")
            self.count -= 1
            return tmp
        else:
            raise ValueError("Empty QUEUE")

    def is_empty(self):
        if self.head is None and self.tail is None:
            return True
        else:
            return False

    def peek(self):
        return self.head.data

        
    def __len__(self):
        return self.count    

    def is_empty(self):
        return self.count == 0
    
    def print(self):
        node = self.head
        while node:
            print(node.value, end = " ")
            node = node.next
        print('')

# 第三部分：Stack-Queue练习题
# 1,使用堆栈实现队列：
class QueueWithTwoStacks:    
    def __init__(self):
        self.insertStack = []
        self.popStack = []

    def enqueue(self, e):
        self.insertStack.append(e)
        return e
    
    def dequeue(self):
        if len(self.insertStack)==0 and len(self.popStack)==0:
            return None        
        if len(self.popStack)==0:
            while len(self.insertStack)!=0:
                self.popStack.append(self.insertStack.pop())        
        return self.popStack.pop()

if __name__ == '__main__':
	mystack = QueueWithTwoStacks()
	e = mystack.enqueue(3)
	print(e)
	e = mystack.enqueue(2)
	print(e)
	e = mystack.enqueue(1)
	print(e)
	e = mystack.dequeue()
	print(e)
	e = mystack.dequeue()
	print(e)

# 2，使用队列实现堆栈：
class StackWithQueue:    
    def __init__(self):
        self.queue = LinkedList()

    # Push element x onto stack.
    def push(self, x):
        self.queue.add_last(x)

    # Removes the element on top of the stack.
    def pop(self):
        size = self.queue.size()
        for i in range(1, size):
            self.queue.add_last(self.queue.remove_first())
        self.queue.remove_first()
        
    def top(self):
        size = self.queue.size()
        for i in range(1, size):
            self.queue.add_last(self.queue.remove_first())
        result = self.queue.remove_first()
        self.queue.add_last(result)
        return result

if __name__ == '__main__':
	stack = StackWithQueue() 
	stack.push(1)
	stack.push(2)
	print(stack.top())

	stack = StackWithQueue()
	stack.push(1)
	stack.push(2)
	stack.pop()
	stack.push(3)
	print(stack.top())

# 3,最小堆栈：
import sys

class NodeWithMin:
    def __init__(self, v, min):
        self._value = v
        self._min = min

class MinStack1(ArrayStack):    
    def __init__(self):
        super(MinStack, self).__init__()
    
    def push(self, v):       
        newMin = min(v, self.min())
        super(MinStack, self).push(NodeWithMin(v, newMin))
    
    def min(self):
        if (super(MinStack, self).is_empty()):
            return sys.maxsize
        else:
            return super(MinStack, self).top()._min;
    
class MinStack2(ArrayStack):
    
    def __init__(self):
        super(MinStack2, self).__init__()
        self.min_stack = ArrayStack()
        
    def push(self, value):
        if value <= self.min():
            self.min_stack.push(value)
        super(MinStack2, self).push(value)
        return value
          
    def min(self):
        if self.min_stack.is_empty():
            return sys.maxsize
        else:
            return self.min_stack.top()    
      
    def pop(self):
        value = super(MinStack2, self).pop()
        if value == self.min():
            self.min_stack.pop()
        return value

if __name__ == '__main__':
	minStack = MinStack2()
	minStack.push(4)
	minStack.push(6)
	minStack.push(8)
	minStack.push(3)
	print(minStack.min())
	minStack.pop()
	minStack.pop()
	print(minStack.min())

# 4,一个数组实现两个堆栈：
class twoStacks:
     
    def __init__(self, n): 
        self.size = n
        self.arr = [None] * n
        self.top1 = -1
        self.top2 = self.size
         
    # Method to push an element x to stack1
    def push1(self, x):
         
        # There is at least one empty space for new element
        if self.top1 < self.top2 - 1 :
            self.top1 = self.top1 + 1
            self.arr[self.top1] = x
 
        else:
            print("Stack Overflow ")
 
    # Method to push an element x to stack2
    def push2(self, x):
 
        # There is at least one empty space for new element
        if self.top1 < self.top2 - 1:
            self.top2 = self.top2 - 1
            self.arr[self.top2] = x
 
        else :
           print("Stack Overflow ")
 
    # Method to pop an element from first stack
    def pop1(self):
        if self.top1 >= 0:
            x = self.arr[self.top1]
            self.top1 = self.top1 -1
            return x
        else:
            print("Stack Underflow ")
 
    # Method to pop an element from second stack
    def pop2(self):
        if self.top2 < self.size:
            x = self.arr[self.top2]
            self.top2 = self.top2 + 1
            return x
        else:
            print("Stack Underflow ")

# 5,堆栈排序：
def sortStack(s):
    r = ArrayStack()    
    while not s.is_empty():
        tmp = s.pop()       
        while not r.is_empty() and r.top() > tmp:
            s.push(r.pop())            
        r.push(tmp)    
    return r

def sortedInsert(s, x):
    if len(s) == 0 or x > s.top():
        s.push(x)
        return
    temp = s.pop()
    sortedInsert(s, x)
    s.push(temp)
    
def sortStack(s):
    if len(s) != 0:
        x = s.pop()
        sortStack(s)
        sortedInsert(s, x)

if __name__ == '__main__':
	s = ArrayStack()
	s.push(30)
	s.push(-5)
	s.push(18)
	s.push(14)
	s.push(-3)
	s.printstack()
	sortStack(s)
	s.printstack()

# 6,反转字符串：
def reverse(s):
    lst = []
    for i in s:
        lst.append(i)
    result = []
    while len(lst) != 0:
        result.append(lst.pop())
    return ''.join(result)

if __name__ == '__main__':
	s = "hello world"
	print(reverse(s))

# 7,回文(Palindrome):
def isPalindrome(s):
    r = reverse(s)
    return r == s

if __name__ == '__main__':
	s = "hello world"
	print(reverse(s))

# 8,有效括号：
def isValid(s):
    stack = []
    for c in s:
        if (c == '(' or c == '[' or c == '{'):
            stack.append(c)
        else:
            if len(stack)==0:
                return False
            if (   (c == ')' and stack[-1] == '(')
                or (c == ']' and stack[-1] == '[')
                or (c == '}' and stack[-1] == '{')):
                stack.pop()
            else:
                return False
    return len(stack)==0

if __name__ == '__main__':
	s = "{{{}}{}{{}}}"
	print(isValid(s))

# 9，简化命令行路径：
def simplifyPath(path):
    lst = []
    splits = path.split("/")
    
    for s in splits:
        if s == "":
            continue
        if s == ".":
            continue
            
        if s == "..":
            if len(lst) != 0:
                lst.pop()
        else:
            lst.append(s)
    
    result = []
    if len(lst) == 0:
        return "/"
    result = ['/' + i for i in lst]
    return ''.join(result)
    
if __name__ == '__main__':
	path = "/home/"
	print(simplifyPath(path))

# 10,解码字符串：
def decodeString(s):
    stack = []
    stack.append(["", 1])
    num = ""
    for ch in s:
        if ch.isdigit():
            num += ch
        elif ch == '[':
            stack.append(["", int(num)])
            num = ""
        elif ch == ']':
            st, k = stack.pop()
            stack[-1][0] += st*k
        else:
            stack[-1][0] += ch
    return stack[0][0]

if __name__ == '__main__':
	s = "2[abc]3[cd]ef"
	print(decodeString(s))

# 11,比赛打分：
def calPoints(ops):
    stack = []
    for op in ops:
        if op == '+':
            stack.append(stack[-1] + stack[-2])
        elif op == 'C':
            stack.pop()
        elif op == 'D':
            stack.append(2 * stack[-1])
        else:
            stack.append(int(op))

    return sum(stack)

if __name__ == '__main__':
	ops = ["5","-2","4","C","D","9","+","+"]
	print(calPoints(ops))

# 12,行星碰撞：
def asteroidCollision(asteroids):
    ans = []
    for new in asteroids:
        while ans and new < 0 < ans[-1]:
            if ans[-1] < -new:
                ans.pop()
                continue
            elif ans[-1] == -new:
                ans.pop()
            break
        else:
            ans.append(new)
    return ans

if __name__ == '__main__':
	asteroids = [10, 2, -5] 
	print(asteroidCollision(asteroids))

# 13,查找下一个较大的元素：没有比当前值大的元素则返回0
def nextGreat(nums):
    if len(nums) == 0:
        return
    stack = []
    stack.append(nums[0])    
    for i in range(1, len(nums)):
        while (len(stack) != 0 and nums[i] > stack[-1]):
            num = stack.pop()
            print(num, ": ", array[i])
        stack.append(nums[i])        
    while len(stack) != 0:
        print(stack.pop(), ": -1")
if __name__ == '__main__':
	array = [6, 4, 5, 2, 25]
	nextGreat(array)

# 14,查找下一个较大的元素：与上面不同的是，这里是循环数组
def nextGreat2(nums):
    stack, r = [], [-1] * len(nums)
    for i in range(len(nums)):
        while stack and (nums[stack[-1]] < nums[i]):
            r[stack.pop()] = nums[i]
        stack.append(i)
    print(r)
    for i in range(len(nums)):
        while stack and (nums[stack[-1]] < nums[i]):
            r[stack.pop()] = nums[i]
        if stack == []:
            break
    return r

if __name__ == '__main__':
	array = [37, 6, 4, 5, 2, 25]
	nextGreat2(array)

# 15,日常温度：根据日常气温列表,制作一个列表,在输入的每一天中,都会告诉您需要等待多长时间,直到温度升高.如果没有可能的将来的日子,返回0
def dailyTemperatures(temperatures):
    if not temperatures: return []
    result, stack = [0] * len(temperatures), []
    stack.append((temperatures[0], 0))

    for i in range(1, len(temperatures)):
        while stack:
            prev = stack[-1]
            if prev[0] < temperatures[i]:
                result[prev[1]] = i - prev[1]
                stack.pop()
            else:
                break
        stack.append((temperatures[i], i))
    return result

def dailyTemperatures2(temperatures):
    if not temperatures: return []
    result, stack = [0] * len(temperatures), []
    stack.append(0)

    for i in range(1, len(temperatures)):
        while stack:
            prev = stack[-1]
            if temperatures[prev] < temperatures[i]:
                result[prev] = i - prev
                stack.pop()
            else:
                break
        stack.append(i)
    return result

if __name__ == '__main__':
	t = [73, 74, 75, 71, 69, 72, 76, 73]
	print(dailyTemperatures(t))

# 16,滑动窗口最大值：
from collections import deque

def movingMax(arr,k):
    n = len(arr)
    Qi = deque()
    
    # Process first k (or first window) 
    # elements of array
    for i in range(k):
        # For every element, the previous 
        # smaller elements are useless
        # so remove them from Qi
        while Qi and arr[i] >= arr[Qi[-1]] :
            Qi.pop()
        
        # Add new element at rear of queue
        Qi.append(i);
        
    # Process rest of the elements, i.e. 
    # from arr[k] to arr[n-1]
    for i in range(k, n):
        
        # The element at the front of the
        # queue is the largest element of
        # previous window, so print it
        print(str(arr[Qi[0]]) + " ", end = "")
        
        # Remove the elements which are 
        # out of this window
        while Qi and Qi[0] <= i-k:
            
            # remove from front of deque
            Qi.popleft() 
        
        # Remove all elements smaller than
        # the currently being added element 
        # (Remove useless elements)
        while Qi and arr[i] >= arr[Qi[-1]] :
            Qi.pop()
        
        # Add current element at the rear of Qi
        Qi.append(i)
    
    # Print the maximum element of last window
    print(str(arr[Qi[0]]))

if __name__ == '__main__':
	arr = [12, 1, 78, 90, 57, 89, 56]
	k = 3
	movingMax(arr, k)

# 17,评估算数表达式：
def infixToPostfix(infixexpr):
    prec = {}
    prec["*"] = 3
    prec["/"] = 3
    prec["+"] = 2
    prec["-"] = 2
    prec["("] = 1
    opStack = []
    postfixList = []
    tokenList = infixexpr.split()

    for token in tokenList:
        if token in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" or token in "0123456789":
            postfixList.append(token)
        elif token == '(':
            opStack.append(token)
        elif token == ')':
            topToken = opStack.pop()
            while topToken != '(':
                postfixList.append(topToken)
                topToken = opStack.pop()
        else:
            while (len(opStack) != 0) and \
               (prec[opStack[-1]] >= prec[token]):
                postfixList.append(opStack.pop())
            opStack.append(token)

    while len(opStack) != 0:
        postfixList.append(opStack.pop())
    return " ".join(postfixList)

def postfixEval(postfixExpr):
    operandStack = []
    tokenList = postfixExpr.split()

    for token in tokenList:
        if token in "0123456789":
            operandStack.append(int(token))
        else:
            operand2 = operandStack.pop()
            operand1 = operandStack.pop()
            result = doMath(token, operand1, operand2)
            operandStack.append(result)
    return operandStack.pop()

def doMath(op, op1, op2):
    if op == "*":
        return op1 * op2
    elif op == "/":
        return op1 / op2
    elif op == "+":
        return op1 + op2
    else:
        return op1 - op2

if __name__ == '__main__':
	print(infixToPostfix("A * B + C * D"))
	print(infixToPostfix("( A + B ) * C - ( D - E ) * ( F + G )"))
	print(infixToPostfix("A + B * C - ( D - E ) * F + G"))

	print(postfixEval('1 2 + 3 * 4 5 - 6 7 + * -'))
	print(postfixEval('1 2 3 * + 4 5 - 6 * - 7 +'))


# 18,双端队列：允许在两端插入和删除
