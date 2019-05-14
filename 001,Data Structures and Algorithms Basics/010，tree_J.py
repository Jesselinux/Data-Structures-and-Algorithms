class Node(object):
    __slots__ = '_item' , '_left' , '_right'

    def __init__ (self, item, left=None, right=None):
        self._item = item
        self._left = left
        self._right = right

class BinarySearchTree(object):

    
    def __init__ (self, root=None):
        self._root = root
        
    # Get methods
    def get(self, key):
        return self.__get(self._root, key);

    def __get(self, node, key): # helper
        if (node is None):
            return None
        if (key == node._item):
            return node._item
        if (key < node._item):
            return self.__get(node._left, key)
        else:
            return self.__get(node._right, key)
        
    
    # add methods
    def add(self, value):
        self._root = self.__add(self._root, value)
        
    def __add(self, node, value): # return node ,helper
        if (node is None):
            return Node(value)
        if (value == node._item):
            pass
        else:
            if (value < node._item):
                node._left = self.__add(node._left, value)
            else:
                node._right = self.__add(node._right, value)
        return node 
    
    # remove methods
    def remove(self, key):
        self._root = self.__remove(self._root, key)
        
    def __remove(self, node, key):  # helper
        if node is None:
            return None
        if (key < node._item):
            node._left = self.__remove(node._left, key)
        elif (key > node._item):
            node._right = self.__remove(node._right, key)
        else:
            if (node._left is None):
                node = node._right  # if right is None,  node = None; case 1: no child  
                                    # if right is not None, node = node._right; case 2: one child
            elif (node._right is None):
                node = node._left
            else:
                node._item = self.__get_max(node._left)
                node._left = self.__remove(node._left, node._item)
                
        return node
    
    # get max/min methods
    def get_max(self):
        return self.__get_max(self._root)
    
    def __get_max(self, node): # helper
        if (node is None):
            return None
        while (node._right is not None):
            node = node._right
        return node._item

    # Traversal Methods  
    def print_inorder(self):
        self._print_inorder(self._root)
        print('')

    def _print_inorder(self, node):
        if (node is None):
            return
        self._print_inorder(node._left)
        print ('[', node._item, ']', end = " ")
        self._print_inorder(node._right)
    
    def print_preorder(self):
        self._print_preorder(self._root)
        print('')

    def _print_preorder(self, node):
        if (node is None):
            return
        print ('[', node._item, ']', end = " ")
        self._print_preorder(node._left)
        self._print_preorder(node._right)    
        
    def print_postorder(self):
        self._print_postorder(self._root)
        print('')

    def _print_postorder(self, node):
        if (node is None):
            return
        self._print_postorder(node._left)
        self._print_postorder(node._right)          
        print ('[', node._item, ']', end = " ")


class Node_ll:
    def __init__ (self, value = None, next = None):
        self.value = value
        self.next = next

class LinkedList:
    
    def __init__(self):
        self.head = Node_ll()
        self.length = 0

    def peek(self):
        if not self.head.next:
            raise ValueError( 'LinkedList is empty' )
        return self.head.next

    def get_first(self):
        if not self.head.next:
            raise ValueError( 'LinkedList is empty' )
        return self.head.next
        
    def get_last(self):
        if not self.head.next:
            raise ValueError( 'LinkedList is empty' )
        node = self.head
        while node.next != None:
            node = node.next
        return node
    
    def get(self, index):
        if (index < 0 or index >= self.length):
            raise ValueError( 'index is out of bound' );
        if not self.head.next:
            raise ValueError( 'LinkedList is empty' )
        node = self.head.next
        for i in range(index):
            node = node.next
        return node
                
    def add_first(self, value):
        node = Node_ll(value, None)
        node.next = self.head.next
        self.head.next = node
        self.length += 1   
        
    def add_last(self, value):
        new_node = Node_ll(value)
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
        new_node = Node_ll(value)
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

if __name__ == "__main__":
    ll = LinkedList()
    for i in range(1, 10):
        ll.add_last(i)
        #ll.add_end(i+1)
        
    mm = LinkedList()
    for i in range(100, 110):
        mm.add_last(i)
        #ll.add_end(i+1)        
    
    ll.printlist()
    mm.printlist()
    
    
    ll.add_first(0)    
    ll.add_first(-1)
    print('Linked List: ')
    ll.printlist()
    
    node = ll.peek()
    print('peek: ' , str(node.value))
    node = ll.get_first()
    print('get first: ' , str(node.value))
    node = ll.get_last()
    print('get last: ' , str(node.value))
    node = ll.get(0)
    print('get position 0: ' , str(node.value))
    node = ll.get(2)
    print('get position 2: ' , str(node.value))
    ll.add(0, -2)
    ll.add(4, 1.5)
    print('Linked List: ')
    ll.printlist()
    node = ll.remove(0)
    print('remove position 0: ' , str(node.value))
    ll.printlist()
    node = ll.remove(3)
    print('remove position 3: ' , str(node.value))
    ll.printlist()
    
    
    ll = LinkedList()
    for i in range(1, 4):
        ll.add_first(i)
        #ll.add_end(i+1)
    print('Linked List: ')
    ll.printlist()
    ll.remove_first()
    ll.remove_first()
    print('Linked List: ')
    ll.printlist()
    ll.remove_first()
    print('Linked List: ')
    ll.printlist()
    
    ll = LinkedList()
    for i in range(1, 10):
        ll.add_last(i)
        
    ll.remove_first()
    ll.remove_last()
    ll.remove_first()
    ll.remove_last()
    print('Linked List: ')
    ll.printlist()