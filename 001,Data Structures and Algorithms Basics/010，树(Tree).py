# 第一部分：创建树类
# 1,创建一个二叉搜索树(Binary Search Tree):
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

if __name__ == '__main__':
    bst = BinarySearchTree()
    numbers = [6, 4, 8, 7, 9, 2, 1, 3, 5, 13, 11, 10, 12]
    for i in numbers:
        bst.add(i)
    bst.print_inorder()
    bst.print_postorder()
    bst.print_preorder()

# 2，创建一个AVL树：AVL树本质上是一颗二叉查找树，但是它又具有以下特点：它是一棵空树或它的左右两个子树的高度差的绝对值不超过1，并且左右两个子树都是一棵平衡二叉树。在AVL树中任何节点的两个子树的高度最大差别为一，所以它也被称为平衡二叉树。
class Node(object):
    def __init__(self,key):
        self.key=key
        self.left=None
        self.right=None
        self.height=0
class AVLTree(object):
    def __init__(self):
        self.root=None
    def find(self,key):
        if self.root is None:
            return None
        else:
            return self._find(key,self.root)
    def _find(self,key,node):
        if node is None:
            return None
        elif key<node.key:
            return self._find(key,self.left)
        elif key>node.key:
            return self._find(key,self.right)
        else:
            return node
    def findMin(self):
        if self.root is None:
            return None
        else:
            return self._findMin(self.root)
    def _findMin(self,node):
        if node.left:
            return self._findMin(node.left)
        else:
            return node
    def findMax(self):
        if self.root is None:
            return None
        else:
            return self._findMax(self.root)
    def _findMax(self,node):
        if node.right:
            return self._findMax(node.right)
        else:
            return node
    def height(self,node):
        if node is None:
            return -1
        else:
            return node.height
     
    def singleLeftRotate(self,node):
        k1=node.left
        node.left=k1.right
        k1.right=node
        node.height=max(self.height(node.right),self.height(node.left))+1
        k1.height=max(self.height(k1.left),node.height)+1
        return k1
    def singleRightRotate(self,node):
        k1=node.right
        node.right=k1.left
        k1.left=node
        node.height=max(self.height(node.right),self.height(node.left))+1
        k1.height=max(self.height(k1.right),node.height)+1
        return k1
    def doubleLeftRotate(self,node):
        node.left=self.singleRightRotate(node.left)
        return self.singleLeftRotate(node)
    def doubleRightRotate(self,node):
        node.right=self.singleLeftRotate(node.right)
        return self.singleRightRotate(node)
    def put(self,key):
        if not self.root:
            self.root=Node(key)
        else:
            self.root=self._put(key,self.root)
    def _put(self,key,node):
        if node is None:
            node=Node(key)
        elif key<node.key:
            node.left=self._put(key,node.left)
            if (self.height(node.left)-self.height(node.right))==2:
                if key<node.left.key:
                    node=self.singleLeftRotate(node)
                else:
                    node=self.doubleLeftRotate(node)
             
        elif key>node.key:
            node.right=self._put(key,node.right)
            if (self.height(node.right)-self.height(node.left))==2:
                if key<node.right.key:
                    node=self.doubleRightRotate(node)
                else:
                    node=self.singleRightRotate(node)
         
         
        node.height=max(self.height(node.right),self.height(node.left))+1
        return node
         
    def delete(self,key):
        self.root=self.remove(key,self.root)
    def remove(self,key,node):
        if node is None:
            raise KeyError('Error,key not in tree')
        elif key<node.key:
            node.left=self.remove(key,node.left)
            if (self.height(node.right)-self.height(node.left))==2:
                if self.height(node.right.right)>=self.height(node.right.left):
                    node=self.singleRightRotate(node)
                else:
                    node=self.doubleRightRotate(node)
            node.height=max(self.height(node.left),self.height(node.right))+1
             
                 
        elif key>node.key:
            node.right=self.remove(key,node.right)
            if (self.height(node.left)-self.height(node.right))==2:
                if self.height(node.left.left)>=self.height(node.left.right):
                    node=self.singleLeftRotate(node)
                else:
                    node=self.doubleLeftRotate(node)
            node.height=max(self.height(node.left),self.height(node.right))+1
         
        elif node.left and node.right:
            if node.left.height<=node.right.height:
                minNode=self._findMin(node.right)
                node.key=minNode.key
                node.right=self.remove(node.key,node.right)
            else:
                maxNode=self._findMax(node.left)
                node.key=maxNode.key
                node.left=self.remove(node.key,node.left)
            node.height=max(self.height(node.left),self.height(node.right))+1
        else:
            if node.right:
                node=node.right
            else:
                node=node.left
         
        return node

# 3,创建一个红黑树：一种近似平衡的二叉查找树，它能够确保任何一个节点的左右子树的高度差不会超过二者中较低那个的一倍
class TreeNode(object):
    def __init__(self, data, left=None, right=None, parent=None, color="RED"):
        self.data = data
        self.left = left
        self.right = right
        self.parent = parent
        self.color = color
class RBTree(object):
    def __init__(self):
        self.root = None
        self.size = 0
    def find(self, key, node):
        if not node:
            return None
        elif key < node.data:
            return self.find(key, node.left)
        elif key > node.data:
            return self.find(key, node.right)
        else:
            return node
    def findMin(self, node):
        """
        找到以 node 节点为根节点的树的最小值节点 并返回
        :param node: 以该节点为根节点的树
        :return: 最小值节点
        """
        temp_node = node
        while temp_node.left:
            temp_node = temp_node.left
        return temp_node
    def findMax(self, node):
        """
        找到以 node 节点为根节点的树的最大值节点 并返回
        :param node: 以该节点为根节点的树
        :return: 最大值节点
        """
        temp_node = node
        while temp_node.right:
            temp_node = temp_node.right
        return temp_node
    def transplant(self, tree, node_u, node_v):
        """
        用 v 替换 u
        :param tree: 树的根节点
        :param node_u: 将被替换的节点
        :param node_v: 替换后的节点
        :return: None
        """
        if not node_u.parent:
            tree.root = node_v
        elif node_u == node_u.parent.left:
            node_u.parent.left = node_v
        elif node_u == node_u.parent.right:
            node_u.parent.right = node_v
        # 加一下为空的判断
        if node_v:
            node_v.parent = node_u.parent
    def left_rotate(self, node):
        '''
             * 左旋示意图：对节点x进行左旋
             *     parent               parent
             *    /                       /
             *   node                   right
             *  / \                     / \
             * ln  right   ----->     node  ry
             *    / \                 / \
             *   ly ry               ln ly
             * 左旋做了三件事：
             * 1. 将right的左子节点ly赋给node的右子节点,并将node赋给right左子节点ly的父节点(ly非空时)
             * 2. 将right的左子节点设为node，将node的父节点设为right
             * 3. 将node的父节点parent(非空时)赋给right的父节点，同时更新parent的子节点为right(左或右)
            :param node: 要左旋的节点
            :return:
        '''
        parent = node.parent
        right = node.right
        # 把右子子点的左子点节   赋给右节点 步骤1
        node.right = right.left
        if node.right:
            node.right.parent = node
        # 把 node 变成基右子节点的左子节点 步骤2
        right.left = node
        node.parent = right
        # 右子节点的你节点更并行为原来节点的父节点。 步骤3
        right.parent = parent
        if not parent:
            self.root = right
        else:
            if parent.left == node:
                parent.left = right
            else:
                parent.right = right
    def right_rotate(self, node):
        '''
             * 左旋示意图：对节点y进行右旋
             *        parent           parent
             *       /                   /
             *      node                left
             *     /    \               / \
             *    left  ry   ----->   ln  node
             *   / \                     / \
             * ln  rn                   rn ry
             * 右旋做了三件事：
             * 1. 将left的右子节点rn赋给node的左子节点,并将node赋给rn右子节点的父节点(left右子节点非空时)
             * 2. 将left的右子节点设为node，将node的父节点设为left
             * 3. 将node的父节点parent(非空时)赋给left的父节点，同时更新parent的子节点为left(左或右)
            :param node:
            :return:
        '''
        parent = node.parent
        left = node.left
        # 处理步骤1
        node.left = left.right
        if node.left:
            node.left.parent = node
        # 处理步骤2
        left.right = node
        node.parent = left
        # 处理步骤3
        left.parent = parent
        if not parent:
            self.root = left
        else:
            if parent.left == node:
                parent.left = left
            else:
                parent.right = left
    def insert(self, node):
        # 找到最接近的节点
        temp_root = self.root
        temp_node = None
        while temp_root:
            temp_node = temp_root
            if node.data == temp_node.data:
                return False
            elif node.data > temp_node.data:
                temp_root = temp_root.right
            else:
                temp_root = temp_root.left
        # 在相应位置插入节点
        if not temp_node:
            # insert_case1
            self.root = node
            node.color = "BLACK"
        elif node.data < temp_node.data:
            temp_node.left = node
            node.parent = temp_node
        else:
            temp_node.right = node
            node.parent = temp_node
        # 调整树
        self.insert_fixup(node)
    def insert_fixup(self, node):
        if node.value == self.root.data:
            return
        # 为什么是这个终止条件？
        # 因为如果不是这个终止条件那就不需要调整
        while node.parent and node.parent.color == "RED":
            # 只要进入循环则必有祖父节点 否则父节点为根节点 根节点颜色为黑色 不会进入循环
            if node.parent == node.parent.parent.left:
                node_uncle = node.parent.parent.right
                # 1. 没有叔叔节点 若此节点为父节点的右子 则先左旋再右旋 否则直接右旋
                # 2. 有叔叔节点 叔叔节点颜色为黑色
                # 3. 有叔叔节点 叔叔节点颜色为红色 父节点颜色置黑 叔叔节点颜色置黑 祖父节点颜色置红 continue
                # 注: 1 2 情况可以合为一起讨论 父节点为祖父节点右子情况相同 只需要改指针指向即可
                if node_uncle and node_uncle.color == "RED":
                    # insert_case3
                    node.parent.color = "BLACK"
                    node_uncle.color = "BLACK"
                    node.parent.parent.color = "RED"
                    node = node.parent.parent
                    continue
                elif node == node.parent.right:
                    # insert_case4
                    self.left_rotate(node.parent)
                    node = node.left
                # insert_case5
                node.parent.color = "BLACK"
                node.parent.parent.color = "RED"
                self.right_rotate(node.parent.parent)
                return
            # 对称情况
            elif node.parent == node.parent.parent.right:
                node_uncle = node.parent.parent.left
                if node_uncle and node_uncle.color == "RED":
                    node.parent.color = "BLACK"
                    node_uncle.color = "BLACK"
                    node.parent.parent.color = "RED"
                    node = node.parent.parent
                    continue
                elif node == node.parent.left:
                    self.right_rotate(node)
                    node = node.right
                node.parent.color = "BLACK"
                node.parent.parent.color = "RED"
                self.left_rotate(node.parent.parent)
                return
        # 最后记得把根节点的颜色改为黑色 保证红黑树特性
        self.root.color = "BLACK"
    def delete(self, node):
        # 找到以该节点为根节点的右子树的最小节点
        node_color = node.color
        if not node.left:
            temp_node = node.right
            self.transplant(node, node.right)
        elif not node.right:
            temp_node = node.left
            self.transplant(node, node.left)
        else:
            # 最麻烦的一种情况 既有左子 又有右子 找到右子中最小的做替换 类似于二分查找树的删除
            node_min = self.findMin(node.right)
            node_color = node_min.color
            temp_node = node_min.right
            if node_min.parent != node:
                self.transplant(node_min, node_min.right)
                node_min.right = node.right
                node_min.right.p = node_min
            self.transplant(node, node_min)
            node_min.left = node.left
            node_min.left.parent = node_min
            node_min.color = node.color
        # 当删除的节点的颜色为黑色时 需要调整红黑树
        if node_color == "BLACK":
            self.delete_fixup(temp_node)
    def delete_fixup(self, node):
        # 实现过程还需要理解 比如为什么要删除 为什么是那几种情况
        while node != self.root and node.color == "BLACK":
            if node == node.parent.left:
                node_brother = node.parent.right
                if node_brother.color == "RED":
                    # delete_case2
                    node_brother.color = "BLACK"
                    node.parent.color = "RED"
                    self.left_rotate(node.parent)
                    node_brother = node.parent.right
                if (not node_brother.left or node_brother.left.color == "BLACK") and \
                        (not node_brother.right or node_brother.right.color == "BLACK"):
                    # delete_case3
                    node_brother.color = "RED"
                    node = node.parent
                else:
                    if not node_brother.right or node_brother.right.color == "BLACK":
                        # delete_case5
                        node_brother.color = "RED"
                        node_brother.left.color = "BLACK"
                        self.right_rotate(node_brother)
                        node_brother = node.parent.right
                    # delete_case6
                    node_brother.color = node.parent.color
                    node.parent.color = "BLACK"
                    node_brother.right.color = "BLACK"
                    self.left_rotate(node.parent)
                node = self.root
                break
            else:
                node_brother = node.parent.left
                if node_brother.color == "RED":
                    node_brother.color = "BLACK"
                    node.parent.color = "RED"
                    self.left_rotate(node.parent)
                    node_brother = node.parent.right
                if (not node_brother.left or node_brother.left.color == "BLACK") and \
                        (not node_brother.right or node_brother.right.color == "BLACK"):
                    node_brother.color = "RED"
                    node = node.parent
                else:
                    if not node_brother.left or node_brother.left.color == "BLACK":
                        node_brother.color = "RED"
                        node_brother.right.color = "BLACK"
                        self.left_rotate(node_brother)
                        node_brother = node.parent.left
                    node_brother.color = node.parent.color
                    node.parent.color = "BLACK"
                    node_brother.left.color = "BLACK"
                    self.right_rotate(node.parent)
                node = self.root
                break
        node.color = "BLACK"


# 第二部分：树的相关概念
# 1,树的大小：
from tree_J import BinarySearchTree, Node
class AdvBST1(BinarySearchTree):
    
    def size(self):
        return self._size(self._root)
    
    def _size(self, node):
        if (not node):
            return 0
        return self._size(node._left) + self._size(node._right) + 1

if __name__ == '__main__':
    bst1 = AdvBST1()
    numbers = [6, 4, 8, 7, 9, 2, 1, 3, 5, 13, 11, 10, 12]
    for i in numbers:
        bst1.add(i)
    bst1.print_inorder()
    print(bst1.size())

# 2,最大深度:
class AdvBST2(AdvBST1):
    def maxDepth(self):
        return self._maxDepth(self._root)

    def _maxDepth(self, node):
        if (not node):
            return 0
        left_depth = self._maxDepth(node._left)
        right_depth = self._maxDepth(node._right)
        return max(left_depth, right_depth) + 1

if __name__ == '__main__':
    bst = AdvBST2()
    numbers = [6, 4, 8, 7, 9, 2, 1, 3, 5, 13, 11, 10, 12]
    for i in numbers:
        bst.add(i)
    bst.print_inorder()
    print(bst.maxDepth())

# 3,是否是平衡树:
class AdvBST3(AdvBST2):    
    def minDepth(self):
        return self._minDepth(self._root)
    
    def _minDepth(self, node):
        if (not node):
            return 0
        left_depth = self._minDepth(node._left)
        right_depth = self._minDepth(node._right)
        return min(left_depth, right_depth) + 1
    
    def isBalanced(self):
        return (self.maxDepth() - self.minDepth()) <= 1

if __name__ == '__main__':
    bst = AdvBST3()
    numbers = [3,1,5]
    for i in numbers:
        bst.add(i)
    bst.print_inorder()
    print(bst.isBalanced())

# 4,查找floor/ceiling:
class AdvBST4(AdvBST3):    
    def floor(self, key):
        return self._floor(self._root, key)
    
    def _floor(self, node, key):
        if (not node):
            return None
        if (key == node._item):
            return node
        if (key < node._item):
            return self._floor(node._left, key)
        t = self._floor(node._right, key)
        if t:
            return t
        return node

if __name__ == '__main__':
    bst = AdvBST4()
    numbers = [40,20,70,50,10,60,30,80]
    for i in numbers:
        bst.add(i)
    print(bst.floor(40)._item)
    print(bst.floor(44)._item)
    print(bst.floor(10)._item)
    print(bst.floor(5))
    print(bst.floor(100)._item)

# 5,是否是二叉搜索树：
import sys
class AdvBST5(AdvBST4):    
    def isBST(self):
        return self._isBST(self._root, -sys.maxsize, sys.maxsize)
    
    def _isBST(self, node, minval, maxval):
        if not node:
            return True
        if (node._item < minval or node._item > maxval):
            return False
        return self._isBST(node._left, minval, node._item) and self._isBST(node._right, node._item, maxval)

if __name__ == '__main__':
    bst = AdvBST5()
    numbers = [1,2,3,4,5,6,7,8]
    for i in numbers:
        bst.add(i)
    print(bst.isBST())

# 6,树的镜像：
class AdvBST6(AdvBST5):    
    def mirror(self):
        self._mirror(self._root)
    
    def _mirror(self, node):
        if (node is not None):
            self._mirror(node._left)   # head recursion
            self._mirror(node._right)
            
            temp = node._left
            node._left = node._right
            node._right = temp
            
if __name__ == '__main__':
    bst = AdvBST6()
    numbers = [6, 4, 8, 7, 9, 5, 1, 3, 2]
    for i in numbers:
        bst.add(i)
    bst.print_inorder()

# 7,结构相同的树：
class AdvBST7(AdvBST6):    
    def sameTree(self, another):
        return self._sameTree(self._root, another._root)
    
    def _sameTree(self, nodeA, nodeB):
        if (nodeA is None and nodeB is None):
            return True
        if (nodeA is not None and nodeB is not None):
            return nodeA._item == nodeB._item and self._sameTree(nodeA._left, nodeB._left) and self._sameTree(nodeA._right, nodeB._right)
        return False

if __name__ == '__main__':
    bst = AdvBST7()
    numbers = [6, 4, 8, 7, 9, 5, 1, 3, 2]
    for i in numbers:
        bst.add(i)
    another = AdvBST7()
    numbers = [6, 4, 8, 7, 9, 5, 1, 3, 2]
    for i in numbers:
        another.add(i)
    print(bst.sameTree(another))

# 8,判断是否是可折叠树：
class AdvBST8(AdvBST7):    
    def isFoldable(self):
        if self._root is None:
            return True
        return self._isFoldable(self._root._left, self._root._right)
    
    def _isFoldable(self, nodeA, nodeB):
        if (nodeA is None and nodeB is None):
            return True
        if (nodeA is None or nodeB is None):
            return False        
        return self._isFoldable(nodeA._left, nodeB._right) and self._isFoldable(nodeA._right, nodeB._left)

if __name__ == '__main__':
    bst = AdvBST8()
    numbers = [3,2,5,1,8]
    for i in numbers:
        bst.add(i)
    bst.isFoldable()

# 第三部分、树的非递归（循环）方式：
# 1,get_iteratively:
class AdvBST1(BinarySearchTree):
    
    def getIterative(self, key):
        node = self._root
        while (node is not None):
            if key == node._item:
                return node._item
            if key < node._item:
                node = node._left
            else:
                node = node._right
        return None
if __name__ == '__main__':
    bst = AdvBST1()
    numbers = [6, 4, 8, 7, 9, 2, 1, 3, 5, 13, 11, 10, 12]
    for i in numbers:
        bst.add(i)
    bst.print_inorder()
    bst.getIterative(5)

# 2,add_iteratively:
class AdvBST2(AdvBST1):
    def addIterative(self, value):
        newNode = Node(value)
        if (self._root is None):
            self._root = newNode
            return
        
        current = self._root
        parent = None
        while True:
            parent = current
            if (value == current._item):
                return
            if (value < current._item):
                current = current._left
                if (current is None):
                    parent._left = newNode
                    return
            else:
                current = current._right
                if (current is None):
                    parent._right = newNode
                    return

if __name__ == '__main__':
    bst = AdvBST2()
    numbers = [6, 4, 8, 7, 9, 2, 1, 3, 5, 13, 11, 10, 12]
    for i in numbers:
        bst.addIterative(i)

    bst2 = AdvBST2()
    numbers = [6, 4, 8, 7, 9, 2, 1, 3, 5, 13, 11, 10, 12]
    for i in numbers:
        bst2.add(i)
    bst.print_inorder()
    bst2.print_inorder()
    bst.print_preorder()
    bst2.print_preorder()

# 3,Inorder traversal method, iteratively:
class AdvBST3(AdvBST2):
    def printInorderIterative(self):
        node = self._root
        stack = []
        
        while True:
            while (node is not None):
                stack.append(node)
                node = node._left
            if len(stack) == 0:
                return
            
            node = stack.pop()
            print ('[', node._item, ']', end = " ")
            node = node._right

if __name__ == '__main__':
    bst = AdvBST3()
    numbers = [6, 4, 8, 7, 9, 2, 1, 3, 5, 13, 11, 10, 12]
    for i in numbers:
        bst.add(i)
    bst.print_inorder()
    bst.printInorderIterative()

# 4,Preorder traversal method, iteratively:
class AdvBST4(AdvBST3):
    def printPreorderIterative(self):
        ret = []
        stack = [self._root]
        while stack:
            node = stack.pop()
            if node:
                ret.append(node._item)
                stack.append(node._right)
                stack.append(node._left)
        return ret

if __name__ == '__main__':
    bst = AdvBST4()
    numbers = [6, 4, 8, 7, 9, 2, 1, 3, 5, 13, 11, 10, 12]
    for i in numbers:
        bst.add(i)
    bst.print_preorder()
    bst.printPreorderIterative()

# 5,Postorder traversal method, iteratively:
class AdvBST5(AdvBST4):
    def printPostorderIterative(self):
        node = self._root
        stack = []
        stack.append(node)
        
        while len(stack) != 0:
            node = stack[-1]
            if node._left is None and node._right is None:
                pop = stack.pop()
                print ('[', node._item, ']', end = " ")
                
            else:
                if node._right is not None:
                    stack.append(node._right)
                    node._right = None
                if node._left is not None:
                    stack.append(node._left)
                    node._left = None
        print('')

    def printPostorderIterative2(self):
        stack = [(self._root, False)]
        while stack:
            node, visited = stack.pop()
            if node:
                if visited:
                    # add to result if visited
                    print ('[', node._item, ']', end = " ")
                else:
                    # post-order
                    stack.append((node, True))
                    stack.append((node._right, False))
                    stack.append((node._left, False))

if __name__ == '__main__':
    bst = AdvBST5()
    numbers = [6, 4, 8, 7]
    for i in numbers:
        bst.add(i)
    bst.print_postorder()
    bst.printPostorderIterative()

    bst = AdvBST5()
    numbers = [6, 4, 8, 7]
    for i in numbers:
        bst.add(i)
    bst.printPostorderIterative2()


# 第四部分、遍历:
# 1,level order traversal(from left to right, level by level):
from collections import deque
class AdvBST1(BinarySearchTree):
    def levelOrder(self):
        if not self._root:
            return []

        ret = []
        level = [self._root]

        while level:
            currentNodes = []
            nextLevel = []
            for node in level:
                currentNodes.append(node._item)
                if node._left:
                    nextLevel.append(node._left)
                if node._right:
                    nextLevel.append(node._right)
            ret.append(currentNodes)
            level = nextLevel

        return ret

if __name__ == '__main__':
    bst = AdvBST1()
    numbers = [6, 4, 8, 7, 9, 2, 1, 3, 5, 13, 11, 10, 12]
    for i in numbers:
        bst.add(i)
    bst.levelOrder()

# 2,the bottom-up level order traversal(from left to right, level by level from leaf to root):
class AdvBST2(BinarySearchTree):
    
    def levelOrder(self):
        if not self._root:
            return []
        ans, level = [], [self._root]
        while level:
            ans.insert(0, [node._item for node in level])
            temp = []
            for node in level:
                temp.extend([node._left, node._right])
            level = [leaf for leaf in temp if leaf]
        
        return ans
    
    
    def levelOrder2(self):
        if not self._root:
            return []
        ans, level = [], [self._root]
        while level:
            ans.append([node._item for node in level])
            temp = []
            for node in level:
                temp.extend([node._left, node._right])
            level = [leaf for leaf in temp if leaf]
        ans.reverse()
        return ans    
    
if __name__ == '__main__':
    bst = AdvBST2()
    numbers = [6, 4, 8, 7, 9, 2, 1, 3, 5, 13, 11, 10, 12]
    for i in numbers:
        bst.add(i)
    bst.levelOrder()

# 3,zigzag level order traversal(from left to right, then right to left for the next level and alternate between):
class AdvBST3(AdvBST2):
    
    def zigzagLevelOrder(self,):
        if not self._root: 
            return []
        res, temp, stack, flag = [], [], [self._root], 1
        while stack:
            for i in range(len(stack)):
                node = stack.pop(0)
                temp += [node._item]
                if node._left:  stack += [node._left]
                if node._right: stack += [node._right]
            res += [temp[::flag]]
            temp = []
            flag *= -1
        return res

if __name__ == '__main__':
    bst = AdvBST3()
    numbers = [6, 4, 8, 7, 9, 2, 1, 3, 5, 13, 11, 10, 12]
    for i in numbers:
        bst.add(i)
    bst.zigzagLevelOrder()

# 4,Construct Binary Tree from Preorder and Inorder Traversal:
def buildTree(preorder, inorder):
    if inorder:
        ind = inorder.index(preorder.pop(0))
        root = Node(inorder[ind])
        root._left = buildTree(preorder, inorder[0:ind])
        root._right = buildTree(preorder, inorder[ind+1:])
        return root

def buildTree2(preorder, inorder, preorderStart = 0, preorderEnd = None, inorderStart = 0, inorderEnd = None):
    if preorderEnd is None:
        preorderEnd = len(preorder) - 1
        
    if inorderEnd is None:
        inorderEnd = len(inorder) - 1

    if preorderStart > len(preorder) - 1 or inorderStart > inorderEnd:
        return None

    rootValue = preorder[preorderStart]
    root = Node(rootValue)
    inorderIndex = inorder.index(rootValue)

    root._left = buildTree2(preorder, inorder, preorderStart+1, inorderIndex, inorderStart, inorderIndex-1)
    root._right = buildTree2(preorder, inorder, preorderStart+inorderIndex+1-inorderStart, preorderEnd, inorderIndex+1, inorderEnd)

    return root

if __name__ == '__main__':
    preorder = [3,9,20,15,7]
    inorder = [9,3,15,20,7]
    root = buildTree2(preorder, inorder)

    bst = BinarySearchTree(root)
    bst.print_preorder()
    bst.print_inorder()

# 5,Construct Binary Tree from Inorder and Postorder Traversal:
def buildTree(inorder, postorder):
    if not inorder or not postorder:
        return None

    root = Node(postorder.pop())
    inorderIndex = inorder.index(root._item)

    root._right = buildTree(inorder[inorderIndex+1:], postorder)
    root._left = buildTree(inorder[:inorderIndex], postorder)

    return root

if __name__ == '__main__':
    inorder = [9,3,15,20,7]
    postorder = [9,15,7,20,3]
    root = buildTree(inorder, postorder)

    bst = BinarySearchTree(root)
    bst.print_inorder()
    bst.print_postorder()

# 6,将有序数组转换成平衡二叉树：
def sortedArrayToBST(num):
    if not num:
        return None

    mid = len(num) // 2

    root = Node(num[mid])
    root._left = sortedArrayToBST(num[:mid])
    root._right = sortedArrayToBST(num[mid+1:])

    return root

if __name__ == '__main__':
    num = [-10,-3,0,5,9]
    root = sortedArrayToBST(num)
    bst = BinarySearchTree(root)
    bst.print_inorder()
    bst.print_preorder()

# 7,将有序列表转换成平衡二叉树：
from tree_J import LinkedList as LL
from tree_J import Node_ll as LN

def sortedListToBST(head):
    if head is None:
        return None
    
    dummy = LN(0)
    dummy.next = head
    head = dummy
    
    fast = head
    slow = head
    left_tail = head
    
    while fast is not None and fast.next is not None:
        fast = fast.next.next
        left_tail = slow
        slow = slow.next
    
    left_tail.next = None
    node = Node(slow.value)
    node._left = sortedListToBST(head.next)
    node._right = sortedListToBST(slow.next)
    return node

if __name__ == '__main__':
    node1 = LN(1)
    node2 = LN(2)
    node3 = LN(3)
    node4 = LN(4)
    node5 = LN(5)
    node6 = LN(6)
    node7 = LN(7)
    node8 = LN(8)
    node1.next = node2
    node2.next = node3
    node3.next = node4
    node4.next = node5
    node5.next = node6
    node6.next = node7
    #node7.next = node8

    root = sortedListToBST(node1)
    bst = BinarySearchTree(root)
    bst.print_inorder()
    bst.print_preorder()

# 第五部分、路径和，共同祖先:
# 1,路径和(1)：判断存在从根到叶的路径使得路径和等于指定数值,返回布尔值
class AdvBST1(BinarySearchTree):
    def hasPathSumHelper(self, node, s):
        if not node:
            return False

        if not node._left and not node._right and node._item == s:
            return True
        
        s -= node._item

        return self.hasPathSumHelper(node._left, s) or self.hasPathSumHelper(node._right, s)
    
    def hasPathSum(self, s):
        return self.hasPathSumHelper(self._root, s)

if __name__ == '__main__':
    bst = AdvBST1()
    numbers = [6, 4, 8, 7, 9, 2, 1, 3, 5, 13, 11, 10, 12]
    for i in numbers:
        bst.add(i)

    print(bst.hasPathSum(15))
    print(bst.hasPathSum(16))

# 2,路径和(2):与上题的区别是，需要返回路径
class AdvBST2(AdvBST1):
    def hasPathSum2Helper(self, node, s):
        if not node:
            return []
        if not node._left and not node._right and s == node._item:
            return [[node._item]]
        tmp = self.hasPathSum2Helper(node._left, s-node._item) + self.hasPathSum2Helper(node._right, s - node._item)
        #print(tmp)
        return [[node._item] + i for i in tmp]
    
    def hasPathSum2(self, s):
        return self.hasPathSum2Helper(self._root, s)

class AdvBST3(AdvBST2):
    def hasPathSum2Helper(self, node, s):
        if not node:
            return []
        res = []
        self.dfs(node, s, [], res)
        return res
    
    def dfs(self, node, s, ls, res):
        if not node._left and not node._right and s == node._item:
            ls.append(node._item)
            res.append(ls)
        if node._left:
            self.dfs(node._left, s-node._item, ls+[node._item], res)
        if node._right:
            self.dfs(node._right, s-node._item, ls+[node._item], res)

if __name__ == '__main__':
    bst = AdvBST3()
    numbers = [6, 4, 8, 7, 9, 2, 1, 3, 5, 13, 11, 10, 12]
    for i in numbers:
        bst.add(i)

    print(bst.hasPathSum2(15))
    print(bst.hasPathSum2(16))

# 3,路径和(3)：不要求必须是从根开始，由叶结束，但需要自上而下
class AdvBST4(AdvBST3):
    
    def pathSum(self, target):
        return self.pathSumHelper(self._root, target)
    
    def findPaths(self, node, target):
        if node:
            return int(node._item == target) + \
                self.findPaths(node._left, target - node._item) + \
                self.findPaths(node._right, target - node._item)
        return 0

    def pathSumHelper(self, node, target):
        if node:
            return self.findPaths(node, target) + \
                self.pathSumHelper(node._left, target) + \
                self.pathSumHelper(node._right, target)
        return 0     

if __name__ == '__main__':
    bst = AdvBST4()
    numbers = [6, 4, 8, 7, 9, 2, 1, 3, 5, 13, 11,12]
    for i in numbers:
        bst.add(i)

    print(bst.pathSum(9))

# 4,最近公共祖先：给定二叉搜索树，两个节点，返回公共祖先
class AdvBST5(AdvBST4):
    
    def lowestCommonAncestor(self, p, q):
        return self.lowestCommonAncestorHelper(self._root, p, q)
    
    def lowestCommonAncestorHelper(self, node, p, q):
        while node:
            if node._item > p._item and node._item > q._item:
                node = node._left
            elif node._item < p._item and node._item < q._item:
                node = node._right
            else:
                return node

if __name__ == '__main__':
    bst = AdvBST5()
    numbers = [6, 4, 8, 7, 9, 2, 1, 3, 5, 13, 11, 12]
    for i in numbers:
        bst.add(i)

    node1 = Node(3)
    node2 = Node(5)
    print(bst.lowestCommonAncestor(node1, node2)._item)

# 5,最近公共祖先：与上题不同的是,这里是二叉树(leetcode236, beats 90%):
class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        x, pok, qok = self.find(root, p, q)
        if x is None:
            return None
        return x

    # find returns three values, x, pok, qok.
    # x returns TreeNode if LCA found, otherwise None.
    # pok returns True if p found under the node, otherwise False.
    # qok returns True if q found under the node, otherwise False.
    def find(self, parent, p, q):
        if not parent:
            return None, False, False
        if parent == p:
            if p == q:
                return parent, True, True
            # parent is p, check q is found among left subtree or right subtree.
            _, _, qok1 = self.find(parent.right, p, q)
            if qok1:
                return parent, True, True
            _, _, qok2 = self.find(parent.left, p, q)
            if qok2:
                return parent, True, True
            return None, True, False

        if parent == q:
            if p == q:
                return parent, True, True
            # parent is q, check p is found among left subtree or right subtree.    
            _, pok1, _ = self.find(parent.right, p, q)
            if pok1:
                return parent, True, True
            _, pok2, _ = self.find(parent.left, p, q)
            if pok2:
                return parent, True, True
            return None, False, True

        # check both right subtree and left subtree
        r, pok1, qok1 = self.find(parent.right, p, q)
        if r:
            return r, True, True
        l, pok2, qok2 = self.find(parent.left, p, q)
        if l:
            return l, True, True
        if (pok1 and qok2) or (
                pok2 and qok1):  # if p and q found on each side, parent is LCA
            return parent, True, True
        return None, pok1 or pok2, qok1 or qok2