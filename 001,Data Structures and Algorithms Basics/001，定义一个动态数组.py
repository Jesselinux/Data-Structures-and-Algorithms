# 1，动态数组的实现：
import ctypes

class DynamicArray:
	def __init__(self):
		"create an empty array."
		self._n = 0
		self._capacity = 10
		self._A = self._make_array(self._capacity)  # 等价于：[None] * self.capacity

	def __len__(self):
		return self._n

	def is_empty(self):
		return self._n == 0

	def __getitem__(self, k):   # O(1) , 索引取值
		if not 0 <= k < self._n:
			raise ValueError('invalid index')
		return self._A[k]

	def append(self, obj):  # O(1)
		if self._n == self._capacity:
			self.resize(2 * self._capacity)
		self._A[self._n] = obj
		self._n += 1

	def _make_array(self, c):
		return (c * ctypes.py_object)( )

	def _resize(self, c):
		B = self._make_array(c)
		for j in range(self._n):
			B[j] = self._A[j]
		self._A = B
		self._capacity = c

	def insert(self, k, value):   # O(n)
		if self._n == self._capacity:
			self._resize(2 * self._capacity)
		for i in range(self._n, k, -1):
			self._A[i] = self._A[i-1]   # index为k+1及其后面的元素向后移
		self._A[k] = value
		self._n += 1

	def remove(self, value):
		for i in range(self._n):
			if self._A[i] == value:
				for j in range(i, self._n - 1):
					self._A[j] = self._A[j+1]   # index为j-1及其后面的元素向前移
				self._A[self._n - 1] = None     # 删除最后一个元素
				self._n -= 1                    # 数组长度减 1
				return
		raise ValueError('value not found')    # 如果上述循环没有运行，则不会运行return，然后就会运行raise语句，即报错

	def _print(self):
		for i in range(self._n):
			print(self._A[i], end = ' ')
		print()


if __name__ == '__main__':
	new_list = DynamicArray()
	print('size was: ', str(len(new_list)))
	new_list.append(1)
	new_list.append(2)
	new_list.append(3)

	new_list.insert(0, 8)
	new_list.insert(1, 18)
	new_list.insert(3, 28)

	new_list._print()

	new_list.remove(3)

	new_list._print()

	print('The length of the list is: ', str(len(new_list)))