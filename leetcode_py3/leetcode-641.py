class MyCircularDeque(object):
	def __init__(self, k):
		self.queue = []
		self.size = k

	def insertFront(self, value):
		if not self.isFull():
			self.queue.insert(0, value)
			return True
		else:
			return False

	def insertLast(self, value):
		if not self.isFull():
			self.queue.append(value)
			return True
		else:
			return False

	def deleteFront(self):
		if not self.isEmpty():
			self.queue.pop(0)
			return True
		else:
			return False

	def deleteLast(self):
		if not self.isEmpty():
			self.queue.pop()
			return True
		else:
			return False

	def getFront(self):
		if self.isEmpty():
			return -1
		else:
			return self.queue[0]

	def getRear(self):
		if self.isEmpty():
			return -1
		else:
			return self.queue[-1]

	def isEmpty(self):
		return 0 == len(self.queue)

	def isFull(self):
		return self.size == len(self.queue)

		