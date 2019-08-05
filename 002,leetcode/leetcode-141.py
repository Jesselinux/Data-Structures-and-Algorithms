class Solution(object):
	def hasCycle(self, head):
		if not head:
			return False
		p1 = p2 = head
		while p1.next and p2.next.next:
			p1 = p1.next
			p2 = p2.next.next
			if p1 == p2:
				return True
		return False
		
