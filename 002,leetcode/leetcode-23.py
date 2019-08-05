class Solution:
	def mergeKLists(self, lists):
		all_node = []
		for i in lists:
			while i is not None:
				all_node.append(x)
				i = i.next

		all_node_sort = sorted(all_node, key = lambda node:node.val)
		pHead = ListNode(-1)
		p = pHead
		for node in all_node_sort:
			p.next = node
			p = node
		p.next = None
		return pHead.next

#第一个会超时，使用第二个可以通过leetcode：
class Solution(object):

    def mergeKLists(self, lists):

        """

        :type lists: List[ListNode]

        :rtype: ListNode

        """

        Len=len(lists)

        for i in range(Len-1,-1,-1):

            if lists[i]==None:del lists[i]

        if len(lists)==0:return []

        

        ans=ListNode(0)

        h=ans;

        while len(lists)>0:

            m=lists[0].val

            cur=0

            for i in range(1,len(lists)):

                if lists[i].val<=m:

                    cur=i

                    m=lists[i].val

            ans.next=ListNode(m)

            ans=ans.next

            lists[cur]=lists[cur].next

            if lists[cur]==None:

                del lists[cur];

        return h.next

            
