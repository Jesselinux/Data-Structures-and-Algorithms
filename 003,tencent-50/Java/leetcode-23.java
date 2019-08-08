/**
     * 优先级队列小顶堆
     * @param lists
     * @return
     */
    public ListNode mergeKLists(ListNode[] lists) {
        if(lists == null || lists.length == 0){
            return null;
        }
        PriorityQueue<ListNode> queue = new PriorityQueue<ListNode>(new Comparator<ListNode>() {
            @Override
            public int compare(ListNode o1, ListNode o2) {
                return o1.val - o2.val;
            }
        });
        
        for(int i = 0; i < lists.length; i++){
            if(lists[i] != null){
                queue.offer(lists[i]);
            }
        }

        ListNode head = null;
        ListNode cur = null;
        while (!queue.isEmpty()){
            ListNode node = queue.poll();
            if(head == null){
                head = node;
                cur = head;
            }else {
                cur.next = node;
                cur = cur.next;
            }
            if(node.next != null){
                queue.offer(node.next);
            }
        }
        return head;// 返回链表的头结点
    }

    /**
     * 自己写堆 效率高点
     * @param lists
     * @return
     */
    public ListNode mergeKLists2(ListNode[] lists) {
        if(lists == null || lists.length == 0){
            return null;
        }

        MinHeap minHeap = new MinHeap(lists.length);
        for(int i = 0; i < lists.length; i++){
            if(lists[i] != null){
                minHeap.insert(lists[i]);
            }
        }
        ListNode head = null;
        ListNode cur = null;
        while (!minHeap.isEmpty()){
            ListNode node = minHeap.remove();
            if(head == null){
                head = node;
                cur = head;
            }else {
                cur.next = node;
                cur = cur.next;
            }
            if(node.next != null){
                minHeap.insert(node.next);
            }
        }
        return head;
    }

    /**
     * 实现一个简易的堆看看效果
     */
    public class MinHeap{
        private ListNode[] lists;
        private int index = 1;      //索引开始为1 根据子节点查找父节点只需要除二 不用判断奇偶

        public MinHeap(int len){
            lists = new ListNode[len + 1];
        }

        public ListNode insert(ListNode node){
            if(index == lists.length){
                return lists[1];
            }
            int pos = index;
            lists[index++] = node;
            //堆化
            while (pos > 1){
                int midPos = pos >> 1;
                if(lists[pos].val < lists[midPos].val){
                    ListNode tmp = lists[midPos];
                    lists[midPos] = lists[pos];
                    lists[pos] = tmp;
                    pos = midPos;
                }else {
                    break;
                }
            }
            return lists[1];
        }

        public ListNode remove(){
            ListNode result = lists[1];
            lists[1] = lists[index - 1];
            lists[index - 1] = null;
            index--;
            int pos = 1;
            while (pos <= (index - 1)/2){
                int minPos = pos;
                int minValue = lists[pos].val;
                if(lists[pos].val > lists[pos * 2].val){
                    minPos = pos * 2;
                    minValue = lists[pos * 2].val;
                }
                if(index - 1 >= 2 * pos + 1){
                    //右节点存在
                    if(minValue > lists[2 * pos + 1].val){
                        minPos = 2 * pos + 1;
                        minValue = lists[2 * pos + 1].val;
                    }
                }
                //和minPos互换
                if(pos != minPos){
                    ListNode tmp = lists[pos];
                    lists[pos] = lists[minPos];
                    lists[minPos] = tmp;
                    pos = minPos;
                }else {
                    break;
                }
            }
            return result;
        }

        public boolean isEmpty(){
            return index <= 1;
        }
    }

