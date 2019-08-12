/** 冒泡排序，a 表示数组，n 表示数组大小
 * 当没有发生数据交换时，证明数据已经是正确的顺序了，提前结束程序。
 */
public void bubbleSort(int[] a, int n) {
  if (n <= 1){
  	return a;
  }
 
 for (int i = 0; i < n; ++i) {
    // 提前退出冒泡循环的标志位
    boolean flag = false;
    for (int j = 0; j < n - i - 1; ++j) {
      if (a[j] > a[j+1]) { // 交换，从小到大排序
        int tmp = a[j];
        a[j] = a[j+1];
        a[j+1] = tmp;
        flag = true;  // 表示有数据交换      
      }
    }
    if (!flag) break;  // 没有数据交换，提前退出
  }
}

/**
 * 冒泡排序改进:在每一轮排序后记录最后一次元素交换的位置,作为下次比较的边界,
 * 对于边界外的元素在下次循环中无需比较.
 */ 
public static void bubbleSort2(int[] a, int n) {
    if (n <= 1) return;

    // 最后一次交换的位置
    int lastExchange = 0;
    // 无序数据的边界,每次只需要比较到这里即可退出
    int sortBorder = n - 1;
    for (int i = 0; i < n; i++) {
        // 提前退出标志位
        boolean flag = false;
        for (int j = 0; j < sortBorder; j++) {
            if (a[j] > a[j + 1]) {
                int tmp = a[j];
                a[j] = a[j + 1];
                a[j + 1] = tmp;
                // 此次冒泡有数据交换
                flag = true;
                // 更新最后一次交换的位置
                lastExchange = j;
            }
        }
        sortBorder = lastExchange;
        if (!flag) break;    // 没有数据交换，提前退出
    }
}
