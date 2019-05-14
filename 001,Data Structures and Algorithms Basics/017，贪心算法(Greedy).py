# 1,找硬币：
def minCoins(V):
    available = [1, 2, 5, 10, 20, 50, 100, 500, 1000]
    result = []
    for i in available[::-1]:
        while (V >= i):
            V -= i
            result.append(i)
    
    return result

if __name__ == '__main__':
	V = 93
	minCoins(V)

# 2,活动时间日程表：
def printMaxActivities(acts):
    n = len(acts)
    sort_acts = sorted(acts, key=lambda tup: tup[1])
    prev = sort_acts[0]
    print(prev)
    for curr in sort_acts:
        if curr[0] >= prev[1]:
            print(curr)
            prev = curr

if __name__ == '__main__':
	acts = [(0,6),(3,4),(1,2),(5,7),(8,9),(5,9)]
	printMaxActivities(acts)

# 3,最小数字：
def findSmallest(m, s):
 
    if (s == 0):
        if(m == 1) :
              print("Smallest number is 0") 
        else : 
              print("Not possible")
        return
  
    # 9999
    if (s > 9 * m):
        print("Not possible")
        return
  
    res = [0 for i in range(m + 1)]
  
    # deduct sum by one to account for cases later 
    # (There must be 1 left for the most significant digit)
    s -= 1
  
    for i in range(m-1,0,-1):
     
        # If sum is still greater than 9, digit must be 9.
        if (s > 9):
            res[i] = 9
            s -= 9
        else:
            res[i] = s
            s = 0
  
    res[0] = s + 1
  
    print("Smallest number is ",end="")
    for i in range(m):
        print(res[i],end="")

if __name__ == '__main__':
	s = 20
	m = 3
	findSmallest(m, s)

# 4,两数最小和：
import heapq
def minSum(a):
    heapq.heapify(a)
    num1 = 0
    num2 = 0
    while a:
        num1 = num1 * 10 + heapq.heappop(a)
        if a:
            num2 = num2 * 10 + heapq.heappop(a)
    
    return num1 + num2           

if __name__ == '__main__':
	a = [5, 3, 0, 7, 4]
	minSum(a)

# 5,最低成本连接绳索：
import heapq
def ropeCost(ropes):
    heapq.heapify(ropes)
    total = 0
    
    while ropes:
        first = heapq.heappop(ropes)
        second = heapq.heappop(ropes)
        local = first + second
        total += local
        if not ropes:
            break
        heapq.heappush(ropes, local)
    return total  

if __name__ == '__main__':
	ropes = [1,3,2,5,4]
	ropeCost(ropes)

# 6,最小平台数：
def findPlatform(arr, dep, n):
 
    arr.sort()
    dep.sort()
  
    # plat_needed indicates number of platforms needed at a time
    plat_needed = 0
    result = 0
    i = 0
    j = 0
  
    # Similar to merge in merge sort to process all events in sorted order
    while (i < n and j < n):
        if (arr[i] < dep[j]):
            plat_needed += 1
            i += 1
  
            result = max(result, plat_needed)
  
        else:
            plat_needed -= 1
            j += 1
         
    return result

if __name__ == '__main__':
	arr = [900, 940, 950, 1100, 1500, 1800]
	dep = [910, 1200, 1120, 1130, 1900, 2000]
	n = len(arr)
	findPlatform(arr, dep, n)

# 7,部分背包问题：
def fracKnapsack(capacity, weights, values):
    numItems = len(values)
    valuePerWeight = sorted([[v / w, w, v] for v,w in zip(values,weights)], reverse=True)
    print(valuePerWeight)
    totalCost = 0.
    for tup in valuePerWeight:
        if capacity >= tup[1]:
            capacity -= tup[1]
            totalCost += tup[2]
        else:
            totalCost += capacity * tup[0]
            break
    return totalCost

if __name__ == '__main__':
	n = 3
	capacity = 50
	values = [72, 100, 120]
	weights = [24, 50, 30]
	fracKnapsack(capacity, weights, values)

# 8,最小成本切割成正方形：
def minimumCostOfBreaking(X, Y, m, n):
 
    res = 0
 
    # sort the horizontal cost in reverse order
    X.sort(reverse = True)
 
    # sort the vertical cost in reverse order
    Y.sort(reverse = True)
 
    # initialize current width as 1
    hzntl = 1; vert = 1
 
    # loop untill one or both
    # cost array are processed
    i = 0; j = 0
    while (i < m and j < n):
     
        if (X[i] > Y[j]):
         
            res += X[i] * vert
 
            # increase current horizontal
            # part count by 1
            hzntl += 1
            i += 1
         
        else:
            res += Y[j] * hzntl
 
            # increase current vertical
            # part count by 1
            vert += 1
            j += 1
 
    # loop for horizontal array, if remains
    total = 0
    while (i < m):
        total += X[i]
        i += 1
    res += total * vert
 
    #loop for vertical array, if remains
    total = 0
    while (j < n):
        total += Y[j]
        j += 1
    res += total * hzntl
 
    return res

if __name__ == '__main__':
	m, n = 5, 3
	X = [2, 1, 3, 1, 4]
	Y = [4, 1, 2]
	 
	print(minimumCostOfBreaking(X, Y, m, n))

# 9,字典中的最小数组：
def minimizeWithKSwaps(arr, n, k):
 
    for i in range(n-1):
        pos = i
        for j in range(i+1, n):
 
            # If we exceed the Max swaps then terminate the loop
            if ( j - i > k):
                break
 
            # Find the minimum value from i+1 to max (k or n)
            if (arr[j] < arr[pos]):
                pos = j
 
        # Swap the elements from Minimum position we found till now to the i index
        for j in range(pos, i, -1):
            arr[j],arr[j-1] = arr[j-1], arr[j]
 
        # Set the final value after swapping pos-i elements
        k -= pos - i

if __name__ == '__main__':
	n, k = 5, 1
	arr = [7, 6, 9, 2, 1]
	minimizeWithKSwaps(arr, n, k)
	 
	# Print the final Array
	for i in range(n):
	    print(arr[i], end = " ")

# 10,最小最大高度差：
def getMinDiff(arr, n, k):
 
    if (n == 1):
        return 0
 
    # Sort all elements
    arr.sort() 
 
    # Initialize result
    ans = arr[n-1] - arr[0] 
 
    # Handle corner elements
    small = arr[0] + k 
    big = arr[n-1] - k 
     
    if (small > big):
        small, big = big, small 
 
    # Traverse middle elements
    for i in range(1, n-1):
     
        subtract = arr[i] - k 
        add = arr[i] + k 
 
        # If both subtraction and addition
        # do not change diff
        if (subtract >= small or add <= big):
            continue
 
        # Either subtraction causes a smaller
        # number or addition causes a greater
        # number. Update small or big using
        # greedy approach (If big - subtract
        # causes smaller diff, update small
        # Else update big)
        if (big - subtract <= add - small):
            small = subtract 
        else:
            big = add 
     
 
    return min(ans, big - small) 

if __name__ == '__main__':
	arr = [ 4, 6 ] 
	n = len(arr) 
	k = 10	 
	print("Maximum difference is", getMinDiff(arr, n, k)) 