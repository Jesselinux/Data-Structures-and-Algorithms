# 2，给一个m×n的矩阵，如果有一个元素为0，则把该元素对应的行与列所有元素全部变成0：
# space complexity: O(m+n):
def zero(matrix):
	m = [None] * len(matrix)
	n = [None] * len(matrix[0])
	for i in range(len(matrix)):
		for j in range(len(matrix[0])):
			if (matrix[i][j] == 0):
				m[i] = 1
				n[j] = 1

	for i in range(len(matrix)):
		for j in range(len(matrix[0])):
			if (m[i] == 1 or n[j] == 1):
				matrix[i][j] = 0

if __name__ == '__main__':
	matrix = [  [ 1, 1, 1, 1, 1, 0, 1, 1, 1, 1 ],
	            [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
	            [ 1, 1, 0, 1, 1, 1, 1, 1, 1, 1 ],
	            [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
	            [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ] ]

	for m in matrix:
		print(m, sep=" ")

	zero(matrix)

	print('***                        ***')
	for m in matrix:
	    print(m, sep=" ")


# 3，九宫图：
def magic_square(n):
	magic = [[0] * (n) for i in range(n)]
	row = n - 1
	col = n//2
	magic[row][col] = 1

	for i in range(2, n*n + 1):
		try_row = (row + 1) % n
		try_col = (col + 1) % n

		if (magic[try_row][try_col] == 0):
			row = try_row
			col = try_col
		else:
			row = (row - 1 + n) % n

		magic[row][col] = i 

	for m in magic:
		print(m, sep=' ')


if __name__ == '__main__':
	magic_square(9)


# 4，数独：满足每一行、每一列、每一个粗线宫（3*3）内的数字均含1-9，不重复;
# 该算法运用了二进制运算
def shudu(matrix):
	n = len(matrix)
	result_row = result_col = result_blk = 0

	for i in range(n):
		result_row = result_col = result_blk = 0
		for j in range(n):
			tmp = matrix[i][j]   # check row
			if ((result_row & (1 << (tmp - 1))) == 0):
				result_row = result_row | (1<<(tmp-1))
			else:
				print('row: ', i, j)
				return False

			tmp = matrix[j][i]   # check column
			if ((result_col & (1<<(tmp-1))) == 0):
				result_col = result_col | (1<<(tmp-1))
			else:
				print('col: ', j, i)
				return False

			idx_row = (i//3) * 3 + j//3
			idx_col = (i%3) * 3 + j%3
			tmp = matrix[idx_row][idx_col]   # check block
			if ((result_blk & (1<<(tmp-1))) == 0):
				result_blk = result_blk | (1<<(tmp-1))
			else:
				print('block: ', idx_row, idx_col)
				return False
	return True

if __name__ == '__main__':
	matrix = [
			    [5,3,4,6,7,8,9,1,2],
			    [6,7,2,1,9,5,3,4,8],
			    [1,9,8,3,4,2,5,6,7],
			    [8,5,9,7,6,1,4,2,3],
			    [4,2,6,8,5,3,7,9,1],
			    [7,1,3,9,2,4,8,5,6],
			    [9,6,1,5,3,7,2,8,4],
			    [2,8,7,4,1,9,6,3,5],
			    [3,4,5,2,8,6,1,7,9]
			]

	print(shudu(matrix))

# 5，旋转数组：给一个n×n的数组，旋转90度
def rotate(matrix):   # 法一
	n = len(matrix)
	result = [[0] * (n) for i in range(n)]

	for i in range(n):
		for j in range(n):
			result[j][n-1-i] = matrix[i][j]

	for i in result:
		print(i, sep=' ')

def rotate_two(matrix):   # 法二
	n = len(matrix)
	for layer in range(n//2):
		first = layer
		last = n - 1 - layer
		for i in range(first, last):
			offset = i - first
			top = matrix[first][i]   # save top

			matrix[first][i] = matrix[last-offset][first]   # left->top
			matrix[last-offset][first] = matrix[last][last - offset]   # bottom -> left
			matrix[last][last - offset] = matrix[i][last]   # right -> bottom
			matrix[i][last] = top  # right <- saved top    # top -> right
	for i in matrix:
		print(i, sep=' ')


if __name__ == '__main__':
	matrix = [[i*5+j for j in range(5)] for i in range(5)]
	
	for i in matrix:
		print(i, sep=' ')

	rotate(matrix)

# 6，反转字符串：
def reverse1(s):
    return s[::-1]

def reverse2(s):
    l = list(s)
    for i in range(len(l)//2):
        l[i], l[len(s)-1-i] = l[len(s)-1-i], l[i]
    return ''.join(l)

def reverse3(s):
    l = list(s)
    begin = 0
    end = len(l) - 1
    while begin <= end:
        l[begin], l[end] = l[end], l[begin]
        begin += 1
        end -= 1
    return ''.join(l)

if __name__ == '__main__':
	str = 'hello world!'
	print(reverse3(str))


# 7，给一个只包含0和1的数组，找出最长的全是1的子数组：Input: [1,1,0,1,1,1] -> Output: 3
def find_consecutive_ones(nums):
    local = maximum = 0
    for i in nums:
        local = local + 1 if i == 1 else 0
        maximum = max(maximum, local)
    return maximum

if __name__ == '__main__':
	nums = [1,1,0,1,1,1,1,0,0,0,0,0,1,1,1,0,0,1]
	print(find_consecutive_ones(nums))


# 8，给定一个数组，数组里有一个数组有且只有一个最大数，
# 判断这个最大数是否是其他数的两倍或更大。如果存在这个数，则返回其index，否则返回-1
def largest_twice(nums):     # O(n) time，O(1) space
    maximum = second = idx = 0
    for i in range(len(nums)):
        if (maximum < nums[i]):
            second = maximum
            maximum = nums[i]
            idx = i
        elif second < nums[i]:
            second = nums[i]
    return idx if (maximum >= second * 2) else -1

if __name__ == '__main__':
	nums = [1, 2,3,8,3,2,1]
	print('the maxmiun number`s index is: %i,\
		and the number is: %i'%(largest_twice(nums), nums[largest_twice(nums)]))


# 9，给定一个数组，1<=a[i]<=n, n是这个数组的长度，整数可以出现多次也可能一次都没有出现，找到所有未出现的整数
# Input: [4,3,2,7,8,2,3,1] -> Output: [5,6] 
def findDisappearedNumbers1(nums):  # time complexity: O(n).
    # For each number i in nums,
    # we mark the number that i points as negative.
    # Then we filter the list, get all the indexes
    # who points to a positive number
    for i in range(len(nums)):
        index = abs(nums[i]) - 1
        nums[index] = - abs(nums[index])

    return [i + 1 for i in range(len(nums)) if nums[i] > 0]


def findDisappearedNumbers2(nums):  # time complexity: O(n^2)
    result = []
    for i in range(1, len(nums) + 1):
        if (i not in nums):
            result.append(i)
    return result


if __name__ == '__main__':
    nums = [4, 3, 2, 7, 8, 2, 3, 1]
    print(findDisappearedNumbers1(nums))

    # 时间复杂度测试：
    import time
    import matplotlib.pyplot as plt
    import random
    import math
    #% matplotlib inline


    def random_list(l):
        return [[i + 1 for i in range(l * n)] for n in range(1, 20)]


    def findDisappearedNumbersTest1(nums):
        start = time.time()
        r = findDisappearedNumbers1(nums)
        t = time.time() - start
        return r, len(nums), t


    def findDisappearedNumbersTest2(nums):
        start = time.time()
        r = findDisappearedNumbers2(nums)
        t = time.time() - start
        return r, len(nums), t


    random_lists = random_list(100)
    rst1 = [findDisappearedNumbersTest1(i) for i in random_lists]
    rst2 = [findDisappearedNumbersTest2(i) for i in random_lists]

    figs = plt.figure(figsize=(20, 6), facecolor='gray')

    ax1 = figs.add_subplot(1, 2, 1)
    ax1.set_title('Time Complexity: O(n)')
    x1 = list(zip(*rst1))[1]
    y1 = list(zip(*rst1))[2]
    plt.plot(x1, y1)

    ax2 = figs.add_subplot(1, 2, 2)
    ax2.set_title('Time Complexity: O(n^2)')
    x2 = list(zip(*rst2))[1]
    y2 = list(zip(*rst2))[2]
    plt.plot(x2, y2)

    plt.show()


# 10，数组加法：整数的list形式实现加一操作
def plusOne(digits):
    if len(digits)==0:
        return False
    addCarry=1
    for i in range(len(digits)-1,-1,-1):
        digits[i]+=addCarry
        if digits[i]==10:
            digits[i]=0
            if i==0:
                digits.insert(0,1)
        else:
            break
    return digits

if __name__ == '__main__':
	digits1 = [1, 2, 3]
	digits2 = [9, 9, 9]

	print(plusOne(digits1), plusOne(digits2))