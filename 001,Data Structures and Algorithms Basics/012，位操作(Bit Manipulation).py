# 1,Set Bit, Clear Bit, Toggle Bit, Test Bit:
def setBit(a, n):
    return a | (1<<n)

def clearBit(a, n):
    return a & (~(1<<n))

def toggleBit(a, n):
    return a ^ (1<<n)

def testBit(a, n):
    result = a & (1<<n)
    return result != 0

if __name__ == '__main__':
	a = 128
	n = 1
	print(bin(a))
	r = setBit(a, n)
	print(r)
	print(bin(r))

	a = 127
	print(bin(a))
	n = 0
	r = clearBit(a, n)
	print(bin(r))

	a = 155
	print(bin(a))
	n = 3
	r = toggleBit(a, n)
	print(bin(r))

	a = 155
	print(bin(a))
	n = 5
	r = testBit(a, n)
	print(r)

# 2,整数转换成二进制：
def toBinary(n):
    sb = []
    if n < 256:
        upper = 128
    else:
        upper = 32768
    i = upper
    while i > 0:
        if n & i != 0:
            sb.append(str(1))
        else:
            sb.append(str(0))
        i = i >> 1
    return ''.join(sb)

if __name__ == '__main__':
	n = 125
	print(toBinary(n))
	print(bin(n))

# 3，二进制转换成整数：
def convertBits2Int(binary):
    length = len(binary)
    result = 0
    if length > 16:
        raise ValueError("Only Supports 16 Bits")
    for i in range(length):
        c = int(binary[i])
        if (c != 0 and c != 1):
            raise ValueError("binary can only be 0 or 1")
        #result += c << (length - i - 1)
        result = (result << 1) + c
        
    return result

if __name__ == '__main__':
	binary = "11111111"
	result = convertBits2Int(binary)
	print(result)

# 4，小数转换成二进制：
def convertDecimal(f):
    str_f = str(f).split(".")
    int_part, dec_part = divmod(f, 1)
    int_part = int(int_part)
    print(int_part, dec_part)
    
    int_s = ""
    while (int_part > 0):
        r = int_part % 2
        int_part >>= 1
        int_s = str(r) + int_s

    dec_s = [] 
    while (dec_part > 0):
        if (len(dec_s) > 32):
            print("".join(dec_s))
            raise ValueError("Not Support")
        if (dec_part == 1):
            dec_s.append(str(dec_part))
            break
        r = dec_part * 2
        
        if (r >= 1):
            dec_s.append("1")
            dec_part = r - 1
        else:
            dec_s.append("0")
            dec_part = r
        
    return int_s + "." + "".join(dec_s)

if __name__ == '__main__':
	f = 3.875
	print(convertDecimal(f))

# 5,十六进制转换成整数：
def hex2int(s):
    digits = "0123456789ABCDEF"
    val = 0
    for i in range(len(s)):
        c = s[i].upper()
        d = digits.index(c)
        val = 16 * val + d
    return val

if __name__ == '__main__':
	s = "DAD"
	print(hex2int(s))

# 6，整数转换成十六进制：
def int2hex(d):
    digits = "0123456789ABCDEF"
    if d == 0:
        return "0"
    hex = ""
    while (d > 0):
        digit = d % 16
        hex = digits[digit] + hex
        d = d // 16
    return hex

if __name__ == '__main__':
	d = 3501
	print(int2hex(d))

# 7，整数的二进制形式包含1的个数：
def bitCountA(n):
    count = 0
    while (n != 0):
        if (n & 1 != 0):
            count += 1
        n = n>>1
    return count

def bitCountB(n):
    count = 0
    while (n != 0):
        n = n & (n - 1)
        count += 1
    return count

if __name__ == '__main__':
	n = 11
	print(bitCountB(n))

# 8，下一个2的幂数：
def next2Power(n):
    while (n & (n-1) != 0):
        n = n & (n-1)
    return n << 1

if __name__ == '__main__':
	n = 555
	print(next2Power(n))

# 9，判断两个整数符号是否一致：
def isOppositeSigns(a, b):
    return (a^b) < 0

if __name__ == '__main__':
	a, b = 10, -20
	print(isOppositeSigns(a, b))

# 10，判断整数是否是正数
def isPositiveInteger(n):
    return (n >> 31) == 0

if __name__ == '__main__':
	n = -1
	print(isPositiveInteger(n))

# 11，绝对值：
def absoluteA(a):
    mask = a >> 31
    result = (a + mask) ^ mask
    return result

def absoluteB(a):
    mask = a >> 31
    result = (a ^ mask) - mask
    return result

if __name__ == '__main__':
	print(absoluteB(-5))

# 12,整数交换：
def swap3(a, b):
    a = a ^ b
    b = a ^ b
    a = a ^ b
    print(a, b)

def swap2(a, b):
    a = b - a
    b = b - a
    a = a + b
    print(a, b)

def swap1(a, b):
    a, b = b, a
    print(a, b)

if __name__ == '__main__':
	a, b = 5, 10
	swap3(a,b)

# 13,计算A转换成B，需要改变的二进制位数：
def convertA2B(a, b):
    count = 0
    c = a ^ b
    while (c != 0):
        c = c & (c - 1)
        count += 1
    return count

if __name__ == '__main__':
	a, b = 5, 10
	print(bin(a))
	print(bin(b))
	print(convertA2B(a, b))

# 14,将N的[i, j]的子串，改变成M：
def amazingMask(n, m, i, j):
    allOne = ~0
    left = allOne - ((1<<(j+1))-1)
    right = (1<<i)-1
    mask = left | right

    return (n & mask) | (m << i)
    
if __name__ == '__main__':
	n = 1024
	m = 21
	i, j = 2, 6
	r = amazingMask(n, m, i, j)
	print(bin(n))
	print(bin(m))
	print(bin(r))

# 15，整数的二进制形式是否是回文：
def bitPalindrome(s):
    for i in range(len(s)//2):
        if (int(s[i]) ^ int(s[-1-i]) == 1):
            return False
    return True

if __name__ == '__main__':
	s = "01000000000000000000000000000010"
	print(bitPalindrome(s))

# 16,加法：
def add(a, b):
    if b == 0:
        return a
    sum = a ^ b
    carry = (a & b) << 1
    return add(sum, carry)

if __name__ == '__main__':
	a, b = 759, 674
	print(add(a, b))

# 17,找到丢失的数字：
def printTwoElements(arr):
    for i in range(len(arr)):
        if arr[abs(arr[i]) - 1] > 0:
            arr[abs(arr[i]) - 1] = -arr[abs(arr[i]) - 1]
        else:
            print("The repeating element is", abs(arr[i]))
             
    for i in range(len(arr)):
        if arr[i] > 0:
            print("and the missing element is", i + 1)

if __name__ == '__main__': 	 
	arr = [7, 3, 4, 5, 5, 6, 2]
	n = len(arr)
	printTwoElements(arr)

# 18，查找下一个最大的整数：
def getBit(n, index):
    return ((n & (1<<index))>0)

def setBit(n, index, b):
    if b:
        return n | (1<<index)
    else:
        return n & (~(1<<index))

def getNext(n):
    if n <= 0: 
        return -1

    index = 0
    countOnes = 0

    # Find first one.
    while (not getBit(n, index)):
        index += 1

    # turn on next zero
    while( getBit(n, index) ):
        index += 1
        countOnes += 1
    
    n = setBit(n, index, True)

    # turn off previous one 
    index -= 1
    n = setBit(n, index, False)
    countOnes -= 1

    # set zeros
    i = index - 1
    while (i >= countOnes):
        n = setBit(n, i, False)
        i -= 1

    # set ones
    i = countOnes - 1
    while (i >= 0):
        n = setBit(n, i, True)
        i -= 1

    return n

def getPrevious(n):
    if (n <= 0):
        return -1

    index = 0
    countZeros = 0

    # find first zero
    while( getBit(n, index) ):
        index += 1

    # turn off next 1
    while( not (getBit(n, index)) ):
        index += 1
        countZeros += 1
    
    n = setBit(n, index, False)

    # turn on previous zero
    index -= 1
    n = setBit(n, index, True)
    countZeros -= 1

    # set ones
    i = index - 1
    while (i >= countZeros):
        n = setBit(n, i, True)
        i -= 1

    # set zeros
    i = countZeros - 1
    while (i >= 0):
        n = setBit(n, i, False)
        i -= 1
    
    return n

if __name__ == '__main__':
	n = 500
	r = getNext(n)
	print(bin(n))
	print(bin(r))

	n = 500
	r = getPrevious(n)
	print(bin(n))
	print(bin(r))

# 19,数据流中，查找一个概率相等的数字：
import random
def reservoirSampling(items, k):
    sample = items[0:k]

    for i in range(k, len(items)):
        j = random.randrange(1, i + 1)
        if j <= k:
            sample[j - 1] = items[i]

    return sample

if __name__ == '__main__':
	items = list(range(1, 100))
	print(reservoirSampling(items,10))

# 20,阶乘的结果中尾部0的个数：
def findTrailingZeros(n):
    count = 0
    i = 5
    while (n / i >= 1):
        count += n//i
        i *= 5
 
    return count

if __name__ == '__main__':
	n = [(n, findTrailingZeros(n)) for n in range(1,131)]
	print(n)

# 21,最大公约数：
def gcd(a, b):
    if b > a:
        return gcd(b, a)

    if a % b == 0:
        return b

    return gcd(b, a % b)  

if __name__ == '__main__':
	print(gcd(39, 91))