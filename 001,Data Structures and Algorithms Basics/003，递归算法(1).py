# 1，求和：使用循环算法
def sum_recursion(n):
	if n == 0:
		return 0
	return n + sum_recursion(n-1)

if __name__ == '__main__':
	print(sum_recursion(10))


# 2，阶乘：
def factorial(n):
	if n == 1:
		return 1
	return n * factorial(n-1)

if __name__ == '__main__':
	print(factorial(10))


# 3,斐波那契数列：
def fibonacci_first(n):
	assert(n >= 0)
	a, b = 0, 1
	for i in range(1, n+1):
		a, b = b, a+b
	return a

def fibonacci_second(n):
	assert(n>=0)
	if (n <= 2):
		return 1
	return fibonacci_second(n-1) + fibonacci_second(n -2)

def fibonacci_third(n):
	assert(n >= 0)
	if (n <= 1):
		return(n, 0)
	(a, b) = fibonacci_third(n-1)
	return (a+b, a)

def fibonacci(n):
	assert(n>=1)
	result = [1, 1]
	for i in range(2, n):
		result.append(result[-2] + result[-1])
	return result

if __name__ == '__main__':
	# time fibonacci_first(20)
	# time fibonacci_second(20)
	# time fibonacci_third(20)
	import time
	start = time.time() 
	print(fibonacci_first(20))
	end = time.time()
	print('fibonacci_first cost time: {:.16f} s.'.format(end - start))

	start = time.time() 
	print(fibonacci_second(20))
	end = time.time()
	print('fibonacci_second cost time: {:.16f} s.'.format(end - start))

	start = time.time() 
	print(fibonacci_third(20))
	end = time.time()
	print('fibonacci_third cost time: {:.16f} s.'.format(end - start))

	print(fibonacci(20))
	print('黄金分割率： {:.16f}.'.format(fibonacci(20)[-1]/fibonacci(20)[-2]))

# 尺子：
def ruler_recursion(n):
	assert(n>=0)
	if n == 1:
		return '1'
	t = ruler_recursion(n-1)
	return t + ' ' + str(n) + ' ' + t

def ruler_for(n):
	result = ''
	for i in range(1, n+1):
		result = result + str(i) + ' ' + result
	return result

class ruler_print(object):
	def draw_line(self, tick_length, tick_label=''):
	    line = '-' * tick_length
	    if tick_label:
	        line += ' ' + tick_label
	    print(line)

	def draw_interval(self, center_length):
	    if center_length > 0:
	        self.draw_interval(center_length - 1)
	        self.draw_line(center_length)
	        self.draw_interval(center_length - 1)
	        
	def draw_rule(self, num_inches, major_length):
	    self.draw_line(major_length, '0')
	    for j in range(1, 1 + num_inches):
	        self.draw_interval(major_length - 1)
	        self.draw_line(major_length, str(j))

if __name__ == '__main__':
	print(ruler_recursion(5))
	print(ruler_for(5))

	my_ruler = ruler_print()
	r = my_ruler.draw_rule(3,5)
	print(r)


# 5, 数学表达式：
def intSeq(a, b):
    if (a == b):
        return str(a)
    
    if (b % 2 == 1):
        return "(" + intSeq(a, b-1) + " + 1)"
    
    if (b < a * 2):
        return "(" + intSeq(a, b-1) + " + 1)"
        
    return intSeq(a, b/2) + " * 2";

if __name__ == '__main__':
	a = 5;
	b = 101;
	print(str(b) + " = " + intSeq(a, b))


# 6，汉诺塔：
def hanoi(n, start, end, by):
    if (n==1):
        print("Move from " + start + " to " + end)
    else:
        hanoi(n-1, start, by, end)
        hanoi(1, start, end, by)
        hanoi(n-1, by, end, start)

if __name__ == '__main__':
	print(hanoi(5, 'start', 'end', 'by'))


# 7, 格雷码：
def moves_ins(n, forward):
    if n == 0: 
        return
    moves_ins(n-1, True)
    print("enter ", n) if forward else print("exit  ", n)
    moves_ins(n-1, False)

if __name__ == '__main__':
	print(moves_ins(3, True))