a = 10
b = 11
c = 15
### 변수의 값 확인
print(a)
print(a > 10)
a + 10
print(a - 10)
print(a * 10)
print(a / 10)
print(a // 10) # 몫
print(a % 10) # 나머지
### 일주일은 몇초인지?
seconds_in_week = 1 * 7 * 24 * 60 * 60
print(seconds_in_week)
a = 'hello world'
b = "hello world"
c = '''hello world'''
d = """hello 

world"""
print(d)
a = 'hello'
b = ' world'
print(a + b)
c = ' hi there'
print(a + b + c)
print(a * 4)
print('-' * 50)
### 인덱싱
a = 'hello world'
a[0]
a[-1]
a[-11]
a[0] = 'j'
a = 'jello world'
### 슬라이싱
a = 'hello world'
print(a[0:3])
print(a[1:6])
print(a[2:9])
print(a[2:-1])
print(a[0:])
print(a[:7])
print(a[:])
print(a[0:9:2])
### formatting(포맷팅)
rain_prob = 60
rain_amount = 30
a = '내일의 강수 확률은 {}%, 강수량은 {}mm입니다.'.format(rain_prob, rain_amount)
print(a)
###  len, replace, split
a = 'hello world'
length = len(a)
print(length)
b = a.replace('h','j')
print(b)
a = 'hello world hi there'
print(a.split(' '))
### 인코딩, 디코딩
a = '한글'
print(a)
b = a.encode('utf-8')
print(b)
c = a.encode('euc-kr')
print(c)
d = b.decode('utf-8')
print(d)
e = c.decode('euc-kr')
print(e)

#------------------------------------------------------------------------------

### 리스트
a = [] # 빈 리스트 생성
print(type(a))
print(a)
b = list()
print(b)
a = [1, 2, 3, 4, 5, 6, 7, 8]
a = 'hello world hi there'
b = a.split(' ')
print(b)
### 리스트 인덱싱
a = [1, 2, 3, 5, 6, 7, 8]
print(a)
a[0] = 9 # 리스트는 자유롭게 값 재할당 가능
print(a)
### 리스트 슬라이싱
a[1:4]
### 리스트 오퍼레이션(함수)
a = [1, 2, 3]
b = [4, 5, 6]
print(a)
print(b)
print(a + b)
print(a * 4)
a.append(4)
print(a)
a.append(b) # 리스트가 들어감
print(a)
print(a[4])
a = [1, 2, 3]
a + b
a.extend(b) # 리스트의 원소값이 들어감
print(a)
### 주어진 리스트에서 임의의 두 원소의 합 최대값 찾기
print(type(a))
a = [1, 17, 8, 29, 10, 11]
a.sort() #정렬
print(a)
print(a[-2] + a[-1])
### 튜플
a = (1, 2, 3)
print(a)
print(type(a))
a = tuple([1, 2, 3])
print(a)
a = 1
b = 2
c = 3
print(a, b, c)
a, b, c = (1, 2, 8)
a, b, c = 1, 2, 8
print(a, b, c)
### 튜플 스와핑
# 튜플은 불변이기 때문에 처음 생성 후, 값 변경 불가
a = 5
b = 4
print(a, b)
tmp = a
a = b
b = tmp
print(a, b)
a = 5
b = 4
a, b = b, a
print(a, b)
a = (1, 2, 3)
print(a[0])
# a[0] = 4
### 딕셔너리(사전)
# 데이터를 key, value로 저장하는 타입
a = {}
print(type(a))
print(a)
a = {1:2, 4:8, 6:9} # key:value
print(a)
print(a[1])
a = {'Canada': 100, 'Korea': 200}
print(a)
a['England'] = 50
print(a)
a['Korea'] = 300 # key의 중복이 없어 이미 할당된 key에 값을 대입하는 경우 값의 업데이트 발생
print(a)
print(a['USA'])
print(a.get('USA'))
print('USA' in a)
print('Korea' in a)
a.pop('England') # key 삭제
print(a)
### Set(집합)
a = set()
a = {1, 1, 2, 3, 4, 3, 3, 8} # 데이터 사이의 순서가 없고, 중복된 데이터 저장 불가능
print(a)
b = {1, 2, 5, 6, 7}
print(a)
print(b)
print(a.intersection(b))
print(a & b)
print(a.union(b))
print(a | b)
a.difference(b)
print(a - b)

### 조건문(condition)
a = 0
if a > 2:
    print('5는 2보다 크다')
    print('이것은 정말로 참이다')
else:
    print('5는 2보다 작다')
    print('이것은 거짓이다')
### if문의 중첩
a = 99
if a % 11 == 0:
    if a % 9 == 1:
        print('당첨입니다')
    else:
        print('아쉽네요 한번더')
else:
    print('하하하 다음 기회에')

print('이것은 다른 코드')
x = 5
y = 4
if x <= y:
    print('x가 y보다 작거나 같다')
else:
    print('그게 아니다')
### and, or, not 사용하기
a = 10
b = 40
x = 20
y = 7
if a > b and x > y:
    print('조건이 맞습니다')
else:
    print('조건이 틀립니다')

if not (a > b):
    print('조건이 맞습니다')
else:
    print('조건이 틀립니다')

### not > and > or 우선순위
a = 10
b = 40
x = 20
y = 7
if a > 10 and b > 40 or not x != 20:
    print('hahaha')

### if, elif, else
a = 98
if a % 4 == 1:
    print('4로 나누면 나머지가 1')
elif a % 4 == 2:
    print('4로 나누면 나머지가 2')
elif a % 4 == 3:
    print('4로 나누면 나머지가 3')
else:
    print('4의 배수')
print('이건 코드의 끝') # 조건문 사이에 코드블락 이외에 다른 코드 사용할 수 없음

a = 97
if a % 4 == 1:
    print('4로 나누면 나머지가 1')
if a % 4 == 2:
    print('4로 나누면 나머지가 2')
if a % 4 == 3:
    print('4로 나누면 나머지가 3')
else:
    print('4의 배수')
print('이건 코드의 끝')
### 한 줄 조건문
value = None
length = 1000
if length > 50:
    value = '길다'
else:
    value = '짧다'
print(value)

value = '길다' if length > 50 else '짧다'
print(value)

#------------------------------------------------------------------------------

### 반복문(while, for)
a = [1, 2, 3, 4, 5]
i = 0
while i < len(a):
    print(a[i])
    i += 1

### break
a = [1, 2, 3, 4, 5]
i = 0
while i < len(a):
    if a[i] % 2 == 0:
        print('break') 
        break # while문 실행 중break 를 만나면 바로 반복 종료
    print(a[i])
    i +=1
print('while문 종료')

### continue
a = 6
while a > 0:
    a -= 1
    print('current value: ', a)
    
a = 6
while a > 0:
    a -= 1
    if a == 3:
        continue # while문 실행 중 continue를 만나면 실행 흐름이 다시 while 조건으로 이동
    print('current value: ', a)

### 짝수 홀수 분리
a = [1, 2, 3, 4, 5, 6, 7, 8]
i = 0
even_nums = []
odd_nums= []
while i < len(a):
    if a[i] % 2 == 0:
        even_nums.append(a[i])
    else:
        odd_nums.append(a[i])
        
    i += 1
print(even_nums)
print(odd_nums)    

### for
# while문과 달리 for은 인덱스로 순회가 가능한 타입(리스트 등)의 각 아이템을 반복적으로 접근하게 함
a = [10, 2, 30, 4 ,5]
for i in a:
    print(i)

### enumerate
a = [10, 2, 30, 4 ,5]
for i, num in enumerate(a): # 인덱스, 원소 추출
    print(i, num)
    
### 중첩 반복(구구단)
a = [2, 3, 4, 5, 6, 7, 8, 9]
b = [1, 2, 3, 4, 5, 6, 7, 8, 9]
for x in a:
    for y in b:
        print(x, '*', y, '=', x * y)
    print('-' * 50)

### range 함수
# range(start, end, step)
# start : 시작수(포함)
# end : 끝수(미포함)
# step : 건너뛰는 수(기본값 = 1)
print(range(101))
print(list(range(101)))
print(list(range(1, 101)))
print(list(range(1, 101, 2)))

### 1-100 사이의 5의 배수만 갖는 리스트 생성
print(list(range(5, 101, 5)))

### 연습문제
### 1. 6277이 소수인지 아닌지 판단하시오
### 2. 주어진 리스트의 평균을 구하시오
### 3. 주어진 리스트에서 최대값, 최소값을 찾으시오(sort함수 사용하지 말고 작성) 
# 1. 2부터 6266까지 순회
# 2. 6277을 각 수로 나눠서
#  2-1 나눠 떨어지는 경우가 한번이라도 있다면 - 소수가 아님(합성수)
#  2-2 한번도 없으면 소수
result = ''
nums = list(range(2, 6277))
for x in nums:
    if 6277 % x == 0:
        result = '합성수'
        break
    else:
        result = '소수'
print(result)

result = '소수'
nums = list(range(2, 6277))
for x in nums:
    if 6277 % x == 0:
        result = '합성수'
        break
print(result)

a = [2, 3, 6, 17, 92, 34]
i = 0
_sum = 0
while i < len(a):
    _sum += a[i]
    i += 1
print(_sum/ i)

print(sum(a) / len(a)) # sum함수 이용하기

a = [2, 3, 6, 17, 1, 92, 34]
# 최소값을 가장 첫번째 값 가정
# 각 아이템 순회하면서 현재 최소값보다 작은 값을 만나면 최소값을 그 값으로 대체
_min = a[0]
for x in a[1:]:
    if x < _min:
        _min = x

_max = a[0]
for x in a[1:]:
    if x > _max:
        _max = x
print(_min, _max)

_min = a[0]
_max = a[0]
for x in a[1:]:
    if x < _min:
        _min = x
    if x > _max:
        _max = x
print(_min, _max)

#------------------------------------------------------------------------------

### 함수의 정의
a = [1, 8, 17, 2, 0, 99]
_min = a[0]
for x in a[1:]:
    if x < _min:
        _min = x
        
b = [11, 81, 170, 2, 10, 199]
_min = b[0]
for x in b[1:]:
    if x < _min:
        _min = x

print(_min)

def get_min(lst):
    _min = lst[0]
    for x in lst[1:]:
        if x < _min:
            _min = x
    return _min

a = [1, 8, 17, 2, 0, 99]
b = [11, 81, 170, 2, 10, 199]
min_of_a = get_min(a)
min_of_b = get_min(b)
print (min_of_a)
print (min_of_b)

def increment(x):
    x = x + 1
    return x
a = 5
b = increment(a)
print(a,b)

### default parameter # 디폴트 파라미터는 마지막 부터 명시 가능
def add_all(x, y = 1, z = 9, a = 8):
    return x + y + z + a
add_all(3, 4)
#def add_all(x, y = 1, z, a = 8):
#    return x + y + z + a SyntaxError 발생

### Named parameter
def sub(x, y):
    return x - y
a = sub(30, 40)
b = sub(y = 30, x = 40)
c = sub(x = 30, y = 40)
print(a, b, c)

### 가변길이 파라미터(입력값의 개수가 미리 정해져 있지 않을 때 사용, 파라미터를 tuple로 사용)
a = 0
b = 1
c = 2
d = 3
e = 4
print(a, b, c, d, e)

def add_all(*x):
    print(type(x))
    total = 0
    for i in x:
        total += i
    return total
add_all()
add_all(20)
add_all(20, 21)

### 키워드 파라미터(입력값의 개수가 미리 정해져 있지 않고, 네임드 파라미터로 사용 가능, 파라미터를 dict로 사용)
def print_all(**kwargs):
    print(type(kwargs))
    print(kwargs)
print_all(a = 1, b = 2, c = 3, d = 4, xx = 5, kwch = 60)

### return keyward
def add(x, y):
    if x == 0:
        return y
    
    return x + y
add(30, 40)
c = add(0, 50)
print(c)

def no_return(x):
    if x == 0:
        return
    
    return x + 1

a = no_return(9)
print(a)
b = no_return(0)
print(b)

### 변수의 scope(유효범위)
a = [1, 2, 3, 4, 5]
print(len(a))

x = 100 # global

x = 100
def increment(x):
    x = x + 1
    return x
increment(200)
print(x)

del x 
print(x)
def increment(x):
    x = x + 1
    return x
increment(200)
print(x)

a = 1
b = 2
def increment(x):
    a = 10
    b = 20
    return a + b + x
increment(100)
print(a, b)

### 함수를 파라미터로 받는 함수
a = [9, 8, 10, 7, 12, 1, 2, 6]
print(a)
a.sort()
print(a)

### 1의 자리수로 정렬하기
a = [9, 8, 10, 7, 12, 1, 2, 6]
def digit(x):
    return x % 10
print(digit(12))
print(digit(24))
print(digit(35))
a.sort(key = digit)
print(a)

### lambda(람다) 함수
def square(x):
    return x * x
print(square(4))

square2 = lambda x:x*x # lamda 파라미터1, 파라미터2,… : 리턴값
print(square2(4))

a = [9, 8, 10, 7, 12, 1, 2, 6]
a.sort(key = lambda x:x%10)
print(a)

### filter(리스트 원소 필터링)
def even(x):
    return x % 2 == 0
print(even(100))
print(even(101))
a = [1, 2, 4, 7, 9, 10, 12, 17]
b = list(filter(even, a))
print(b)

a = [1, 2, 4, 7, 9, 10, 12, 17]
b = list(filter(lambda y:y%2==0, a))
print(b)
c = list(filter(lambda i:i>10, a))
print(c)

### 연습문제
### 1. 리스트를 입력받아 짝수만 갖는 리스트를 반환
### 2. 주어진 수가 소수인지 아닌지 판단
### 3. 주어진 문자열에서 모음의 개수를 출력
#1 
def even_filter(lst):
    even_lst = []
    for x in lst:
        if x % 2 == 0:
            even_lst.append(x)
    return even_lst
print(even_filter([1, 2, 4, 5, 8, 9, 10]))

#2
# 2 부터 num-1까지 나누기
# 한번이라도 나누어 지면, 소수가 아님 -> return False
# return True
def is_prime_number(num):
    for x in range(2, num):
        if num % x == 0:
            return False
    return True
print(is_prime_number(59))

#3
def count_vowel(string):
    count = 0
    for ch in string:
        if ch == 'a' or \
        ch == 'i' or \
        ch == 'e' or \
        ch == 'o' or \
        ch == 'u':
            count += 1
    return count
print(count_vowel("python"))

def count_vowel(string):
    count = 0
    for ch in string:
        if ch in 'aeiou':
            count += 1
    return count
print(count_vowel("python"))

#------------------------------------------------------------------------------

### 모듈의 이해와 활용
import math
print(math.pi)
print(math.cos(1))
import math as m
print(m.pi)

### 특정한 값만 import 하기
from math import pi
print(pi)
from math import cos
print(cos(2))
from math import cos as c
print(c(2))
from math import * # math 패키지 안의 모든 것들 import
print(sin(1))
print(sinh(1))

### 클래스
a = list()
a.append(1)
a.append(2)
a.append(3)

class Person:
    pass         # 나중에 구현하고 싶을 때 pass 사용

alice = Person()
bob = Person()
print(type(alice))
print(type(bob))

class Person:
    def __init__(self, name, height, weight):
        print('객체가 생성되고 있습니다.')
        self.name = name
        self.height = height
        self.weight = weight
        
alice = Person('Alice', 200, 200)
bob = Person('Bob', 100, 100)

### method 정의
class Person:
    def __init__(self, name, height, weight):
        print('객체가 생성되고 있습니다.')
        self.name = name
        self.height = height
        self.weight = weight
    
    def print_person_data(self):
        message = '이름: {}, 키: {}, 몸무게: {}'.format(self.name, self.height, self.weight)
        print(message)
        
alice = Person('alice', 200, 200)
alice.print_person_data()
bob = Person('bob', 100, 100)
bob.print_person_data()

### 클래스 상속
class Person:
    def __init__(self, name):
        self.name = name
    
    def sleep(self, time):
        print('{}은 {}시간동안 잠을 잤습니다.'.format(self.name, time))
        
bob = Person('Bob')
bob.sleep(10)

class Student(Person):
    def __init__(self, name):
        self.name = name
    
    def study(self, time):
        super().sleep(time) # 오버라이딩 한 후 부모클래스의 기능 호출
        print('{}은 {}시간동안 공부를 했습니다'.format(self.name, time))
    
    def sleep(self, time):
        print('{}은 {}시간동안 잠을 열심히 잤습니다.'.format(self.name, time)) # 부모클래스의 기능 오버라이딩

alice = Student('Alice')
alice.sleep(2)
alice.study(5)

bob.study(6)

alice.sleep(10)

### special method
# https://docs.python.org/3/reference/datamodel.html#object.__add__
class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def add(self, pt):
        return Point(self.x + pt.x, self.y + pt.y)
        
    def print_point(self):
        print('x: {}, y: {}'.format(self.x, self.y))

p1 = Point(3, 4)
p2 = Point(4, 6)

p1.print_point()
p2.print_point()

p3 = p1.add(p2)
p3.print_point()

p3 = p1 + p2 # 가능하게 하기

class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __add__(self, pt):
        return Point(self.x + pt.x, self.y + pt.y)
    def __sub__(self, pt):
        return Point(self.x - pt.x, self.y - pt.y)
    def __len__(self):
        return self.x ** 2 + self.y ** 2
    def __getitem__(self, index):
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
    def __str__(self):
         return('x: {}, y: {}'.format(self.x, self.y))

        
    def print_point(self):
        print('x: {}, y: {}'.format(self.x, self.y))

p1 = Point(3, 4)
p2 = Point(4, 6)

p1.print_point()
p2.print_point()

p3 = p1 + p2
p3.print_point()

p3 = p1 - p2
p3.print_point()

print(len("hello world"))
print(len([1, 2, 3]))

p1 = Point(3, 4)
p2 = Point(4, 6)
print(len(p1))

a = "hello world"
a[0]
p1 = Point(3, 4)
p2 = Point(4, 6)
print(p1[0])
print(p1[1])

print([1, 2, 3, 4, 5])
p1 = Point(3, 4)
p2 = Point(4, 6)
print(p1)

#------------------------------------------------------------------------------

### 정규표현식 (Regular Expression)
a = 'python\n\n' # escape 문자열
print(a)
b = r'python\n\n'
print(b)
#https://docs.python.org/3/library/re.html

### search method
# . 개행을 제외한 아무 문자 한개
# ^ 텍스트의 시작 점
# & 텍스트의 종료 점
# + 앞의 문자 1번 이상 발생
# ? 앞의 문자 0번 또는 1번만 발생
# * 앞의 문자 0번 이상 발생
# \d 모든 숫자
# \D 숫자 외의 문자
# \w 숫자, 알파벳, 한글 등을 포함한 문자
# \W 문자 외의 기호
# [abcd] 대괄호 안의 abcd 문자 중 하나
# [^abcd] 대괄호 안의 abcd 문자가 아닌 문자 하나
# [0-9] 0~9 사이 모든 문자
# [A-Z] A~Z 사이 모든 문자
# [a-z] a~z 사이 모든 문자
# [가-힣] 가~힣 사이 모든 문자
# {n} {n}앞의 문자 n번 발생
# {m, n} {m, n}앞의 문자 m~n번 발생
import re
re.search(r'test', 'test is hard')
print(re.search(r'\d', 'hi there 45k'))
print(re.search(r'\d\d', 'hi there 45k'))
print(re.search(r'\w\w\w', 'hi there'))

### metacharacters
print(re.search(r'[abc]at', 'cat'))
print(re.search(r'[abc]at', 'bat'))
print(re.search(r'[abc]at', 'zat'))

match = re.search(r'[abc]at', 'cat')
print(match.start())
print(match.end())
print(match.group())

print(re.search(r'[0-5]at', '1at'))
print(re.search(r'[0-5]at', '8at'))
print(re.search(r'[0-9]at', '8at'))

print(re.search(r'[^abc]at', 'aat'))
print(re.search(r'[^abc]at', 'bat'))
print(re.search(r'[^abc]at', 'cat'))
print(re.search(r'[^abc]at', 'dat'))
print(re.search(r'[^abc]at', '1at'))

print(re.search(r'\d\d\d', 'a124k'))
print(re.search(r'\d\d\D', 'a124k'))
print(re.search(r'\w\w\w', '@@abc123'))
print(re.search(r'\W\w\w', '@@abc123'))

print(re.search(r'...a.', 'klpa0'))
print(re.search(r'...a.', 'kl#a0'))
print(re.search(r'...a.', 'klpb0'))
print(re.search(r'...a\.', 'klpa.'))

print(re.search(r'a[bcd]+b', 'abcbdccb'))
print(re.search(r'a[bcd]+b', 'abcbdcc'))
print(re.search(r'a[bcd]*b', 'ab'))
print(re.search(r'a[bcd]+b', 'ab'))
print(re.search(r'a[bcd]?b', 'ab'))
print(re.search(r'a[bcd]?b', 'acb'))
print(re.search(r'a[bcd]?b', 'accb'))

print(re.search(r'b\w+a', 'cabana'))
print(re.search(r'^b\w+a', 'cabana'))
print(re.search(r'^b\w+a', 'babana'))
print(re.search(r'b\w+a$', 'cabana'))
print(re.search(r'b\w+a$', 'cabanap'))

### grouping
m = re.search(r'\w+@.+', 'my email address is test@test.com')
print(m)
m = re.search(r'(\w+)@(.+)', 'my email address is test@test.com')
print(m)
print(m.group())
print(m.group(0))
print(m.group(1))
print(m.group(2))

print(re.search(r'010-\d{4}-\d{4}', '010-1111-1111'))
print(re.search(r'010-\d{4}-\d{4}', '010-11110-1111'))
print(re.search(r'010-\d{4,5}-\d{4}', '010-11110-1111'))
print(re.search(r'010-\d{4,5}-\d{4}', '010-111-1111'))

### 미니멈 매칭
# html
# <a>
# <img>
# <hl><\hl>
print(re.search(r'<.+>', '<html>Title<\html>'))
print(re.search(r'<.+?>', '<html>Title<\html>'))

print(re.search(r'a{3,5}', 'aaaaaa'))
print(re.search(r'a{3,5}?', 'aaaaaa'))

print(re.search(r'010\d\d', 'hahah 01016'))
print(re.match(r'010\d\d', 'hahah 01016'))
print(re.match(r'010\d\d', '01016'))
print(re.search(r'^010\d\d', 'hahah 01016'))
print(re.search(r'^010\d\d', '01016'))

print(re.search(r'010\d\d', 'hahaha 01023 01034 010 56'))
print(re.findall(r'010\d\d', 'hahaha 01023 01034 010 56'))

print(re.sub('\d+', 'number', '010 hahah nice good great 99 112 nice good'))
print(re.sub('\d+', 'number', '010 hahah nice good great 99 112 nice good', count = 1))
print(re.sub('\d+', 'number', '010 hahah nice good great 99 112 nice good', count = 2))

email_re = re.compile(r'\w+@.+')
print(email_re.search('test@gmail.com'))
print(email_re.search('test2@gmail.com'))

print(email_re.sub('number', 'test@gmail.com'))

### 연습문제
# 아래 뉴스에서 이메일 주소를 추출해 보세요
# 다음 중 올바른 (http, https)웹페이지만 찾으시오
import requests
from bs4 import BeautifulSoup
# 위의 두 모듈이 없는 경우에는 pip install requests bs4 실행

def get_news_content(url):
    response = requests.get(url)
    content = response.text
    
    soup = BeautifulSoup(content, 'html5lib')
    
    div = soup.find('div', attrs = {'id' : 'harmonyContainer'})
    
    content = ''
    for paragraph in div.find_all('p'):
        content += paragraph.get_text()
        
    return content

news1 = get_news_content('https://news.v.daum.net/v/20190617073049838')
print(news1)

webs = ['http://www.test.co.kr',
        'https://www.test1.com',
        'http://www.test.com',
        'ftp://www.test.com',
        'http:://www.test.com',
        'htp://www.test.com',
        'http://www.google.com',
        'https://www.homepage.com.']

email_pattern = '/w+@.+'
# 아이디 : 문자, -,. , 숫자도 올 수 있음
# 도메인 : 문자, 숫자, ., gmail.com, yahoo.co.kr

email_re = re.compile(r'^\w+[\w.-]*@[\w.]+\w+$')
print(email_re.search('test-gmail@gmail.co.kr'))


web_re = re.compile(r'https?://[\w.]+\w+$')
for w in webs:
    print(web_re.search(w))

#------------------------------------------------------------------------------

### request 모듈 사용하여 http request/response 확인하기
''' 
requests 모듈
1. http request/response를 위한 모듈
2. HTTP method를 메소드 명으로 사용하여 request 요청 예) get, post
'''
import requests
'''
 get 요청하기
1. http get 요청하기
2. query parameter 이용하여 데이터 전달하기
'''
url = 'https://news.v.daum.net/v/20190728165812603'
resp = requests.get(url)
resp
resp.text

'''
 post 요청하기
1. http post 요청하기
2. post data 이용하여 데이터 전달하기
'''
url = 'https://www.kangcom.com/member/member_check.asp'
data = {
        'id' : 'macmath22',
        'pwd' : 'Test1357!'
}

resp = requests.post(url, data=data)
resp.text

'''
 HTTP header 데이터 이용하기
1. header 데이터 구성하기
2. header 데이터 전달하기
'''
url = 'https://news.v.daum.net/v/20190728165812603'
headers = {
        'user-agent' : 'Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.142 Safari/537.36'
}

resp = requests.get(url, headers=headers)
resp.text

'''
 HTTP response 처리하기
1. response 객체의 이해
2. status_code 확인하기
3. text 속성 확인하기
'''
url = 'https://news.v.daum.net/v/20190728165812603'
resp = requests.get(url)
resp.text
resp.status_code
if resp.status_code == 200:
    resp.headers
else:
    print('error')

#------------------------------------------------------------------------------
    
### OPEN API를 활용하여 json 데이터 추출하기(공공데이터 API)
'''
 공공데이터 포털 OPEN API 사용하기
1. 공공데이터 포털 회원가입/로그인(https://www.data.go.kr/)
2. API 사용 요청 / 키 발급
3. API 문서(specification) 확인
4. API 테스트 및 개발
id: adam0215
pw: alclsehowl7&
'''

'''
 Endpoint 확인하기
API가 서비스되는 서버의 IP 혹은 domain 주소
'''
endpoint = 'http://api.visitkorea.or.kr/openapi/service/rest/EngService/areaCode?serviceKey={}&numOfRows=10&pageSize=10&pageNo=1&MobileOS=ETC&MobileApp=AppTest'.format(serviceKey)
print(endpoint) # 주소 붙여넣기 XML형식으로 데이터를 받아 옴

'''
 key 값 확인하기
서비스 호출 트래킹할 목적이나 악의적인 사용을 금지할 목적으로 주로 사용
새로 발급받은 키는 1시간 이후 사용 가능
'''
serviceKey = 'neVxMy3ZO8apaGsXlXAV5V746RodMG0VBkyIbANFXFlbMTcvWhx6n8SNIxs7R1AiTIGzdVDuTcsqaUjGD%2Fq1IA%3D%3D'

'''
 Parameter 확인하기
API 호출에 필요한 parameter 값 확인 및 구성
'''
''' 
요청 및 Response 확인
requests 모듈을 활용하여 API 호출
response 확인하여 원하는 정보 추출
json 데이터의 경우, python dictionary로 변경하여 사용가능
'''
endpoint = 'http://api.visitkorea.or.kr/openapi/service/rest/EngService/areaCode?serviceKey={}&numOfRows=10&pageSize=10&pageNo={}&MobileOS=ETC&MobileApp=AppTest&_type=json'.format(serviceKey, 1)
resp = requests.get(endpoint)

print(resp.status_code)
print(resp.text)

resp.json()
type(resp.json())

data = resp.json()
print(data['response']['body']['items']['item'][0])

#------------------------------------------------------------------------------

### beautifulsoup 모듈 - beautifulsoup모듈 사용하여 HTML 파싱하기(parsing)
from bs4 import BeautifulSoup

# html 문자열 파싱
# 문자열로 정의된 html 데이터 파싱하기
html = '''
    <html>
      <head>
        <title>BeautifulSoup test</title>
      </head>
      <body>
        <div id='upper' class='test' custom='good'>
          <h3 title='Good Content Title'>Contents Title</h3>
          <p>Test contents</p>
        </div>
        <div id='lower' class='test' custom='nice'>
          <p>Test Test Test 1</p>
          <p>Test Test Test 2</p>
          <p>Test Test Test 3</p>
        </div>
      </body>
    </html>'''

# find 함수(특정 html tag를 검색, 검색 조건을 명시하여 찾고자하는 tag를 검색)
soup = BeautifulSoup(html)
soup.find('h3')
soup.find('p')
soup.find('div')
soup.find('div', custom='nice')
soup.find('div', id='lower')
soup.find('div', class='test') # 클래스는 키워드기 때문에 오류 발생
soup.find('div', class_='test') # 클래스 뒤에 언더스코어 사용

attrs = {'id': 'upper', 'class': 'test'} # multiple한 조건을 명시
soup.find('div', attrs=attrs)

# find_all 함수 (find가 조건에 만족하는 하나의 tag만 검색한다면, find_all은 조건에 맞는 모든 tag를 리스트로 반환)
soup.find_all('div')
soup.find_all('p')
soup.find_all('div', class_='test')

# get_text 함수(tag 안의 value를 추출, 부모 tag의 경우 모든 자식 tag의 value를 추출)
tag = soup.find('h3')
print(tag)
tag.get_text()

tag = soup.find('p')
print(tag)
tag.get_text()

tag = soup.find('div', id='upper')
print(tag)
tag.get_text()
tag.get_text().strip()

# attribute 값 추출하기
'''
경우에 따라 추출하고자 하는 값이 attribute 에도 존재함
이 경우에는 검색한 tag에 attribute 이름은 []연산을 통해 추추가능
예) div.find('h3')['title']
'''
tag = soup.find('h3')
print(tag)
tag['title']

### beautifulsoup 모듈 - id, class 속성을 이용하여 원하는 값 추출하기
'''
1. beautifulsoup 모듈 사용하기
2. id, class 속성으로 tag 찾기
3. CSS를 이용하여 tag 찾기
4. 속성 값으로 tag 찾기
5. 정규표현식으로 tag 찾기
6. 개발자도구를 이용하여 동적으로 로딩되는 데이터 추출하기
'''
import requests
from bs4 import BeautifulSoup

# 다음 뉴스 데이터 추출
'''
뉴스기사에서 제목, 작성자 작성일, 댓글 개수 추출
뉴스링크 = https://news.v.daum.net/v/20190728165812603
tag 추출할때는 가장 그 tag를 쉽게 특정할 수 있는 속성을 사용(id 의 경우 원칙적으로 한 html 문서 내에서 유일)
'''
# id, class 속성으로 tag 찾기(타이틀, 작성자, 작성일)
url = 'https://news.v.daum.net/v/20190728165812603'
resp = requests.get(url)

resp.text
soup = BeautifulSoup(resp.text)
soup
title = soup.find('h3', class_='tit_view')
title.get_text()

url = 'https://news.v.daum.net/v/20190728165812603'
resp = requests.get(url)

soup = BeautifulSoup(resp.text)
soup.find_all('span', class_='txt_info')
soup.find_all('span', class_='txt_info')[0]
soup.find_all('span', class_='txt_info')[1]

info = soup.find('span', class_='info_view')
info.find('span', class_='txt_info')

url = 'https://news.v.daum.net/v/20190728165812603'
resp = requests.get(url)

soup = BeautifulSoup(resp.text)
container = soup.find('div', id='harmonyContainer')
container
contents = ''
for p in container.find_all('p'):
    contents += p.get_text().strip()

contents

### beautifulsoup 모듈 - CSS를 이용하여 원하는 값 추출하기
# CSS를 이용하여 tag 찾기
'''
select, select_one함수 사용
css selector 사용법
(1) 태그명 찾기 tag
(2) 자손 태그 찾기 - 자손 관계 (tag tag)
(3) 자식 태그 찾기 - 다이렉트 자식 관계 (tag > tag)
(4) 아이디 찾기 #id
(5) 클래스 찾기 .class
(6) 속성값 찾기 [name='test']
    속성값 prefix 찾기 [name^='test']
    속성값 suffix 찾기 [name$='test']
    속성값 substring 찾기 [name*='test']
(7) n번째 자식 tag 찾기 :nth-child(n)
'''
url = 'https://news.v.daum.net/v/20190728165812603'
resp = requests.get(url)

soup = BeautifulSoup(resp.text)

soup.select('h3')
soup.select('#harmonyContainer')
soup.select('div#harmonyContainer')
soup.select('#harmonyContainer p')
soup.select('#harmonyContainer > p')

soup.select('h3')
soup.select('h3.tit_view')
soup.select('.tit_view')

soup.select('h3')
soup.select('h3[class="tit_view"]')
soup.select('h3[class^="t"]')
soup.select('h3[class^="tx"]')
soup.select('h3[class$="view"]')
soup.select('h3[class$="_view"]')
soup.select('h3[class*="_"]')

soup.select('span.txt_info')
soup.select('span.txt_info')[1]
soup.select('span.txt_info:nth-child(1)')
soup.select('span.txt_info:nth-child(2)')

### beautifulsoup 모듈 - 정규표현식을 이용하여 원하는 값 추출하기
# 정규표현식으로 tag 찾기
import re
soup.find_all(re.compile('h\d'))

soup.find_all('img', attrs={'src': re.compile('.+\.jpg')})
soup.find_all('img', attrs={'src': re.compile('.+\.gif')})
soup.find_all('img', attrs={'src': re.compile('.+\.png')})

soup.find_all('h3', class_='tit_view')
soup.find_all('h3', class_=re.compile('.+view$'))
soup.find_all('h3', class_=re.compile('.+newsview$'))

#------------------------------------------------------------------------------

### selenium 모듈 - 사이트에 로그인하여 데이터 크롤링하기
# 다음 뉴스 댓글 개수 크롤링하기
import requests
# XHR 전체 웹사이트를 다시 로딩하는게 아니라 부분적으로 필요한 정보만 비동기적 요청
# response에서 commentCount":43 확인
url = 'https://comment.daum.net/apis/v1/posts/@20190728165812603'
resp = requests.get(url)
print(resp)

'''
HTTP 상태 코드
1xx(정보) : 요청을 받았으며 프로세스를 계속한다
2xx(성공) : 요청을 성공적으로 받았으며 인식했고 수용하였다
3xx(리다이렉션) : 요청 완료를 위해 추가 작업 조치가 필요하다
4xx(클라이언트 오류) : 요청의 문법이 잘못되었거나 요청을 처리할 수 없다
5xx(서버 오류) : 서버가 명백히 유효한 요청에 대해 충족을 실패했다
'''
# 클라이언트 오류일때 헤더 추가하면 될 가능성 있음
headers = {
       'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJncmFudF90eXBlIjoiYWxleF9jcmVkZW50aWFscyIsInNjb3BlIjpbXSwiZXhwIjoxNTY4MzA2NjM2LCJhdXRob3JpdGllcyI6WyJST0xFX0NMSUVOVCJdLCJqdGkiOiJlY2E0MDFkYi0xMDY0LTQ5YjQtYjhjYS05YWI1Njk3MzFjOTkiLCJjbGllbnRfaWQiOiIyNkJYQXZLbnk1V0Y1WjA5bHI1azc3WTgifQ.4BgSqqPFyo-cUqn8UpeWwhfo7ecnn-MZmdQA_cEtW5E',
       'Origin': 'https://news.v.daum.net',
    'Referer': 'https://news.v.daum.net/v/20190728165812603',
    'Sec-Fetch-Mode': 'cors',
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36'
}
resp = requests.get(url, headers=headers)
print(resp)
resp.json()['commentCount']

# 로그인하여 데이터 크롤링하기
'''
1. endpoint 찾기 (개발자 도구의 network를 활용)
2. id와 password가 전달되는 form data 찾기
3. session 객체 생성하여 login 진행
4. 이후 session 객체로 원하는 페이지로 이동하여 크롤링
'''
import requests
from bs4 import BeautifulSoup
# endpoint 찾기
# 로그인 endpoint
url = 'https://www.kangcom.com/member/member_check.asp'

# id, password로 구성된 form data 생성하기
data = {
        'id': 'macmath22',
        'pwd': 'Test1357!'
        }

# login
# endpoint(url)과 data를 구성하여 post 요청
# login의 경우 post로 구성하는 것이 정상적인 웹사이트
s = requests.Session()
resp = s.post(url, data=data)
resp.text
print(resp)

# crawling
# login시 사용했던 session을 다시 사용하여 요청
my_page = 'https://www.kangcom.com/mypage/'
resp = s.get(my_page)
soup = BeautifulSoup(resp.text)
soup.select('td.a_bbslist55')
soup.select('td.a_bbslist55')[2]
td = soup.select_one('td.a_bbslist55:nth-child(3)')
mileage = td.get_text()
mileage

### selenium 모듈 - selenium 모듈로 웹사이트 크롤링하기
# selenium 모듈 사용법
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

from bs4 import BeautifulSoup
import time

# selenium 예제
'''
python.org로 이동하여 자동으로 검색해보기
1. python.org 사이트 오픈
2. input 필드를 검색하여 Key 이벤트 전달
'''
chrome_driver = 'C:/Users/USER/chromedriver'
driver = webdriver.Chrome(chrome_driver)
driver.get('https://www.python.org')
time.sleep(2)
driver.close()

chrome_driver = 'C:/Users/USER/chromedriver'
driver = webdriver.Chrome(chrome_driver)
driver.get('https://www.python.org')
search = driver.find_element_by_id('id-search-field')
search.clear()
time.sleep(3)
search.send_keys('lambda')
time.sleep(3)
search.send_keys(Keys.RETURN)
time.sleep(3)
driver.close()

# selenium을 이용한 다음뉴스 웹사이트 크롤링
# driver 객체의 find_xxx_by 함수 활용
chrome_driver = 'C:/Users/USER/chromedriver'
driver = webdriver.Chrome(chrome_driver)
url = 'https://news.v.daum.net/v/20190728165812603'
driver.get(url)
src = driver.page_source
soup = BeautifulSoup(src)
driver.close()
soup

chrome_driver = 'C:/Users/USER/chromedriver'
driver = webdriver.Chrome(chrome_driver)
url = 'https://news.v.daum.net/v/20190728165812603'
driver.get(url)
src = driver.page_source
soup = BeautifulSoup(src)
driver.close()
comment = soup.select_one('span.alex-count-area')
comment.get_text()

### selenium 모듈 - 웹사이트의 필요한 데이터가 로딩 된 후 크롤링하기
# selenium을 활용하여 특정 element의 로딩 대기
'''
WebDriverWait 객체를 이용하여 해당 element가 로딩 되는 것을 대기
실제로 해당 기능을 활용하여 거의 모든 사이트의 크롤링이 가능
WebDriverWait(driver, 시간(초)).until(EC.presence_of_element_located((By.CSS_SELECTOR,'CSS_RULE')))
'''
chrome_driver = 'C:/Users/USER/chromedriver'
driver = webdriver.Chrome(chrome_driver)
url = 'https://news.naver.com/main/read.nhn?mode=LSD&mid=sec&sid1=001&oid=081&aid=0003018031'
driver.get(url)
src = driver.page_source
soup = BeautifulSoup(src)
driver.close()
comment = soup.select_one('span.u_cbox_count')
comment.get_text()

chrome_driver = 'C:/Users/USER/chromedriver'
driver = webdriver.Chrome(chrome_driver)
url = 'https://news.naver.com/main/read.nhn?mode=LSD&mid=sec&sid1=001&oid=081&aid=0003018031'
driver.get(url)
WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'span.u_cbox_count')))
src = driver.page_source
soup = BeautifulSoup(src)
driver.close()
comment = soup.select_one('span.u_cbox_count')
comment.get_text()

### selenium 모듈 - 실전 웹 크롤링 연습문제 풀이
# 다음 뉴스와 그 뉴스의 댓글 크롤링하기
import requests
from bs4 import BeautifulSoup

def get_daum_news_title(news_id):
    url = 'https://news.v.daum.net/v/{}'.format(news_id)
    resp = requests.get(url)
    
    soup = BeautifulSoup(resp.text)
    
    title_tag = soup.select_one('h3.tit_view')
    if title_tag:
        return title_tag.get_text()
    return ""

get_daum_news_title('20190728165812603')
get_daum_news_title('20190801114158041')

# 뉴스 본문 크롤링
def get_daum_news_content(news_id):
    url = 'https://news.v.daum.net/v/{}'.format(news_id)
    resp = requests.get(url)
    
    soup = BeautifulSoup(resp.text)
    
    content = ''
    for p in soup.select('div#harmonyContainer p'):
                         content += p.get_text()
    return content

get_daum_news_content('20190728165812603')
get_daum_news_content('20190801114158041')

# 뉴스 댓글 크롤링
url = 'https://comment.daum.net/apis/v1/posts/@20190728165812603/comments?parentId=0&offset=33&limit=10&sort=RECOMMEND&isInitial=false'
requests.get(url)
headers = {
        'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJncmFudF90eXBlIjoiYWxleF9jcmVkZW50aWFscyIsInNjb3BlIjpbXSwiZXhwIjoxNTY4MzExNDQ4LCJhdXRob3JpdGllcyI6WyJST0xFX0NMSUVOVCJdLCJqdGkiOiI2NTg5Nzc2ZC1lMTc1LTRlNjAtYWFjNy01MzkxZjU0NTFhYjgiLCJjbGllbnRfaWQiOiIyNkJYQXZLbnk1V0Y1WjA5bHI1azc3WTgifQ.zQvsd6-MruYk_uMXg-YrI-6QEl0RXPsIsZnlCrEZ4JQ',
'Origin': 'https://news.v.daum.net',
'Referer': 'https://news.v.daum.net/v/20190728165812603',
'User-Agent': 'Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36'
}
requests.get(url, headers=headers)
resp = requests.get(url, headers=headers)
resp.text
resp.json()

def get_daum_news_comments(news_id):
    headers = {
        'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJncmFudF90eXBlIjoiYWxleF9jcmVkZW50aWFscyIsInNjb3BlIjpbXSwiZXhwIjoxNTY4MzExNDQ4LCJhdXRob3JpdGllcyI6WyJST0xFX0NMSUVOVCJdLCJqdGkiOiI2NTg5Nzc2ZC1lMTc1LTRlNjAtYWFjNy01MzkxZjU0NTFhYjgiLCJjbGllbnRfaWQiOiIyNkJYQXZLbnk1V0Y1WjA5bHI1azc3WTgifQ.zQvsd6-MruYk_uMXg-YrI-6QEl0RXPsIsZnlCrEZ4JQ',
'Origin': 'https://news.v.daum.net',
'Referer': 'https://news.v.daum.net/v/20190728165812603',
'User-Agent': 'Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36'
}
    url_template = 'https://comment.daum.net/apis/v1/posts/@{}/comments?parentId=0&offset={}&limit=10&sort=RECOMMEND&isInitial=false'
    offset = 0
    comments = []
    
    while True:
        url = url_template.format(news_id, offset)
        resp = requests.get(url, headers=headers)
        data = resp.json()
        if not data:
            break
        
        comments.extend(data)
        offset += 10
        
    return comments

len(get_daum_news_comments('20190728165812603'))
len(get_daum_news_comments('20190801114158041'))

#------------------------------------------------------------------------------

### numpy 모듈 & ndarray 이해하기
import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

plt.plot(x, y)

### ndarray 데이터 생성하기(numpy 모듈 함수 이용)
# np.array 함수로 생성하기
x = np.array([1, 2, 3, 4])
print(x)
 y = np.array([[2, 3, 4], [1, 2, 5]])
print(y)

print(type(y))

# np.arrange 함수로 생성하기
np.arange(10)
np.arange(1, 10)
np.arange(1, 10, 2)
np.arange(5, 101, 5)

# np.ones, np.zeros로 생성하기
np.ones((4, 5)) # 튜플을 이용해 명시
np.ones((2, 3, 4))
np.zeros((2, 3))
np.zeros((2, 3, 8))
np.zeros((2, 3, 8, 8))

# np.empty, np.full로 생성하기
np.empty((3, 4))
np.full((3, 4), 7)
np.full((3, 4, 2), 7)

# np.eye로 생성하기 (단위 행렬 생성)
np.eye(5)
np.eye(3)

# np.linspace로 생성하기
np.linspace(1, 10, 3)
np.linspace(1, 10, 4)
np.linspace(1, 10, 5)

# reshape 함수 활용 (ndarray의 형태, 차원을 바꾸기 위해 사용)
x = np.arange(1, 16)
print(x)
x.shape
x.reshape(3, 5)
x.reshape(5, 3)
x.reshape(5, 3, 1)

### random 서브모듈 함수를 통해 ndarray 생성하기
# rand 함수 - 0, 1 사이의 분포로 랜덤한 ndarray 생성
np.random.rand(2, 3)
np.random.rand(2)
np.random.rand(10)
np.random.rand(4, 5, 3)

# randn 함수 (정규분포로 샘플링된 랜덤 ndarray 생성)
np.random.randn(3, 4)
np.random.randn(3, 4, 2)
np.random.randn(5)

# randint 함수 (특정 정수 사이에서 랜덤하게 샘플링)
np.random.randint(1, 100, size=(3, 5))
np.random.randint(1, 100, size=(3, 5, 2))
np.random.randint(1, 100, size=(5,))

# seed 함수 - 랜덤한 값을 동일하게 다시 생성하고자 할 때 사용
np.random.seed(100)
np.random.randn(3, 4)

# choice
''' 
주어진 1차원 ndarray로 부터 랜덤으로 샘플링
정수가 주어진 경우, np.arange(해당숫자)로 간주
'''
np.random.choice(100, size=(3, 4))

x = np.array([1, 2, 3, 1.5, 2.6, 4.9])
np.random.choice(x, size=(2, 2))
np.random.choice(x, size=(2, 2), replace=False)

# 확률분포에 따른 ndarray 생성 (uniform, normal 등등)
np.random.uniform(1.0, 3.0, size=(4, 5))
np.random.normal(size = (3, 4)) # np.random.randn(3, 4)와 같은 함수

### ndarray 인덱싱 & 슬라이싱 이해하기
# 인덱싱
'''
파이썬 리스트와 동일한 개념으로 사용
,를 사용하여 각 차원의 인덱스에 접근 가능
'''
# 1차원 벡터 인덱싱
x = np.arange(10)
print(x)
x[0]
x[9]
x[-1]
x[3] = 100
print(x)

# 2차원 행렬 인덱싱
x = np.arange(10).reshape(2, 5)
print(x)
x[0]
x[1]
x[0, 2]
x[1, 2]

# 3차원 텐서 인덱싱
x = np.arange(36).reshape(3, 4, 3)
print(x)
x[0]
x[1, 2]
x[1, 2, 1]

# 슬라이싱
'''
리스트, 문자열 slicing과 동일한 개념으로 사용
,를 사용하여 각 차원 별로 슬라이싱 가능
'''
# 1차원 벡터 슬라이싱
x = np.arange(10)
print(x)
x[1:7]
x[1:]
x[:]

# 2차원 행렬 슬라이싱
x = np.arange(10).reshape(2, 5)
print(x)
x[:, 1:4]
x[:, 0:2]
x[0, :2] # 벡터 (인덱싱은 차원의 갯수를 하나 줄여줌)
x[:1, :2] # 행렬

# 3차원 텐서 슬라이싱
x = np.arange(54).reshape(2, 9, 3)
print(x)
x[:1, :2, :]
x[0, :2, :] # 2차원 행렬

### ndarray 데이터 형태를 바꿔보기(reshape, flatten 등 함수 이용)
# ravel, np.ravel
'''
다차원배열을 1차원으로 변경
'order' 파라미터
'C' - row 우선 변경
'F' - column 우선 변경
'''
x = np.arange(15).reshape(3, 5)
print(x)
x.ravel()
np.ravel(x)

# flatten
'''
다차원배열을 1차원으로 변경
ravel과의 차이점: copy를 생성하여 변경함(즉 원본 데이터가 아닌 복사본을 반환)
'order' 파라미터
'C' - row 우선 변경
'F' - column 우선 변경
'''
y = np.arange(15).reshape(3, 5)
print(x)
y.flatten()

temp = x.ravel()
print(temp)
temp[0] = 100
print(temp)
print(x)

t2 = y.flatten()
print(t2)
t2[0] = 100
print(t2)
print(y)

np.ravel(x, order='C')
np.ravel(x, order='F')

t2 = y.flatten(order='C')
print(t2)
t2 = y.flatten(order='F')
print(t2)

x = np.arange(30).reshape(2, 3, 5)
print(x)
x.ravel()

# reshape 함수
'''
array의 shape을 다른 차원으로 변경
주의할 점은 reshape한 후의 결과의 전체 원소 개수와 이전 개수가 같아야 가능
사용 예) 이미지 데이터 벡더화 - 이미지는 기본적으로 2차원 혹은 3차원(RGB)이나 트레이닝을 위해 1차원으로 변경하여 사용 됨
'''
x= np.arange(36)
print(x)
print(x.shape)
print(x.ndim)

x.reshape(6, 6)
x.reshape(6, -1)
x.reshape(-1, 6)

y = x.reshape(6, 6)
print(y.shape)
print(y.ndim)

k = x.reshape(3, 3, 4) # k = x.reshape(3, 3, -1)도 같음
print(k)
print(k.shape)
print(k.ndim)
#------------------------------------------------------------------------------
### ndarray 기본 함수 사용하기
import numpy as np
# https://numpy.org/devdocs/reference/
x = np.arange(15).reshape(3, 5)
y = np.random.rand(15).reshape(3, 5)
print(x)
print(y)

# 연산 함수 (add, subtract, multiply, divide)
np.add(x, y)
np.subtract(x, y)
np.multiply(x, y)
np.divide(x, y)

x + y
x - y
x * y
x / y

# 통계 함수
print(y)
np.mean(y)
y.mean()
np.max(y)
np.argmax(y)
np.var(y), np.median(y), np.std(y)

# 집계 함수
y
np.sum(y)
sum(y)
np.sum(y, axis=0)
np.cumsum(y)

# any, all 함수
'''
any: 특정 조건을 만족하는 것이 하나라도 있으면 Trun, 아니면 False
all: 모든 원소가 특정 조건을 만족한다면 True, 아니면 False
'''
z = np.random.randn(10)
print(z)
z > 0
np.any(z > 0)
np.all(z != 0)

# where 함수
'''
조건에 따라 선별적으로 값을 선택 가능
사용 예 ) 음수인 경우는 0, 나머지는 그대로 값을 쓰는 경우
'''
z = np.random.randn(10)
print(z)
np.where(z > 0 , z, 0)

### axis(축) 이해 및 axis를 파라미터로 갖는 함수 활용하기
# axis 이해하기
'''
몇몇 함수에는 axis keyword 파라미터가 존재
axis값이 없는 경우에는 전체 데이터에 대해 적용
axis값이 있는 경우에는, 해당 axis를 따라서 연산 적용
'''
x = np.arange(15)
print(x)
np.sum(x)
np.sum(x, axis=0)

y = x.reshape(3, 5)
print(y)
np.sum(y)
np.sum(y, axis=0)
np.sum(y, axis=1)

z = np.arange(36).reshape(3, 4, 3)
print(z)
np.sum(z)
np.sum(z, axis=0)
np.sum(z, axis=1)
np.sum(z, axis=2)
np.sum(z, axis=-1)

# axis 값이 튜플일 경우 해당 튜플에 명시된 모든 axis에 대해서 연산
print(z)
np.sum(z, axis=(0, 1))
np.sum(z, axis=(0, 2))

### Boolean indexing으로 조건에 맞는 데이터 선택하기
# ndarray 인덱싱 시, bool 리스트를 전달하여 True인 경우만 필터링
# 브로드캐스팅을 활용하여 ndarray로부터 bool list 얻기 (예) 짝수인 경우만 찾아보기
x = np.random.randint(1, 100, size=10)
print(x)
even_mask = x % 2 == 0
print(even_mask)
x[even_mask]
x[x % 2 == 0]
x[x > 30]

# 다중조건 사용하기
'''
파이썬 논리 연산자인 and, or, not 키워드 사용 불가
& - and
| - or
'''
x % 2 == 0
x < 30
x[(x % 2 == 0) & (x < 30)]
x[(x % 2 == 0) | (x < 30)]

# 예제) 2019년 7월 서울 평균기온 데이터
'''
평균기온이 25도를 넘는 날수는?
평균기온이 25도를 넘는 날의 평균 기온은?
'''
temp = np.array(
        [23.9, 24.4, 24.1, 25.4, 27.6, 29.7,
         26.7, 25.1, 25.0, 22.7, 21.9, 23.6,
         24.9, 25.9, 23.8, 24.7, 25.6, 26.9,
         28.6, 28.0, 25.1, 26.7, 28.1, 26.5,
         26.3, 25.9, 28.4, 26.1, 27.5, 28.1, 25.8])
len(temp)
len(temp[temp > 25.0])
np.sum(temp > 25.0) # True는 정수 연산에 사용될 경우 1의 값을 가짐
np.mean(temp[temp > 25.0])

### broadcasting 이해 및 활용하기
# 브로드캐스팅
'''
shape이 같은 두 ndarray에 대한 연산은 각 원소별로 진행
연산되는 두 ndarray가 다른 shape을 갖는 경우 브로드캐스팅(shape을 맞춤) 후 진행
'''
# https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html#general-broadcasting-rules
# 뒷 차원에서부터 비교하여 Shape이 같거나, 차원 중 값이 1인 것이 존재하면 가능
# https://www.tutorialspoint.com/numpy/images/array.jpg

# Shape이 같은 경우의 연산
x = np.arange(15).reshape(3, 5)
y = np.random.rand(15).reshape(3, 5)
print(x)
print(y)
x + y

# Scalar(상수)와의 연산
x + 2

# Shape이 다른 경우 연산
a = np.arange(12).reshape(4, 3)
b = np.arange(100, 103)
c = np.arange(1000, 1004)
d = b.reshape(1, 3)
print(a)
print(b)
print(c)
print(d)
print(a.shape)
print(b.shape)
print(c.shape)
print(d.shape)
a + b
a + c
a + d

### linalg 서브모듈 사용하여 선형대수 연산하기
# np.linalg.inv
'''
역행렬을 구할 때 사용
모든 차원의 값이 같아야 함
'''
x = np.random.rand(3, 3)
print(x)
np.linalg.inv(x)
x @ np.linalg.inv(x) # 행렬곱은 @ 사용
np.matmul(x, np.linalg.inv(x)) # 이것도 같은 함수

# np.linalg.solve
'''
Ax = B 형태의 선형대수식 솔루션을 제공
예제) 호랑이와 홍합의 합 : 25 호랑이 다리와 홍합 다리의 합은 64
x + y = 25
2x + 4y = 64
'''
A = np.array([[1, 1], [2, 4]])
B = np.array([25, 64])
x = np.linalg.solve(A, B)
print(x)
np.allclose(A@x, B)

### ndarray 데이터를 이용하여 다양한 그래프 표현하기
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# 그래프 데이터 생성
x = np.linspace(0, 10, 11)
y = x ** 2 + x + np.random.randn(11)
print(x)
print(y)

# 그래프 출력하기
'''
plot함수(선 그래프), scatter(점 그래프), hist(히스토그램)등 사용
함수의 parameter 호근 plt의 다른 함수로 그래프 형태 및 설정을 변경 가능
기본적으로 x, y에 해당하는 값이 필요
'''
plt.plot(x, y)
plt.scatter(x, y)

# 그래프에 주석 추가
# x, y 축 및 타이틀
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('X-Y relation')
plt.plot(x, y)

# grid 추가
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('X-Y relation')
plt.grid(True)
plt.plot(x, y)

# x, y축 범위 지정
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('X-Y relation')
plt.grid(True)
plt.xlim(0, 20)
plt.ylim(0, 200)
plt.plot(x, y)

# plot 함수 parameters
'''
그래프의 형태에 대한 제어 가능
https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot
'''

# 그래프의 색상 변경
plt.plot(x, y, 'r')
plt.plot(x, y, 'b')
plt.plot(x, y, 'y')
plt.plot(x, y, 'k')
plt.plot(x, y, '#ff00ff')

# 그래프 선스타일 변경
plt.plot(x, y, '-.')
plt.plot(x, y, 'g^')
plt.plot(x, y, 'm:')

# 그래프의 두께 변경
# linewidth 명시
plt.plot(x, y, 'm:', linewidth=3)
plt.plot(x, y, 'm:', linewidth=9)

'''
keyword parameter 이용하여 모든 속성 설정
color
linestyle
marker
markerfacecolor
markersize 등등
'''
plt.plot(x, y, color='black',
         linestyle='--', marker='^',
         markerfacecolor='blue', markersize=6)

# subplot으로 여러 그래프 출력하기
plt.subplot(2, 2, 1)
plt.plot(x, y, 'r')

plt.subplot(2, 2, 2)
plt.plot(x, y, 'g')

plt.subplot(2, 2, 3)
plt.plot(y, x, 'k')

plt.subplot(2, 2, 4)
plt.plot(x, np.exp(x), 'b')

# hist함수 사용
'''
histogram 생성
bins로 histogram bar 개수 설정
'''
data = np.random.randint(1, 100, size=200)
print(data)

plt.hist(data, bins=20, alpha=0.3)
plt.xlabel('값')
plt.ylabel('개수')
plt.grid(True)

# 연습문제
'''
1. 로또 번호 자동 생성기(함수로)를 만드시오
2. numpy를 이용하여 pi(원주율) 값을 계산하시오
 몬테 카를로 방법 이용하기
 http://mathfaculty.fullerton.edu/mathews/n2003/montecarlopi/MonteCarloPiMod/Images/MonteCarloPiMod_gr_25.gif
'''
# 1
np.random.choice(np.arange(1, 46), size=6) # 중복 발생
np.random.choice(np.arange(1, 46), size=6, replace=False)

def generate_lotto_nums():
    return np.random.choice(np.arange(1, 46), size=6, replace=False)
    
generate_lotto_nums()

# 2
# pi/4 : 1 = (4분원 안에 생성된 점의 개수) : 전체 시도 횟수
# pi = 4 * (4분원 안에 생성된 점의 개수) / 1e7(천만 번)
total = int(1e7)
points = np.random.rand(total, 2)
4 * np.sum(np.sum(points ** 2, axis=1) < 1) / total

#------------------------------------------------------------------------------

### Series 데이터 생성하기
import numpy as np
import pandas as pd

# Series
'''
pandas의 기본 객체 중 하나
numpy의 ndarray를 기반으로 인덱싱 기능을 추가하여 1차원 배열을 나타냄
index를 지정하지 않을 시, 기본적으로 ndarray와 같이 0-based 인덱스 생성, 지정할 경우 명시적으로 지정된 index를 사용
같은 타입의 0개 이상의 데이터를 가질 수 있음
'''
# data로만 생성하기
# index는 기본적으로 0부터 자동적으로 생성
s1 = pd.Series([1, 2, 3])
s1
s2 = pd.Series(['a', 'b', 'c'])
s2
s3 = pd.Series(np.arange(200))
s3

# data, index 함께 명시하기
s4 = pd.Series([1, 2, 3], [100, 200, 300])
s4
s5 = pd.Series([1, 2, 3], ['a', 'm', 'k'])
s5

# data, index, data type 함께 명시하기
s6 = pd.Series(np.arange(5), np.arange(100, 105), dtype=np.int32)
s6

# 인덱스 활용하기
s6.index
s6.values

# 1. 인덱스를 통한 데이터 접근
s6[104]
s6[105]

# 2. 인덱스를 통한 데이터 업데이트
s6[104] = 70
s6
s6[105] = 90
s6

# 3. 인덱스 재사용하기
s7 = pd.Series(np.arange(6), s6.index)
s7

### Series 데이터 심플 분석 (개수, 빈도 등 계산하기)
# Series size, shape, unique, count, value_couns 함수
'''
size: 개수 반환
shape: 튜플형태로 shape 반환
unique: 유일한 값만 ndarray로 반환
count: NaN을 제외한 개수를 반환
mean: NaN을 제외한 평균
value_counts: NaN을 제외하고 각 값들의 빈도를 반환
'''
s = pd.Series([1, 1, 2, 1, 2, 2, 2, 1, 1, 3, 3, 4, 5, 5, 7, np.NaN])
s
len(s)
s.size
s.shape
s.unique()
s.count()
a = np.array([2, 2, 2, 2, np.nan])
a.mean() # NaN을 반환
b = pd.Series(a)
b.mean() # numpy와 다르게 NaN 값을 무시하고 평균을 반환
s.mean()
s.value_counts()

# index를 활용하여 멀티플한 값에 접근
s[[5, 7, 8, 10]]
s[[5, 7, 8, 10]].value_counts()

# head, tail 함수
'''
head: 상위 n개 출력 기본 5개
tail: 하위 n개 출력 기본 5개
'''
s.head(n=7)
s.tail()
s

### Series 데이터 연산하기
# index를 기준으로 연산
s1 = pd.Series([1, 2, 3, 4], ['a', 'b', 'c', 'd'])
s2 = pd.Series([6, 3, 2, 1], ['d', 'c', 'b', 'a'])
s1
s2
s1 + s2

# 산술연산
'''
Series의 경우에도 스칼라와의 연산은 각 원소별로 스칼라와의 연산이 적용
Series와의 연산은 각 인덱스에 맞는 값끼리 연산이 적용
이때, 인덱스의 pair가 맞지 않으면, 결과는 NaN
'''
s1 ** 2
s1 ** s2

# index pair가 맞지 않는 경우 해당 index에 대해선 NaN값 생성
s1['k'] = 7
s2['e'] = 9
s1
s2
s1 + s2

### Series 데이터 Boolean Selection으로 데이터 선택하기
# Boolean selection
'''
boolean Series가 []와 함께 사용되면 True 값에 해당하는 값만 새로 반환되는 Series 객체에 포함됨
다중조건의 경우 &(and), |(or)를 사용하여 연결 가능
'''
s = pd.Series(np.arange(10), np.arange(10)+1)
s
s > 5
s[s>5]
s[s % 2 == 0]
s
s[s.index > 5]
s[(s > 5) & (s < 8)]

(s >= 7).sum()
(s[s >= 7]).sum()

### Series 데이터 변경 & 슬라이싱하기
# Series 값 변경
'''
추가 및 업데이트: 인덱스를 이용
삭제: drop 함수 이용
'''
s = pd.Series(np.arange(100, 105), ['a', 'b', 'c', 'd', 'e'])
s
s['a'] = 200
s
s['k'] = 300
s
s.drop('k')
s # drop 함수는 s 자체에 아무 영향을 주지 않음
s.drop('k', inplace=True)
s # s에서도 'k'가 지워짐
s[['a', 'b']] = [300, 900]
s

# Slicing (리스트, ndarray와 동일하게 적용)
s1 = pd.Series(np.arange(100, 105))
s1
s1[1:3]
s2 = pd.Series(np.arange(100, 105), ['a', 'c', 'b', 'd', 'e'])
s2
s2[1:3]
s2['c':'d'] # 문자열로 이루어진 인덱스의 경우에는 마지막까지 포함
#------------------------------------------------------------------------------
### DataFrame 데이터 살펴보기

'''
DataFrame
Series가 1차원이라면 DataFrame은 2차원으로 확대된 버전
Excel spreadsheet라고 생각하면 이해하기 쉬움
2차원이기 때문에 인덱스가 row, column으로 구성됨
row는 각 개별 데이터를, column은 개별 속성을 의미
Data Analysis, Machine Learning에서 data 변형을 위해 가장 많이 사용
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# data 출처: https://www.kaggle.com/hesh97/titanicdataset-traincsv/data
train_data = pd.read_csv('./train.csv')

# head, tail 함수
# 데이터 전체가 아닌, 일부(처음부터, 혹은 마지막부터를 간단히 보기 위한 함수)
train_data.head(n=3)
train_data.tail(n=10)

# dataframe 데이터 파악하기
'''
shape 속성 (row, column)
describe 함수 - 숫자형 데이터의 통계치 계산
info 함수 -데이터 타입, 각 아이템의 개수 등 출력
'''
train_data.shape
train_data.describe()
train_data.info(0)

### DataFrame 구조 이해하기
# 인덱스(index)
'''
index 속성
각 아이템을 특정할 수 있는 고유의 값을 저장
복잡한 데이터의 경우, 멀티 인덱스로 표현 가능
'''
train_data.index

# 컬럼(column)
'''
column 속성
각각의 특성(feature)을 나타냄
복잡한 데이터의 경우, 멀티 컬럼으로 표현 가능
'''
train_data.columns

### DataFrame 데이터 생성하기
# DataFrame 생성하기
'''
일반적으로 분석을 위한 데이터는 다른 데이터 소스(database, 외부 파일)을 통해 dataframe을 생성
여기서는 실습을 통해, dummy 데이터를 생성하는 방법을 다룰 예정
'''
# dictionary로 부터 생성하기
# dict의 key -> column
data = {'a' : 100, 'b' : 200, 'c' : 300}
pd.DataFrame(data, index={'x', 'y', 'z'})
data = {'a' : [1, 2, 3], 'b' : [4, 5, 6], 'c' : [10, 11, 12]}
pd.DataFrame(data, index=[0, 1, 2])

# Series로 부터 생성하기
# 각 Series의 인덱스 -> column
a = pd.Series([100, 200, 300], ['a', 'b', 'c'])
b = pd.Series([101, 201, 301], ['a', 'b', 'c'])
c = pd.Series([110, 210, 310], ['a', 'b', 'c'])
pd.DataFrame([a, b, c], index=[100, 101, 102])

a = pd.Series([100, 200, 300], ['a', 'b', 'd'])
b = pd.Series([101, 201, 301], ['a', 'b', 'k'])
c = pd.Series([110, 210, 310], ['a', 'b', 'c'])
pd.DataFrame([a, b, c], index=[100, 101, 102])

### 샘플 csv 데이터로 DataFrame 데이터 생성하기
# csv 데이터로부터 DataFrame 생성
'''
데이터 분석을 위해, dataframe을 생성하는 가장 일반적인 방법
데이터 소스로부터 추출된 csv(comma separated values) 파일로부터 생성
pandas.read_csv 함수 사용
'''
train_data = pd.read_csv('./train.csv')
train_data.head()

# read_csv 함수 파라미터
'''
sep - 각 데이터 값을 구별하기 위한 구분자(separator) 설정
header - header를 부시할 경우, None 설정
index_col - index로 사용할 column 설정
usecols - 실제로 dataframe에 로딩할 columns만 설정
'''
train_data = pd.read_csv('./train.csv', sep='#')
train_data
train_data = pd.read_csv('./train.csv', sep=',')
train_data

train_data = pd.read_csv('./train.csv', header=None)
train_data

train_data = pd.read_csv('./train.csv', index_col='PassengerId')
train_data
train_data.columns

train_data = pd.read_csv('./train.csv', usecols=['Survived', 'Pclass', 'Name'])
train_data

### DataFrame 원하는 column(컬럼)만 선택하기
# column 선택하기
'''
기본적으로 []는 column을 추출
컬럼 인덱스일 경우 인덱스의 리스트 사용 가능
리스트를 전달할 경우 결과는 Dataframe
하나의 컬럼명을 전달할 경우 결과는 Series
'''
train_data = pd.read_csv('./train.csv')
train_data

# 하나의 컬럼 선택하기
train_data[0]
train_data['Survived']

# 복수의 컬럼 선택하기
train_data[['Survived', 'Age', 'Name']]
train_data[['Survived']]

### DataFrame 원하는 row(데이터)만 선택하기
# dataframe slicing
'''
dataframe의 경우 기본적으로 []연산자가 column 선택에 사용
하지만, slicing은 row 레벨로 지원
'''
train_data[:10]

# row 선택하기
'''
Series의 경우 []로 row 선택이 가능하나, DataFrame의 경우는 기본적으로 column을 선택하도록 설계
.loc, .iloc로 row 선택 가능
loc - 인덱스 자체를 사용
iloc - 0 based index로 사용
이 두 함수는 ,를 사용하여 column 선택도 가능
'''
train_data.head()
train_data.index =  np.arange(100,991)
train_data.head()
train_data.loc[986]
train_data.loc[[986, 100, 110, 990]]
train_data.head()
train_data.iloc[0]
train_data.iloc[[0, 100, 200, 2]]
train_data.loc[[0, 100, 200, 2]]

# row, column 동시에 선택하기
# loc, iloc 속성을 이용할 때, 콤마를 이용하여 둘 다 명시 가능
train_data.loc[[986, 100, 110, 990], ['Survived', 'Name', 'Sex', 'Age']]
train_data.iloc[[101, 110, 200, 102], [1, 4, 5]]

### DataFrame Boolean Selection으로 데이터 선택하기
train_data = pd.read_csv('./train.csv')
train_data.head()
# boolean selection으로 row 선택하기
# numpy에서와 동일한 방식으로 해당 조건에 맞는 row만 선택

# 30대이면서 1등석에 탄 사람 선택하기
class_ = train_data['Pclass'] == 1
age_ = (train_data['Age'] >= 30) & (train_data['Age'] < 40)

train_data[class_ & age_]

### DataFrame에 새 column(컬럼) 추가 & 삭제하기
train_data = pd.read_csv('./train.csv')
train_data.head()

# 새 column 추가하기
'''
[] 사용하여 추가하기
insert 함수 사용하여 원하는 위치에 추가하기
'''
train_data['Age_double'] = train_data['Age'] * 2
train_data.head()
train_data['Age_tripple'] = train_data['Age_double'] + train_data['Age']
train_data.head()

train_data.insert(3, 'Fare10', train_data['Fare'] / 10)
train_data.head()

# column 삭제하기
# drop 함수 사용하여 삭제
# 리스트를 사용하여 멀티플 삭제 가능
train_data.drop('Age_tripple', axis=1)
train_data.head()

train_data.drop(['Age_double', 'Age_tripple'], axis=1)
train_data.head()

train_data.drop(['Age_double', 'Age_tripple'], axis=1, inplace=True) # inplace 원본 자체에 실행
train_data.head()

### DataFrame column(컬럼)간 상관관계 계산하기
# 변수(column) 사이의 상관계수(correlation)
'''
corr 함수를 통해 상관계수 연산(-1, 1 사이의 결과)
연속형(숫자형)데이터에 대해서만 연산
인과관계를 의미하진 않음
'''
train_data = pd.read_csv('./train.csv')
train_data.head()
train_data.corr()
plt.matshow(train_data.corr())

### DataFrame NaN 데이터 처리
# NaN 값 확인
# info함수를 통하여 개수 확인
# isna함수를 통해 boolean 타입으로 확인
train_data = pd.read_csv('./train.csv')
train_data.head()

train_data.info()
train_data.isna()
train_data['Age'].isna()

# NaN 처리 방법
# 데이터에서 삭제(dropna함수)
# 다른 값으로 치환(fillna함수)
train_data.dropna()
train_data.dropna(subset=['Age'])
train_data.dropna(axis=1)

# NaN 값 대체하기
# 평균으로 대체하기
train_data['Age'].fillna(train_data['Age'].mean())

# 생존자/사망자 별 평균으로 대체하기
# 생존자 나이 평균
mean1 = train_data[train_data['Survived'] == 1]['Age'].mean()
# 사망자 나이 평균
mean0 = train_data[train_data['Survived'] == 0]['Age'].mean()

train_data[train_data['Survived'] == 1]['Age'].fillna(mean1)
train_data[train_data['Survived'] == 0]['Age'].fillna(mean0)

train_data.loc[train_data['Survived'] == 1, 'Age'] = train_data[train_data['Survived'] == 1]['Age'].fillna(mean1)
train_data.loc[train_data['Survived'] == 0, 'Age'] = train_data[train_data['Survived'] == 0]['Age'].fillna(mean0)
train_data

### 숫자 데이터와 범주형 데이터의 이해
train_data = pd.read_csv('./train.csv')
train_data.head()

# info함수로 각 변수의 데이터 타입 확인
# 타입 변경은 astype함수를 사용
train_data.info()

# 숫자형(Numerical Type) 데이터
# 연속성을 띄는 숫자로 이루어진 데이터(Age, Fare 등)
# 범주형(Categorical Type) 데이터
# 연속적이지 않은 값을 갖는 데이터를 의미(Name, Sex, Ticket, Cabin, Embarked)
# 어떤 경우, 숫자형 타입이라 할지라도 개념적으로 범주형으로 처리해야할 경우가 있음(Pclass)

### 숫자 데이터의 범주형 데이터화
# Pclass 변수 변환하기
# astype 사용하여 간단한 타입만 변환
train_data.info()
train_data['Pclass'] = train_data['Pclass'].astype(str)
train_data.info()

# Age 변수 변환하기
# 변환 로직을 함수로 만든 후, apply 함수로 적용
import math
def age_categorize(age):
    if math.isnan(age):
        return -1
    return math.floor(age / 10) * 10
train_data['Age'].apply(age_categorize)

### 범주형 데이터 전처리 하기(one-hot encoding)
# One-hot encoding
'''
범주형 데이터는 분석단계에서 계산이 어렵기 때문에 숫자형으로 변경이 필요함
범주형 데이터의 각 범주(category)를 column레벨로 변경
해당 범주에 해당하면 1, 아니면 0으로 채우는 인코딩 기법
pandas.get_dummies 함수 사용
drop_first : 첫번째 카테고리 값은 사용하지 않음
'''
train_data = pd.read_csv('./train.csv')
train_data.head()
pd.get_dummies(train_data)
pd.get_dummies(train_data, columns=['Pclass', 'Sex', 'Embarked'])
pd.get_dummies(train_data, columns=['Pclass', 'Sex', 'Embarked'], drop_first=True)

#------------------------------------------------------------------------------
### DataFrame group by 이해하기
import pandas as pd
import numpy as np

# group by
'''
아래의 세 단계를 적용하여 데이터를 그룹화(groupping)(SQL의 group by와 개념적으로는 동일, 사용법은 유사)
데이터 분할
operation 적용
데이터 병합
'''
df = pd.read_csv('./train.csv')
df.head()

# GroupBy groups 속성
# 각 그룹과 그룹에 속한 index를 dict 형태로 표현
class_group = df.groupby('Pclass')
class_group
class_group.groups

gender_group = df.groupby('Sex')
gender_group.groups

# groupping 함수
'''
그룹 데이터에 적용 가능한 통계 함수(NaN은 제외하여 연산)
count - 데이터 개수
sum - 데이터의 합
mean, std, var - 평균, 표준편차, 분산
min, max - 최소, 최대값
'''
class_group.count()
class_group.sum()
class_group.mean()['Age']
class_group.mean()['Survived']
class_group.min()

# 성별에 따른 생존률 구해보기
df.groupby('Sex').mean()['Survived']

# 복수 columns로 groupping 하기
'''
groupby에 column 리스트를 전달
통계함수를 적용한 결과는 multiindex를 갖는 dataframe
클래스와 성별에 따른 생존률 구해보기
'''
df.groupby(['Pclass', 'Sex']).mean()
df.groupby(['Pclass', 'Sex']).mean().index
df.groupby(['Pclass', 'Sex']).mean().loc[(2, 'female')]
df.groupby(['Pclass', 'Sex']).mean()['Survived']

# index를 이용한 group by
'''
index가 있는 경우, groupby 함수에 level 사용 가능
level은 index의 depth를 의미하며, 가장 왼쪽부터 0부터 증가
set_index 함수
column 데이터를 index 레벨로 변경
reset_index 함수
인덱스 초기화
'''
df.head()
df.set_index('Pclass')
df.set_index(['Pclass', 'Sex'])
df.set_index(['Pclass', 'Sex']).reset_index()

df.set_index('Age').groupby(level=0).mean()

# 나이대별로 생존율 구하기
import math
def age_categorize(age):
    if math.isnan(age):
        return -1
    return math.floor(age / 10) * 10
df.set_index('Age').groupby(age_categorize).mean()['Survived']

# Multiindex를 이용한 groupping
df.set_index(['Pclass', 'Sex']).groupby(level=[0, 1]).mean()['Age']

# aggregate(집계)함수 사용하기
# groupby 결과에 집계함수를 적용하여 그룹별 데이터 확인 가능
df.set_index(['Pclass', 'Sex']).groupby(level=[0, 1]).aggregate([np.mean, np.sum, np.max])

### transform 함수의 이해 및 활용하기
df = pd.read_csv('./train.csv')
df.head()
# transform 함수
'''
groupby 후 trasform 함수를 사용하면 원래의 index를 유지한 상태로 통계함수를 적용
전체 데이터의 집계가 아닌 각 그룹에서의 집계를 계산
따라서 새로 생성된 데이터를 원본 dataframe과 합치기 쉬움
'''
df.groupby('Pclass').mean()
df.groupby('Pclass').transform(np.mean)
df['Age2'] = df.groupby('Pclass').transform(np.mean)['Age']
df
df.groupby(['Pclass', 'Sex']).mean()
df['Age3'] = df.groupby(['Pclass', 'Sex']).transform(np.mean)['Age']
df

### pivot, pivot_table 함수의 이해 및 활용하기
df = pd.DataFrame({
        '지역': ['서울', '서울', '서울', '경기', '경기', '부산', '서울', '서울', '부산', '경기', '경기', '경기'],
        '요일': ['월요일', '화요일', '수요일', '월요일', '화요일', '월요일', '목요일', '금요일', '화요일', '수요일', '목요일', '금요일'],
        '강수량': [100, 80, 1000, 200, 200, 100, 50, 100, 200, 100, 50, 100],
        '강수확률':[80, 70, 90, 10, 20, 30, 50, 90, 20, 80, 50, 10]
        })
df

# pivot
'''
dataframe의 형태를 변경
인덱스, 컬럼, 데이터로 사용할 컬럼을 명시
'''
df.pivot('지역', '요일')
df.pivot('요일', '지역')
df.pivot('요일', '지역', '강수량')
# pivot_table
'''
기능적으로 pivot과 동일
pivot과의 차이점
중복되는 모호한 값이 있을 경우, aggregation 함수 사용하여 값을 채움
'''
df = pd.DataFrame({
        '지역': ['서울', '서울', '서울', '경기', '경기', '부산', '서울', '서울', '부산', '경기', '경기', '경기'],
        '요일': ['월요일', '월요일', '수요일', '월요일', '화요일', '월요일', '목요일', '금요일', '화요일', '수요일', '목요일', '금요일'],
        '강수량': [100, 80, 1000, 200, 200, 100, 50, 100, 200, 100, 50, 100],
        '강수확률':[80, 70, 90, 10, 20, 30, 50, 90, 20, 80, 50, 10]
        })
df
# 중복 허용 후 호출
df.pivot('요일', '지역') # 서울, 월요일에 중복된 값 있어서 에러
df.pivot_table(df, index='요일', columns='지역', aggfunc=np.mean)

### stack, unstack 함수의 이해 및 활용하기
df = pd.DataFrame({
        '지역': ['서울', '서울', '서울', '경기', '경기', '부산', '서울', '서울', '부산', '경기', '경기', '경기'],
        '요일': ['월요일', '화요일', '수요일', '월요일', '화요일', '월요일', '목요일', '금요일', '화요일', '수요일', '목요일', '금요일'],
        '강수량': [100, 80, 1000, 200, 200, 100, 50, 100, 200, 100, 50, 100],
        '강수확률':[80, 70, 90, 10, 20, 30, 50, 90, 20, 80, 50, 10]
        })
df
# stack & unstack
'''
stack: 컬럼 레벨에서 인덱스 레벨로 dataframe 변경
즉, 데이터를 쌓아올리는 개념
unstack: 인덱스 레벨에서 컬럼 레벨로 dataframe 변경
stack의 반대 operation
둘은 역의 관계에 있음
'''
new_df = df.set_index(['지역', '요일'])
new_df
new_df.index
new_df.unstack(0)
new_df.unstack(1)

new_df.unstack(0).stack(0)
new_df.unstack(0).stack(1)
new_df.stack

#------------------------------------------------------------------------------
### Concat 함수로 데이터 프레임 병합하기
# concat 함수 사용하여 dataframe 병합하기
'''
pandas concat 함수
축을 따라 dataframe을 병합 가능
기본 axis=0 -> 행단위 병합
'''
# colum명이 같은 경우
df1 = pd.DataFrame({'key1' : np.arange(10), 'value1' : np.random.randn(10)})
df2 = pd.DataFrame({'key1' : np.arange(10), 'value1' : np.random.randn(10)})
df1
pd.concat([df1, df2])
pd.concat([df1, df2], ignore_index=True)
pd.concat([df1, df2], axis=1)

# colum명이 다른 경우
df3 = pd.DataFrame({'key2' : np.arange(10), 'value2' : np.random.randn(10)})
pd.concat([df1, df3])
pd.concat([df1, df3], axis=1)

### Merge & join 함수로 데이터 프레임 병합하기
'''
dataframe merge
SQL의 join처럼 특정한 column을 기준으로 병합
join 방식 : how 파라미터를 통해 명시
inner: 기본값, 일치하는 값이 있는 경우
left: left outer join
right: right outer join
outer: full outer join
pandas.merge 함수가 사용됨
'''
customer = pd.DataFrame({'customer_id' : np.arange(6),
                         'name' : ['철수'"", '영희', '길동', '영수', '수민', '동건'],
                         '나이' : [40, 20, 21, 30, 31, 18]})
customer
orders = pd.DataFrame({'customer_id' : [1, 1, 2, 2, 2, 3, 3, 1, 4, 9],
                       'item' : ['치약', '칫솔', '이어폰', '헤드셋', '수건', '생수', '수건', '치약', '생수', '케이스'],
                       'quantity' : [1, 2, 1, 1, 3, 2, 2, 3, 2, 1]})
orders.head()

# on(join 대상이 되는 column 명시)
pd.merge(customer, orders, on='customer_id', how='inner')
pd.merge(customer, orders, on='customer_id', how='left')
pd.merge(customer, orders, on='customer_id', how='right')
pd.merge(customer, orders, on='customer_id', how='outer')

# index 기준으로 join
cust1 = customer.set_index('customer_id')
order1 = orders.set_index('customer_id')
cust1
order1
pd.merge(cust1, order1, left_index=True, right_index=True)

'''
연습문제
1. 가장 많이 팔린 아이템은?
2. 영희가 가장 많이 구매한 아이템은?
'''
#1
pd.merge(customer, orders, on='customer_id').groupby('item').sum().sort_values(by='quantity', ascending=False)
#2
pd.merge(customer, orders, on='customer_id').groupby(['name', 'item']).sum().loc['영희', 'quantity']

# join 함수
'''
기본적으로 pandas.merge 함수 사용
기본적으로 index를 사용하여 left join
'''
cust1.join(order1)
cust1.join(order1, how='inner')
