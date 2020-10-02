############## Python for Finance - PART 1 ##############
# 변수와 값, 기본 데이터 타입, 간단한 출력

### 다양한 출력 포맷팅
# 1) % 포맷팅 (파이썬 2)
a, b = 'Hello', 3.14
"%s %.2f" % (a, b)
# 2) str.format() 포맷팅 (파이썬 2, 파이썬 3)
a, b = 'Hello', 3.14
"{} {:.2f}".format(a, b)
# 3) f-string (파이썬 3.6 이상)
a, b = 'Hello', 3.14
f"{a} {b}"

### % 포맷팅 (파이썬 2)
# %f - float
# %d - int
# %s - str
"%d" % (100)

a = 10
b = 30

"%d, %d" % (a, b) # 순서대로 %d에 매칭되어 출력
# 특히, %f의 경우 소수점 이하 몇 자리까지 출력할 것인지 지정할 수 있습니다.

print("%s is short, You need %s!" % ("Life", "파이썬"))
print("Today 14 Mar is Pi (%.2f) day  !" % (3.1415926535))
print("x= %.2f, y = %d, z = %.3f" % (3.1415, 2.8, 0.0150))

### str.format() 포맷팅 (파이썬 2, 파이썬 3)
# 파이썬3 에서는 str.format() 함수를 사용하여 좀 더 복잡한 표현이 가능한 포맷 문자열을 제공합니다.

# "포맷문자열".format(값1, 값2, 값3)
# 포맷문자열의 {} 부분에순서 변수명등을 지정할 수 있을 뿐만 아니라 타입과 다양한 포맷 방법을 제공합니다.

# str.format()  포맷팅 (Python3 스타일)

# 변수에 값을 할당
a, b, c = 3.1415, 2.8, 0.015

# 순서대로 할당
print("x = {:.2f}, y = {}, z = {}".format(a, b, c))

# 번호로 순서를 지정
print("x = {0:.2f}, y = {2}, z = {1}".format(a, b, c))

# 변수명을 사용하여 지정
print("x = {x:.2f}, z = {z}, y = {y}".format(x=a, y=b, z=c))

### f-string (파이썬 3.6 이상)
# %를 사용하여 문자열을 포맷팅(파이썬2, 파이썬3)하는 방법과 str.format()을 사용하는 문자열 포맷팅(파이썬3)하는 방법은 모두 장단점이 있다. 출력하는 변수가 많아지면 상당히 복잡해 집니다.

# f-string 포맷팅

# 변수에 값을 할당
a, b, c = 3.1415, 2.8, 0.015

# 순서대로 할당
print(f"x = {a:.2f}, y = {b}, z = {c}")

# {}안에서 변수 연산도 가능
print(f"x = {a}, w = {b + c}")

### 수치 값들을 자리 맞추어 출력하기
# 다양한 수치 값이 다양한 자리수로 출력되어 읽기 불편
for x in range(1, 200, 20):
    print("{} {} {}".format(x, x**2, x/x**2))
    
# % 포맷팅 (파이썬2, 파이썬3)
for x in range(1, 200, 20):
    print("%5d | %5d | %5.4f" % (x, x**2, x/x**2))
    
# str.format() 함수 (파이썬3)
for x in range(1, 200, 20):
    print("{:5d} | {:5d} | {:5.4f}".format(x, x**2, x/x**2))
    
# f-string (파이썬3.6+)
for x in range(1, 200, 20):
    print(f"{x:5d} | {x**2 :5} | {x/x**2 :.4f}")
    
############## Python for Finance - PART 2 ##############
# 연산자, 문자열

### 문자열과 이스케이프 문자
s = 'Hello, Python!'
type(s)

# new line(\n) 문자
s = "Hello, \nPython!"
print(s)

print("인생은 짧아요.\n파이썬을 쓰세요.") # 줄바꿈 표시
print("인생은 짧아요.\\n파이썬을 쓰세요.") # 역슬래시 차체를 표시
print('인생은 짧아요.\'파이썬을\' 쓰세요.') # 따옴표 표시

### 문자열의 포함 여부 확인
# in: 문자열에 문자열 포함여부 테스트하는 연산자
'파이썬' in '인생은 짧아요, 파이썬 쓰세요'

### 문자열 자르기와 합치기 그리고 바꾸기
# 문자열 분리(split)
s = "화학,출판,전기제품,제약,은행"
s.split(',')

# 문자열 합치기 (join)
sectors = ['화학', '출판', '전기제품', '제약', '은행']
"|".join(sectors)

# 문자열 대체 (replace)
s = "Hello, Python!"
w = s.replace('Python', 'World')
w

s = '8,832,934원'
price = int(s.replace(',', '').replace('원', ''))

print(price)

############## Python for Finance - PART 3 ##############
# 자료구조, 튜플, 딕셔너리

lst = ['AAPL', 'GOOGL', 'MSFT'] # 리스트
tup = (7541, 6556, 5429, 4607, 4296) # 튜플
dic = {'005930': '삼성전자', '000660': 'SK하이닉스', '005380': '현대차',} # 딕셔너리

print( lst[0] )
print( tup[0] )
print( dic['005930'] )

### 리스트의 요소값 바꾸기
stocks = ['삼성전자', 'SK하이닉스', '현대차', '셀트리온', 'LG화학', 'NAVER']

stocks[5] = '네이버'
print(stocks)

### 리스트의 요소값 위치 얻기
stocks = ['삼성전자', 'SK하이닉스', '현대차', '셀트리온', 'LG화학', 'NAVER']

stocks.index('현대차')

### 리스트에 특정 요소값이 있는지 확인하기
stocks = ['삼성전자', 'SK하이닉스', '현대차', '셀트리온', 'LG화학', 'NAVER']

'셀트리온' in stocks

'신한지주' not in stocks

### 리스트에 요소 추가
stocks = ['삼성전자', 'SK하이닉스', '현대차', '셀트리온', 'LG화학', 'NAVER']
stocks.append('한국전력')
stocks.append('현대모비스')

print(stocks)

### 리스트에 요소 삽입, 삭제
관심종목 =  ['삼성전자', '현대차', 'NAVER', '한국전력', '현대모비스']
print(관심종목)

# 요소 삽입(insert)
관심종목.insert(1, 'SK하이닉스')
print(관심종목)

# 요소 삭제(remove)
관심종목.remove('현대모비스')
print(관심종목)

관심종목 =  ['삼성전자', '현대차', 'NAVER', '한국전력', '현대모비스']
del 관심종목[2] # 2번요소 'NAVER' 삭제

print(관심종목)

### 리스트 소트하기
# 리스트를 소트(sort) 할 수 있습니다. reverse 인자를 True로 주면 역순으로 소트합니다.
관심종목 =  ['삼성전자', '현대차', 'NAVER', '한국전력', '현대모비스']
print(관심종목)

관심종목.sort(reverse=True)
print(관심종목)

# 리스트의 메소드인 sort()는 리스트 자체를 소트하고 반환값이 없는데 비해 내장함수 sorted()는 리스트를 소트하여 그 결과를 리스트로 반환합니다.
sorted(관심종목) # 올림차순
['NAVER', '삼성전자', '한국전력', '현대모비스', '현대차']
sorted(관심종목, reverse=True) # 내림차순

### 리스트 덧셈 (합치기)
# 리스트는 덧셈(+) 연산으로 간단하게 병합(merge)할 수 있습니다.

관심종목01 = ['삼성전자', 'SK하이닉스']
관심종목02 = ['현대차', 'NAVER', '한국전력']
관심종목 = 관심종목01 + 관심종목02
print(관심종목)

관심종목 += ['아모레퍼시픽']
print(관심종목)

### 리스트와 enumerate 활용
# 리스트를 enumerate 객체로 만들면 위치값과 함께 사용할 수 있습니다.

관심종목 =  ['삼성전자', '현대차', 'NAVER', '한국전력', '현대모비스']
for 종목 in 관심종목:
    print(종목)

관심종목 =  ['삼성전자', '현대차', 'NAVER', '한국전력', '현대모비스']
for i, item in enumerate(관심종목):
    print (i, item)
    
### 딕셔너리, 요소추가
#딕셔너리에 요소를 추가할 때는 바로 키를 인덱스로 지정하고 값을 할당합니다.
stocks = {'005930':'삼성전자', '000660':'SK하이닉스', '005380':'현대차'}

# 요소추가
stocks['035420'] = 'NAVER' 
stocks

stocks = {'005930':'삼성전자', '000660':'SK하이닉스', '005380':'현대차'}

# 삭제
del stocks['005380']

stocks

stock = {
    'name': '삼성전자',
    'market': '코스피',
    'close': [
        ('2019-01-02', 38750),
        ('2019-01-03', 37600),
        ('2019-01-04', 37450),
        ('2019-01-07', 38750),
    ],
    'market-cap': 2766994,
    'PER': 8.55,
}

print (stock['name'])
print (stock['PER'])
print ("시가총액 %d (억원)" % stock['market-cap'])

stock.keys()

for p in stock['close']:
    print( "날짜: %s, 종가: %d" % p)
    
### 딕셔너리와 반복문
corps = {}
corps['AAPL'] = 'AAPL Apple Inc.'
corps['GOOG'] = 'Alphabet Inc.'
corps['GOOGL'] = 'Alphabet Inc.'
corps['MSFT'] = 'Microsoft Corporation'
corps['AMZN'] = 'Amazon.com, Inc.'

corps

for key in corps:
    print(key, ':', corps[key])
    
corps.items()

for key, val in corps.items():
    print (key, ':', val)
    
### zip
# zip은 주어진 두개 리스트의 각 항목을 짝지워 줍니다.
z = zip(['a', 'b', 'c'], 
        ['d', 'e', 'f'])

list(z)

stock_list = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'FB']
marcap_list = [7541, 6556, 5429, 4607, 4296]

list(zip(stock_list, marcap_list))

for stock, marcap in zip(stock_list, marcap_list):
    print(stock, marcap)
    
############## Python for Finance - PART 4 ##############
# 흐름 제어 - 조건문과 반복문

# 가끔 아무내용도 없는 빈 블럭이 필요한 경우가 있습니다. 예를 들어, if 블럭에서 할 작업이 없는 경우 입니다.
if 0 > 1:
    pass
else:
    print("문장02")    
    
### for 와 enumerate
# enumerate 객체는 (순서값, 데이터)를 짝을 반환합니다.
for ix, x in enumerate(range(-3,3)):
    print(ix, x)
    
### break, continue
# break: 반복문을 종료
# continue: 반복문의 시작으로 이동
for ch in "인생은 짧아요":
    if ch == "은":
        break
    print(ch)

print(">> 반복문 종료")

for ch in "인생은 짧아요":
    if ch == "은":
        continue
    print(ch)

print(">> 반복문 종료")

### 리스트 컴프리헨션
# 리스트 컴프리헨션 (list comprehension)을 사용하면 축약된 형식으로 리스트를 생성할 수 있습니다.

# [표현식 for 항목 in 순회_가능_객체]

# 새로운 리스트를 생성하는데 주로 사용합니다.

#  0~9까지 각 수의 제곱
[x ** 2 for x in range(0, 10)]

even_squares = [x ** 2 for x in range(1, 10) if x % 2 == 0]
even_squares

code_list = ['005930', '000660', '005380']
name_list = ['삼성전자', 'SK하이닉스', '현대차']

dic = {code:name for code, name in zip(code_list,name_list)}
dic

############## Python for Finance - PART 5 ##############
# 함수와 모듈

### all(), any() - 다수의 True/False 조건을 테스트하는 함수
#여러개의 조건을 한번에 테스트 할 때 사용한다

# all(이터러블) - 이터러블의 모든 값이 참이면 True를 반환
# any(이터러블) - 이터러블중에 하나라고 참이면 True를 반환
print( all([True, True, True]) )
print( any([True, False, False]) )

############## Python for Finance - PART 6 ##############
# 에러와 예외 처리

############## Python for Finance - PART 9 ##############
# 파이썬 문법 총정리

### 파이썬의 주요 키워드
'''
True, False: 참/거짓을 표현. 비교연산(==, < 등)의 결과는 항상 True 혹은 False.
None: 값 없음을 표현
del: 객체를 삭제
and, or, not: 논리연산 AND, OR, NOT
def: 함수를 정의
return: 함수를 종료하고 값을 반환
if, else, elif: 조건에 참/거짓 여부에 따라 블럭을 실행
for, while: 보통 순회가능와 함께 반복(for), 조건이 True인 동안 실행(while)
break, continue: 반복문을 종료(break)하거나, 처음으로 이동(continue)
from, import: 모듈을 읽어 메모리에 적재
try, except: 예외 처리
with, as: 오픈된 파일의 범위(컨텍스트)
in: 리스트에 요소값이 존재하는지 테스트
is: 같은 객체인지 테스트
pass: 빈 블럭(실행할 것이 없음)
'''

### 나머지 키워드
'''
assert: 조건이 False면 에러(AssertError)를 발생(주로 테스트에 사용)
finally: 예외처리의 try..except에서 처리하지 않는 내용을 처리하는 블럭
raise: 예외를 발생시킴
global, nonlocal: 전역변수로 지정, 로컬변수가 아님을 지정
lambda: 이름이 없는(익명) 함수로 실행 코드만 사용 (주로 map() 같은 함수와 함께 사용)
class: 새로운 클래스를 정의
yield: 값을 호출한 곳에 반환(제너레이터를 반환)하고 다시 돌아옴
'''


### 한 번에 정리하는 파이썬 프로그래밍

## print()함수로 여러가지 출력
print('Hello, Python')  # Hello, Python

"""
따옴표 세개로 묶어 긴 설명 문서를 작성할 수 있습니다
"""

# 변수와 값, 변수(variable)에 값(value)을 할당(assign)
원금 = 1250
수익률 = 0.25
잔고 = 원금 + 원금 * 수익률

# 데이터 타입
print(type(원금))    # <class 'int'>
print(type(수익률))  # <class 'float'>

# 출력을 위해 문자열 포맷팅
출력_문자열 = '원금={}, 수익률={:.2f}, 잔고 = {:.2f}'.format(원금, 수익률, 잔고)
print(출력_문자열) # 원금=1250, 수익률=0.25, 잔고 = 1562.50

# 문자열 슬라이싱
s = 'Hello, Python'
print(s)       # Hello, Python
print(s[4])    # o
print(s[:])    # Hello, Python
print(s[0:5])  # Hello
print(s[7:9])  # Py
print(s[-1])   # n

# 문자열 split(), join()
print("화학,출판,제약".split(','))        # ['화학', '출판', '제약']
print("|".join(['화학', '출판', '제약'])) # 화학|출판|제약

# 리스트, append(), remove(), sort(), 병합
관심종목 = ['삼성전자', '한국전력']
관심종목.append('현대차')
관심종목.append('셀트리온')
print(관심종목)                 # ['삼성전자', '한국전력', '현대차', '셀트리온']
print(관심종목.index('현대차')) # 1
print('셀트리온' in 관심종목)   # True

관심종목.remove('한국전력')
print(관심종목)                 # ['삼성전자', '현대차', '셀트리온']
          
관심종목.sort()
print(관심종목)                 # ['삼성전자', '셀트리온', '현대차']

관심종목 += ['아모레퍼시픽']    # ['삼성전자', '셀트리온', '현대차', '아모레퍼시픽']
print(관심종목)

# 리스트 컴프리헨션
lst = [x ** 2 for x in range(0, 10)]
print(lst) # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# for와 enumerate 함께 사용하기
for i, item in enumerate(관심종목):
    print (i, item) # 0 삼성전자, 1 셀트리온, 2 현대차, 3 아모레퍼시픽
    
# 딕셔너리: 키(key)와 값(value)
stocks = {'005930':'삼성전자', '000660':'SK하이닉스', '005380':'현대차'}

# 딕셔너리에 요소추가
stocks['035420'] = 'NAVER' 

# zip + 딕셔너리 컴프리핸션: 두 리스트 데이터의 순서쌍 만들기 
code_list = ['005930', '000660', '005380']
name_list = ['삼성전자', 'SK하이닉스', '현대차']

dic = {code:name for code, name in zip(code_list,name_list)}
print(dic) # {'005930': '삼성전자', '000660': 'SK하이닉스', '005380': '현대차'}

# if 문 사용하기
ret = 0.1
if ret > 0.0:
    print("상승")
else:
    print("하락")
    
# for와  range와 함께 사용하기
for i in range(1, 10): # 1~9
    print(i, end=', ')

# for와 리스트 함께 사용하기
tech_stocks = ['AAPL', 'GOOGL', 'AMZN', 'FB', 'SBUX']
for i in tech_stocks:
    print(i, ' 처리하기' )
    
# 필요한 모듈을 임포트하는 방법들
import requests 
import numpy as np
from datetime import datetime 

# 함수 만들기: def 함수이름(매개변수1, 매개변수2) 
def stat_func(a, b):
    return a + b, (a + b) / 2  # 다수 리턴값

print(stat_func(10, 20))  # (30, 15.0)

# 내장함수들: len(), sum(), min(), max(), 
data = [10, 20, 30, 50, 110, 2, 90, 30, 8]
print(len(data), sum(data), min(data), max(data))
print('데이터 평균값:', sum(data) / len(data))

# 예외 처리: try...except
x = [10, 20, 30]

try:
    print(x[4])      # IndexError
    print(x[1] / 0)  # ZeroDivisionError
except IndexError as e:
    # IndexError 예외 처리하기
    print(e)
except Exception as e:
    # 그밖에 모든 예외 처리하기
    print(e)

# 날짜와 시간
from datetime import datetime, timedelta
dt = datetime(2019, 3, 10, 11, 30)

# strftime(): 시간을 문자열로
print(dt.strftime("%Y-%m-%d %H:%M:%S")) # 2019-03-10 11:30:00

# strptime(): 문자열을 시간으로
s = "2019년 03월 10일 11시 30분"
print(datetime.strptime(s, "%Y년 %m월 %d일 %H시 %M분")) # 2019-03-10 11:30:00

# 오늘 부터 100일 후
print(datetime.now() - timedelta(days=100)) # 2018-10-29 23:42:12.772049

# collections.Counter
from collections import Counter
stocks = ['삼성전자', '셀트리온', '셀트리온']

cntr = Counter(stocks)
print(cntr.most_common())  # [('셀트리온', 2), ('삼성전자', 1)]

# 파일 입출력, 쓰기
data = "6, 12, 3, 2, 13"
with open("output.txt", "w") as f:
    f.write(data)
    
# 파일 입출력, 읽기
with open("output.txt", "r") as f:
    contents = f.read()

print(contents)  # 6, 12, 3, 2, 13

print('Happy Python Coding!')