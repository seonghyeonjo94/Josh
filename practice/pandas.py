import pandas as pd
import numpy as np
import FinanceDataReader as fdr

# pd.Series(data, index, dtype)
s = pd.Series(np.arange(5), np.arange(100,105), dtype=np.int32)
s

s.index
s.values

s2 = pd.Series(np.arange(100,105), s.index)

s = pd.Series([1, 1, 2, 1, 2, 2, 2, 1, 1, 3, 3, 4, 5, 5, 7, np.NaN])
len(s)

s.size # 개수 반환
s.shape # 튜플형태로 shape 반환
s.unique() # 유일한 값만 ndarray로 반환
s.count() # NaN을 제외한 개수를 반환
s.mean() # NaN을 제외한 평균
s.value_counts() # NaN을 제외하고 각 값들의 빈도를 반환

s.head()
s.tail()

# Series 데이터 연산
s1 = pd.Series([1, 2, 3, 4], ['a', 'b', 'c', 'd'])
s2 = pd.Series([6, 3, 2, 1], ['d', 'c', 'b', 'a'])

s1 + s2 # index를 기준으로 연산
s1 ** 2
s1 ** s2

s1['k'] = 7
s2['e'] = 9
s1
s2

s1 + s2 # index의 pair가 맞지 않으면, 결과는 NaN

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
s[s.index > 5]

s[(s > 5) & (s < 8)]

(s >= 7).sum()
(s[s >= 7]).sum()

# Series 데이터 변경 & 슬라이싱
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

s[['a', 'b']] = [300, 900]
s

# DataFrame
# pd.read_excel(io, sheet_name, index_col)
df = pd.read_excel(r'C:\Users\a\Downloads\data.xls')
df = pd.read_excel(r'C:\Users\a\Downloads\data.xls', index_col='종목코드')
df.to_excel(r'C:\Users\a\Downloads\data.xls')
df.head()
df.tail()

df.shape # 속성 (row, column)

df.describe() # 숫자형 데이터의 통계치 계산

df.info() # 데이터 타입, 각 아이템의 개수 등 출력

df.index

df.columns

# DataFrame 데이터 생성
# pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)
# dictionary로 부터 생성
# dict의 key -> column
data = {'a' : [1, 2, 3], 'b' : [4, 5, 6], 'c' : [10, 11, 12]}

pd.DataFrame(data, index=[0, 1, 2])

# Series로 부터 생성
# 각 Series의 인덱스 -> column
a = pd.Series([100, 200, 300], ['a', 'b', 'd'])
b = pd.Series([101, 201, 301], ['a', 'b', 'k'])
c = pd.Series([110, 210, 310], ['a', 'b', 'c'])

pd.DataFrame([a, b, c], index=[100, 101, 102])

### DataFrame 원하는 column(컬럼)만 선택하기
# column 선택하기
'''
기본적으로 []는 column을 추출
컬럼 인덱스일 경우 인덱스의 리스트 사용 가능
리스트를 전달할 경우 결과는 Dataframe
하나의 컬럼명을 전달할 경우 결과는 Series
'''
df = pd.read_excel(r'C:\Users\a\Downloads\data.xls')
df['종목코드']
df[['종목코드', '종목명']]
df[:10]

df.loc[[0, 1, 3, 5], ['종목코드', '종목명']]
df.iloc[[0, 1, 3, 5], [0, 3, 5]]

df.index = df['종목명']
df = df['시가총액']

for i in range(len(df)):
    df.iloc[i] = df.iloc[i].replace(',', '')
df = pd.to_numeric(df)
large_cap = df[df.rank(ascending=False) < 5]

large_cap

# 새 column 추가
df = pd.read_excel(r'C:\Users\a\Downloads\data.xls')
df = df[['종목코드', '종목명', '시가총액']]
for i in range(len(df)):
    df['시가총액'][i] = df['시가총액'][i].replace(',', '')
df['시가총액'] = pd.to_numeric(df['시가총액'])
df['시가총액 * 2'] = df['시가총액'] * 2
df.head(3)

df.insert(0, '시가총액/1000', df['시가총액'] / 1000)
df.head(3)

df.drop('시가총액 * 2', axis=1).head()

df.drop(['시가총액/1000', '시가총액 * 2'], axis=1, inplace=True)
df.head()

x = np.arange(10)
y = [0, 1, 3, 0, -2, 10, 9, -3, 9, 3]
mat = pd.DataFrame([x, y])
mat = mat.T

mat.corr() # 컬럼간 상관관계 계산

price = pd.read_csv(r'C:\Users\a\Downloads\quant\Python\data\price.csv', index_col='Unnamed: 0')
symbols = ['005930', '068270', '005380', '055550', '017670']
price = price[symbols]
rets = (price / price.shift(1)) - 1

rets.head()

rets.isna().head() # boolean 타입으로 확인

# NaN 처리
rets.dropna().head() # 데이터에서 삭제

# rets.fillna(method=bfill,ffill)
rets.fillna(method='bfill').head()

# read_csv 함수 파라미터
'''
sep - 각 데이터 값을 구별하기 위한 구분자(separator) 설정
header - header를 부시할 경우, None 설정
index_col - index로 사용할 column 설정
usecols - 실제로 dataframe에 로딩할 columns만 설정
'''
train_data = pd.read_csv(r'C:\Users\a\Downloads\train.csv', sep='#')
train_data
train_data = pd.read_csv(r'C:\Users\a\Downloads\train.csv', sep=',')
train_data

train_data = pd.read_csv(r'C:\Users\a\Downloads\train.csv', header=None)
train_data

train_data = pd.read_csv(r'C:\Users\a\Downloads\train.csv', index_col='PassengerId')
train_data
train_data.columns

train_data = pd.read_csv(r'C:\Users\a\Downloads\train.csv', usecols=['Survived', 'Pclass', 'Name'])
train_data

# 30대이면서 1등석에 탄 사람 선택하기
train_data = pd.read_csv(r'C:\Users\a\Downloads\train.csv')
class_ = train_data['Pclass'] == 1
age_ = (train_data['Age'] >= 30) & (train_data['Age'] < 40)

train_data[class_ & age_]

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
train_data = pd.read_csv(r'C:\Users\a\Downloads\train.csv')
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
# Pclass 변수 변환
# astype 사용하여 간단한 타입만 변환
train_data.info()
train_data['Pclass'] = train_data['Pclass'].astype(str)
train_data.info()

# Age 변수 변환
# 변환 로직을 함수로 만든 후, apply 함수로 적용
import math
def age_categorize(age):
    if math.isnan(age):
        return -1
    return math.floor(age / 10) * 10
train_data['Age'].apply(age_categorize)

### 범주형 데이터 전처리
# One-hot encoding
'''
범주형 데이터는 분석단계에서 계산이 어렵기 때문에 숫자형으로 변경이 필요함
범주형 데이터의 각 범주(category)를 column레벨로 변경
해당 범주에 해당하면 1, 아니면 0으로 채우는 인코딩 기법
pandas.get_dummies 함수 사용
drop_first : 첫번째 카테고리 값은 사용하지 않음
'''
train_data = pd.read_csv(r'C:\Users\a\Downloads\train.csv')
train_data.head()
pd.get_dummies(train_data)
pd.get_dummies(train_data, columns=['Pclass', 'Sex', 'Embarked'])
pd.get_dummies(train_data, columns=['Pclass', 'Sex', 'Embarked'], drop_first=True)

# group by
'''
아래의 세 단계를 적용하여 데이터를 그룹화(groupping)(SQL의 group by와 개념적으로는 동일, 사용법은 유사)
데이터 분할
operation 적용
데이터 병합
'''
df = pd.read_csv(r'C:\Users\a\Downloads\train.csv')
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
class_group.max()

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
df = pd.read_csv(r'C:\Users\a\Downloads\train.csv')
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

### pivot, pivot_table 함수의 이해 및 활용
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

# stack, unstack 함수의 이해 및 활용
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

# 데이터 프레임 병합
# colum명이 같은 경우
df1 = pd.DataFrame({'key1' : np.arange(10), 'value1' : np.random.randn(10)})
df2 = pd.DataFrame({'key1' : np.arange(10), 'value1' : np.random.randn(10)})

pd.concat([df1, df2], ignore_index=True)
pd.concat([df1, df2], axis=1) # 기본 axis=0 -> 행단위 병합

# colum명이 다른 경우
df3 = pd.DataFrame({'key2' : np.arange(10), 'value2' : np.random.randn(10)})

pd.concat([df1, df3],sort=False)
pd.concat([df1, df3], axis=1)

# Merge 함수로 데이터 프레임 병합
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
