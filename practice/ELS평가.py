'''
S(t+dt) = S(t)exp[((r-(σ^2)/2)dt + σ√(dt)Z)], Z ~ N(0,1)
'''

# stock_process.py (무위험 이자율이 1.65%, 변동성은 7.9%, 현재시점 주가 100인 경우 주가 경로 시뮬레이션)
import numpy as np
import matplotlib.pyplot as plt
N = 180; S = np.zeros([N, 1]); S[0] = 100
vol = 0.079; r = 0.0165; T = 1; dt = T/N
t = np.linspace(0, T, N)
z = np.random.normal(0, 1, N)
for i in range(N-1):
    S[i+1, 0] = S[i, 0 ]*np.exp((r-0.5*vol**2)*dt+vol*z[i]*np.sqrt(dt))
plt.plot(t,S[:,0], 'ko-')
plt.xlabel('Time')
plt.ylabel('Stock Process')
plt.show()

# stock_process50.py (50개의 주가 경로 시뮬레이션)
N = 365; S = np.zeros([N, 1]); S[0] = 100
vol = 0.079; r = 0.021; T = 1; dt = T/N
t = np.linspace(0, T, N)
z = np.random.normal(0, 1, N)
plt.xlabel('Time')
plt.ylabel('Stock Process')
for k in range(0, 50):
    z = np.random.normal(0, 1, N)
    for i in range(N-1):
        S[i+1, 0] = S[i, 0 ]*np.exp((r-0.5*vol**2)*dt+vol*z[i]*np.sqrt(dt))
    plt.plot(t[:],S[:], 'k-', linewidth=0.3)
plt.show()

# 기초자산이 1개인 주가연계증권(ELS)에 대한 몬테카를로 시뮬레이션
# 미래에셋대우(ELS)22903(조기상환형)
# https://www.miraeassetdaewoo.com/ > 금융상품 > ELS/DLS/ETN > 검색 > 청약완료 상품 > 상품명:미래에셋대우(ELS)22903(조기상환형)
'''
조기 행사 만기 / 조기 행사가 / 쿠폰 이자율 / 상환금액
2018년 09월 19일 / 최초기준가격의 95% / 2.2% / 액면금액 * 102.2%
2019년 03월 20일 / 최초기준가격의 95% / 4.4% / 액면금액 * 104.4%
2019년 09월 19일 / 최초기준가격의 95% / 6.6% / 액면금액 * 106.6%
2020년 03월 19일 / 최초기준가격의 90% / 8.8% / 액면금액 * 108.8%
2020년 09월 21일 / 최초기준가격의 90% / 11% / 액면금액 * 111.0%
2021년 03월 19일 / 최초기준가격의 85% / 13.2% / 액면금액 * 113.2% 
(dummy = 13.2%, Knock-in Barrier = 65%)
'''
# ELS_MC_1D.py
import numpy as np
from datetime import date

n = 10000; r = 0.0165; vol = 0.1778 # 시뮬레이션 횟수, 이자율, 변동성
n0 = date.toordinal(date(2018,3,23)) # 최조기준가격평가일
n1 = date.toordinal(date(2018,9,19)) # 1차조기상환일
n2 = date.toordinal(date(2019,3,20)) # 2차조기상환일
n3 = date.toordinal(date(2019,9,19)) # 3차조기상환일
n4 = date.toordinal(date(2020,3,19)) # 4차조기상환일
n5 = date.toordinal(date(2020,9,21)) # 5차조기상환일
n6 = date.toordinal(date(2021,3,19)) # 만기상환일
check_day = np.array([n1-n0, n2-n0, n3-n0, n4-n0, n5-n0, n6-n0]) # 조기상환일 벡터
oneyear = 365; tot_date = n6 - n0 # 1년의 일수, 만기상환일
dt = 1/oneyear # 시간 격자 간격
S = np.zeros([tot_date + 1, 1]) # 주가 벡터 생성
S[0] = 100.0 # 기초자산의 초깃값
strike_price = np.array([0.95, 0.95, 0.95, 0.90, 0.90, 0.85]) * S[0] # 조기행사가격 벡터
repay_n = len(strike_price) # 조기상환 횟수
coupon_rate = np.array([0.022, 0.044, 0.066, 0.088, 0.11, 0.132]) # 조기행사시 받게 되는 이자율 벡터
payment = np.zeros([repay_n, 1]) # 조기상환시 페이오프 벡터
facevalue = 10 ** 4 # 액면금액
tot_payoff = np.zeros([repay_n, 1]) # 전체 페이오프 벡터
payoff = np.zeros([repay_n, 1]) # 페이오프 벡터
discount_payoff = np.zeros([repay_n, 1]) # 현가 할인 된 페이오프 벡터
kib = 0.65 * S[0]; dummy = 0.132 # 낙인 배리어, 더미 이자율
# 조기상환 했을 때의 페이오프 벡터 생성
for j in range(repay_n):
    payment[j] = facevalue * (1 + coupon_rate[j])
# 몬테카를로 시뮬레이션을 이용한 ELS 가격 결정
for i in range(n):
    # 만기상환일 만큼의 랜덤넘버 생성
    z = np.random.normal(0, 1, size=[tot_date, 1])
    # 임의의 주가 경로 생성
    for j in range(tot_date):
        S[j+1] = S[j] * np.exp((r - 0.5 * vol ** 2) * dt + vol * np.sqrt(dt) * z[j])
    # 조기상환일 체크하여 페이오프 결정
    S_checkday = S[check_day]
    payoff = np.zeros([repay_n, 1])
    repay_event = 0 # 조기상환이 되지 않은 상태를 의미. for 문을 빠져나감
    for j in range(repay_n):
        if S_checkday[j] >= strike_price[j]:
            payoff[j] = payment[j]
            # 조기상환이 된 상태를 의미
            repay_event = 1
            break
    # 조기상환이 되지 않고 만기까지 온 경우
    if repay_event == 0:
        # 낙인 배리어 라래로 내려간 적이 없는 경우
        if min(S) > kib:
            payoff[-1] = facevalue * (1 + dummy)
        # 낙인 배리어 아래로 내려간 적이 있는 경우
        else:
            payoff[-1] = facevalue * (S[-1] / S[0])
    # 시뮬레이션마다 페이오프를 더함
    tot_payoff = tot_payoff + payoff
# 모든 시뮬레이션의 페이오프 평균을 구함
mean_payoff = tot_payoff / n
# 페이오프를 무위험 이자율로 할인하여 현재 가격을 구함
for j in range(repay_n):
    discount_payoff[j] = mean_payoff[j] * np.exp(-r * check_day[j] / oneyear)
# ELS 가격을 구함
price = np.sum(discount_payoff)
print(price)

# 기초자산이 2개인 주가연계증권
# 미래에셋대우(ELS)26043

'''
조기 행사 만기 / 조기 행사가 / 쿠폰 이자율 / 상환금액
2018년 12월 21일 / 최초기준가격의 90% / 2.5% / 액면금액 * 102.5%
2019년 06월 25일 / 최초기준가격의 90% / 5.0% / 액면금액 * 105.0%
2019년 12월 23일 / 최초기준가격의 85% / 7.5% / 액면금액 * 107.5%
2020년 06월 24일 / 최초기준가격의 85% / 10.0% / 액면금액 * 110.0%
2020년 12월 23일 / 최초기준가격의 80% / 12.5% / 액면금액 * 112.5%
2021년 06월 24일 / 최초기준가격의 75% / 15.0% / 액면금액 * 115.0% 
(dummy = 15.0%, Knock-in Barrier = 50%)
'''
# stock_process_with_correlation2.py (2개의 기초자산이 상관관계를 갖는 주가 경로 생성)
import numpy as np
import matplotlib.pyplot as plt
x_vol = 0.079; y_vol = 0.105
r = 0.021
N = 100; T = 1
S1 = np.zeros((N+1, 1))
S2 = np.zeros((N+1, 1))
S1[0] = 100; S2[0] = 100
dt = T/N; t = np.linspace(0, T, N+1)
rho = 0.3
correlation = np.array([[1, rho], [rho, 1]])
cholesky = np.linalg.cholesky(correlation)
z0 = np.random.normal(0, 1, size=[N, 2])
np.random.seed(56)
z0 = np.transpose(z0)
z = np.matmul(cholesky, z0)
Worst_performer = np.zeros((N+1, 1))
for i in range(N):
    S1[i+1] = S1[i] * np.exp((r - 0.5 * x_vol ** 2) * dt + x_vol * z[0, i] * np.sqrt(dt))
    S2[i+1] = S2[i] * np.exp((r - 0.5 * y_vol ** 2) * dt + y_vol * z[1, i] * np.sqrt(dt))
    Worst_performer[i] = min(S1[i, 0], S2[i, 0])
    Worst_performer[-1] = min(S1[-1, 0], S2[-1, 0])
plt.plot(t, S1[:], 'k-', label='asset1', linewidth=1, markersize=3.5)
plt.plot(t, S2[:], 'k--', label='asset2', linewidth=1, markersize=3.5)
plt.plot(t, Worst_performer[:], 'k+-', label='min(S1, S2)', linewidth=1, markersize=3.5)
plt.legend()
plt.xlim(0, 1.0)
plt.ylim(70, 130)
plt.xlabel('Time')
plt.ylabel('Stock Process')
plt.legend(prop={'size':12})
plt.show()

# 기초자산이 2개인 주가연계증권 가격 구하기
# ELS_MC_2D.py
import numpy as np
from datetime import date
n = 10000; r = 0.0165 # 시뮬레이션 횟수, 이자율
x_vol = 0.249; y_vol = 0.2182 # 기초자산 1의 변동성, 기초자산 2의 변동성
n0 = date.toordinal(date(2018,6,29)) # 최조기준가격평가일
n1 = date.toordinal(date(2018,12,21)) # 1차조기상환일
n2 = date.toordinal(date(2019,6,25)) # 2차조기상환일
n3 = date.toordinal(date(2019,12,23)) # 3차조기상환일
n4 = date.toordinal(date(2020,6,24)) # 4차조기상환일
n5 = date.toordinal(date(2020,12,23)) # 5차조기상환일
n6 = date.toordinal(date(2021,6,24)) # 만기상환일
check_day = np.array([n1-n0, n2-n0, n3-n0, n4-n0, n5-n0, n6-n0]) # 조기상환일 벡터
rho = 0.0981; corr = np.array([[1, rho], [rho, 1]]) # 두 기초자산의 상관계수, 상관계수 행렬
coupon_rate = ([0.025, 0.05, 0.075, 0.10, 0.125, 0.15]) # 조기행사시 받는 쿠폰 이자율
oneyear = 365; tot_date = n6-n0; dt = 1/oneyear # 1년의 일수, 만기, 시간 격자 간격
k = np.linalg.cholesky(corr) # 촐레스키 분해
S1 = np.zeros((tot_date + 1, 1))
S2 = np.zeros((tot_date + 1, 1))
S1[0] = 100; S2[0] = 100 # 기초자산 1, 2의 초깃값
ratio_S1 = S1[0]; ratio_S2 = S2[0] # [만기평가가격/최조기준가격]의 비율 결정하기 위해 기초자산의 초깃값 저장
strike_price = ([0.90, 0.90, 0.85, 0.85, 0.80, 0.75]) # 조기행사가격 벡터
repay_n = len(strike_price) # 조기상환 횟수
payment = np.zeros([repay_n, 1]) # 조기상환시 페이오프 벡터
payoff = np.zeros([repay_n, 1]) # 페이오프 벡터
tot_payoff = np.zeros([repay_n, 1]) # 전체 페이오프 벡터
discount_payoff = np.zeros([repay_n, 1]) # 현가 할인 된 페이오프 벡터
face_value = 10 ** 4; dummy = 0.15; kib = 0.50 # 액면금액, 더미 이자율, 낙인 배리어
# 조기상환 했을 때의 페이오프 벡터 생성
for j in range(repay_n):
    payment[j] = face_value * (1 + coupon_rate[j])
# 몬테카를로 시뮬레이션을 이용한 ELS 가격 결정
for i in range(n):
    # 촐레스키 분해된 행렬 k를 이용하여
    # 상관관계가 있는 난수 생성
    w0 = np.random.normal(0, 1, size=[tot_date, 2])
    # 난수 행렬을 전치시켜 줌
    w0 = np.transpose(w0)
    w = np.matmul(k, w0)
    for j in range(tot_date):
        S1[j + 1] = S1[j] * np.exp((r - 0.5 * x_vol ** 2) * dt + x_vol * np.sqrt(dt) * w[0, j])
        S2[j + 1] = S2[j] * np.exp((r - 0.5 * y_vol ** 2) * dt + y_vol * np.sqrt(dt) * w[1, j])
    # [만기평가가격/최초기준가격]의 비율이 더 낮은 가격을
    # 갖는 기초자산 결정
    R1 = S1 / ratio_S1; R2 = S2 / ratio_S2
    WP = np.minimum(R1, R2)
    # 조기상환일 체크하여 페이오프 결정
    WP_checkday = WP[check_day]
    payoff = np.zeros([repay_n, 1])
    # 조기상환이 되지 않은 상태를 의미함
    repay_event = 0
    for j in range(repay_n):
        if WP_checkday[j] >= strike_price[j]:
            payoff[j] = payment[j]
            # 조기상환이 된 상태를 의미함. for 문을 빠져나감
            repay_event = 1
            break
    # 조기상환 되지 않고 만기까지 온 경우
    if repay_event == 0:
        if min(WP) > kib:
            # 낙인 배리어 아래로 내려간 적이 없는 경우
            payoff[-1] = face_value * (1 + dummy)
        else:
            # 낙인 배리어 아래로 내려간 적이 있는 경우
            payoff[-1] = face_value * WP[-1]
    # 시뮬레이션마다 페이오프를 더함
    tot_payoff = tot_payoff + payoff
# 모든 시뮬레이션의 페이오프 평균을 구함
mean_payoff = tot_payoff / n
# 페이오프를 무위험 이자율로 할인하여 현재 가격을 구함
for j in range(repay_n):
    discount_payoff[j] = mean_payoff[j] * np.exp(-r * check_day[j] / oneyear)
# ELS 가격을 구함
price = np.sum(discount_payoff)
print(price)

# 기초자산이 3개인 주가연계증권
# 미래에셋대우(ELS)22345(조기상환형)

'''
조기 행사 만기 / 조기 행사가 / 쿠폰 이자율 / 상환금액
2018년 06월 08일 / 최초기준가격의 95% / 4.8% / 액면금액 * 104.8%
2018년 12월 11일 / 최초기준가격의 95% / 9.6% / 액면금액 * 109.6%
2019년 06월 11일 / 최초기준가격의 90% / 14.4% / 액면금액 * 114.4%
2019년 12월 10일 / 최초기준가격의 90% / 19.2% / 액면금액 * 119.2%
2020년 06월 09일 / 최초기준가격의 85% / 24.0% / 액면금액 * 124.0%
2020년 12월 09일 / 최초기준가격의 85% / 28.8% / 액면금액 * 128.8% 
(dummy = 6.0%, Knock-in Barrier = 50%)
'''
# stock_process_with_correlation3.py
import numpy as np
import matplotlib.pyplot as plt
x_vol = 0.2662; y_vol = 0.2105; z_vol = 0.2111; r = 0.0165
N = 80; T = 1; dt = T/N
S1 = np.zeros((N+1, 1))
S2 = np.zeros((N+1, 1))
S3 = np.zeros((N+1, 1))
S1[0] = 100; S2[0] = 100; S3[0] = 100
t = np.linspace(0, T, N+1)
rho_xy = 0.279; rho_xz = 0.2895; rho_yz = 0.5256
correlation = np.array([[1, rho_xy, rho_xz], [rho_xy, 1, rho_yz], [rho_xz, rho_yz, 1]])
cholesky = np.linalg.cholesky(correlation)
z0 = np.random.normal(0, 1, size=[N, 3])
np.random.seed(42)
z0 = np.transpose(z0)
z = np.matmul(cholesky, z0)
Worst_performer = np.zeros((N+1, 1))
plt.xlim(0, 1)
plt.ylim(60, 170)
for i in range(N):
    S1[i+1] = S1[i] * np.exp((r - 0.5 * x_vol ** 2) * dt + x_vol * z[0, i] * np.sqrt(dt))
    S2[i+1] = S2[i] * np.exp((r - 0.5 * y_vol ** 2) * dt + y_vol * z[1, i] * np.sqrt(dt))
    S3[i+1] = S3[i] * np.exp((r - 0.5 * z_vol ** 2) * dt + z_vol * z[2, i] * np.sqrt(dt))
    Worst_performer[i] = min(S1[i, 0], S2[i, 0], S3[i, 0])
    Worst_performer[-1] = min(S1[-1, 0], S2[-1, 0], S3[-1, 0])
plt.plot(t, S1[:], 'k', label='asset1', linewidth=1, markersize=4)
plt.plot(t, S2[:], 'k--', label='asset2', linewidth=1, markersize=4)
plt.plot(t, S3[:], 'k+-', label='asset3', linewidth=1, markersize=4)
plt.plot(t, Worst_performer[:], 'k*-', label='min(S1, S2, S3)', linewidth=1, markersize=4)
plt.xlabel('Time')
plt.ylabel('Stock Process')
plt.legend(prop={'size':12})
plt.show()

# 기초자산이 3개인 주가연계증권 가격 구하기
# ELS_MC_3D.py
import numpy as np
from datetime import date
n = 10000; r = 0.0165 # 시뮬레이션 횟수, 이자율
x_vol = 0.2662; y_vol = 0.2105; z_vol = 0.2111 # 기초자산 1, 2, 3의 변동성
n0 = date.toordinal(date(2017,12,14)) # 최조기준가격평가일
n1 = date.toordinal(date(2018,6,8)) # 1차조기상환일
n2 = date.toordinal(date(2018,12,11)) # 2차조기상환일
n3 = date.toordinal(date(2019,6,11)) # 3차조기상환일
n4 = date.toordinal(date(2019,12,10)) # 4차조기상환일
n5 = date.toordinal(date(2020,6,9)) # 5차조기상환일
n6 = date.toordinal(date(2020,12,9)) # 만기상환일
check_day = np.array([n1-n0, n2-n0, n3-n0, n4-n0, n5-n0, n6-n0]) # 조기상환일 벡터
rho_xy = 0.279; rho_xz = 0.2895; rho_yz = 0.5256 # 기초자산들의 상관계수
corr = np.array([[1, rho_xy, rho_xz], [rho_xy, 1, rho_yz], [rho_xz, rho_yz, 1]]) # 상관계수 행렬
k = np.linalg.cholesky(corr) # 촐레스키 분해
oneyear = 365; tot_date = n6-n0; dt = 1/oneyear # 1년의 일수, 만기, 시간 격자 간격
S1 = np.zeros((tot_date + 1, 1)) # 기초자산1의 벡터생성
S2 = np.zeros((tot_date + 1, 1)) # 기초자산2의 벡터생성
S3 = np.zeros((tot_date + 1, 1)) # 기초자산3의 벡터생성
S1[0] = 100; S2[0] = 100; S3[0] = 100 # 기초자산 1, 2, 3의 초깃값
ratio_S1 = S1[0]; ratio_S2 = S2[0]; ratio_S3 = S3[0] # [만기평가가격/최조기준가격]의 비율 결정하기 위해 기초자산의 초깃값 저장
strike_price = ([0.95, 0.95, 0.90, 0.90, 0.85, 0.85]) # 조기행사가격 벡터
repay_n = len(strike_price) # 조기상환 횟수
coupon_rate = ([0.048, 0.096, 0.144, 0.192, 0.24, 0.288]) # 조기행사시 받는 쿠폰 이자율
payment = np.zeros([repay_n, 1]) # 조기상환시 페이오프 벡터
payoff = np.zeros([repay_n, 1]) # 페이오프 벡터
tot_payoff = np.zeros([repay_n, 1]) # 전체 페이오프 벡터
discount_payoff = np.zeros([repay_n, 1]) # 현가 할인 된 페이오프 벡터
face_value = 10 ** 4; dummy = 0.06; kib = 0.50 # 액면금액, 더미 이자율, 낙인 배리어
# 조기상환 했을 때의 페이오프 벡터 생성
for j in range(repay_n):
    payment[j] = face_value * (1 + coupon_rate[j])
# 몬테카를로 시뮬레이션을 이용한 ELS 가격 결정
for i in range(n):
    # 촐레스키 분해된 행렬 k를 이용하여
    # 상관관계가 있는 난수 생성
    w0 = np.random.normal(0, 1, size=[tot_date, 3])
    # 난수 행렬을 전치시켜 줌
    w0 = np.transpose(w0)
    w = np.matmul(k, w0)
    payoff = np.zeros([repay_n, 1])
    repay_event = 0
    for j in range(tot_date):
        S1[j + 1] = S1[j] * np.exp((r - 0.5 * x_vol ** 2) * dt + x_vol * np.sqrt(dt) * w[0, j])
        S2[j + 1] = S2[j] * np.exp((r - 0.5 * y_vol ** 2) * dt + y_vol * np.sqrt(dt) * w[1, j])
        S3[j + 1] = S3[j] * np.exp((r - 0.5 * z_vol ** 2) * dt + z_vol * np.sqrt(dt) * w[2, j])
    # [만기평가가격/최초기준가격]의 비율이 더 낮은 가격을
    # 갖는 기초자산 결정
    R1 = S1 / ratio_S1; R2 = S2 / ratio_S2 ; R3 = S3 / ratio_S3
    WP = np.minimum(R1, R2, R3)
    # 조기상환일 체크하여 페이오프 결정
    WP_checkday = WP[check_day]
    for j in range(repay_n):
        if WP_checkday[j] >= strike_price[j]:
            payoff[j] = payment[j]
            # 조기상환이 된 상태를 의미함. for 문을 빠져나감
            repay_event = 1
            break
    # 조기상환 되지 않고 만기까지 온 경우
    if repay_event == 0:
        if min(WP) > kib:
            # 낙인 배리어 아래로 내려간 적이 없는 경우
            payoff[-1] = face_value * (1 + dummy)
        else:
            # 낙인 배리어 아래로 내려간 적이 있는 경우
            payoff[-1] = face_value * WP[-1]
    # 시뮬레이션마다 페이오프를 더함
    tot_payoff = tot_payoff + payoff
# 모든 시뮬레이션의 페이오프 평균을 구함
mean_payoff = tot_payoff / n
# 페이오프를 무위험 이자율로 할인하여 현재 가격을 구함
for j in range(repay_n):
    discount_payoff[j] = mean_payoff[j] * np.exp(-r * check_day[j] / oneyear)
# ELS 가격을 구함
price = np.sum(discount_payoff)
print(price)
