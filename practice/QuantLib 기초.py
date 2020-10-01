# pip install QuantLib-Python
import QuantLib as ql

##### QuantLib
### 1. Date
'''
ql.Date(Day, Month, Year)
ql.Date(serialNumber)
'''
# Construction
date1 = ql.Date(11, 4, 2020)
date2 = ql.Date(43932)

print(date1)
print(date2)

# 1) dayofMonth() / dayOfYear() / month() / year()
# 2) serialNumber()
# 3) weekday()
# Basic Functions
date = ql.Date(11, 4, 2020)

dayOfMonth = date.dayOfMonth()
dayOfYear = date.dayOfYear()
month = date.month()
year = date.year()
serialNumber = date.serialNumber()
weekday = date.weekday()

print('Day Of Month = {}'.format(dayOfMonth))
print('Day Of Year = {}'.format(dayOfYear))
print('Month = {}'.format(month))
print('Year = {}'.format(year))
print('Serial Number = {}'.format(serialNumber))
print('Weekday = {}'.format(weekday))

# 4) todaysDate()
# 5) isLeap() / isEndOfMonth()
# 6) endOfMonth()
# 7) nextWeekday()
# 8) nthWeekday()
# Advanced Functions
date = ql.Date(12, 4, 2020)

todaysDate = date.todaysDate()
isLeap = date.isLeap(date.year()) # 윤년 확인
isEndOfMonth = date.isEndOfMonth(date)
endOfMonth = date.endOfMonth(date)
nextWeekday = date.nextWeekday(date, 4) # 2020년 4월 12일을 기준으로 다음 수요일
nthWeekday = date.nthWeekday(3 ,5, 7, 2020) # 몇번째(Size), 무슨요일(Weekday), 월(Month), 연도(Year)

print("Today's Date = {}".format(todaysDate))
print('is Leap? = {}'.format(isLeap))
print('isEndofMonth? = {}'.format(isEndOfMonth))
print('End of Month = {}'.format(endOfMonth))
print('nextWeekday = {}'.format(nextWeekday))
print('Nth Weekday = {}'.format(nthWeekday))

### 2. Peariod
'''
ql.Period(Integer, TimeUnits)
ql.Period(Frequency)

# TimeUnits
ql.Days
ql.Weeks
ql.Months
ql.Years

# Frequency
ql.Annual
ql.Semiannual
ql.Quarterly
ql.Monthly
ql.Biweekly
ql.Weekly
ql.Daily
ql.Once
'''
# Construction
period1 = ql.Period(3, ql.Months)
period2 = ql.Period(ql.Semiannual)

# Functions
date1 = ql.Date(11 ,4, 2020)
date2 = ql.Date(31, 12, 2020)

three_weeks = ql.Period(3, ql.Weeks)
three_months = ql.Period(3, ql.Months)
three_years = ql.Period(3, ql.Years)

print('After 3 Weeks : {}'.format(date1 + three_weeks))
print('After 3 Months : {}'.format(date1 + three_months))
print('After 3 Years : {}'.format(date1 + three_years))

print('Days between Date2 and Date1 = {}'.format(date2-date1))

### 3. Calendar
# Construction
us = ql.UnitedStates()
eu = ql.TARGET()
kr = ql.SouthKorea()
jp = ql.Japan()
cn = ql.China()

# 1) holidayList()
date1 = ql.Date(1, 1, 2020)
date2 = ql.Date(31, 12, 2020)

kr_holidayList = kr.holidayList(date1, date2) # 시작일(Date), 종료일(Date)
print(kr_holidayList)

# 2) addHoliday() / removeHoliday()
kr.addHoliday(ql.Date(27, 1, 2020))
kr.addHoliday(ql.Date(15, 4, 2020))
kr.removeHoliday(ql.Date(6, 5, 2020))
print(kr_holidayList)

# 3) businessDaysBetween()
kr.businessDaysBetween(date1, date2)

# 4) isBusinessDay() / isHoliday()
kr.isBusinessDay(date1)
kr.isHoliday(date1)

# 5 advance()
'''
Business Day Convention
ql.Unadjusted
ql.Preceding
ql.ModifiedPreceding
ql.Following
ql.ModifiedFollowing
'''
kr.advance(date1, ql.Period(6, ql.Months), ql.ModifiedFollowing, True) # 영업일 관행(Business Day Convention), 월말 기준(End of Month)

#6 JointCalendar()
new_calendar = ql.JointCalendar(us, eu, kr)
print(new_calendar.holidayList(date1, date2))

### 4. DayCounter
# Construction
act360 = ql.Actual360() # Actual/360
act365 = ql.Actual365Fixed() # Actual/365
actact = ql.ActualActual() # Actual/Actual
thirty360 = ql.Thirty360() # 30/360
b252 = ql.Business252() # BusinessDay/252

# 1) dayCount()
date1 = ql.Date(12, 2, 2020)
date2 = ql.Date(14, 5, 2020)

# DayCount
print("Day Count by Actual/360 = {}".format(act360.dayCount(date1, date2)))
print("Day Count by Actual/365 = {}".format(act365.dayCount(date1, date2)))
print("Day Count by Actual/Actual = {}".format(actact.dayCount(date1, date2)))
print("Day Count by 30/360 = {}".format(thirty360.dayCount(date1, date2)))
print("Day Count by BusinessDay/252 = {}".format(b252.dayCount(date1, date2)))

# 2) yearFraction()
print("Year Fraction by Actual/360 = {}".format(round(act360.yearFraction(date1, date2), 4)))
print("Year Fraction by Actual/365 = {}".format(round(act365.yearFraction(date1, date2), 4)))
print("Year Fraction by Actual/Actual = {}".format(round(actact.yearFraction(date1, date2), 4)))
print("Year Fraction by 30/360 = {}".format(round(thirty360.yearFraction(date1, date2), 4)))
print("Year Fraction by BusinessDay/252 = {}".format(round(b252.yearFraction(date1, date2), 4)))

### 5. Schedule
'''
# 효력발생일, 만기일, 이자지급주기, 달력, 이자결제일의 영업일관행, 만기일의 영업일관행, 날짜생성방식, 월말기준
ql.Schedule(Date effectiveDate,
            Date terminationDate,
            Period tenor,
            Calendar calendar,
            BusinessDayConvention convention,
            BusinessDayConvention terminationDateConvention,
            DateGeneration rule,
            Bool endOfMonth)
'''
'''
DateGeneration
ql.backward # 만기일부터 효력발생일까지 후진(Backward)방식으로 이자지급 스케줄을 생성
ql.Forward # 효력발생일부터 만기일까지 전진(Forward)방식으로 이자지급 스케줄을 생성
ql.Zero # 효력발생일과 만기일 사이에 어떠한 결제일도 존재하지 않음
ql.ThirdWednesday # 효력발생일과 만기일을 제외한 모든 중간 이자지급일을 해당 월의 세번째 수요일로 지정
ql.Twentieth # 효력발생일을 제외한 모든 이자지급일을 해당 월의 20일로 지정
ql.TwentiethIMM # 효력발생일을 제외한 모든 이자지급일을 3,6,9,12월 20일로 지정
'''
# Components
effectiveDate = ql.Date(13, 4, 2020)
maturityDate = ql.Date(15, 4, 2023)
tenor = ql.Period(3, ql.Months)
calendar = ql.SouthKorea()
convention = ql.ModifiedFollowing
rule = ql.DateGeneration.Backward
endOfMonth = False

# Construction
schedule = ql.Schedule(effectiveDate,
                       maturityDate,
                       tenor,
                       calendar,
                       convention,
                       convention,
                       rule,
                       endOfMonth)

ref_date = ql.Date(4, 10, 2021)

# Functions
print("Next Payment Date From {} : {}".format(ref_date, schedule.nextDate(ref_date)))
print("Previous Payment Date From {} : {}".format(ref_date, schedule.previousDate(ref_date)))

### 6. Quote
# Construction
quote = ql.SimpleQuote(2767.88)

# Functions
print(quote.value())

quote.setValue(2800.6)
print(quote.value())

### 7. InterestRate
'''
# 금리(rate), 이자 일수 계산방식(dayCounter), 복리계산방식(Compounding), 이자지급주기(Frequency)
ql.InterestRate(Real rate,
                DayCounter daycounter,
                Compounding compounding,
                Frequency frequency)
'''
'''
Compounding
ql.Simple # 단리
ql.Compounded # 복리
ql.Continuous # 연속복리
ql.SimpleThenCompounded # 차기이표일까지는 단리, 이후부터는 복리
ql.CompoundedThenSimple # 차기이표일까지는 복리, 이후부터는 단리
'''
# Components
rate = 0.0148
dc = ql.ActualActual()
comp = ql.Compounded
freq = ql.Annual

# Construction
ir = ql.InterestRate(rate,
                     dc,
                     comp,
                     freq)

# Discount & Compound Factor
start_date = ql.Date(19, 4, 2020)
end_date = ql.Date(19, 4, 2021)

t = 1
print("Discount Factor between {} and {} = {}".format(start_date, end_date, round(ir.discountFactor(1), 4)))
print("Compounding Factor between {} and {} = {}".format(start_date, end_date, round(ir.compoundFactor(t), 4)))

# Equivalent Rate
new_dc = ql.ActualActual()
new_comp = ql.Compounded
new_freq = ql.Quarterly
print("Equivalent Rate = {}".format(ir.equivalentRate(new_dc, new_comp, new_freq, start_date, end_date)))

# Implied Rate
comp_factor = 1.05
new_dc = ql.ActualActual()
new_comp = ql.Compounded
new_freq = ql.Annual
print("Implied Rate = {}".format(ir.impliedRate(comp_factor, new_dc, new_comp, new_freq, start_date, end_date)))

### 8. IborIndex
'''
IborIndex - (1) Libor - 1) USDLibor
                      - 2) JPYLibor
                      - 3) GBPLibor
                      - 4) CHFLibor
                      - 5) CADLibor
                      - 6) AUDLibor
                      - 7) NZDLibor
                      - 8) SEKLibor
          - (2) Euribor
          - (3) OvernightIndex - 1) FedFunds 
                               - 2) EONIA
                               - 3) SONIA
                               - 4) AONIA
'''
'''
ql.IborIndex(String name,
             Period tenor,
             Integer settlementDays,
             Currency currency,
             calendar fixingCalendar,
             BusinessDayConvention convention,
             Bool endOfMonth,
             DayCounter daycounter,
             Handle forecastYieldTermStructure)
# name:준거금리의 이름을 나타내며 단순히 string형태의 데이터를 입력(e.g. "USD_3M_LIBOR")
# tenor:준거금리의 픽싱이 되는 주기를 나타내며 Period 클래스를 인자로 받음. 픽싱(Fixing)이란 
        스왑거래를 할 때 변동금리가 주기적으로 확정되는 것을 의미하며, 일반적인 경우 스왑거래는
        3개월 혹은 6개월의 주기를 가진다
# settlemnetDays:일반적인 플레인 바닐라 스왑의 경우 결제일은 거래일(Trade Date) 혹은 금리결정일(Fixing Date)
                 와 정확히 일치하지 않으며, 보통 2일정도 후에 위치한다. 여기에는 integer 형태의 데이터를 입력(e.g. 1 or 2)
# currency:해당 준거금리의 기준통화가 무엇이냐를 나타내는 인자
# Currency()
# ql.USDCurrency()
# ql.EURCurrency()
# ql.KRWCurrency()
# ql.GBPCurrency()
# ql.USDCurrency()
# ql.USDCurrency()

# fixingCalendar:calendar 클래스를 인자로 받으며, 준거금리의 픽싱 스케줄을 정할 때 기준이 되는 달력을 입력받아야 함
# convention:BusinessDayConvention 열거형을 인자로 받으며, 영업일 관행 방식을 결정하는 인자
# endOfMonth:bool형태의 값을 인자로 받으며, 월말인자 기준을 어떻게 설정할 것인가에 대한 인자
# dayCounter:DayCounter클래스를 인자로 받으며, 이자일수 계산방식에 대한 기준을 제공하는 인자
# forecastYieldTermStructure:디폴트 값으로 빈 Handle 클래스가 주어져있는 인자이며, 만약 특정 금리커브에 대한
                             Handle이 주어진다면 이를 통해 미래 변동금리들을 추정한다
'''
# Components
name = "USD_3M_Libor"
tenor = ql.Period(ql.Quarterly)
settlementDays = 2
currency = ql.USDCurrency()
calendar = ql.UnitedStates()
convention = ql.ModifiedFollowing
endOfMonth = False
dayCounter = ql.Actual360()

# Construction
usd_3m_libor = ql.IborIndex(name,
                            tenor,
                            settlementDays,
                            currency,
                            calendar,
                            convention,
                            endOfMonth,
                            dayCounter)

usd_3m_libor.addFixing(ql.Date(27, 4, 2020), 0.0135)
usd_3m_libor.clearFixings()

### 9. TermStructure
# 1) YieldTermStructure
# ql.YieldTermStructure()의 자식 클래스들
'''
ql.FlatForward(), ql.DiscountCurve(), ql.NaturalCubicDiscountCurve()
ql.MonotonicLogCubicDiscountCurve(), ql.ZeroCurve(), ql.LogLinearZeroCurve()
ql.CubicZeroCurve(), ql.NaturalCubicZeroCurve(), ql.LogCubicZeroCurve()
ql.MonotonicCubicZeroCurve(), ql.ZeroSpreadedTermStructure(), ql.ForwardCurve()
ql.ForwardSpreadedTermStructure(), ql.PiecewiseConvexMonotoneZero()
ql.PiecewiseCubicZero(), ql.PiecewiseFlatForward(), ql.PiecewiseKrugerLogDiscount()
ql.PiecewiseKrugerZero(), ql.PiecewiseLinearForward(), ql.PiecewiseLinearZero()
ql.PiecewiseLogCubicDiscount(), ql.PiecewiseLogLinearDiscount()
ql.PiecewiseSplineCubicDiscount(), ql.ImpliedTermStructure()
ql.FittedBondDiscountCurve(), ql.SpreadedLinearZeroInterpolationTermStructure()
'''
# 2) VolatilityTermStructure
# ql.VolatilityTermStructure()의 자식 클래스들
'''
Equity/FX - ql.BlackConstantVol()
          - ql.BlackVarianceCurve()
          - ql.BlackVarianceSurface()
          - ql.AndreasenHugeVolatilityAdapter()
          - ql.LocalConstantVol()
          - ql.LocalVolSurface()
          - ql.AndreasenHugeLocalVolAdapter()
Interest Rate - ql.CapFloorTermVolCurve()
              - ql.CapFloorTermVolSurface()
              - ql.ConstantOptionletVolatility()
              - ql.OptionletStripper1()
              - ql.StrippedOptionletAdapter()
              - ql.StrippedOptionletBase()
              - ql.ConstatsSwaptionVolatility()
              - ql.SwaptionVolatilityDiscrete()
              - ql.SwaptionVolatilityMatrix()
              - ql.SwaptionVolCube1()
              - ql.SwaptionVolCube2()              
'''
# 3) DefaultProbabilityTermStructure
# ql.DefaultProbabilityTermStructure()의 자식 클래스들
'''
ql.FlatHazardRate()
ql.PiecewiseFlatHazardRate()
ql.SurvivalProbabilityCurve()
'''
# 4) InflationTermStructure
# ql.InflationTermStructure()의 자식 클래스들
'''
ql.ZeroInflationCurve()
ql.PiecewiseZeroInflation()
ql.YoYInflationCurve()
ql.PiecewiseYoYInflation()
'''

### 10. Handle
'''
Handle <------------------------------------ RelinkableHandle
ql.QuoteHandle()                             ql.RelinkableQuoteHandle()
ql.YieldTermStrcture()                       ql.RelinkableYieldTermStrcture()
ql.BlackVolTermStructureHandle()             ql.RelinkableBlackVolTermStructureHandle()
ql.LocalVolTermStructureHandle()             ql.RelinkableLocalVolTermStructureHandle()
ql.CapFloorTermVolatilityStructureHandle()   ql.RelinkableCapFloorTermVolatilityStructureHandle()
ql.OptionletVolatilityStructureHandle()      ql.RelinkableOptionletVolatilityStructureHandle()
ql.SwaptionVolatilityStructureHandle()       ql.RelinkableSwaptionVolatilityStructureHandle()
ql.DefaultProbabilityTermStructureHandle()   ql.RelinkableDefaultProbabilityTermStructureHandle()
ql.ZeroInflationTermStructureHandle()        ql.RelinkableZeroInflationTermStructureHandle()
ql.YoYInflationTermStructureHandle()         ql.RelinkableYoYInflationTermStructureHandle()
'''

### 11. BootstrapHelper
# BootstrapHelper클래스의 자식 클래스들
'''
ql.DepositRateHelper()
ql.OISRateHelper()
ql.DatedOISRateHelper()
ql.FuturesRateHelper()
ql.FraRateHelper()
ql.FixedRateBondHelper()
ql.SwapRateHelper()
ql.FxSwapRateHelper()
ql.SpreadCdsHelper()
ql.UpfrontCdsHelper()
ql.YearOnYearInflationSwapHelper()
ql.ZeroCouponInflationSwapHelper()
'''

### Black-Scholes Option Model
valuationDate = ql.Date(14, 6, 2019)
ql.Settings.instance().evaluationDate = valuationDate
calendar = ql.SouthKorea()
dayCount = ql.ActualActual()

# Simple Quote Objects
underlying_qt = ql.SimpleQuote(270.48) # Underlying Price
dividend_qt = ql.SimpleQuote(0.0) # Dividend Yield
riskfreerate_qt = ql.SimpleQuote(0.01) # Risk-free Rate
volatility_qt = ql.SimpleQuote(0.13) # Volatility

# Quote Handle Objects
u_qhd = ql.QuoteHandle(underlying_qt)
q_qhd = ql.QuoteHandle(dividend_qt)
r_qhd = ql.QuoteHandle(riskfreerate_qt)
v_qhd = ql.QuoteHandle(volatility_qt)

# Term-Structure Objects
r_ts = ql.FlatForward(valuationDate, r_qhd, dayCount)
d_ts = ql.FlatForward(valuationDate, q_qhd, dayCount)
v_ts = ql.BlackConstantVol(valuationDate, calendar, v_qhd, dayCount)

# Term-Structure Handle Objects
r_thd = ql.YieldTermStructureHandle(r_ts)
d_thd = ql.YieldTermStructureHandle(d_ts)
v_thd = ql.BlackVolTermStructureHandle(v_ts)

# Process & Engine
process = ql.BlackScholesMertonProcess(u_qhd, d_thd, r_thd, v_thd)
engine = ql.AnalyticEuropeanEngine(process)

# Option Objects
option_type = ql.Option.Call
strikePrice = 272
expiryDate = ql.Date(12, 12, 2019)
exercise = ql.EuropeanExercise(expiryDate)
payoff = ql.PlainVanillaPayoff(option_type, strikePrice)
option = ql.VanillaOption(payoff, exercise)

# Pricing
option.setPricingEngine(engine)

# Price & Greeks Results
print('Option Premium =', round(option.NPV(), 2)) # option premium
print('Option Delta =', round(option.delta(), 4)) # delta
print('Option Gamma =', round(option.gamma(), 4)) # gamma
print('Option Theta =', round(option.thetaPerDay(), 4)) # theta
print('Option Vega =', round(option.vega() / 100, 4)) # vega
print('Option Rho =', round(option.rho() / 100, 4)) # rho

# Automatic Re-Pricing
underlying_qt.setValue(275)
print('Option Premium =', round(option.NPV(), 2)) # option premium
print('Option Delta =', round(option.delta(), 4)) # delta
print('Option Gamma =', round(option.gamma(), 4)) # gamma
print('Option Theta =', round(option.thetaPerDay(), 4)) # theta
print('Option Vega =', round(option.vega() / 100, 4)) # vega
print('Option Rho =', round(option.rho() / 100, 4)) # rho

# Implied Volatility
underlying_qt.setValue(270.48)
mkt_price = 8.21
implied_volatility = option.impliedVolatility(mkt_price, process)
volatility_qt.setValue(implied_volatility)
print('Option Premium =', round(option.NPV(), 2)) # option premium
print('Option Delta =', round(option.delta(), 4)) # delta
print('Option Gamma =', round(option.gamma(), 4)) # gamma
print('Option Theta =', round(option.thetaPerDay(), 4)) # theta
print('Option Vega =', round(option.vega() / 100, 4)) # vega
print('Option Rho =', round(option.rho() / 100, 4)) # rho

### Black Model
valuationDate = ql.Date(2, 3, 2020)
ql.Settings.instance().evaluationDate = valuationDate
calendar = ql.SouthKorea()
dayCount = ql.ActualActual()

# Simple Quote Objects
futures_qt = ql.SimpleQuote(270.90) # KOSPI 200 Futures Mar20
riskfreerate_qt = ql.SimpleQuote(0.01) # Risk-free Rate
volatility_qt = ql.SimpleQuote(0.40) # Volatility

# Quote Handle Objects
f_qhd = ql.QuoteHandle(futures_qt)
r_qhd = ql.QuoteHandle(riskfreerate_qt)
v_qhd = ql.QuoteHandle(volatility_qt)

# Term-Structure Objects
r_ts = ql.FlatForward(valuationDate, r_qhd, dayCount)
v_ts = ql.BlackConstantVol(valuationDate, calendar, v_qhd, dayCount)

# Term-Structure Handle Objects
r_thd = ql.YieldTermStructureHandle(r_ts)
v_thd = ql.BlackVolTermStructureHandle(v_ts)

# Process & Engine
process = ql.BlackProcess(f_qhd, r_thd, v_thd)
engine = ql.AnalyticEuropeanEngine(process)

# Option Objects
option_type = ql.Option.Call
strikePrice = 267.50
expiryDate = ql.Date(12, 3, 2020)

exercise = ql.EuropeanExercise(expiryDate)
payoff = ql.PlainVanillaPayoff(option_type, strikePrice)
option = ql.VanillaOption(payoff, exercise)

# Pricing
option.setPricingEngine(engine)

# Price & Greeks Results
print('Option Premium =', round(option.NPV(), 2)) # option premium
print('Option Delta =', round(option.delta(), 4)) # delta
print('Option Gamma =', round(option.gamma(), 4)) # gamma
print('Option Theta =', round(option.thetaPerDay(), 4)) # theta
print('Option Vega =', round(option.vega() / 100, 4)) # vega
print('Option Rho =', round(option.rho() / 100, 4)) # rho

# Implied Volatility
mkt_price = 8.95
implied_volatility = option.impliedVolatility(mkt_price, process)
volatility_qt.setValue(implied_volatility)
print('Option Premium =', round(option.NPV(), 2)) # option premium
print('Option Delta =', round(option.delta(), 4)) # delta
print('Option Gamma =', round(option.gamma(), 4)) # gamma
print('Option Theta =', round(option.thetaPerDay(), 4)) # theta
print('Option Vega =', round(option.vega() / 100, 4)) # vega
print('Option Rho =', round(option.rho() / 100, 4)) # rho

### Black-Scholes Option Fomular
import numpy as np
import scipy.stats as stat

def europian_option(S, K, T, r, sigma, option_type):
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        V = S * stat.norm.cdf(d1) - K * np.exp(-r * T) * stat.norm.cdf(d2)
    else:
        V = K * np.exp(-r * T) * stat.norm.cdf(-d2) - S * stat.norm.cdf(-d1)
        
    return V

print(europian_option(100, 100, 1, 0.02, 0.2, 'call'))
print(europian_option(100, 100, 1, 0.02, 0.2, 'put'))

### Visualization
import numpy as np
import scipy.stats as stat
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

# Parameters
K = 100
r = 0.01
sigma = 0.25

# Variables
T = np.linspace(0, 1, 100)
S = np.linspace(0, 200, 100)
T, S = np.meshgrid(T, S)

# Output
Call_Value = europian_option(S, K, T, r, sigma, 'call')
Put_Value = europian_option(S, K, T, r, sigma, 'put')

# Call Option
trace = go.Surface(x=T, y=S, z=Call_Value)
data = [trace]
layout = go.Layout(title='Call Option',
                   scene={'xaxis':{'title':'Maturity'}, 'yaxis':{'title':'Spot Price'}, 'zaxis':{'title':'Option Price'}}
                   )
fig = go.Figure(data=data, layout=layout)
plot(fig)

# Put Option
trace = go.Surface(x=T, y=S, z=Put_Value)
data = [trace]
layout = go.Layout(title='Put Option',
                   scene={'xaxis':{'title':'Maturity'}, 'yaxis':{'title':'Spot Price'}, 'zaxis':{'title':'Option Price'}}
                   )
fig = go.Figure(data=data, layout=layout)
plot(fig)

# 3D Plot Decomposition

# 1. Call Option
# X-axis : Spot Price
S = np.linspace(0, 200, 100)

# Maturity 2 ~ 10
data1 = []
for i in range(2, 11, 2):
    T = i
    Z = europian_option(S, K, T, r, sigma, 'call')
    trace = go.Scatter(x=S, y=Z, name=('Maturity = '+ str(T)))
    data1.append(trace)
    
# Maturity 0 ~ 2
data2 = []
for i in range(0, 11, 2):
    T = i / 10
    Z = europian_option(S, K, T, r, sigma, 'call')
    trace = go.Scatter(x=S, y=Z, name=('Maturity = '+ str(T)))
    data2.append(trace)
    
# Plotting
layout = go.Layout(width=800, height=400, xaxis=dict(title='Spot Price'), yaxis=dict(title='Option Value'))
fig1 = dict(data=data1, layout=layout)
fig2 = dict(data=data2, layout=layout)

plot(fig1)
plot(fig2)

# X-axis : Maturity
# Maturity 0 ~ 10
data1 = []
for S in range(0, 201, 50):
    T = np.linspace(0, 10, 100)
    Z = europian_option(S, K, T, r, sigma, 'call')
    trace = go.Scatter(x=T, y=Z, name=('Spot Price = '+ str(S)))
    data1.append(trace)
    
# Maturity 0 ~ 1
data2 = []
for S in range(0, 201, 50):
    T = np.linspace(0, 1, 100)
    Z = europian_option(S, K, T, r, sigma, 'call')
    trace = go.Scatter(x=T, y=Z, name=('Spot Price = '+ str(S)))
    data2.append(trace)
    
# Plotting
layout = go.Layout(width=900, height=400, xaxis=dict(title='Maturity'), yaxis=dict(title='Option Value'))
fig1 = dict(data=data1, layout=layout)
fig2 = dict(data=data2, layout=layout)

plot(fig1)
plot(fig2)

# 2. Put Option
# X-axis : Spot Price
S = np.linspace(0, 200, 100)

# Maturity 2 ~ 10
data1 = []
for i in range(2, 11, 2):
    T = i
    Z = europian_option(S, K, T, r, sigma, 'put')
    trace = go.Scatter(x=S, y=Z, name=('Maturity = '+ str(T)))
    data1.append(trace)
    
# Maturity 0 ~ 1
data2 = []
for i in range(0, 11, 2):
    T = i / 10
    Z = europian_option(S, K, T, r, sigma, 'put')
    trace = go.Scatter(x=S, y=Z, name=('Maturity = '+ str(T)))
    data2.append(trace)
    
# Plotting
layout = go.Layout(width=800, height=400, xaxis=dict(title='Spot Price'), yaxis=dict(title='Option Value'))
fig1 = dict(data=data1, layout=layout)
fig2 = dict(data=data2, layout=layout)

plot(fig1)
plot(fig2)

# X-axis : Maturity
# Maturity 0 ~ 10
data1 = []
for S in range(0, 201, 50):
    T = np.linspace(0, 10, 100)
    Z = europian_option(S, K, T, r, sigma, 'put')
    trace = go.Scatter(x=T, y=Z, name=('Spot Price = '+ str(S)))
    data1.append(trace)
    
# Maturity 0 ~ 1
data2 = []
for S in range(0, 201, 50):
    T = np.linspace(0, 1, 100)
    Z = europian_option(S, K, T, r, sigma, 'put')
    trace = go.Scatter(x=T, y=Z, name=('Spot Price = '+ str(S)))
    data2.append(trace)
    
# Plotting
layout = go.Layout(width=900, height=400, xaxis=dict(title='Maturity'), yaxis=dict(title='Option Value'))
fig1 = dict(data=data1, layout=layout)
fig2 = dict(data=data2, layout=layout)

plot(fig1)
plot(fig2)
