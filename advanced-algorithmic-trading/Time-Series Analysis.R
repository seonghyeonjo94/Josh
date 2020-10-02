################# Serial Correlation #################

# Create a scatter plot of two sequences
# of normally distributed random variables
set.seed(1)
x <- seq(1,100) + 20.0*rnorm(1:100)
set.seed(2)
y <- seq(1,100) + 20.0*rnorm(1:100)
plot(x,y)

# Calculate their covariance
cov(x,y)

# Calculate their correlation
cor(x,y)

# Plot a correlogram of a sequence of
# normally distributed random variables
set.seed(1)
w <- rnorm(100)
acf(w)

# Plot a correlogram of a sequence of 
# integers from 1 to 100
w <- seq(1, 100)
acf(w)

# Plot a correlogram of a repeating
# sequence of integers from 1 to 10
w <- rep(1:10, 10)
acf(w)

################# Random Walks and White Noise #################

# Plot a correlogram of a sequence of
# normally distributed random variables
set.seed(1)
acf(rnorm(1000))

# Calculate the sample variance of a
# sequence of 1000 normally distributed
# random variables
set.seed(1)
var(rnorm(1000, mean=0, sd=1))

# Plot a realisation of a random walk
set.seed(4)
x <- w <- rnorm(1000)
for (t in 2:1000) x[t] <- x[t-1] + w[t]
plot(x, type="l")

# Plot the correlogram of the random walk
acf(x)

# Take differences of the random walk and
# plot its correlogram
acf(diff(x))

# Install quantmod
install.packages('quantmod')
library('quantmod')

# Obtain Microsoft (MSFT) daily prices from Yahoo
# and plot the differences in adjusted closing prices
getSymbols('MSFT', src='yahoo')
acf(diff(Ad(MSFT)), na.action = na.omit)

# Obtain S&P500 (^GSPC) daily prices from Yahoo
# and plot the differences in adjusted closing prices
getSymbols('^GSPC', src='yahoo')
acf(diff(Ad(GSPC)), na.action = na.omit)

################# ARMA models #################

##### AR models

# Create an AR(1) proces, plotting its values
# and correlogram, for alpha_1 = 0.6
set.seed(1)
x <- w <- rnorm(100)
for (t in 2:100) x[t] <- 0.6*x[t-1] + w[t]
layout(1:2)
plot(x, type="l")
acf(x)

# Fit an autoregressive model to the series
# previously generated, outputting its order,
# parameter estimates and confidence intervals
x.ar <- ar(x, method = "mle")
x.ar$order
x.ar$ar
x.ar$ar + c(-1.96, 1.96)*sqrt(x.ar$asy.var)

# Create an AR(1) proces, plotting its values
# and correlogram, for alpha_1 = -0.6
set.seed(1)
x <- w <- rnorm(100)
for (t in 2:100) x[t] <- -0.6*x[t-1] + w[t]
layout(1:2)
plot(x, type="l")
acf(x)

# Fit an autoregressive model to the series
# previously generated, outputting its order,
# parameter estimates and confidence intervals
x.ar <- ar(x, method = "mle")
x.ar$order
x.ar$ar
x.ar$ar + c(-1.96, 1.96)*sqrt(x.ar$asy.var)

# Create an AR(2) proces, plotting its values
# and correlogram, for alpha_1 = 0.666 and
# alpha_2 = -0.333
set.seed(1)
x <- w <- rnorm(100)
for (t in 3:100) x[t] <- 0.666*x[t-1] - 0.333*x[t-2] + w[t]
layout(1:2)
plot(x, type="l")
acf(x)

# Fit an autoregressive model to the series
# previously generated, outputting its order and
# parameter estimates
x.ar <- ar(x, method = "mle")
x.ar$order
x.ar$ar

# Plot daily closing prices for Amazon Inc. (AMZN)
require(quantmod)
getSymbols("AMZN")
plot(Cl(AMZN))

# Create differenced log returns of AMZN
# and plot their values and correlogram
amznrt = diff(log(Cl(AMZN)))
plot(amznrt)
acf(amznrt, na.action=na.omit)

# Fit an autoregressive model to AMZN log returns
# outputting its order and parameter estimates
# and confidence intervals
amznrt.ar <- ar(amznrt, na.action=na.omit)
amznrt.ar$order
amznrt.ar$ar
amznrt.ar$asy.var
-0.0278 + c(-1.96, 1.96)*sqrt(4.59e-4)
-0.0687 + c(-1.96, 1.96)*sqrt(4.59e-4)

# Plot daily closing prices for the S&P500 (^GSPC)
getSymbols("^GSPC")
plot(Cl(GSPC))

# Create differenced log returns of ^GSPC
# and plot their values and correlogram
gspcrt = diff(log(Cl(GSPC)))
plot(gspcrt)
acf(gspcrt, na.action=na.omit)

# Fit an autoregressive model to ^GSPC log returns
# outputting its order and parameter estimates
gspcrt.ar <- ar(gspcrt, na.action=na.omit)
gspcrt.ar$order
gspcrt.ar$ar

##### MA models

# Create an MA(1) proces, plotting its values
# and correlogram, for beta_1 = 0.6
set.seed(1)
x <- w <- rnorm(100)
for (t in 2:100) x[t] <- w[t] + 0.6*w[t-1]
layout(1:2)
plot(x, type="l")
acf(x)

# Fit an ARIMA(0, 0, 1) model (i.e. MA(1) ) 
# to the series previously generated, 
# outputting its order, parameter estimates 
# and confidence intervals
x.ma <- arima(x, order=c(0, 0, 1))
x.ma
0.6023 + c(-1.96, 1.96)*0.0827

# Create an MA(1) proces, plotting its values
# and correlogram, for beta_1 = -0.6
set.seed(1)
x <- w <- rnorm(100)
for (t in 2:100) x[t] <- w[t] - 0.6*w[t-1]
layout(1:2)
plot(x, type="l")
acf(x)

# Fit an ARIMA(0, 0, 1) model (i.e. MA(1) ) 
# to the series previously generated, 
# outputting its order, parameter estimates 
# and confidence intervals
x.ma <- arima(x, order=c(0, 0, 1))
x.ma
-0.730 + c(-1.96, 1.96)*0.1008

# Create an MA(3) proces, plotting its values
# and correlogram, for beta_1 = 0.6, beta_2 = 0.4
# and beta_3 = 0.3
set.seed(3)
x <- w <- rnorm(1000)
for (t in 4:1000) x[t] <- w[t] + 0.6*w[t-1] + 0.4*w[t-2] + 0.3*w[t-3]
layout(1:2)
plot(x, type="l")
acf(x)

# Fit an ARIMA(0, 0, 3) model (i.e. MA(3) ) 
# to the series previously generated, 
# outputting its order, parameter estimates 
# and confidence intervals
x.ma <- arima(x, order=c(0, 0, 3))
x.ma
0.544 + c(-1.96, 1.96)*0.0309
0.345 + c(-1.96, 1.96)*0.0349
0.298 + c(-1.96, 1.96)*0.0311

# Create differenced log returns of AMZN
require(quantmod)
getSymbols("AMZN")
amznrt = diff(log(Cl(AMZN)))

# Fit an ARIMA(0, 0, 1) model (i.e. MA(1) ) 
# and plot the correlogram of the residuals
amznrt.ma <- arima(amznrt, order=c(0, 0, 1))
amznrt.ma
acf(amznrt.ma$res[-1])

# Fit an ARIMA(0, 0, 2) model (i.e. MA(2) ) 
# and plot the correlogram of the residuals
amznrt.ma <- arima(amznrt, order=c(0, 0, 2))
amznrt.ma
acf(amznrt.ma$res[-1])

# Fit an ARIMA(0, 0, 3) model (i.e. MA(3) ) 
# and plot the correlogram of the residuals
amznrt.ma <- arima(amznrt, order=c(0, 0, 3))
amznrt.ma
acf(amznrt.ma$res[-1])

# Create differenced log returns 
# of the S&P500 (^GPSC)
getSymbols("^GSPC")
gspcrt = diff(log(Cl(GSPC)))

# Fit an ARIMA(0, 0, 1) model (i.e. MA(1) ) 
# and plot the correlogram of the residuals
gspcrt.ma <- arima(gspcrt, order=c(0, 0, 1))
gspcrt.ma
acf(gspcrt.ma$res[-1])

# Fit an ARIMA(0, 0, 2) model (i.e. MA(2) ) 
# and plot the correlogram of the residuals
gspcrt.ma <- arima(gspcrt, order=c(0, 0, 2))
gspcrt.ma
acf(gspcrt.ma$res[-1])

# Fit an ARIMA(0, 0, 3) model (i.e. MA(3) ) 
# and plot the correlogram of the residuals
gspcrt.ma <- arima(gspcrt, order=c(0, 0, 3))
gspcrt.ma
acf(gspcrt.ma$res[-1])

##### ARMA models

# Simulate an ARMA(1,1) model with alpha = 0.5 and 
# beta = 0.5, then plot its values and correlogram
set.seed(1)
x <- arima.sim(n=1000, model=list(ar=0.5, ma=-0.5))
plot(x)
acf(x)

# Determine the parameters and calculate confidence
# intervals using the arima function
arima(x, order=c(1, 0, 1))
-0.396 + c(-1.96, 1.96)*0.373
0.450 + c(-1.96, 1.96)*0.362

# Simulate an ARMA(2,2) model with alpha_1 = 0.5,
# alpha_2 = -0.25, beta_1 = 0.5 and beta_2 = -0.3 
# then plot its values and correlogram
set.seed(1)
x <- arima.sim(n=1000, model=list(ar=c(0.5, -0.25), ma=c(0.5, -0.3)))
plot(x)
acf(x)

# Determine the parameters and calculate confidence
# intervals using the arima function
arima(x, order=c(2, 0, 2))
0.653 + c(-1.96, 1.96)*0.0802
-0.229 + c(-1.96, 1.96)*0.0346
0.319 + c(-1.96, 1.96)*0.0792
-0.552 + c(-1.96, 1.96)*0.0771

# Create an ARMA(3,2) model
set.seed(3)
x <- arima.sim(n=1000, model=list(ar=c(0.5, -0.25, 0.4), ma=c(0.5, -0.3)))

# Loop over p = 0 to 4, q = 0 to 4 and create each
# ARMA(p,q) model, then fit to the previous ARMA(3,2)
# realisation, using the AIC to find the best fit
final.aic <- Inf
final.order <- c(0,0,0)
for (i in 0:4) for (j in 0:4) {
  current.aic <- AIC(arima(x, order=c(i, 0, j)))
  if (current.aic < final.aic) {
    final.aic <- current.aic
    final.order <- c(i, 0, j)
    final.arma <- arima(x, order=final.order)
  }
}

# Output the results of the fit
final.aic
final.order
final.arma

# Plot the residuals of the final model
acf(resid(final.arma))

# Carry out a Ljung-Box test for realisation
# of discrete white noise
Box.test(resid(final.arma), lag=20, type="Ljung-Box")

# Create S&P500 differenced log returns
require(quantmod)
getSymbols("^GSPC")
sp = diff(log(Cl(GSPC)))

# Loop over p = 0 to 4, q = 0 to 4 and create each
# ARMA(p,q) model, then fit to the previous S&P500 
# returns, using the AIC to find the best fit
spfinal.aic <- Inf
spfinal.order <- c(0,0,0)
for (i in 0:4) for (j in 0:4) {
  spcurrent.aic <- AIC(arima(sp, order=c(i, 0, j)))
  if (spcurrent.aic < spfinal.aic) {
    spfinal.aic <- spcurrent.aic
    spfinal.order <- c(i, 0, j)
    spfinal.arma <- arima(sp, order=spfinal.order)
  }
}

# Output the results of the fit
spfinal.order

# Plot the residuals of the final model
acf(resid(spfinal.arma), na.action=na.omit)

# Carry out a Ljung-Box test for realisation
# of discrete white noise
Box.test(resid(spfinal.arma), lag=20, type="Ljung-Box")

################# ARIMA, GARCH models #################

##### ARIMA models

# Simulate an ARIMA(1,1,1) model with alpha = 0.6
# and beta = 0.5, then plot the values
set.seed(2)
x <- arima.sim(list(order = c(1,1,1), ar = 0.6, ma=-0.5), n = 1000)
plot(x)

# Fit an ARIMA(1,1,1) model to the realisation above
# and calculate confidence intervals
x.arima <- arima(x, order=c(1, 1, 1))
x.arima
0.6470 + c(-1.96, 1.96)*0.1065
-0.5165 + c(-1.96, 1.96)*0.1189

# Plot the residuals of the fitted model
acf(resid(x.arima))

# Calculate the Ljung-Box test
Box.test(resid(x.arima), lag=20, type="Ljung-Box")

# Install the forecast library
install.packages("forecast")
library(forecast)

# Obtain differenced log prices for AMZN
require(quantmod)
getSymbols("AMZN", from="2013-01-01")
amzn = diff(log(Cl(AMZN)))

# Calculate the best fitting ARIMA model
azfinal.aic <- Inf
azfinal.order <- c(0,0,0)
for (p in 1:4) for (d in 0:1) for (q in 1:4) {
  azcurrent.aic <- AIC(arima(amzn, order=c(p, d, q)))
  if (azcurrent.aic < azfinal.aic) {
    azfinal.aic <- azcurrent.aic
    azfinal.order <- c(p, d, q)
    azfinal.arima <- arima(amzn, order=azfinal.order)
  }
}

# Output the best ARIMA order
azfinal.order

# Plot a correlogram of the residuals, calculate 
# the Ljung-Box test and predict the next 25 daily
# values of the series
acf(resid(azfinal.arima), na.action=na.omit)
Box.test(resid(azfinal.arima), lag=20, type="Ljung-Box")
plot(forecast(azfinal.arima, h=25))

# Obtain differenced log prices for the S&P500
getSymbols("^GSPC", from="2013-01-01")
sp = diff(log(Cl(GSPC)))

# Calculate the best fitting ARIMA model
spfinal.aic <- Inf
spfinal.order <- c(0,0,0)
for (p in 1:4) for (d in 0:1) for (q in 1:4) {
  spcurrent.aic <- AIC(arima(sp, order=c(p, d, q)))
  if (spcurrent.aic < spfinal.aic) {
    spfinal.aic <- spcurrent.aic
    spfinal.order <- c(p, d, q)
    spfinal.arima <- arima(sp, order=spfinal.order)
  }
}

# Output the best ARIMA order
spfinal.order

# Plot a correlogram of the residuals, calculate 
# the Ljung-Box test and predict the next 25 daily
# values of the series
acf(resid(spfinal.arima), na.action=na.omit)
Box.test(resid(spfinal.arima), lag=20, type="Ljung-Box")
plot(forecast(spfinal.arima, h=25))

##### GARCH models

# Create a GARCH(1,1) model, with alpha_0 = 0.2,
# alpha_1 = 0.5 and beta_1 = 0.3
set.seed(2)
a0 <- 0.2
a1 <- 0.5
b1 <- 0.3
w <- rnorm(10000)
eps <- rep(0, 10000)
sigsq <- rep(0, 10000)
for (i in 2:10000) {
  sigsq[i] <- a0 + a1 * (eps[i-1]^2) + b1 * sigsq[i-1]
  eps[i] <- w[i]*sqrt(sigsq[i])
}

# Plot the correlograms of both the residuals
# and the squared residuals
acf(eps)
acf(eps^2)

# Include the tseries time series library
require(tseries)

# Fit a GARCH model to the series and calculate 
# confidence intervals for the parameters at the 
# 97.5% level
eps.garch <- garch(eps, trace=FALSE)
confint(eps.garch)

# Obtain the differenced log values of the FTSE100
# and plot the values
require(quantmod)
getSymbols("^FTSE")
ftrt = diff(log(Cl(FTSE)))
plot(ftrt)

# Remove the NA value created by the diff procedure
ft <- as.numeric(ftrt)
ft <- ft[!is.na(ft)]

# Fit a suitable ARIMA(p,d,q) model to the 
# FTSE100 returns series
ftfinal.aic <- Inf
ftfinal.order <- c(0,0,0)
for (p in 1:4) for (d in 0:1) for (q in 1:4) {
  ftcurrent.aic <- AIC(arima(ft, order=c(p, d, q)))
  if (ftcurrent.aic < ftfinal.aic) {
    ftfinal.aic <- ftcurrent.aic
    ftfinal.order <- c(p, d, q)
    ftfinal.arima <- arima(ft, order=ftfinal.order)
  }
}

# Output the order of the fit
ftfinal.order

# Plot both the residuals and the squared residuals
acf(resid(ftfinal.arima))
acf(resid(ftfinal.arima)^2)

# Fit a GARCH model
ft.garch <- garch(ft, trace=F)
ft.res <- ft.garch$res[-1]

# Plot the residuals and squared residuals
acf(ft.res)
acf(ft.res^2)

################# Cointegration #################

##### Simulation

## SIMULATED DATA

## Create a simulated random walk
set.seed(123)
z <- rep(0, 1000)
for (i in 2:1000) z[i] <- z[i-1] + rnorm(1)
plot(z, type="l")

# Plot the autocorrelation of the
# series and its differences
layout(1:2)
acf(z)
acf(diff(z))

# For  x and y series that are
# functions of the z series
x <- y <- rep(0, 1000)
x <- 0.3*z + rnorm(1000)
y <- 0.6*z + rnorm(1000)
layout(1:2)
plot(x, type="l")
plot(y, type="l")

# Form the linear combination "comb"
# and plot its correlogram
comb <- 2*x - y
layout(1:2)
plot(comb, type="l")
acf(comb)

# Carry out the unit root tests
library("tseries")
adf.test(comb)
pp.test(comb)
po.test(cbind(2*x,-1.0*y))

# Form a non-stationary linear combination
# and test for unit root with ADF test
badcomb <- -1.0*x + 2.0*y
layout(1:2)
plot(badcomb, type="l")
acf(diff(badcomb))
adf.test(badcomb)

##### CADF

library("quantmod")
library("tseries")

## Set the random seed to 123
set.seed(123)

## Create two non-stationary series based on the
## simulated random walk
z <- rep(0, 1000)
for (i in 2:1000) z[i] <- z[i-1] + rnorm(1)
p <- q <- rep(0, 1000)
p <- 0.3*z + rnorm(1000)
q <- 0.6*z + rnorm(1000)

## Perform a linear regression against the two
## simulated series in order to assess the hedge ratio
## and calculate the ADF test
comb <- lm(p~q)
comb
adf.test(comb$residuals, k=1)

## FINANCIAL DATA - EWA/EWC

## Obtain EWA and EWC for dates corresponding to Chan (2013)
getSymbols("EWA", from="2006-04-26", to="2012-04-09")
getSymbols("EWC", from="2006-04-26", to="2012-04-09")

## Utilise the backwards-adjusted closing prices
ewaAdj = unclass(EWA$EWA.Adjusted)
ewcAdj = unclass(EWC$EWC.Adjusted)

## Plot the ETF backward-adjusted closing prices
plot(ewaAdj, type="l", xlim=c(0, 1500), ylim=c(5.0, 35.0), xlab="April 26th 2006 to April 9th 2012", ylab="ETF Backward-Adjusted Price in USD", col="blue")
par(new=T)
plot(ewcAdj, type="l", xlim=c(0, 1500), ylim=c(5.0, 35.0), axes=F, xlab="", ylab="", col="red")
par(new=F)

## Plot a scatter graph of the ETF adjusted prices
plot(ewaAdj, ewcAdj, xlab="EWA Backward-Adjusted Prices", ylab="EWC Backward-Adjusted Prices")

## Carry out linear regressions twice, swapping the dependent
## and independent variables each time, with zero drift
comb1 = lm(ewcAdj~ewaAdj)
comb2 = lm(ewaAdj~ewcAdj)

## Plot the residuals of the first linear combination
plot(comb1$residuals, type="l", xlab="April 26th 2006 to April 9th 2012", ylab="Residuals of EWA and EWC regression")
comb1
comb2

## Now we perform the ADF test on the residuals,
## or "spread" of each model, using a single lag order
adf.test(comb1$residuals, k=1)
adf.test(comb2$residuals, k=1)

## FINANCIAL DATA - RDS-A/RDS-B

## Obtain RDS equities prices for a recent ten year period
getSymbols("RDS-A", from="2006-01-01", to="2015-12-31")
getSymbols("RDS-B", from="2006-01-01", to="2015-12-31")

## Avoid the hyphen in the name of each variable
RDSA <- get("RDS-A")
RDSB <- get("RDS-B")

## Utilise the backwards-adjusted closing prices
rdsaAdj = unclass(RDSA$"RDS-A.Adjusted")
rdsbAdj = unclass(RDSB$"RDS-B.Adjusted")

## Plot the ETF backward-adjusted closing prices
plot(rdsaAdj, type="l", xlim=c(0, 2517), ylim=c(25.0, 80.0), xlab="January 1st 2006 to December 31st 2015", ylab="RDS-A and RDS-B Backward-Adjusted Closing Price in GBP", col="blue")
par(new=T)
plot(rdsbAdj, type="l", xlim=c(0, 2517), ylim=c(25.0, 80.0), axes=F, xlab="", ylab="", col="red")
par(new=F)

## Plot a scatter graph of the
## Royal Dutch Shell adjusted prices
plot(rdsaAdj, rdsbAdj, xlab="RDS-A Backward-Adjusted Prices", ylab="RDS-B Backward-Adjusted Prices")

## Carry out linear regressions twice, swapping the dependent
## and independent variables each time, with zero drift
comb1 = lm(rdsaAdj~rdsbAdj)
comb2 = lm(rdsbAdj~rdsaAdj)

## Plot the residuals of the first linear combination
plot(comb1$residuals, type="l", xlab="January 1st 2006 to December 31st 2015", ylab="Residuals of RDS-A and RDS-B regression")

## Now we perform the ADF test on the residuals,
## or "spread" of each model, using a single lag order
adf.test(comb1$residuals, k=1)
adf.test(comb2$residuals, k=1)

##### Johansen Test

library("quantmod")
library("tseries")
library("urca")

set.seed(123)

## Simulated cointegrated series

z <- rep(0, 10000)
for (i in 2:10000) z[i] <- z[i-1] + rnorm(1)

p <- q <- r <- rep(0, 10000)

p <- 0.3*z + rnorm(10000)
q <- 0.6*z + rnorm(10000)
r <- 0.8*z + rnorm(10000)

jotest=ca.jo(data.frame(p,q,r), type="trace", K=2, ecdet="none", spec="longrun")
summary(jotest)

s = 1.000*p + 1.791324*q - 1.717271*r
plot(s, type="l")

adf.test(s)

## EWA, EWC and IGE

getSymbols("EWA", from="2006-04-26", to="2012-04-09")
getSymbols("EWC", from="2006-04-26", to="2012-04-09")
getSymbols("IGE", from="2006-04-26", to="2012-04-09")

ewaAdj = unclass(EWA$EWA.Adjusted)
ewcAdj = unclass(EWC$EWC.Adjusted)
igeAdj = unclass(IGE$IGE.Adjusted)

jotest=ca.jo(data.frame(ewaAdj,ewcAdj,igeAdj), type="trace", K=2, ecdet="none", spec="longrun")
summary(jotest)

## SPY, IVV and VOO

getSymbols("SPY", from="2015-01-01", to="2015-12-31")
getSymbols("IVV", from="2015-01-01", to="2015-12-31")
getSymbols("VOO", from="2015-01-01", to="2015-12-31")

spyAdj = unclass(SPY$SPY.Adjusted)
ivvAdj = unclass(IVV$IVV.Adjusted)
vooAdj = unclass(VOO$VOO.Adjusted)

jotest=ca.jo(data.frame(spyAdj,ivvAdj,vooAdj), type="trace", K=2, ecdet="none", spec="longrun")
summary(jotest)

################# Hidden Markov Models #################

# Import the necessary packages and set
# random seed for replication consistency
library('depmixS4')
library('quantmod')
set.seed(1)

# Create the parameters for the bull and
# bear market returns distributions
Nk_lower <- 50
Nk_upper <- 150
bull_mean <- 0.1
bull_var <- 0.1
bear_mean <- -0.05
bear_var <- 0.2

# Create the list of durations (in days) for each regime
days <- replicate(5, sample(Nk_lower:Nk_upper, 1))

# Create the various bull and bear markets returns
market_bull_1 <- rnorm( days[1], bull_mean, bull_var )
market_bear_2 <- rnorm( days[2], bear_mean, bear_var )
market_bull_3 <- rnorm( days[3], bull_mean, bull_var )
market_bear_4 <- rnorm( days[4], bear_mean, bear_var )
market_bull_5 <- rnorm( days[5], bull_mean, bull_var )

# Create the list of true regime states and full returns list
true_regimes <- c( rep(1,days[1]), rep(2,days[2]), rep(1,days[3]), rep(2,days[4]), rep(1,days[5]))
returns <- c( market_bull_1, market_bear_2, market_bull_3, market_bear_4, market_bull_5)

# Create and fit the Hidden Markov Model
hmm <- depmix(returns ~ 1, family = gaussian(), nstates = 2, data=data.frame(returns=returns))
hmmfit <- fit(hmm, verbose = FALSE)
post_probs <- posterior(hmmfit)

# Output both the true regimes and the
# posterior probabilities of the regimes
layout(1:2)
plot(post_probs$state, type='s', main='True Regimes', xlab='', ylab='Regime')
matplot(post_probs[,-1], type='l', main='Regime Posterior Probabilities', ylab='Probability')
legend(x='topright', c('Bull','Bear'), fill=1:2, bty='n')

# Obtain S&P500 data from 2004 onwards and
# create the returns stream from this
getSymbols( "^GSPC", from="2004-01-01" )
gspcRets = diff( log( Cl( GSPC ) ) )
returns = as.numeric(gspcRets)
plot(gspcRets)

# Fit a Hidden Markov Model with two states
# to the S&P500 returns stream
hmm <- depmix(returns ~ 1, family = gaussian(), nstates = 2, data=data.frame(returns=returns))
hmmfit <- fit(hmm, verbose = FALSE)
post_probs <- posterior(hmmfit)

# Plot the returns stream and the posterior
# probabilities of the separate regimes
layout(1:2)
plot(returns, type='l', main='Regime Detection', xlab='', ylab='Returns')
matplot(post_probs[,-1], type='l', main='Regime Posterior Probabilities', ylab='Probability')
legend(x='topright', c('Regime #1','Regime #2'), fill=1:2, bty='n')

# Fit a Hidden Markov Model with three states
# to the S&P500 returns stream
hmm <- depmix(returns ~ 1, family = gaussian(), nstates = 3, data=data.frame(returns=returns))
hmmfit <- fit(hmm, verbose = FALSE)
post_probs <- posterior(hmmfit)

# Plot the returns stream and the posterior
# probabilities of the separate regimes
layout(1:2)
plot(returns, type='l', main='Regime Detection', xlab='', ylab='Returns')
matplot(post_probs[,-1], type='l', main='Regime Posterior Probabilities', ylab='Probability')
legend(x='bottomleft', c('Regime #1','Regime #2', 'Regime #3'), fill=1:3, bty='n')
