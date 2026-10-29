
# The data-generating-process is simple.
# We assume the returns are governed by the market factor $mkt_t$ and two characteristics $c1_{i,t}, c2_{i,t}$.
# $ r_{i,t} = \beta_i mkt_t + b1 * c1_{i,t} + b2_i*c2_{i,t} + \epsilon_{i,t} $.
# The characteristics follow Uniform[-1,1] with a normal(0,0.1) fluctuation.
# The market beta is a function of characteristics $c2, c3$ and macroeconomic indicator $m2$.
# The characteristics are independent.
# The return and volatility of the market factor is conditional on a macroeconomic indicator $m1$.
# Other noise variables include $c5, m2$

# parameters

set.seed(20220215)
# set.seed(89)

N <- 1000 # number of asset
T <- 100 # number of time period
b1 <- 1/100 #c1的真实定价系数
b2 <- 2/100

# functions

## function that generates characteristics
#资产特征
char <- function(N,T){
  c <- runif(n=N, min = -1, max = 1)
  c_matrix <- t(matrix(rep(c,T), nrow=N)) + matrix(rnorm(n=N*T, mean=0, sd=0.1), nrow=T)
  return(c_matrix)
}

## function that generate market beta
#市场beta
betaF <- function(c1, c2, c3, c4, c5, m1, m2){
    beta <- 1 + 0.5 * ( c2 + c3 + (1-(m1>0)*2) )
    return(beta)
}

## function that reshape matrix into a stacked list
#矩阵按列堆叠成向量 训练模型的时候好切分
m2l <- function(m){
  d <- dim(m)
  t <- d[1] # nrow
  k <- d[2] # ncol
  l <- rep(0, t*k)
  for (i in c(1:k)){
    left <- t*(i-1)+1
    right <- t*i
    # print(left, right)
    l[left:right] <- m[,i]
  }
  return(l)
}

## function that reshape matrix into a stacked list

# simulation

## simulate the characteristics of assets
#模拟特征 c1和c2有定价效应 c3影响beta c4和c5是噪声特征
c1 <- char(N,T)
c2 <- char(N,T)
c3 <- char(N,T)
c4 <- char(N,T)
c5 <- char(N,T)

# simulate macroeconomic indicator
#宏观指标
m1 <- rnorm(n=T, mean=0, sd=1/100)
m2 <- rnorm(n=T, mean=0, sd=1/100)

## simualte market factor
#市场因子
mkt <- rnorm(n=T, mean=1/100, sd=2/100)
mkt <- matrix(mkt, nrow=T)

## simulate the market beta of assets
#计算T*N的beta矩阵
beta <- betaF(c1, c2, c3, c4, c5, m1, m2)

## simulate asset returns
#资产收益率
r <- matrix(0, T, N)
dim(r)
for (i in c(1:N)){
    r[,i] <- beta[,i]*mkt + c1[,i]*b1 + c2[,i]*b2 + rnorm(n=T, mean=0, sd=10/100)
}

# save
## stack data and save as rda
#堆叠成面板数据并保存
da <- data.frame(
      xret = m2l(r),
      id   = ceiling(1:(N*T)/N),
      date = rep(c(1:T),N),
      mkt  = rep(mkt,N),
      m1   = rep(m1,N),
      m2   = rep(m2,N),
      c1   = m2l(c1),
      c2   = m2l(c2),
      c3   = m2l(c3),
      c4   = m2l(c4),
      c5   = m2l(c5)
)

f = as.matrix(cbind(mkt))
xt = as.matrix(cbind(m1,m2))
save(da, f, xt, beta, file = "simu_data.rda")