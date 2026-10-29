#加载包
library(TreeFactor)
library(rpart)
library(ranger)

#定义残差函数
tf_residual = function(fit,Y,Z,H,months,no_H){
  # Tree Factor Models
  regressor = Z #暴露
  for(j in 1:dim(Z)[2])
  {
    regressor[,j] = Z[,j] * fit$ft[months + 1] #Z*估计的因子收益
  }
  if(!no_H)
  {
    regressor = cbind(regressor, H) #可选是否加入宏观H（让树不是从0开始 而是基于mkt因子学额外的因子）
  }
  # print(fit$R2*100)
  x <- as.matrix(regressor)
  y <- Y
  b_tf = solve(t(x)%*%x)%*%t(x)%*%y
  haty <- (x%*%b_tf)[,1]
  print(b_tf)
  return(Y-haty)
}

###### parameters #####
#超参数
start = 1
split = 80 #训练期1-80
end   = 100 #测试期 81-100

case='demo' 
max_depth=4 #树深
min_leaf_size = 10 #叶最小样本（至少有多少资产）
max_depth_boosting = 3 #提升树深度变浅
num_iter = 1000
num_cutpoints = 4 #boosting 迭代次数、切分点数（切分考虑多少个阈值）
equal_weight = TRUE
no_H = TRUE
abs_normalize = TRUE
weighted_loss = FALSE
stop_no_gain = FALSE
nu = 1 

# this tiny regularization ensures the matrix inversion
# penalty for the sigma (sigma + lambda I)^{-1} * mu
#正则项
lambda = 1e-4
eta=1

##### load data #####

load("../../data/simu_data.rda")
print(names(da))

data <- da
data['lag_me'] = 1 #资产设置为等权

tmp = data[,c('id', 'date','xret','lag_me', 
              # 5
              'c1', 'c2', 'c3', 'c4', 'c5', # 9
              'm1','m2', # 11
              'mkt' # 12
              )] #允许在c1-c5中选择特征作为切分标准
data = tmp
rm(tmp)
rm(da)

# chars

all_chars <- names(data)[c(5:9)]
top5chars <- c(1:5)
instruments = all_chars[top5chars]
splitting_chars <- all_chars

first_split_var = c(1:5)-1
second_split_var = c(1:5)-1

first_split_var_boosting = c(1:5)-1
second_split_var_boosting = c(1:5)-1

##### train-test split #####

data1 <- data[(data[,c('date')]>=start) & (data[,c('date')]<=split), ]
data2 <- data[(data[,c('date')]>split) & (data[,c('date')]<=end), ]

# rm(data)

###### train data for all boosting steps #####
X_train = data1[,splitting_chars]
R_train = data1[,c("xret")]
months_train = as.numeric(as.factor(data1[,c("date")]))
months_train = months_train - 1 # start from 0
stocks_train = as.numeric(as.factor(data1[,c("id")])) - 1
Z_train = data1[, instruments]
Z_train = cbind(1, Z_train) #暴露矩阵添加一列常数项作为截距
portfolio_weight_train = data1[,c("lag_me")]
loss_weight_train = data1[,c("lag_me")]
num_months = length(unique(months_train))
num_stocks = length(unique(stocks_train))

###### train data 1 #####
# the first H is the mkt
Y_train1 = data1[,c("xret")]
H_train1 = data1[,c("mkt")]
H_train1 = H_train1 * Z_train

# train 1 
#对收益拟合
t = proc.time()
fit1 = TreeFactor_APTree(R_train, Y_train1, X_train, Z_train, H_train1, portfolio_weight_train, 
loss_weight_train, stocks_train, months_train, first_split_var, second_split_var, num_stocks, 
num_months, min_leaf_size, max_depth, num_iter, num_cutpoints, lambda, eta, equal_weight, 
no_H, abs_normalize, weighted_loss, stop_no_gain)
t = proc.time() - t
print(t)

# in sample check
insPred1 = predict(fit1, X_train, R_train, months_train, portfolio_weight_train) #样本内检查
sum((insPred1$ft - fit1$ft)^2)
print(fit1$R2)

# residual 1
res1 = tf_residual(fit1,Y_train1,Z_train,H_train1,months_train,no_H)

# beta
x <- as.matrix(fit1$ft[months_train+1]*Z_train)
y <- as.matrix(Y_train1)
beta_bf1 = solve(t(x)%*%x)%*%t(x)%*%y
# print(beta_bf1)

###### train data 2 #####
# the first H is the mkt
#对残差boosting
Y_train2 = res1
no_H2 = TRUE

# train
t = proc.time()
fit2 = TreeFactor_APTree(R_train, Y_train2, X_train, Z_train, H_train1, portfolio_weight_train, 
loss_weight_train, stocks_train, months_train, first_split_var_boosting, second_split_var_boosting, num_stocks, 
num_months, min_leaf_size, max_depth_boosting, num_iter, num_cutpoints, lambda, eta, equal_weight, 
no_H2, abs_normalize, weighted_loss, stop_no_gain)
t = proc.time() - t
print(t)
print(fit2$R2)

# in sample check
insPred2 = predict(fit2, X_train, R_train, months_train, portfolio_weight_train)
sum((insPred2$ft - fit2$ft)^2)

# residual
res2 = tf_residual(fit2, Y_train2 ,Z_train,H_train1,months_train,no_H2)

# beta
x <- as.matrix(fit2$ft[months_train+1]*Z_train)
y <- as.matrix(Y_train2)
beta_bf2 = solve(t(x)%*%x)%*%t(x)%*%y
# print(beta_bf2)

############# Train Period tf2 #############

###### test data #####
#测试集
X_test = data2[,splitting_chars]
R_test = data2[,c("xret")]
months_test = as.numeric(as.factor(data2[,c("date")]))
months_test = months_test - 1 # start from 0
stocks_test = as.numeric(as.factor(data2[,c("id")])) - 1
Z_test = data2[,instruments]
Z_test = cbind(1, Z_test)
H_test = data2[,c("mkt")]
H_test = H_test * Z_test
portfolio_weight_test = data2[,c("lag_me")]
loss_weight_test = data2[,c("lag_me")]

rm(data2)

############# Test Period Factors #############

pred1 = predict(fit1, X_test, R_test, months_test, portfolio_weight_test)
pred2 = predict(fit2, X_test, R_test, months_test, portfolio_weight_test)

tf_test <- cbind(pred1$ft, pred2$ft)
avg_tf_test <- colMeans(tf_test)

print("####### 1/N ########")
ewsdf = rowMeans(pred1$portfolio)
wsdf = pred1$portfolio %*% fit1$leaf_weight
print("1/N sdf sr: \n")
print(mean(ewsdf)/sd(ewsdf)*sqrt(length(ewsdf)))
print("weighted sdf sr: \n")
print(mean(wsdf)/sd(wsdf)*sqrt(length(wsdf)))
print("####### 1/N ########")


# ############# START OUTPUT #############

# tf_train <- cbind(fit1$ft, fit2$ft)
# # output factors
# write.csv(data.frame(tf_train), paste0(case,"_bf_train",".csv"))
# write.csv(data.frame(tf_test), paste0(case,"_bf_test",".csv"))

# write.csv(data.frame(insPred1$leaf_index), paste0(case,"_leaf_index",".csv"))
# write.csv(data.frame(months_train), paste0(case,"_months_train",".csv"))
# write.csv(data.frame(R_train), paste0(case,"_R_train",".csv"))
# write.csv(data.frame(portfolio_weight_train), paste0(case,"_weight_train",".csv"))

# write.csv(data.frame(fit1$leaf_weight), paste0(case,"_leaf_weight",".csv"))

# ### output test,for plot mve ###

# write.csv(data.frame(pred1$leaf_index), paste0(case,"_leaf_index_test",".csv"))
# write.csv(data.frame(months_test), paste0(case,"_months_test",".csv"))

# write.csv(data.frame(R_test), paste0(case,"_R_test",".csv"))
# write.csv(data.frame(portfolio_weight_test), paste0(case,"_weight_test",".csv"))

# ### output basis portfolio returns

# write.csv(data.frame(fit1$portfolio), paste0(case,"_portfolio_fit1",".csv"))
# write.csv(data.frame(pred1$portfolio), paste0(case,"_portfolio_pred1",".csv"))

# write.csv(data.frame(fit2$portfolio), paste0(case,"_portfolio_fit2",".csv"))
# write.csv(data.frame(pred2$portfolio), paste0(case,"_portfolio_pred2",".csv"))

# ############# END OUTPUT #############