#!/usr/bin/env Rscript
# =============================================================================
# panel_to_r_input.R
#
# 作用：把 data/model_input_v1.csv（已经是"股票×月"面板，740,221 行）拆成
#       TreeFactor_APTree() 要求的 13 个分开的入参，做断言，可选训练一棵树。
#
# 它【不做】数据加工：不排名、不缩尾、不减无风险利率、不过滤样本、不重算特征。
#   那些在 pipeline/build_monthly_panel.py（L2）和 pipeline/panel_export.py（L3）已完成。
#   如果这里发现要补这些，应该回去改 L2/L3，而不是在这里打补丁。
#
# 用法：
#   Rscript pipeline/panel_to_r_input.R                  # 只构建 + 校验，保存入参 RDS
#   Rscript pipeline/panel_to_r_input.R --train           # 额外训练一棵树
#   Rscript pipeline/panel_to_r_input.R --train --train-end 2012-12
#
# 输出：
#   data/r_input_v1.rds      13 个入参的对象（供以后直接 load 用）
#   data/tree_fit_v1.rds     训练后的 fit 对象（仅在 --train 时）
# =============================================================================

suppressPackageStartupMessages({
  library(TreeFactor)
})

# ---------------------------------------------------------------- 命令行参数
args <- commandArgs(trailingOnly = TRUE)
do_train   <- "--train" %in% args
train_end  <- "2015-12"
if ("--train-end" %in% args) {
  i <- which(args == "--train-end")
  if (i < length(args)) train_end <- args[[i + 1]]
}

# 项目根目录：脚本在 <root>/pipeline/ 下
script_path <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])
root <- if (!is.na(script_path)) normalizePath(file.path(dirname(script_path), "..")) else getwd()

csv_path <- file.path(root, "data", "model_input_v1.csv")
if (!file.exists(csv_path)) stop("找不到 ", csv_path, "，请先运行 pipeline/panel_export.py")

# R 库在用户目录（R_LIBS_USER 已由环境变量提供）
if (!nzchar(Sys.getenv("R_LIBS_USER"))) {
  warning("R_LIBS_USER 未设置，TreeFactor 可能找不到")
}

cat("== 读取面板 ==================================================\n")
# 文件是 UTF-8 with BOM；用 UTF-8-BOM 编码读，避免首列名带上 BOM
df <- read.csv(csv_path, fileEncoding = "UTF-8-BOM", check.names = FALSE,
               stringsAsFactors = FALSE)
names(df)[1] <- sub("^\ufeff", "", names(df)[1])
cat("  文件   :", csv_path, "\n")
cat("  规模   :", nrow(df), "行 x", ncol(df), "列\n")
cat("  股票   :", length(unique(df$Stkcd)), "  月份:", length(unique(df$t)), "\n")
cat("  区间   :", min(df$Trdmnt), "~", max(df$Trdmnt), "\n")

# ---------------------------------------------------------------- 拆参数
FEAT <- grep("^X_", names(df), value = TRUE)
if (length(FEAT) == 0) stop("没有找到 X_ 开头的特征列")
cat("\n== 拆分 13 个入参 ============================================\n")
cat("  特征数 :", length(FEAT), "\n")

# 排序：先月后股，保证同一月份的观测连续（不是必需，但便于核对）
df <- df[order(df$t, df$id), ]
rownames(df) <- NULL

X <- as.matrix(df[, FEAT, drop = FALSE])
storage.mode(X) <- "double"

R_vec <- as.numeric(df$ret_ex)      # 叶组合用的收益（超额）
Y_vec <- as.numeric(df$ret_ex)      # 准则回归的目标；单因子版与 R 相同
W_vec <- as.numeric(df$me_lag)      # 叶内加权：t-1 月末总市值

N <- nrow(df)
Z <- matrix(1, N, 1)                # 常数因子暴露（β = b0）
H <- matrix(0, N, 1)                # no_H = TRUE 时不读取，但不能是 NULL

stocks_idx <- as.integer(df$id)     # 0 基
months_idx <- as.integer(df$t)      # 0 基
num_stocks <- max(stocks_idx) + 1L
num_months <- max(months_idx) + 1L

p <- ncol(X)
first_split_var  <- as.numeric(0:(p - 1))
second_split_var <- as.numeric(0:(p - 1))

cat("  X            :", paste(dim(X), collapse = " x "), "\n")
cat("  R / Y        :", length(R_vec), "（两列相同，单因子设定）\n")
cat("  weights      : me_lag，中位", format(median(W_vec), scientific = TRUE), "\n")
cat("  Z / H        :", paste(dim(Z), collapse = " x "), "/", paste(dim(H), collapse = " x "), "\n")
cat("  num_stocks   :", num_stocks, "  num_months:", num_months, "\n")
cat("  split vars   : 0 ..", p - 1, "\n")

# ---------------------------------------------------------------- 断言
cat("\n== 断言（任一条失败就停）=====================================\n")
fail <- 0
check <- function(name, ok, detail = "") {
  cat(sprintf("  [%s] %-34s %s\n", if (isTRUE(ok)) "OK" else "FAIL", name, detail))
  if (!isTRUE(ok)) fail <<- fail + 1
  invisible(ok)
}

check("主键唯一 (Stkcd, Trdmnt)", !anyDuplicated(paste(df$Stkcd, df$Trdmnt)))
check("X 无 NA",       sum(is.na(X)) == 0,       paste0("NA=", sum(is.na(X))))
check("X 无 Inf",      all(is.finite(X)),        paste0("非有限=", sum(!is.finite(X))))
check("X 在 [-1,1]",   min(X) >= -1 && max(X) <= 1,
      sprintf("[%.4f, %.4f]", min(X), max(X)))
check("R / Y 无 NA",   sum(is.na(R_vec)) == 0 && sum(is.na(Y_vec)) == 0)
check("weights 无 NA 且 > 0", sum(is.na(W_vec)) == 0 && min(W_vec) > 0,
      sprintf("min=%.3g", min(W_vec)))
check("stocks 从 0 连续", min(stocks_idx) == 0 && length(unique(stocks_idx)) == num_stocks)
check("months 从 0 连续", min(months_idx) == 0 && length(unique(months_idx)) == num_months)

by_month <- table(months_idx)
check("每月 >= 200 只（10 叶 x 20）", min(by_month) >= 200,
      sprintf("min=%d, median=%d", min(by_month), as.integer(median(by_month))))

# 每个特征在月内是否非常数（常数特征切不动）
const_feat <- FEAT[apply(X, 2, function(v) length(unique(v)) <= 1)]
check("特征月内有变异", length(const_feat) == 0,
      if (length(const_feat)) paste(const_feat, collapse = ",") else "")

if (fail > 0) stop(sprintf("%d 项断言失败，已停止", fail))

# ---------------------------------------------------------------- 保存入参
out <- list(
  X = X, Y = Y_vec, R = R_vec, Z = Z, H = H,
  portfolio_weight = W_vec, loss_weight = W_vec,
  stocks = stocks_idx, months = months_idx,
  first_split_var = first_split_var, second_split_var = second_split_var,
  num_stocks = num_stocks, num_months = num_months,
  features = FEAT, stkcd = df$Stkcd, trdmnt = df$Trdmnt,
  n_obs = N
)
saveRDS(out, file.path(root, "data", "r_input_v1.rds"))
cat("\n  已保存 data/r_input_v1.rds\n")

# ---------------------------------------------------------------- 可选训练
if (!do_train) {
  cat("\n== 未训练（加 --train 才会跑树）=============================\n")
  cat("  入参已就绪，可直接调用：\n")
  cat("  fit <- TreeFactor_APTree(R=..., Y=..., X=..., Z=..., H=...,\n")
  cat("           portfolio_weight=..., loss_weight=..., stocks=..., months=...,\n")
  cat("           first_split_var=..., second_split_var=...,\n")
  cat("           num_stocks=..., num_months=..., min_leaf_size=20, max_depth=4,\n")
  cat("           num_iter=9, num_cutpoints=4, eta=1, equal_weight=FALSE,\n")
  cat("           no_H=TRUE, abs_normalize=TRUE, weighted_loss=FALSE,\n")
  cat("           stop_no_gain=FALSE, lambda_mean=0, lambda_cov=1e-4)\n")
  quit(status = 0)
}

cat("\n== 训练 ======================================================\n")
is_train <- df$Trdmnt <= train_end
cat("  训练期 :", min(df$Trdmnt[is_train]), "~", train_end,
    sprintf("(%d 行, %d 月)", sum(is_train), length(unique(df$t[is_train]))), "\n")
cat("  测试期 :", min(df$Trdmnt[!is_train]), "~", max(df$Trdmnt),
    sprintf("(%d 行, %d 月)", sum(!is_train), length(unique(df$t[!is_train]))), "\n")

tr <- which(is_train)
# 训练集内部重新编码为 0 基连续
tr_months <- as.integer(factor(df$t[tr])) - 1L
tr_stocks <- as.integer(factor(df$id[tr])) - 1L

t0 <- Sys.time()
fit <- TreeFactor_APTree(
  R = R_vec[tr], Y = Y_vec[tr], X = X[tr, , drop = FALSE],
  Z = Z[tr, , drop = FALSE], H = H[tr, , drop = FALSE],
  portfolio_weight = W_vec[tr], loss_weight = W_vec[tr],
  stocks = tr_stocks, months = tr_months,
  first_split_var = first_split_var, second_split_var = second_split_var,
  num_stocks = max(tr_stocks) + 1L, num_months = max(tr_months) + 1L,
  min_leaf_size = 20, max_depth = 4, num_iter = 9, num_cutpoints = 4,
  eta = 1, equal_weight = FALSE, no_H = TRUE,
  abs_normalize = TRUE, weighted_loss = FALSE, stop_no_gain = FALSE,
  lambda_mean = 0, lambda_cov = 1e-4
)
cat("  耗时   :", round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 2), "分钟\n")
cat("  树结构 :\n"); print(fit$tree)
cat("  R2     :", round(fit$R2, 6), "\n")
cat("  叶权重 :", paste(round(fit$leaf_weight, 4), collapse = ", "), "\n")

# 训练期因子收益的夏普（月度年化，仅作 sanity check）
ft <- as.numeric(fit$ft)
sr <- mean(ft) / sd(ft) * sqrt(12)
cat(sprintf("  训练期年化 SR（未扣成本）: %.2f\n", sr))

saveRDS(fit, file.path(root, "data", "tree_fit_v1.rds"))
cat("  已保存 data/tree_fit_v1.rds\n")
cat("\n完成。\n")
