library(TreeFactor)
dir.create("results/single_tree_complete",recursive=TRUE,showWarnings=FALSE)
o <- "results/single_tree_complete"
a <- readRDS("data/r_input_v1.rds")
keep <- c("X_ME","X_BM","X_EP","X_LEV","X_CASH","X_ROA","X_ROE","X_TURN","X_DOLVOL","X_MOM1M","X_OP")
a$X <- a$X[,keep,drop=FALSE]
a$features <- keep
stopifnot(all(is.finite(a$X)))
tr <- which(a$trdmnt<="2015-12")
te <- which(a$trdmnt>"2015-12")
m <- as.integer(factor(a$trdmnt[tr]))-1L
s <- as.integer(factor(a$stkcd[tr]))-1L
fit <- TreeFactor_APTree(R=a$R[tr],Y=a$Y[tr],X=a$X[tr,,drop=FALSE],
 Z=a$Z[tr,,drop=FALSE],H=a$H[tr,,drop=FALSE],
 portfolio_weight=a$portfolio_weight[tr],loss_weight=a$loss_weight[tr],
 stocks=s,months=m,first_split_var=as.numeric(seq_len(ncol(a$X))-1),second_split_var=as.numeric(seq_len(ncol(a$X))-1),
 num_stocks=length(unique(s)),num_months=length(unique(m)),min_leaf_size=20,
 max_depth=10,num_iter=9,num_cutpoints=4,eta=1,equal_weight=FALSE,
 no_H=TRUE,abs_normalize=TRUE,weighted_loss=FALSE,stop_no_gain=FALSE,
 lambda_mean=0,lambda_cov=1e-4)
saveRDS(fit,file.path(o,"fit.rds"))
print(names(fit))
print(fit$leaf_weight)
writeLines(capture.output(str(fit)),file.path(o,"fit_structure.txt"))

# Evaluate fixed training tree on both return definitions.
p <- read.csv("data/panel_v1.csv",fileEncoding="UTF-8-BOM",colClasses=ifelse(names(read.csv("data/panel_v1.csv",nrows=0,fileEncoding="UTF-8-BOM")) %in% c("Stkcd","Trdmnt","ret","rf"),"character","NULL"))
key <- paste(as.integer(p$Stkcd),p$Trdmnt)
stopifnot(!anyDuplicated(key))
idx <- match(paste(as.integer(a$stkcd),a$trdmnt),key)
stopifnot(!anyNA(idx))
p <- p[idx,]
raw <- as.numeric(p$ret)-as.numeric(p$rf)
monthly <- list()
leaves <- list()
for (period in c("Train","Test")) {
 ii <- if(period=="Train") tr else te
 mm <- as.integer(factor(a$trdmnt[ii]))-1L
 for (kind in c("Winsorized","Raw")) {
 rr <- if(kind=="Raw") raw[ii] else as.numeric(a$R[ii])
 pr <- predict(fit,a$X[ii,,drop=FALSE],rr,mm,a$portfolio_weight[ii])
 counts <- table(factor(mm,levels=sort(unique(mm))),factor(pr$leaf_index,levels=fit$leaf_id))
 stopifnot(all(counts>0),all(is.finite(pr$portfolio)))
 if(period=="Train" && kind=="Winsorized") stopifnot(max(abs(pr$ft-fit$ft))<1e-8)
 w <- a$portfolio_weight[ii]
 ew <- tapply(rr,mm,mean)
 vw <- tapply(rr*w,mm,sum)/tapply(w,mm,sum)
 monthly[[paste(period,kind)]] <- data.frame(month=sort(unique(a$trdmnt[ii])),period,kind,
 Tree=as.numeric(pr$ft),EW=as.numeric(ew),VW=as.numeric(vw))
 write.csv(pr$portfolio,file.path(o,paste0("leaf_returns_",period,"_",kind,".csv")),row.names=FALSE)
 write.csv(counts,file.path(o,paste0("leaf_counts_",period,"_",kind,".csv")),row.names=FALSE)
 leaves[[paste(period,kind)]] <- pr$portfolio
 }
}
d <- do.call(rbind,monthly)
write.csv(d,file.path(o,"monthly_returns.csv"),row.names=FALSE)
stats <- do.call(rbind,lapply(split(d,list(d$period,d$kind)),function(z)
 do.call(rbind,lapply(c("Tree","EW","VW"),function(n) {
 r<-z[[n]]; wealth<-cumprod(1+r); dd<-wealth/cummax(c(1,wealth))[-1]-1
 data.frame(period=z$period[1],kind=z$kind[1],strategy=n,n=length(r),
 annual_mean=mean(r)*12,annual_vol=sd(r)*sqrt(12),Sharpe=mean(r)/sd(r)*sqrt(12),
 max_drawdown=min(dd))
 }))))
write.csv(stats,file.path(o,"performance.csv"),row.names=FALSE)
png(file.path(o,"performance.png"),width=1500,height=1000,res=150)
par(mfrow=c(2,2),mar=c(4,4,3,1))
for(kind in c("Winsorized","Raw")) {
 z<-d[d$period=="Test"&d$kind==kind,]
 curves<-apply(as.matrix(z[c("Tree","EW","VW")]),2,function(r)cumprod(1+r))
 matplot(as.Date(paste0(z$month,"-01")),curves,type="l",lty=1,col=c("#005F73","#EE9B00","#777777"),
 xlab="Test month",ylab="Compounded excess-return index",main=paste(kind,"out of sample"))
 legend("topleft",c("P-Tree","Equal weight","Value weight"),col=c("#005F73","#EE9B00","#777777"),lty=1,bty="n")
 z<-stats[stats$period=="Test"&stats$kind==kind,]
 barplot(z$Sharpe,names.arg=z$strategy,col=c("#005F73","#EE9B00","#777777"),ylab="Annualized Sharpe",main=kind)
}
dev.off()
writeLines(fit$tree,file.path(o,"tree.txt"))
write.csv(data.frame(index=seq_len(ncol(a$X))-1,feature=a$features),file.path(o,"feature_index.csv"),row.names=FALSE)
write.csv(data.frame(leaf=as.numeric(fit$leaf_id),weight=as.numeric(fit$leaf_weight)),file.path(o,"leaf_weights.csv"),row.names=FALSE)
print(stats)
# Mean-variance frontier: descriptive, estimated on training returns only.
png(file.path(o,"frontier_train.png"),width=1200,height=900,res=150)
sets <- list(P_Tree=leaves[["Train Winsorized"]],EW_VW=as.matrix(d[d$period=="Train"&d$kind=="Winsorized",c("EW","VW")]))
front <- lapply(sets,function(x) {
 mu<-colMeans(x); V<-cov(x)+diag(1e-4,ncol(x)); inv<-solve(V); one<-rep(1,length(mu))
 A<-as.numeric(t(one)%*%inv%*%one); B<-as.numeric(t(one)%*%inv%*%mu); C<-as.numeric(t(mu)%*%inv%*%mu)
 target<-seq(B/A,max(max(mu)*1.6,B/A+.01),length.out=150)
 data.frame(vol=sqrt((A*target^2-2*B*target+C)/(A*C-B^2))*sqrt(12),mean=target*12)
})
plot(front[[1]],type="l",col="#005F73",lwd=2,xlim=range(unlist(lapply(front,function(x)x$vol))),ylim=range(unlist(lapply(front,function(x)x$mean))),xlab="Annualized volatility",ylab="Annualized mean excess return",main="Training mean-variance frontiers (ridge 1e-4)")
lines(front[[2]],col="#EE9B00",lwd=2)
legend("topleft",names(front),col=c("#005F73","#EE9B00"),lty=1,bty="n")
dev.off()
# Plot the fitted split structure using heap node IDs.
lines_tree<-strsplit(fit$tree,"
")[[1]]
nodes<-read.table(text=paste(lines_tree[-1],collapse="
"),header=FALSE)
names(nodes)<-c("id","variable","cut","a","b")
depth<-floor(log2(nodes$id))
xx<-(nodes$id-2^depth+.5)/2^depth
yy<-max(depth)-depth
png(file.path(o,"tree.png"),width=2200,height=1200,res=140)
plot(xx,yy,type="n",axes=FALSE,xlab="",ylab="",xlim=c(0,1),ylim=c(-.6,max(yy)+.6),main="Single P-Tree / complete characteristics / training 2000-2015")
for(i in seq_len(nrow(nodes))) {
 parent<-match(floor(nodes$id[i]/2),nodes$id)
 if(!is.na(parent))segments(xx[parent],yy[parent],xx[i],yy[i],col="grey60")
}
for(i in seq_len(nrow(nodes))) {
 internal<-(nodes$id[i]*2) %in% nodes$id
 label<-if(internal)paste0(sub("X_","",a$features[nodes$variable[i]+1]),"
<= ",round(nodes$cut[i],2)) else paste0("Leaf ",nodes$id[i])
 text(xx[i],yy[i],label,cex=.8,col=if(internal)"#005F73" else "black")
}
dev.off()
writeLines(capture.output(sessionInfo()),file.path(o,"sessionInfo.txt"))
