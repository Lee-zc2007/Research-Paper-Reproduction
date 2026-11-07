o <- "results/single_tree_13"
fit <- readRDS(file.path(o,"fit.rds"))
a <- list(features=read.csv(file.path(o,"feature_index.csv"))$feature)
# Plot the fitted split structure using heap node IDs.
lines_tree<-strsplit(fit$tree,"
")[[1]]
nodes<-read.table(text=paste(lines_tree[-1],collapse="
"),header=FALSE)
names(nodes)<-c("id","variable","cut","a","b")
depth<-floor(log2(nodes$id))
# Parenthesize arithmetic before membership testing.
leaf <- !((nodes$id*2) %in% nodes$id)
xx <- rep(NA_real_,nrow(nodes))
xx[leaf] <- seq(.05,.95,length.out=sum(leaf))
for (i in rev(seq_len(nrow(nodes)))) {
 if (!leaf[i]) xx[i] <- mean(xx[match(c(nodes$id[i]*2,nodes$id[i]*2+1),nodes$id)])
}
yy<-max(depth)-depth
png(file.path(o,"tree.png"),width=2200,height=1200,res=140)
plot(xx,yy,type="n",axes=FALSE,xlab="",ylab="",xlim=c(0,1),ylim=c(-.6,max(yy)+.6),main="Single P-Tree / 13 complete characteristics / training 2000-2015")
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
