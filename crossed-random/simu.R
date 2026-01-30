simcross = function(N,rho=0.88,kappa=0.57,seed=20260114,sigsqA=1,sigsqB=1,sigsqE=1,mu=0,verbose=FALSE){
# Simulate data for crossed random effects    
# Designed for 0 < kappa < rho < 1 with rho + kappa > 1
# Given rho and kappa roughly match some Stitchfix data

# Some args are variances because stdevs, like `sigA', can be misread as variances

set.seed(seed)

R = ceiling(N^rho)
C = ceiling(N^kappa)    

if(verbose)
  cat("There are",N,"observations in up to",R,"rows and",C,"columns.\n")

if(verbose && N >= 10^6)
  cat("The verbose option can almost double the computation time.\n")

colsizes = rmultinom(1,N,rep(1/C,C)) # number obs in each column

if( max(colsizes) > R )
  stop("Cannot have more than R obs in a column.\nKeeping kappa < rho should reduce or remove this issue.")

Zlist = list()
for( j in 1:C )
  Zlist[[j]] = sample(R,colsizes[j])

a = rnorm(R)*sqrt(sigsqA)
b = rnorm(C)*sqrt(sigsqB)

ans = matrix(0,N,3)
colnames(ans) = c("i","j","Y")

ind = 1
for( j in 1:C ){
  for( i in Zlist[[j]] ){
    y = mu + a[i] + b[j] + rnorm(1)*sqrt(sigsqE)
    ans[ind,] = c(i,j,y)
    ind = ind+1
  }
}
ans = ans[order(ans[,"i"]),]

if(verbose){
  cat("There are",length(unique(ans[,"i"])),"non-empty rows.\n")
  cat("There are",length(unique(ans[,"j"])),"non-empty columns.\n")

  nid = table(ans[,"i"])
  ndj = table(ans[,"j"])

  cat("The smallest non-empty row has",min(nid),"observations.\n")
  cat("The largest  non-empty row has",max(nid),"observations.\n")
  cat("The smallest non-empty column has",min(ndj),"observations.\n")
  cat("The largest  non-empty column has",max(ndj),"observations.\n")
}

ans
}
