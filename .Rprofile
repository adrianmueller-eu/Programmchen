listPackages <- function() {
  ip = as.data.frame(installed.packages()[,c(1,3:4)])
  ip[is.na(ip$Priority),1:2,drop=FALSE]
}
