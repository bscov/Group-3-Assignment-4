library(ggplot2)
library(gridExtra)
library(knitr)
library(dplyr)
library(MonteCarlo)
library(tidyr)

####Import Fund Data####

start_time <- Sys.time()

data_active_1 <- read.csv("PCCOX.csv", header = TRUE, sep = ",", stringsAsFactors = TRUE)
data_active_2 <- read.csv("PRILX.csv", header = TRUE, sep = ",", stringsAsFactors = TRUE)
data_active_3 <- read.csv("RWMGX.csv", header = TRUE, sep = ",", stringsAsFactors = TRUE)
data_passive <- read.csv("WFSPX.csv", header = TRUE, sep = ",", stringsAsFactors = TRUE)

#inspect data
head(data_active_1)
str(data_active_1)
head(data_active_2)
str(data_active_2)
head(data_active_3)
str(data_active_3)
head(data_passive)
str(data_passive)

# Record end time
end_time <- Sys.time()
execution_time <- end_time - start_time
cat(paste("Execution Time: ", execution_time, " seconds\n"))

# Get memory usage
memory_info <- pryr::mem_used()
cat(paste("Memory Usage: ", memory_info / 1024 / 1024, " MB\n"))

####Data Transformation####

##Fund Transformation##

start_time <- Sys.time()

pct_change1 <- diff(data_active_1$Adj.Close) / lag(data_active_1$Adj.Close)
adj_change1 <- 1 + pct_change1
log_return_active_1 <- log(adj_change1)

pct_change2 <- diff(data_active_2$Adj.Close) / lag(data_active_2$Adj.Close)
adj_change2 <- 1 + pct_change2
log_return_active_2 <- log(adj_change2)

pct_change3 <- diff(data_active_3$Adj.Close) / lag(data_active_3$Adj.Close)
adj_change3 <- 1 + pct_change3
log_return_active_3 <- log(adj_change3)

pct_change_passive <- diff(data_passive$Adj.Close) / lag(data_passive$Adj.Close)
adj_change_passive <- 1 + pct_change_passive
log_return_passive <- log(adj_change_passive)

# Record end time
end_time <- Sys.time()
execution_time <- end_time - start_time
cat(paste("Execution Time: ", execution_time, " seconds\n"))

# Get memory usage
memory_info <- pryr::mem_used()
cat(paste("Memory Usage: ", memory_info / 1024 / 1024, " MB\n"))

##Fund Transformation Plots##

start_time <- Sys.time()

par(mfrow = c(2,2))

hist(log_return_active_1, breaks = 1000, main = "Log of PCCOX Returns", xlim = c(-0.1, 0.1))
hist(log_return_active_2, breaks = 1000,  main = "Log of PRILX Returns")
hist(log_return_active_3, breaks = 1000,  main = "Log of RWMGX Returns")
hist(log_return_passive, breaks = 1000,  main = "Log of WFSPX Returns")

par(mfrow = c(1,1))

# Record end time
end_time <- Sys.time()
execution_time <- end_time - start_time
cat(paste("Execution Time: ", execution_time, " seconds\n"))

# Get memory usage
memory_info <- pryr::mem_used()
cat(paste("Memory Usage: ", memory_info / 1024 / 1024, " MB\n"))

####Simulation####

##Compute Drift##

start_time <- Sys.time()

mean_active_1 <- mean(log_return_active_1, na.rm = T)
var_active_1 <- var(log_return_active_1, na.rm = T)
drift_active_1 <- mean_active_1 - (0.5 * var_active_1)

mean_active_2 <- mean(log_return_active_2, na.rm = T)
var_active_2 <- var(log_return_active_2, na.rm = T)
drift_active_2 <- mean_active_2 - (0.5 * var_active_2)

mean_active_3 <- mean(log_return_active_3, na.rm = T)
var_active_3 <- var(log_return_active_3, na.rm = T)
drift_active_3 <- mean_active_3 - (0.5 * var_active_3)

mean_passive <- mean(log_return_passive, na.rm = T)
var_passive <- var(log_return_passive, na.rm = T)
drift_passive <- mean_passive - (0.5 * var_passive)

# Record end time
end_time <- Sys.time()
execution_time <- end_time - start_time
cat(paste("Execution Time: ", execution_time, " seconds\n"))

# Get memory usage
memory_info <- pryr::mem_used()
cat(paste("Memory Usage: ", memory_info / 1024 / 1024, " MB\n"))

##Compute Variance & Daily Returns##

start_time <- Sys.time()

days <- 251
trials <- 10000

stdev_active_1 <- sd(log_return_active_1, na.rm = T)
Z_active_1 <- matrix(qnorm(runif(days * trials)), nrow = days, ncol = trials)
daily_returns_active_1 <- exp(drift_active_1 + stdev_active_1 * Z_active_1)

stdev_active_2 <- sd(log_return_active_2, na.rm = T)
Z_active_2 <- matrix(qnorm(runif(days * trials)), nrow = days, ncol = trials)
daily_returns_active_2 <- exp(drift_active_2 + stdev_active_2 * Z_active_2)

stdev_active_3 <- sd(log_return_active_3, na.rm = T)
Z_active_3 <- matrix(qnorm(runif(days * trials)), nrow = days, ncol = trials)
daily_returns_active_3 <- exp(drift_active_3 + stdev_active_3 * Z_active_3)

stdev_passive <- sd(log_return_passive, na.rm = T)
Z_passive  <- matrix(qnorm(runif(days * trials)), nrow = days, ncol = trials)
daily_returns_passive <- exp(drift_passive  + stdev_passive  * Z_passive)

# Record end time
end_time <- Sys.time()
execution_time <- end_time - start_time
cat(paste("Execution Time: ", execution_time, " seconds\n"))

# Get memory usage
memory_info <- pryr::mem_used()
cat(paste("Memory Usage: ", memory_info / 1024 / 1024, " MB\n"))

##Simulate Price Path for Trials##

start_time <- Sys.time()

price_paths_active_1 <- matrix(0, nrow = days, ncol = trials)
price_paths_active_1[1, ] <- data_active_1[nrow(data_active_1), 6]
for (t in 2:days) {
  price_paths_active_1[t, ] <- price_paths_active_1[t-1, ] * (daily_returns_active_1[t, ])
}

price_paths_active_2 <- matrix(0, nrow = days, ncol = trials)
price_paths_active_2[1, ] <- data_active_2[nrow(data_active_2), 6]
for (t in 2:days) {
  price_paths_active_2[t, ] <- price_paths_active_2[t-1, ] * (daily_returns_active_2[t, ])
}

price_paths_active_3 <- matrix(0, nrow = days, ncol = trials)
price_paths_active_3[1, ] <- data_active_3[nrow(data_active_3), 6]
for (t in 2:days) {
  price_paths_active_3[t, ] <- price_paths_active_3[t-1, ] * (daily_returns_active_3[t, ])
}

price_paths_passive <- matrix(0, nrow = days, ncol = trials)
price_paths_passive[1, ] <- data_passive[nrow(data_passive), 6]
for (t in 2:days) {
  price_paths_passive[t, ] <- price_paths_passive[t-1, ] * (daily_returns_passive[t, ])
}

# Record end time
end_time <- Sys.time()
execution_time <- end_time - start_time
cat(paste("Execution Time: ", execution_time, " seconds\n"))

# Get memory usage
memory_info <- pryr::mem_used()
cat(paste("Memory Usage: ", memory_info / 1024 / 1024, " MB\n"))

#inspect price path array
dim(price_paths_active_1)

##Price Path Matrices to Dataframes##

start_time <- Sys.time()

ncol <- 251
col_names <- paste('Day', 1:ncol, sep = "_")

#price path 1
df1 <- as.data.frame(price_paths_active_1)
df1 <- t(df1)
colnames(df1) <- col_names

#price path 2
df2 <- as.data.frame(price_paths_active_2)
df2 <- t(df2)
colnames(df2) <- col_names

#price path 3
df3 <- as.data.frame(price_paths_active_3)
df3 <- t(df3)
colnames(df3) <- col_names

#price path passive
df_passive <- as.data.frame(price_paths_passive)
df_passive <- t(df_passive)
colnames(df_passive) <- col_names

# Record end time
end_time <- Sys.time()
execution_time <- end_time - start_time
cat(paste("Execution Time: ", execution_time, " seconds\n"))

# Get memory usage
memory_info <- pryr::mem_used()
cat(paste("Memory Usage: ", memory_info / 1024 / 1024, " MB\n"))

##Calculate Return & Volatility##

start_time <- Sys.time()

#PCCOX - data active 1
df1 <- transform(df1, Volatility=apply(df1, 1, sd, na.rm=TRUE))
df1$Return <- (df1$Day_251 - df1$Day_1) / df1$Day_1

#PRILX - data active 2
df2 <- transform(df2, Volatility=apply(df2, 1, sd, na.rm=TRUE))
df2$Return <- (df2$Day_251 - df2$Day_1) / df2$Day_1

#RWMGX - data active 3
df3 <- transform(df3, Volatility=apply(df3, 1, sd, na.rm=TRUE))
df3$Return <- (df3$Day_251 - df3$Day_1) / df3$Day_1

#WFSPX - data passive
df_passive <- transform(df_passive, Volatility=apply(df_passive, 1, sd, na.rm=TRUE))
df_passive$Return <- (df_passive$Day_251 - df_passive$Day_1) / df_passive$Day_1

# Record end time
end_time <- Sys.time()
execution_time <- end_time - start_time
cat(paste("Execution Time: ", execution_time, " seconds\n"))

# Get memory usage
memory_info <- pryr::mem_used()
cat(paste("Memory Usage: ", memory_info / 1024 / 1024, " MB\n"))

##Calculate Average Return##

start_time <- Sys.time()

#average returns
PCCOX_returns <- mean(df1$Return)
PRILX_returns <- mean(df2$Return)
RWMGX_returns <- mean(df3$Return)
WFSPX_returns <- mean(df_passive$Return)

#create vectors
Avg_Annual_Return <- c(PCCOX_returns, PRILX_returns, RWMGX_returns, WFSPX_returns)
Fund <- c("PCCOX", "PRILX", "RWMGX", "WFSPX")

#create returns table
returns_tab <- cbind.data.frame(Fund, Avg_Annual_Return)
returns_tab

# Record end time
end_time <- Sys.time()
execution_time <- end_time - start_time
cat(paste("Execution Time: ", execution_time, " seconds\n"))

# Get memory usage
memory_info <- pryr::mem_used()
cat(paste("Memory Usage: ", memory_info / 1024 / 1024, " MB\n"))

##Dataframes to CSV##

start_time <- Sys.time()

write.csv(df1, 'PCCOX_returns_R.csv', row.names = FALSE)
write.csv(df2, 'PRILX__returns_R.csv', row.names = FALSE)
write.csv(df3, 'RWMGX_returns_R.csv', row.names = FALSE)
write.csv(df_passive, 'WFSPX_returns_R.csv', row.names = FALSE)

# Record end time
end_time <- Sys.time()
execution_time <- end_time - start_time
cat(paste("Execution Time: ", execution_time, " seconds\n"))

# Get memory usage
memory_info <- pryr::mem_used()
cat(paste("Memory Usage: ", memory_info / 1024 / 1024, " MB\n"))

