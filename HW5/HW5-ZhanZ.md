HW5-ZhanZ
================
Zhan, Zhuoyi
Sat Mar 26 15:18:45 2022

``` r
library('MASS') ## for 'mcycle'
library('manipulate') ## for 'manipulate'
library('ggplot2')
library('dplyr')
```

    ## 
    ## Attaching package: 'dplyr'

    ## The following object is masked from 'package:MASS':
    ## 
    ##     select

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

``` r
data(mcycle)
```

## 1

Randomly split the mcycle data into training (75%) and validation (25%)
subsets.

``` r
library(caret)
```

    ## Loading required package: lattice

``` r
index = createDataPartition(mcycle$accel, p = 0.75, list = FALSE)
train = mcycle[index, ]
test = mcycle[-index, ]
```

## 2

Using the mcycle data, consider predicting the mean acceleration as a
function of time. Use the Nadaraya-Watson method with the k-NN kernel
function to create a series of prediction models by varying the tuning
parameter over a sequence of values. (hint: the script already
implements this)

``` r
y <- mcycle$accel
x <- matrix(mcycle$times, length(mcycle$times), 1)

plot(x, y, xlab="Time (ms)", ylab="Acceleration (g)")
```

![](HW5-ZhanZ_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

``` r
## Epanechnikov kernel function
## x  - n x p matrix of training inputs
## x0 - 1 x p input where to make prediction
## lambda - bandwidth (neighborhood size)
kernel_epanechnikov <- function(x, x0, lambda=1) {
  d <- function(t)
    ifelse(t <= 1, 3/4*(1-t^2), 0)
  z <- t(t(x) - x0)
  d(sqrt(rowSums(z*z))/lambda)
}

## k-NN kernel function
## x  - n x p matrix of training inputs
## x0 - 1 x p input where to make prediction
## k  - number of nearest neighbors
kernel_k_nearest_neighbors <- function(x, x0, k=1) {
  ## compute distance betwen each x and x0
  z <- t(t(x) - x0)
  d <- sqrt(rowSums(z*z))

  ## initialize kernel weights to zero
  w <- rep(0, length(d))
  
  ## set weight to 1 for k nearest neighbors
  w[order(d)[1:k]] <- 1
  
  return(w)
}

## Make predictions using the NW method
## y  - n x 1 vector of training outputs
## x  - n x p matrix of training inputs
## x0 - m x p matrix where to make predictions
## kern  - kernel function to use
## ... - arguments to pass to kernel function
nadaraya_watson <- function(y, x, x0, kern, ...) {
  k <- t(apply(x0, 1, function(x0_) {
    k_ <- kern(x, x0_, ...)
    k_/sum(k_)
  }))
  yhat <- drop(k %*% y)
  attr(yhat, 'k') <- k
  return(yhat)
}
```

``` r
## Helper function to view kernel (smoother) matrix
matrix_image <- function(x) {
  rot <- function(x) t(apply(x, 2, rev))
  cls <- rev(gray.colors(20, end=1))
  image(rot(x), col=cls, axes=FALSE)
  xlb <- pretty(1:ncol(x))
  xat <- (xlb-0.5)/ncol(x)
  ylb <- pretty(1:nrow(x))
  yat <- (ylb-0.5)/nrow(x)
  axis(3, at=xat, labels=xlb)
  axis(2, at=yat, labels=ylb)
  mtext('Rows', 2, 3)
  mtext('Columns', 3, 3)
}

## Compute effective df using NW method
## y  - n x 1 vector of training outputs
## x  - n x p matrix of training inputs
## kern  - kernel function to use
## ... - arguments to pass to kernel function
effective_df <- function(y, x, kern, ...) {
  y_hat <- nadaraya_watson(y, x, x,
    kern=kern, ...)
  sum(diag(attr(y_hat, 'k')))
}

## loss function
## y    - train/test y
## yhat - predictions at train/test x
loss_squared_error <- function(y, yhat)
  (y - yhat)^2

## test/train error
## y    - train/test y
## yhat - predictions at train/test x
## loss - loss function
error <- function(y, yhat, loss=loss_squared_error)
  mean(loss(y, yhat))



## make predictions using NW method at training inputs
y_hat <- nadaraya_watson(y, x, x,
  kernel_epanechnikov, lambda=5)

## view kernel (smoother) matrix
matrix_image(attr(y_hat, 'k'))
```

![](HW5-ZhanZ_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

``` r
## create a grid of inputs 
x_plot <- matrix(seq(min(x),max(x),length.out=100),100,1)

## make predictions using NW method at each of grid points
y_hat_plot <- nadaraya_watson(y, x, x_plot,
  kernel_epanechnikov, lambda=1)

## plot predictions
plot(x, y, xlab="Time (ms)", ylab="Acceleration (g)")
lines(x_plot, y_hat_plot, col="#882255", lwd=2) 
```

![](HW5-ZhanZ_files/figure-gfm/unnamed-chunk-4-2.png)<!-- -->

## 3

With the squared-error loss function, compute and plot the training
error, AIC, BIC, and validation error (using the validation data) as
functions of the tuning parameter.

``` r
## k-NN kernel function
## x  - n x p matrix of training inputs
## x0 - 1 x p input where to make prediction
## k  - number of nearest neighbors
kernel_k_nearest_neighbors <- function(x, x0, k=1) {
  ## compute distance betwen each x and x0
  z <- t(t(x) - x0)
  d <- sqrt(rowSums(z*z))

  ## initialize kernel weights to zero
  w <- rep(0, length(d))
  
  ## set weight to 1 for k nearest neighbors
  w[order(d)[1:k]] <- 1
  
  return(w)
}
```

``` r
x_train = matrix(train$times, length(train$times), 1)
y_train = train$accel
x_test = matrix(test$times, length(test$times), 1)
y_test = test$accel
```

``` r
set.seed(777)
training.matrix <- matrix(0, nrow = 20, ncol = length(x_train))

for (i in 1:20) {
  training.matrix[i,] <- nadaraya_watson(y = y_train,
                        x = x_train,
                        x0 = x_train,
                        kern = kernel_k_nearest_neighbors,
                        k = i)
}

training.error <- apply(training.matrix, MARGIN = 1, FUN = error, y = y_train, loss = loss_squared_error)

data.frame(k=1:20, method = "Training error",score = training.error) %>%
  ggplot(aes(x =k, y=score))+
  geom_line(col = " red") +
  geom_point(col = "blue") +
  labs(title = "Training Error") 
```

![](HW5-ZhanZ_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

``` r
df <- rep(0, 20)
for(i in 1:20){
  df[i] <- effective_df(y = y_train, 
                        x = x_train, 
                        kern = kernel_k_nearest_neighbors, 
                        k=i)
}

## y    - training y
## yhat - predictions at training x
## d    - effective degrees of freedom
aic <- function(y, yhat, d)
  error(y, yhat) + 2/length(y)*d


plot_aic <- rep(0, 20)
for( i in 1:20){
  plot_aic[i] <- aic(y = y_train, yhat = training.matrix[i,], d = plot_aic[i])
}

data.frame(k=1:20, method = "Training error", score = plot_aic) %>%
  ggplot(aes(x =k, y=score))+
  geom_line(col = " red") +
  geom_point(col = "blue") +
  labs(title = "AIC") 
```

![](HW5-ZhanZ_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

``` r
## y    - training y
## yhat - predictions at training x
## d    - effective degrees of freedom
bic <- function(y, yhat, d)
  error(y, yhat) + log(length(y))/length(y)*d


plot_bic <- rep(0, 20)
for(i in 1:20){
  plot_bic[i] <- bic(y = y_train, yhat = training.matrix[i,], d=plot_bic[i])
}

data.frame(k=1:20, method = "Training error", score = plot_bic) %>%
  ggplot(aes(x =k, y=score))+
  geom_line(col = " red") +
  geom_point(col = "blue") +
  labs(title = "BIC") 
```

![](HW5-ZhanZ_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

``` r
prediction.matrix <- matrix(0, nrow = 20, ncol = length(x_test))

for (i in 1:20) {
  prediction.matrix[i,] <- nadaraya_watson(y = y_test,
                        x = x_test,
                        x0 = x_test,
                        kern = kernel_k_nearest_neighbors,
                        k = i)
}

validation.error <- apply(prediction.matrix, MARGIN  = 1, FUN = error, y = y_test, loss = loss_squared_error)

data.frame(k=1:20,method = "Validation error",score = validation.error) %>%
  ggplot(aes(x =k, y=score))+
  geom_line(col = " red") +
  geom_point(col = "blue") +
  labs(title = "Validation Error") 
```

![](HW5-ZhanZ_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

### 4,5,6

### For each value of the tuning parameter, Perform 5-fold cross-validation using the combined training and validation data. This results in 5 estimates of test error per tuning parameter value.

### Plot the CV-estimated test error (average of the five estimates from each fold) as a function of the tuning parameter. Add vertical line segments to the figure (using the segments function in R) that represent one “standard error” of the CV-estimated test error (standard deviation of the five estimates from each fold).

### Interpret the resulting figures and select a suitable value for the tuning parameter.

``` r
set.seed(777)
#clean.data <- mcycle[complete.cases(mcycle),]
resample_rows <-  sample(nrow(mcycle))
resample_dat <- mcycle[resample_rows, ]
# Dividing the data into 5 equal parts
n <-  ceiling(nrow(mcycle)/5)


#splitting the data into 5 parts
df_1 <- resample_dat[1:27, ]
df_2 <- resample_dat[28:54, ]
df_3 <- resample_dat[55:81, ]
df_4 <- resample_dat[82:108, ]
df_5 <- resample_dat[109:133, ]
cross_val_dat <- list(df_1 , df_2, df_3, df_4, df_5)

#choose 50 tuning parameters to fit using CV

num_tuned = 50
#test = rep(0, 50)
#train = rep(0, 50)
err_cv <- matrix(0, nrow = num_tuned, ncol=5)

for(i in 1:num_tuned){
  for(j in 1:5){
    test <- cross_val_dat[[j]]
    train <- resample_dat[ setdiff(rownames(resample_dat),rownames(cross_val_dat[[j]])) , ]
    pred <- nadaraya_watson(y=train[,2], 
                            x = as.matrix(train[,1]), 
                            x0 = as.matrix(test[,1]), 
                            kern = kernel_k_nearest_neighbors, 
                            k=i)
    
    #err_cv[i, j] <- 1
    err_cv[i, j] <- error(y=test[,2], yhat = pred, loss=loss_squared_error)
  }
}
```

``` r
average_error <- apply(err_cv, 1, mean)
stds <- apply(err_cv, 1, sd)
plot(x=1:num_tuned, y = average_error, type="l", col="red", pch=20, xlab = "k", ylab = "Average CV error", main=" CV-estimated test error")
segments(x0=1:50, y0 = average_error - stds, y1= average_error+stds, lty =2)
points(x=1:num_tuned, y = average_error, pch=20, col="blue", cex=.5)

abline(h=(average_error+stds)[which.min(average_error)],lty=2,col ="black")
```

![](HW5-ZhanZ_files/figure-gfm/unnamed-chunk-12-1.png)<!-- -->

## Interpret the resulting figures and select a suitable value for the tuning parameter.

### The test error first decrease and then increase as the tuning parameter, k (number of neighbors) increases. According to the one standard error rule, At k=11, the model has the lowest averaged test error, 669. K =29 is the most parsimonious model whode error is no more than one standard error above the error of the best model, 993 &lt; 669+325=994. A larger k is a simpler model because it leads to smoother desicion boundary.
