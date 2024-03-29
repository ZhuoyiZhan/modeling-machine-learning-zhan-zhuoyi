Homework 2
================
Zhan, Zhuoyi
Tue Feb 8 16:11:34 2022

``` r
## load prostate data
prostate <- 
  read.table(url(
    'https://web.stanford.edu/~hastie/ElemStatLearn/datasets/prostate.data'))

## subset to training examples
prostate_train <- subset(prostate, train==TRUE)
```

``` r
## plot lcavol vs lpsa
plot_psa_data <- function(dat=prostate_train) {
  plot(dat$lpsa, dat$lcavol,
       xlab="log Prostate Screening Antigen (psa)",
       ylab="log Cancer Volume (lcavol)",
       pch = 20)
}
plot_psa_data()
```

![](HW2-ZhanZ_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

``` r
############################
## regular linear regression
############################

## L2 loss function
L2_loss <- function(y, yhat)
  (y-yhat)^2

## fit simple linear model using numerical optimization
fit_lin <- function(y, x, loss=L2_loss, beta_init = c(-0.51, 0.75)) {
  err <- function(beta)
    mean(loss(y,  beta[1] + beta[2]*x))
  beta <- optim(par = beta_init, fn = err)
  return(beta)
}

## make predictions from linear model
predict_lin <- function(x, beta)
  beta[1] + beta[2]*x

## fit linear model
lin_beta <- fit_lin(y=prostate_train$lcavol,
                    x=prostate_train$lpsa,
                    loss=L2_loss)

## compute predictions for a grid of inputs
x_grid <- seq(min(prostate_train$lpsa),
              max(prostate_train$lpsa),
              length.out=100)
lin_pred <- predict_lin(x=x_grid, beta=lin_beta$par)

## plot data
plot_psa_data()

## plot predictions
lines(x=x_grid, y=lin_pred, col='darkgreen', lwd=2)

## do the same thing with 'lm'
lin_fit_lm <- lm(lcavol ~ lpsa, data=prostate_train)

## make predictins using 'lm' object
lin_pred_lm <- predict(lin_fit_lm, data.frame(lpsa=x_grid))

## plot predictions from 'lm'
lines(x=x_grid, y=lin_pred_lm, col='pink', lty=2, lwd=2)
```

![](HW2-ZhanZ_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

``` r
##################################
## try modifying the loss function
##################################

## custom loss function
custom_loss <- function(y, yhat)
  (y-yhat)^2 + abs(y-yhat)

## plot custom loss function
err_grd <- seq(-1,1,length.out=200)
plot(err_grd, custom_loss(err_grd,0), type='l',
     xlab='y-yhat', ylab='custom loss')
```

![](HW2-ZhanZ_files/figure-gfm/unnamed-chunk-3-2.png)<!-- -->

``` r
## fit linear model with custom loss
lin_beta_custom <- fit_lin(y=prostate_train$lcavol,
                    x=prostate_train$lpsa,
                    loss=custom_loss)

lin_pred_custom <- predict_lin(x=x_grid, beta=lin_beta_custom$par)

## plot data
plot_psa_data()

## plot predictions from L2 loss
lines(x=x_grid, y=lin_pred, col='darkgreen', lwd=2)

## plot predictions from custom loss
lines(x=x_grid, y=lin_pred_custom, col='pink', lwd=2, lty=2)
```

![](HW2-ZhanZ_files/figure-gfm/unnamed-chunk-3-3.png)<!-- -->

## Homework

### Write functions that implement the L1 loss and tilted absolute loss functions.

``` r
## L1 loss function
L1_loss <- function(y, yhat)
  abs(y-yhat)

## Tilted absolute loss
ab_loss <- function(y, yhat,tau){
  tau = 0.25
  ifelse((y- yhat)<=0,((tau-1)*(y- yhat)),tau*(y- yhat))
}

ab_loss75 <- function(y, yhat,tau){
  tau = 0.75
  ifelse((y- yhat)<=0,((tau-1)*(y- yhat)),tau*(y- yhat))
}
```

### \* Create a figure that shows lpsa (x-axis) versus lcavol (y-axis). Add and label (using the ‘legend’ function) the linear model predictors associated with L2 loss, L1 loss, and tilted absolute value loss for tau = 0.25 and 0.75.

``` r
## fit simple linear model using numerical optimization
fit_lin1 <- function(y, x, loss=L1_loss, beta_init = c(-0.51, 0.75)) {
  err <- function(beta)
    mean(loss(y,  beta[1] + beta[2]*x))
  beta <- optim(par = beta_init, fn = err)
  return(beta)
}

## make predictions from linear model
predict_lin1 <- function(x, beta)
  beta[1] + beta[2]*x

## fit linear model
lin_beta1 <- fit_lin1(y=prostate_train$lcavol,
                    x=prostate_train$lpsa,
                    loss=L1_loss)

## compute predictions for a grid of inputs
x_grid <- seq(min(prostate_train$lpsa),
              max(prostate_train$lpsa),
              length.out=100)
lin_pred1 <- predict_lin1(x=x_grid, beta=lin_beta1$par)
```

``` r
## Tilted absolute loss

## make predictions from linear model
tau =0.25
fit_lin2 <- function(y, x, loss=ab_loss, beta_init = c(-0.51, 0.75)) {
  err <- function(beta)
    mean(loss(y,  beta[1] + beta[2]*x))
  beta <- optim(par = beta_init, fn = err)
  return(beta)
}

predict_lin2 <- function(x, beta)
  beta[1] + beta[2]*x

  
## fit linear model
lin_beta2 <- fit_lin2(y=prostate_train$lcavol,
                    x=prostate_train$lpsa,
                    loss=ab_loss)

lin_pred2 <- predict_lin2(x=x_grid, beta=lin_beta2$par)

# fit simple linear model using numerical optimization
tau = 0.75
fit_lin3 <- function(y, x, loss=ab_loss75, beta_init = c(-0.51, 0.75)) {
  err <- function(beta)
    mean(loss(y,  beta[1] + beta[2]*x))
  beta <- optim(par = beta_init, fn = err)
  return(beta)
}

## make predictions from linear model
predict_lin3 <- function(x, beta)
  beta[1] + beta[2]*x

## fit linear model
lin_beta3 <- fit_lin2(y=prostate_train$lcavol,
                    x=prostate_train$lpsa,
                    loss=ab_loss75)

lin_pred3 <- predict_lin3(x=x_grid, beta=lin_beta3$par)
```

``` r
## plot data
plot_psa_data()

## plot predictions
lines(x=x_grid, y=lin_pred, col='darkgreen', lwd=2)
lines(x=x_grid, y=lin_pred1, col='steelblue', lwd=2)
lines(x=x_grid, y=lin_pred2, col='red', lwd=2)
lines(x=x_grid, y=lin_pred3, col='yellow', lwd=2)
#legend('topright',paste0("L1 Loss"), bty='n')
legend('bottomright', legend=c("L2 Loss", "L1 Loss","tilted (tau=0.25)","tilted (tau=0.75)"),
       col=c("darkgreen", "steelblue", "red", "yellow"),lty=1, cex=0.8)
```

![](HW2-ZhanZ_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

### Write functions to fit and predict from a simple nonlinear model with three parameters defined by ‘beta\[1\] + beta\[2\]*exp(-beta\[3\]*x)’. Hint: make copies of ‘fit\_lin’ and ‘predict\_lin’ and modify them to fit the nonlinear model. Use c(-1.0, 0.0, -0.3) as ‘beta\_init’.

``` r
## fit simple linear model using numerical optimization
fit_lin <- function(y, x, loss=L2_loss, beta_init = c(-1.0, 0.0, -0.3)) {
  err <- function(beta)
    mean(loss(y,  beta[1] + beta[2]*exp(-beta[3]*x)))
  beta <- optim(par = beta_init, fn = err)
  return(beta)
}

## make predictions from linear model
predict_lin <- function(x, beta)
  beta[1] + beta[2]*exp(-beta[3]*x)

## fit linear model
lin_beta <- fit_lin(y=prostate_train$lcavol,
                    x=prostate_train$lpsa,
                    loss=L2_loss)

## compute predictions for a grid of inputs
x_grid <- seq(min(prostate_train$lpsa),
              max(prostate_train$lpsa),
              length.out=100)
lin_pred <- predict_lin(x=x_grid, beta=lin_beta$par)
```

``` r
## fit simple linear model using numerical optimization
fit_lin1 <- function(y, x, loss=L1_loss, beta_init = c(-1.0, 0.0, -0.3)) {
  err <- function(beta)
    mean(loss(y,  beta[1] + beta[2]*exp(-beta[3]*x)))
  beta <- optim(par = beta_init, fn = err)
  return(beta)
}

## make predictions from linear model
predict_lin1 <- function(x, beta)
  beta[1] + beta[2]*exp(-beta[3]*x)

## fit linear model
lin_beta1 <- fit_lin1(y=prostate_train$lcavol,
                    x=prostate_train$lpsa,
                    loss=L1_loss)

## compute predictions for a grid of inputs
x_grid <- seq(min(prostate_train$lpsa),
              max(prostate_train$lpsa),
              length.out=100)
lin_pred1 <- predict_lin1(x=x_grid, beta=lin_beta1$par)
```

``` r
## Tilted absolute loss

## make predictions from linear model
tau = 0.25
fit_lin2 <- function(y, x, loss=ab_loss, beta_init = c(-1.0, 0.0, -0.3)){
  err <- function(beta)
    mean(loss(y,  beta[1] + beta[2]*exp(-beta[3]*x)))
  beta <- optim(par = beta_init, fn = err)
  return(beta)
}

predict_lin2 <- function(x, beta)
  beta[1] + beta[2]*exp(-beta[3]*x)

  
## fit linear model
lin_beta2 <- fit_lin2(y=prostate_train$lcavol,
                    x=prostate_train$lpsa,
                    loss=ab_loss)

lin_pred2 <- predict_lin2(x=x_grid, beta=lin_beta2$par)

# fit simple linear model using numerical optimization
fit_lin3 <- function(y, x, loss=ab_loss75, beta_init = c(-1.0, 0.0, -0.3)){
  err <- function(beta)
    mean(loss(y,  beta[1] + beta[2]*exp(-beta[3]*x)))
  beta <- optim(par = beta_init, fn = err)
  return(beta)
}

## make predictions from linear model
predict_lin3 <- function(x, beta)
  beta[1] + beta[2]*exp(-beta[3]*x)

## fit linear model
lin_beta3 <- fit_lin2(y=prostate_train$lcavol,
                    x=prostate_train$lpsa,
                    loss=ab_loss75)

lin_pred3 <- predict_lin3(x=x_grid, beta=lin_beta3$par)
```

### \* Create a figure that shows lpsa (x-axis) versus lcavol (y-axis). Add and label (using the ‘legend’ function) the nonlinear model predictors associated with L2 loss, L1 loss, and tilted absolute value loss for tau = 0.25 and 0.75.

``` r
## plot data
plot_psa_data()

## plot predictions
lines(x=x_grid, y=lin_pred, col='darkgreen', lwd=2)
lines(x=x_grid, y=lin_pred1, col='steelblue', lwd=2)
lines(x=x_grid, y=lin_pred2, col='red', lwd=2)
lines(x=x_grid, y=lin_pred3, col='yellow', lwd=2)
#legend('topright',paste0("L1 Loss"), bty='n')
legend('bottomright', legend=c("L2 Loss", "L1 Loss","tilted (tau=0.25)","tilted (tau=0.75)"),
       col=c("darkgreen", "steelblue", "red", "yellow"),lty=1, cex=0.8)
```

![](HW2-ZhanZ_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->
