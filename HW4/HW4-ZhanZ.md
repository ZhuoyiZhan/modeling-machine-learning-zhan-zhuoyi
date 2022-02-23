Homework 4
================
Zhan, Zhuoyi
Tue Feb 22 19:30:37 2022

Question 4.4

1.  On average, what fraction of the available observations will we use
    to make the prediction?

Becuase there is only 1 feature and it is uniformly distributed, so on
average we will use 10% of the obervation

2.  

1%

3.  

0.01^100 \*100 = 10^-98 %

4.  

The fraction of training data near test test observation decrease
exponentially.

5.  l = 0.1^(1/p) the volumn of hypercube’ will be 10% of the total
    volume of space in the p-dimension cube.

Question 4.10

1.  

``` r
#install.packages("ISLR")
#install.packages('caret')
library(ISLR)
library(caret)
```

    ## Loading required package: ggplot2

    ## Loading required package: lattice

``` r
library(MASS)
library(class)
library(e1071)
summary(Weekly)
```

    ##       Year           Lag1               Lag2               Lag3         
    ##  Min.   :1990   Min.   :-18.1950   Min.   :-18.1950   Min.   :-18.1950  
    ##  1st Qu.:1995   1st Qu.: -1.1540   1st Qu.: -1.1540   1st Qu.: -1.1580  
    ##  Median :2000   Median :  0.2410   Median :  0.2410   Median :  0.2410  
    ##  Mean   :2000   Mean   :  0.1506   Mean   :  0.1511   Mean   :  0.1472  
    ##  3rd Qu.:2005   3rd Qu.:  1.4050   3rd Qu.:  1.4090   3rd Qu.:  1.4090  
    ##  Max.   :2010   Max.   : 12.0260   Max.   : 12.0260   Max.   : 12.0260  
    ##       Lag4               Lag5              Volume            Today         
    ##  Min.   :-18.1950   Min.   :-18.1950   Min.   :0.08747   Min.   :-18.1950  
    ##  1st Qu.: -1.1580   1st Qu.: -1.1660   1st Qu.:0.33202   1st Qu.: -1.1540  
    ##  Median :  0.2380   Median :  0.2340   Median :1.00268   Median :  0.2410  
    ##  Mean   :  0.1458   Mean   :  0.1399   Mean   :1.57462   Mean   :  0.1499  
    ##  3rd Qu.:  1.4090   3rd Qu.:  1.4050   3rd Qu.:2.05373   3rd Qu.:  1.4050  
    ##  Max.   : 12.0260   Max.   : 12.0260   Max.   :9.32821   Max.   : 12.0260  
    ##  Direction 
    ##  Down:484  
    ##  Up  :605  
    ##            
    ##            
    ##            
    ## 

``` r
pairs(Weekly)
```

![](HW4-ZhanZ_files/figure-gfm/unnamed-chunk-2-1.png)<!-- --> Dimension
and year may have a curved, non-linear pattern. and direction has linear
pattern with other vriables.

2.  

``` r
lr.fit = glm(Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume, data = Weekly, 
    family = binomial)
summary(lr.fit)
```

    ## 
    ## Call:
    ## glm(formula = Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + 
    ##     Volume, family = binomial, data = Weekly)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -1.6949  -1.2565   0.9913   1.0849   1.4579  
    ## 
    ## Coefficients:
    ##             Estimate Std. Error z value Pr(>|z|)   
    ## (Intercept)  0.26686    0.08593   3.106   0.0019 **
    ## Lag1        -0.04127    0.02641  -1.563   0.1181   
    ## Lag2         0.05844    0.02686   2.175   0.0296 * 
    ## Lag3        -0.01606    0.02666  -0.602   0.5469   
    ## Lag4        -0.02779    0.02646  -1.050   0.2937   
    ## Lag5        -0.01447    0.02638  -0.549   0.5833   
    ## Volume      -0.02274    0.03690  -0.616   0.5377   
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 1496.2  on 1088  degrees of freedom
    ## Residual deviance: 1486.4  on 1082  degrees of freedom
    ## AIC: 1500.4
    ## 
    ## Number of Fisher Scoring iterations: 4

Lag 2 has a pr(&gt;\|z\|) of 0.0296 which is statistically significant.

3.  

``` r
glm.probs = predict(lr.fit, type = "response")
glm.pred = rep("Down", length(glm.probs))
glm.pred[glm.probs > 0.5] = "Up"
table(glm.pred, Weekly$Direction)
```

    ##         
    ## glm.pred Down  Up
    ##     Down   54  48
    ##     Up    430 557

The Accuracy rate is (54+557)/(54+557+48+430) = 56.1%. The True positive
rate is 557/(48+557) = 92.1% So 92.1% of time, the logistic regression
predict as up is right.

4.  

``` r
train = (Weekly$Year < 2009)
hold = Weekly[!train, ]
glm.fit = glm(Direction ~ Lag2, data = Weekly, family = binomial, subset = train)
glm.probs = predict(glm.fit, hold, type = "response")
glm.pred = rep("Down", length(glm.probs))
glm.pred[glm.probs > 0.5] = "Up"
Direction.hold = Weekly$Direction[!train]
table(glm.pred, Direction.hold)
```

    ##         Direction.hold
    ## glm.pred Down Up
    ##     Down    9  5
    ##     Up     34 56

``` r
mean(glm.pred == Direction.hold)
```

    ## [1] 0.625

overall fraction of correct predictions is (56+9)/(5+56+34+9) = 62.5%

5.  

``` r
lda.fit = lda(Direction ~ Lag2, data = Weekly, subset = train)
lda.pred = predict(lda.fit, hold)
table(lda.pred$class, Direction.hold)
```

    ##       Direction.hold
    ##        Down Up
    ##   Down    9  5
    ##   Up     34 56

``` r
mean(lda.pred$class == Direction.hold)
```

    ## [1] 0.625

overall fraction of correct predictions is 62.5%

6.  

``` r
qda.fit = qda(Direction ~ Lag2, data = Weekly, subset = train)
qda.class = predict(qda.fit, hold)$class
table(qda.class, Direction.hold)
```

    ##          Direction.hold
    ## qda.class Down Up
    ##      Down    0  0
    ##      Up     43 61

``` r
mean(qda.class == Direction.hold)
```

    ## [1] 0.5865385

overall fraction of correct predictions is 58.65%

7.  

``` r
train.X = as.matrix(Weekly$`Lag2`[train])
test.X = as.matrix(Weekly$`Lag2`[!train])
train.Direction = Weekly$Direction[train]
set.seed(1)
knn.pred = knn(train.X, test.X, train.Direction, k = 1)
table(knn.pred, Direction.hold)
```

    ##         Direction.hold
    ## knn.pred Down Up
    ##     Down   21 30
    ##     Up     22 31

``` r
mean(knn.pred == Direction.hold)
```

    ## [1] 0.5

overall fraction of correct predictions is 50%

8.  

``` r
nb.fit = naiveBayes(Direction ~ Lag2, data = Weekly, subset = train)
nb.class = predict(nb.fit, hold)
table(nb.class, Direction.hold)
```

    ##         Direction.hold
    ## nb.class Down Up
    ##     Down    0  0
    ##     Up     43 61

``` r
mean(nb.class == Direction.hold)
```

    ## [1] 0.5865385

1.  

``` r
# LOgistic regression and LDA 
```

10. 

``` r
# KNN k =10
knn.pred = knn(train.X, test.X, train.Direction, k = 10)
table(knn.pred, Direction.hold)
```

    ##         Direction.hold
    ## knn.pred Down Up
    ##     Down   17 18
    ##     Up     26 43

``` r
mean(knn.pred == Direction.hold)
```

    ## [1] 0.5769231

``` r
# KNN k =50
knn.pred = knn(train.X, test.X, train.Direction, k = 50)
table(knn.pred, Direction.hold)
```

    ##         Direction.hold
    ## knn.pred Down Up
    ##     Down   20 22
    ##     Up     23 39

``` r
mean(knn.pred == Direction.hold)
```

    ## [1] 0.5673077

``` r
#Logistic regression with sqrt(abs(Lag2))

glm.fit = glm(Direction ~ Lag2 + sqrt(abs(Lag2)), data = Weekly, family = binomial, subset = train)
glm.probs = predict(glm.fit, hold, type = "response")
glm.pred = rep("Down", length(glm.probs))
glm.pred[glm.probs > 0.5] = "Up"
Direction.hold = Weekly$Direction[!train]
table(glm.pred, Direction.hold)
```

    ##         Direction.hold
    ## glm.pred Down Up
    ##     Down    2  1
    ##     Up     41 60

``` r
mean(glm.pred == Direction.hold)
```

    ## [1] 0.5961538

``` r
# LDA with Lag2 interaction with Lag1
lda.fit = lda(Direction ~ Lag2:Lag1, data = Weekly, subset = train)
lda.pred = predict(lda.fit, hold)
table(lda.pred$class, Direction.hold)
```

    ##       Direction.hold
    ##        Down Up
    ##   Down    0  1
    ##   Up     43 60

``` r
mean(lda.pred$class == Direction.hold)
```

    ## [1] 0.5769231
