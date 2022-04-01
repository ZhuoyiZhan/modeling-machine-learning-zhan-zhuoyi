HW6-ZhanZ
================
Zhan, Zhuoyi
Tue Mar 29 14:51:49 2022

``` r
vowel <- read.csv("https://hastie.su.domains/ElemStatLearn/datasets/vowel.train")
head(vowel)
```

    ##   row.names y    x.1   x.2    x.3   x.4    x.5   x.6    x.7    x.8    x.9
    ## 1         1 1 -3.639 0.418 -0.670 1.779 -0.168 1.627 -0.388  0.529 -0.874
    ## 2         2 2 -3.327 0.496 -0.694 1.365 -0.265 1.933 -0.363  0.510 -0.621
    ## 3         3 3 -2.120 0.894 -1.576 0.147 -0.707 1.559 -0.579  0.676 -0.809
    ## 4         4 4 -2.287 1.809 -1.498 1.012 -1.053 1.060 -0.567  0.235 -0.091
    ## 5         5 5 -2.598 1.938 -0.846 1.062 -1.633 0.764  0.394 -0.150  0.277
    ## 6         6 6 -2.852 1.914 -0.755 0.825 -1.588 0.855  0.217 -0.246  0.238
    ##     x.10
    ## 1 -0.814
    ## 2 -0.488
    ## 3 -0.049
    ## 4 -0.795
    ## 5 -0.396
    ## 6 -0.365

``` r
library(dplyr)
```

    ## 
    ## Attaching package: 'dplyr'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

``` r
library(caret)
```

    ## Loading required package: ggplot2

    ## Loading required package: lattice

``` r
vowels = select(vowel, -1)
```

### Convert the response variable in the “vowel.train” data frame to a factor variable prior to training, so that “randomForest” does classification rather than regression.

``` r
#install.packages("randomForest")
#install.packages("gpairs")
library('randomForest')
```

    ## Warning: package 'randomForest' was built under R version 4.1.2

    ## randomForest 4.7-1

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

    ## The following object is masked from 'package:dplyr':
    ## 
    ##     combine

``` r
library('magrittr') ## for '%<>%' operator
library('gpairs')   ## pairs plot
library('viridis')
```

    ## Loading required package: viridisLite

``` r
vowels$y <- as.factor(vowels$y)
```

### Fit the random forest model to the vowel data using all of the 11 features using the default values of the tuning parameters.

``` r
set.seed(42)
rf = randomForest(as.factor(y) ~ ., data=vowels)
```

### Use 5-fold CV and tune the model by performing a grid search for the following tuning parameters: 1) the number of variables randomly sampled as candidates at each split; consider values 3, 4, and 5, and 2) the minimum size of terminal nodes; consider a sequence (1, 5, 10, 20, 40, and 80).

``` r
#x <- vowels[, 2:11]
#y <- vowels[, 1]
# Create model with default paramters
control <- trainControl(method="cv", number=5, search ="grid")
metric <- "Accuracy"
mtrys <- c(3,4,5)
#nodesizes <- c(1, 5, 10, 20, 40,80)
#tunegrid <- expand.grid(.mtry = mtrys,
#                        .splitrule = "variance",
#                       .min.node.size = c(1, 5, 10, 20, 40, 80))
tunegrid <- expand.grid(mtry=(3:5))
rf_gs <- train(y~., data=vowels, method="rf", metric=metric,  trControl=control,tuneGrid = tunegrid)
print(rf_gs)
```

    ## Random Forest 
    ## 
    ## 528 samples
    ##  10 predictor
    ##  11 classes: '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 425, 422, 425, 419, 421 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy   Kappa    
    ##   3     0.9469268  0.9416118
    ##   4     0.9450233  0.9395133
    ##   5     0.9432091  0.9375192
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was mtry = 3.

``` r
tunegrid1 <- expand.grid(mtry=3)
modellist <- list()
for (nodesize in c(1, 5, 10, 20, 40, 80)){
  rf_default <- train(y~., data=vowels, method="rf", metric=metric,  trControl=control,tuneGrid = tunegrid1, nodesize=nodesize)#
  key <- toString(nodesize)
  modellist[[key]] <- rf_default
}

#Compare results
results <- resamples(modellist)
summary(results)
```

    ## 
    ## Call:
    ## summary.resamples(object = results)
    ## 
    ## Models: 1, 5, 10, 20, 40, 80 
    ## Number of resamples: 5 
    ## 
    ## Accuracy 
    ##         Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
    ## 1  0.9423077 0.9619048 0.9622642 0.9620784 0.9629630 0.9809524    0
    ## 5  0.9230769 0.9266055 0.9439252 0.9433725 0.9523810 0.9708738    0
    ## 10 0.8715596 0.9056604 0.9142857 0.9133173 0.9333333 0.9417476    0
    ## 20 0.7450980 0.7870370 0.8130841 0.8103451 0.8411215 0.8653846    0
    ## 40 0.5514019 0.6666667 0.6981132 0.6689092 0.7087379 0.7196262    0
    ## 80 0.4245283 0.4761905 0.5333333 0.5093888 0.5480769 0.5648148    0
    ## 
    ## Kappa 
    ##         Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
    ## 1  0.9365338 0.9580922 0.9584680 0.9582778 0.9592530 0.9790419    0
    ## 5  0.9153611 0.9192593 0.9383048 0.9377007 0.9476257 0.9679527    0
    ## 10 0.8586906 0.8961802 0.9056980 0.9046217 0.9266540 0.9358855    0
    ## 20 0.7194541 0.7657267 0.7943889 0.7913545 0.8252306 0.8519723    0
    ## 40 0.5071017 0.6329771 0.6680368 0.6358827 0.6798923 0.6914055    0
    ## 80 0.3670713 0.4242847 0.4866806 0.4606222 0.5038067 0.5212676    0

``` r
#print(rf_default)
#$results 
#finalModel
```

we get optimal mtry value as 3 and optimal nodesize is 1.

### With the tuned model, make predictions using the majority vote method, and compute the misclassification rate using the ‘vowel.test’ data.

``` r
vowel.test <- read.csv("https://hastie.su.domains/ElemStatLearn/datasets/vowel.test")
vowel.test$y <- as.factor(vowel.test$y)
vowels.test = select(vowel.test, -1)
rf_gs1 <- train(y~., data=vowels, method="rf",  trControl=control,tuneGrid = tunegrid1,imortance = TRUE,nodesize=1)
```

``` r
prediction <- predict(rf_gs1, vowel.test)
print(mean(prediction != vowel.test$y))
```

    ## [1] 0.4155844
