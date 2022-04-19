HW6-ZhanZ
================
Zhan, Zhuoyi
Tue Apr 12 16:11:07 2022

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

tunegrid <- expand.grid(mtry = c(3,4,5),
                        splitrule = "gini",
                       min.node.size = c(1, 5, 10, 20, 40, 80))
rf_gs <- train(y~., data=vowels, method="ranger", metric=metric,  trControl=control,tuneGrid = tunegrid, classification=TRUE)
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
    ##   mtry  min.node.size  Accuracy   Kappa    
    ##   3      1             0.9507034  0.9457640
    ##   3      5             0.9432051  0.9375173
    ##   3     10             0.9355657  0.9291107
    ##   3     20             0.9107429  0.9018004
    ##   3     40             0.7615862  0.7377157
    ##   3     80             0.6421898  0.6064614
    ##   4      1             0.9449850  0.9394742
    ##   4      5             0.9393049  0.9332253
    ##   4     10             0.9297954  0.9227663
    ##   4     20             0.8998239  0.8897943
    ##   4     40             0.7501474  0.7251406
    ##   4     80             0.6271423  0.5899659
    ##   5      1             0.9450752  0.9395721
    ##   5      5             0.9393775  0.9333032
    ##   5     10             0.9146960  0.9061486
    ##   5     20             0.8770064  0.8646903
    ##   5     40             0.7386411  0.7124786
    ##   5     80             0.6134215  0.5748741
    ## 
    ## Tuning parameter 'splitrule' was held constant at a value of gini
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final values used for the model were mtry = 3, splitrule = gini
    ##  and min.node.size = 1.

``` r
#tunegrid1 <- expand.grid(.mtry=c(3,4,5),splitrule = "entropy",
#                       .min.node.size = c(1, 5, 10, 20, 40, 80))
#modellist <- list()
#for (nodesize in c(1, 5, 10, 20, 40, 80)){
#  rf_default <- train(y~., data=vowels, method="rf", metric=metric,  #trControl=control,tuneGrid = tunegrid1, nodesize=nodesize)#
#  key <- toString(nodesize)
#  modellist[[key]] <- rf_default
#}

#Compare results
#results <- resamples(modellist)
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
rf_gs1 <- randomForest(as.factor(y) ~ ., data=vowels, nodesize=1, mtry=3)
```

``` r
prediction <- predict(rf_gs1, vowel.test)
print(mean(prediction != vowel.test$y))
```

    ## [1] 0.4112554

The miscalssification rate is 0.41.
