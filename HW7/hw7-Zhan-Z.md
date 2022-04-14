HW7-ZhanZ
================
Zhan, Zhuoyi
Thu Apr 14 16:42:54 2022

Use the Keras library to re-implement the simple neural network
discussed during lecture for the mixture data (see nnet.R). Use a single
10-node hidden layer; fully connected.

``` r
#install.packages("keras")
library(keras)
```

    ## Warning: package 'keras' was built under R version 4.1.2

``` r
#install.packages('rgl')
library('rgl')
library('nnet')
library('dplyr')
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
library(devtools)
```

    ## Loading required package: usethis

``` r
install_url("https://cran.r-project.org/src/contrib/Archive/ElemStatLearn/ElemStatLearn_2015.6.26.tar.gz")
```

    ## Downloading package from url: https://cran.r-project.org/src/contrib/Archive/ElemStatLearn/ElemStatLearn_2015.6.26.tar.gz

    ##      checking for file ‘/private/var/folders/k2/hbqsynqx0l3_f0n6dd27gs680000gn/T/RtmparigAN/remotes101f515b0b5fb/ElemStatLearn/DESCRIPTION’ ...  ✓  checking for file ‘/private/var/folders/k2/hbqsynqx0l3_f0n6dd27gs680000gn/T/RtmparigAN/remotes101f515b0b5fb/ElemStatLearn/DESCRIPTION’ (378ms)
    ##   ─  preparing ‘ElemStatLearn’:
    ##      checking DESCRIPTION meta-information ...  ✓  checking DESCRIPTION meta-information
    ##   ─  checking for LF line-endings in source and make files and shell scripts
    ##   ─  checking for empty or unneeded directories
    ##   ─  building ‘ElemStatLearn_2015.6.26.tar.gz’
    ##      
    ## 

``` r
library('ElemStatLearn')
data(mixture.example)
```

``` r
dat <- mixture.example
```

``` r
fit <- nnet(x=dat$x, y=dat$y, size=10,entropy=TRUE, decay=0)
```

    ## # weights:  41
    ## initial  value 140.862225 
    ## iter  10 value 95.802312
    ## iter  20 value 83.505860
    ## iter  30 value 72.169437
    ## iter  40 value 66.653490
    ## iter  50 value 63.735117
    ## iter  60 value 60.956945
    ## iter  70 value 58.358047
    ## iter  80 value 55.879414
    ## iter  90 value 55.423273
    ## iter 100 value 55.325144
    ## final  value 55.325144 
    ## stopped after 100 iterations

``` r
dim(dat$x)
```

    ## [1] 200   2

``` r
head(dat$x)
```

    ##             [,1]       [,2]
    ## [1,]  2.52609297  0.3210504
    ## [2,]  0.36695447  0.0314621
    ## [3,]  0.76821908  0.7174862
    ## [4,]  0.69343568  0.7771940
    ## [5,] -0.01983662  0.8672537
    ## [6,]  2.19654493 -1.0230141

``` r
n <- dat$x
```

``` r
range01 <- function(x){(x-min(x))/(max(x)-min(x))}
xtrain <- range01(dat$x)
xtest <- range01(dat$px1)
```

``` r
xtrain
```

    ##               [,1]       [,2]
    ##   [1,] 0.754219974 0.42469433
    ##   [2,] 0.431554318 0.38141772
    ##   [3,] 0.491520043 0.48393842
    ##   [4,] 0.480344274 0.49286128
    ##   [5,] 0.373751541 0.50631997
    ##   [6,] 0.704971705 0.22383485
    ##   [7,] 0.360403015 0.29741896
    ##   [8,] 0.240332330 0.55846959
    ##   [9,] 0.750297078 0.31420774
    ##  [10,] 0.660313126 0.52223579
    ##  [11,] 0.332576464 0.37816154
    ##  [12,] 0.274857032 0.29427632
    ##  [13,] 0.748972434 0.54832232
    ##  [14,] 0.368731194 0.44906113
    ##  [15,] 0.479474808 0.48207319
    ##  [16,] 0.567935707 0.33621224
    ##  [17,] 0.329939545 0.56706712
    ##  [18,] 0.797101662 0.55122814
    ##  [19,] 0.424535583 0.37928661
    ##  [20,] 0.442103105 0.43853229
    ##  [21,] 0.740298926 0.18029510
    ##  [22,] 0.312508332 0.33041720
    ##  [23,] 0.413886395 0.50456644
    ##  [24,] 0.779302867 0.07785417
    ##  [25,] 0.595963485 0.65730660
    ##  [26,] 0.532239700 0.36398537
    ##  [27,] 0.381695134 0.66517722
    ##  [28,] 0.634604991 0.30857658
    ##  [29,] 0.548496488 0.23926080
    ##  [30,] 0.694559132 0.54919876
    ##  [31,] 0.823267119 0.47719329
    ##  [32,] 0.536897611 0.13118025
    ##  [33,] 0.538834563 0.39210478
    ##  [34,] 0.416683027 0.43093741
    ##  [35,] 0.568431767 0.56753845
    ##  [36,] 0.396141253 0.45803888
    ##  [37,] 0.748797187 0.57973420
    ##  [38,] 0.499023655 0.43335449
    ##  [39,] 0.605699424 0.21499533
    ##  [40,] 0.482601927 0.38007303
    ##  [41,] 0.604004210 0.26570286
    ##  [42,] 0.509354392 0.50875563
    ##  [43,] 0.612382252 0.26539170
    ##  [44,] 0.726823179 0.43028754
    ##  [45,] 0.348131865 0.74937697
    ##  [46,] 0.638117858 0.32322834
    ##  [47,] 0.251129654 0.65792687
    ##  [48,] 0.518115430 0.38766909
    ##  [49,] 0.513496687 0.24195537
    ##  [50,] 0.578268269 0.38724452
    ##  [51,] 0.814560224 0.48713743
    ##  [52,] 0.425016556 0.41974912
    ##  [53,] 0.686272062 0.25261724
    ##  [54,] 0.331543476 0.50915620
    ##  [55,] 0.405451018 0.30506658
    ##  [56,] 0.268060205 0.59450670
    ##  [57,] 0.396793096 0.51146406
    ##  [58,] 0.365191834 0.45220228
    ##  [59,] 0.781817489 0.50772338
    ##  [60,] 1.000000000 0.53808832
    ##  [61,] 0.425399482 0.42872110
    ##  [62,] 0.407379954 0.74342950
    ##  [63,] 0.621824135 0.61709870
    ##  [64,] 0.364822878 0.42503230
    ##  [65,] 0.786499332 0.13755628
    ##  [66,] 0.423160253 0.67679247
    ##  [67,] 0.718944136 0.39724774
    ##  [68,] 0.475333570 0.19116347
    ##  [69,] 0.453892079 0.45519558
    ##  [70,] 0.747771417 0.26655095
    ##  [71,] 0.534106444 0.24168545
    ##  [72,] 0.633043608 0.22507331
    ##  [73,] 0.557633929 0.36040433
    ##  [74,] 0.536922999 0.35202747
    ##  [75,] 0.508031502 0.25908506
    ##  [76,] 0.518874780 0.21294416
    ##  [77,] 0.440299836 0.40434755
    ##  [78,] 0.694295558 0.26286411
    ##  [79,] 0.389809278 0.26712317
    ##  [80,] 0.718230193 0.52750175
    ##  [81,] 0.447074499 0.38051755
    ##  [82,] 0.916436370 0.45002257
    ##  [83,] 0.761804216 0.16216496
    ##  [84,] 0.648225682 0.20310297
    ##  [85,] 0.808905585 0.61967599
    ##  [86,] 0.513727687 0.30469683
    ##  [87,] 0.365645443 0.51415797
    ##  [88,] 0.462539020 0.30877112
    ##  [89,] 0.288687860 0.40769878
    ##  [90,] 0.768937754 0.16656194
    ##  [91,] 0.386628767 0.61295446
    ##  [92,] 0.622225709 0.33429262
    ##  [93,] 0.758008801 0.58865147
    ##  [94,] 0.588854203 0.37227064
    ##  [95,] 0.269822620 0.54228013
    ##  [96,] 0.441648641 0.36707019
    ##  [97,] 0.726967403 0.51285427
    ##  [98,] 0.714860155 0.48036907
    ##  [99,] 0.519258603 0.42543913
    ## [100,] 0.570567047 0.21030631
    ## [101,] 0.656306685 0.47453528
    ## [102,] 0.920364950 0.53201107
    ## [103,] 0.904135740 0.52017873
    ## [104,] 0.346725498 0.71768026
    ## [105,] 0.537658407 0.32729028
    ## [106,] 0.316825387 0.46426556
    ## [107,] 0.321605424 0.53197449
    ## [108,] 0.617797457 0.65468990
    ## [109,] 0.571170062 0.48518080
    ## [110,] 0.561840044 0.53396401
    ## [111,] 0.598787271 0.63298777
    ## [112,] 0.540912921 0.52047125
    ## [113,] 0.480162606 0.71374732
    ## [114,] 0.538076883 0.48764186
    ## [115,] 0.066875370 0.63606090
    ## [116,] 0.075206453 0.78325411
    ## [117,] 0.578627385 0.47380917
    ## [118,] 0.455725584 0.62293271
    ## [119,] 0.637005128 0.57956559
    ## [120,] 0.489175878 0.49762191
    ## [121,] 0.353877898 0.74032529
    ## [122,] 0.187692818 0.58355377
    ## [123,] 0.576132811 0.33271416
    ## [124,] 0.559640883 0.44706528
    ## [125,] 0.157829848 0.74642347
    ## [126,] 0.619379905 0.41272202
    ## [127,] 0.052546476 0.68694296
    ## [128,] 0.449755144 0.69682664
    ## [129,] 0.206869457 0.68347319
    ## [130,] 0.032422443 0.70660720
    ## [131,] 0.373787945 0.64650324
    ## [132,] 0.148203593 0.77490801
    ## [133,] 0.366061479 0.59498228
    ## [134,] 0.508259292 0.56245267
    ## [135,] 0.378804132 0.56847765
    ## [136,] 0.441511391 0.56521676
    ## [137,] 0.566405031 0.62076386
    ## [138,] 0.590737020 0.42251616
    ## [139,] 0.351680885 0.79144586
    ## [140,] 0.365959979 0.59010590
    ## [141,] 0.330142710 0.68174150
    ## [142,] 0.709779465 0.51662691
    ## [143,] 0.111168836 0.69185967
    ## [144,] 0.533249963 0.53213492
    ## [145,] 0.256520798 0.54643236
    ## [146,] 0.534877531 0.59935886
    ## [147,] 0.385853664 0.49648206
    ## [148,] 0.395399390 0.55090447
    ## [149,] 0.006486084 0.53011895
    ## [150,] 0.592277249 0.60767486
    ## [151,] 0.530411570 0.60849812
    ## [152,] 0.289687500 0.63358938
    ## [153,] 0.083693410 0.67593953
    ## [154,] 0.808296115 0.37980836
    ## [155,] 0.584642134 0.49832004
    ## [156,] 0.108275563 0.49204494
    ## [157,] 0.849994056 0.45994710
    ## [158,] 0.303894045 0.52303605
    ## [159,] 0.452650067 0.63931996
    ## [160,] 0.614526490 0.40887759
    ## [161,] 0.583808231 0.55622876
    ## [162,] 0.461422190 0.35189321
    ## [163,] 0.321376211 0.80349275
    ## [164,] 0.664626422 0.39926141
    ## [165,] 0.291469920 0.53965451
    ## [166,] 0.900229668 0.55963008
    ## [167,] 0.602496850 0.74701274
    ## [168,] 0.321498191 0.58363318
    ## [169,] 0.604212844 0.40453548
    ## [170,] 0.385112267 0.75319856
    ## [171,] 0.462177623 0.61252656
    ## [172,] 0.377499737 0.69437064
    ## [173,] 0.545083848 0.47623114
    ## [174,] 0.667444541 0.44860952
    ## [175,] 0.562687147 0.50408864
    ## [176,] 0.314781032 0.53484744
    ## [177,] 0.338289040 0.53888785
    ## [178,] 0.569110667 0.41745153
    ## [179,] 0.591356018 0.55227596
    ## [180,] 0.188950985 0.70273080
    ## [181,] 0.566550393 0.63138577
    ## [182,] 0.415403582 0.55157119
    ## [183,] 0.603612622 0.43077419
    ## [184,] 0.502351346 0.61066345
    ## [185,] 0.337150234 0.59965816
    ## [186,] 0.276822231 0.54168698
    ## [187,] 0.598823572 0.51393055
    ## [188,] 0.415486194 0.56357177
    ## [189,] 0.908820561 0.58578345
    ## [190,] 0.202950495 0.56366335
    ## [191,] 0.224018649 0.73865165
    ## [192,] 0.201979790 0.49820735
    ## [193,] 0.407419705 0.77849898
    ## [194,] 0.000000000 0.67784626
    ## [195,] 0.444763080 0.43709614
    ## [196,] 0.415085190 0.71947648
    ## [197,] 0.664417438 0.40138173
    ## [198,] 0.571280441 0.52499211
    ## [199,] 0.377931007 0.71180403
    ## [200,] 0.347388546 0.45911873

``` r
model <- keras_model_sequential()
```

    ## Loaded Tensorflow version 2.8.0

``` r
model %>%
  layer_flatten(input_shape = 2) %>%
  layer_dense(units = 10, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')
```

``` r
model %>% compile(
  optimizer = 'adam', 
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)
```

``` r
model %>% fit(xtrain, dat$y, epochs = 5, verbose = 2)
```