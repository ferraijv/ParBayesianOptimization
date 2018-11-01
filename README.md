
<!-- README.md is generated from README.Rmd. Please edit that file -->
ParBayesianOptimization
=======================

Machine learning projects will commonly require a user to "tune" a model's hyperparameters to find a good balance between bias and variance. Several tools are available in a data scientist's toolbox to handle this task, the most blunt of which is a grid search. A grid search gauges the model performance over a pre-defined set of hyperparameters without regard for past performance. As models increase in complexity and training time, grid searches become unwieldly.

Idealy, we would use the information from prior model evaluations to guide us in our future parameter searches. This is precisely the idea behind Bayesian Optimization, in which our prior response distribution is iteratively updated based on our best guess of where the best parameters are. The `ParBayesianOptimization` package does exactly this in the following process:

1.  Initial parameter-score pairs are found
2.  Gaussian Process is fit/updated
3.  Numerical methods are used to estimate the best parameter set
4.  New parameter-score pairs are found
5.  Repeat steps 2-4 until some stopping criteria is met

Installation
------------

You can install ParBayesianOptimization from github with:

``` r
# install.packages("devtools")
devtools::install_github("AnotherSamWilson/ParBayesianOptimization")
```

Practical Example
-----------------

In this example, we will be using the agaricus.train dataset provided in the XGBoost package. Here, we load the packages, data, and create a folds object to be used in the scoring function.

``` r
library("xgboost")
library("ParBayesianOptimization")

data(agaricus.train, package = "xgboost")

Folds <- list(  Fold1 = as.integer(seq(1,nrow(agaricus.train$data),by = 3))
                , Fold2 = as.integer(seq(2,nrow(agaricus.train$data),by = 3))
                , Fold3 = as.integer(seq(3,nrow(agaricus.train$data),by = 3)))
```

Now we need to define the scoring function. This function should, at a minimum, return a list with a `Score` element, which is the model evaluation metric we want to maximize. We can also retain other pieces of information created by the scoring function by including them as named elements of the returned list. In this case, we want to retain the optimal number of rounds determined by the `xgb.cv`:

``` r
scoringFunction <- function(max_depth, min_child_weight, subsample) {

  dtrain <- xgb.DMatrix(agaricus.train$data,label = agaricus.train$label)
  
  Pars <- list( booster = "gbtree"
                , eta = 0.01
                , max_depth = max_depth
                , min_child_weight = min_child_weight
                , subsample = subsample
                , objective = "binary:logistic"
                , eval_metric = "auc")

  xgbcv <- xgb.cv(params = Pars,
                  data = dtrain
                  , nround = 100
                  , folds = Folds
                  , prediction = TRUE
                  , showsd = TRUE
                  , early_stopping_rounds = 5
                  , maximize = TRUE
                  , verbose = 0)

  return(list(Score = max(xgbcv$evaluation_log$test_auc_mean)
             , nrounds = xgbcv$best_iteration
             )
         )
}
```

Some other objects we need to define are the bounds, GP kernel and acquisition function.

-   The `bounds` will tell our process its search space.
-   The kernel is passed to the `GauPro` function `GauPro_kernel_model` and defines the covariance function.
-   The acquisition function defines the utility we get from using a certain parameter set.

``` r
bounds <- list( max_depth = c(2L, 10L)
              , min_child_weight = c(1L, 100L)
              , subsample = c(0.25, 1))

kern <- "Matern52"

acq <- "ei"
```

We are now ready to put this all into the `BayesianOptimization` function.

``` r
ScoreResult <- BayesianOptimization(FUN = scoringFunction
                                  , bounds = bounds
                                  , initPoints = 10
                                  , bulkNew = 1
                                  , nIters = 12
                                  , kern = kern
                                  , acq = acq
                                  , kappa = 2.576
                                  , verbose = 1
                                  , parallel = FALSE)
#> 
#> Running initial scoring function 10 times in 1 thread(s).
#> 
#> Starting round number 1
#>   1) Fitting Gaussian process...
#>   2) Running global optimum search...
#>   3) Running scoring function 1 times in 1 thread(s)...
#> Starting round number 2
#>   1) Fitting Gaussian process...
#>   2) Running global optimum search...
#>   3) Running scoring function 1 times in 1 thread(s)...
```

The console informs us that the process initialized by running `scoringFunction` 10 times. It then fit a Gaussian process to the parameter-score pairs, found the global optimum of the acquisition function, and ran `scoringFunction` again. This process continued until we had 12 parameter-score pairs. You can interrogate the `ScoreResult` object to see the results:

``` r
ScoreResult$ScoreDT
#>     Iteration max_depth min_child_weight subsample Elapsed     Score nrounds
#>  1:         0         3               63 0.3648822    0.33 0.9723497       6
#>  2:         0         8               56 0.8793980    0.89 0.9897887      21
#>  3:         0         2               92 0.6521302    0.34 0.9770927       2
#>  4:         0         2               43 0.4545734    0.15 0.9779723       1
#>  5:         0         4               29 0.3914788    0.23 0.9898880       3
#>  6:         0         3               14 0.9288760    1.37 0.9971037      59
#>  7:         0         4               11 0.9976826    1.61 0.9995433      56
#>  8:         0        10                5 0.3882349    0.79 0.9993440      19
#>  9:         0         8                2 0.2575953    0.56 0.9999563      11
#> 10:         0         4               45 0.3494564    0.18 0.9786310       2
#> 11:         1        10                1 1.0000000    0.26 0.9984757       1
#> 12:         2         7                1 0.6452534    0.27 0.9984757       2
```

``` r
ScoreResult$BestPars
#>    Iteration max_depth min_child_weight subsample     Score nrounds elapsedSecs
#> 1:         0         8                2 0.2575953 0.9999563      11      8 secs
#> 2:         1         8                2 0.2575953 0.9999563      11     18 secs
#> 3:         2         8                2 0.2575953 0.9999563      11     29 secs
```
