Practical ML Project
================
Farzad Ehtemam

Executive summary
-----------------

-   Weight Lifting Exercises Dataset (WLED) was used to develope a context specific model for human activity recongnition (HAR) to differentiate between five types of activity.
-   Exploratory data analysis showed that out of 160 measured/calculated variables in the training set, 100 had more than 25% of their observations missing (i.e., NA). These variables were removed from analysis since high amounts of missing values will make these variables poor predictors.
-   The final out of bag error (OOB) was 0.29% using a random forest method. This is a satisfactory precision to predict the 20 cases from the test dataset for the quiz.

Exploratory data analysis and data cleaning
-------------------------------------------

After downloading the data we read the data into a training and a test dataset:

``` r
traind <- read.csv('pml-training.csv', header = T, na.strings = c ('NA', ''))
testd <- read.csv('pml-testing.csv', header = T, na.strings = c ('NA', ''))
```

We take a look at missing values for columns with more than 25% of missing data:

``` r
miss_col <- apply(traind, 2, function(x) mean(is.na(x))) > 0.25
```

This shows that 100 columns have more than 25% missing values. We remove these columns. We also remove the first 7 columns as these variables are for identification purposes:

``` r
miss_col <- apply(traind, 2, function(x) mean(is.na(x))) > 0.25
traind2 <- traind[, -which(miss_col, miss_col == FALSE)]
# Cleaned training dataset
traind_cl <- traind2[,8:60]
```

This leaves of with 53 variables to use as predictors. Before selecting a model we will take a look at possible correlations between these predictors using corrplot:

``` r
library(corrplot)
```

    ## corrplot 0.84 loaded

``` r
corrMat <- cor(traind_cl[,-53])
corrplot(corrMat, method = "circle", type = "lower", tl.cex = 0.6)
```

![](PML_project_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-4-1.png)

As it can be seen in the figure only a few variables show some correlation. This means that it is unlikely using dimentionality reduction techniques (e.g., PCA) results in substantial improvement in final results. So we move forward with 53 predictors.

Model selection, tuning and training
------------------------------------

Since there is no direct linear relationship between accelerations and the error of the movement, using a method such as random forest that is sutable for nonlinear problems is good first guess for our model. If the model performs unsatisfactory we can mix the model with other methods. We will construct our random forest model using some default values. We will take advantage of parallel processing and we apply k-fold cross-validation with k=5 and we set mtry=7. Depending on the performance of the model we can tune these values further to improve the accuracy.

``` r
set.seed(42)
library(doParallel)
cl <- makePSOCKcluster(detectCores() - 1)
registerDoParallel(cl)
library(caret)
modFit <- train(classe ~ ., data=traind_cl, method="rf", trControl=trainControl(method="cv", number=5), tuneGrid=data.frame(mtry=7), na.action=na.exclude)
```

Conclusion
----------

The results showed OOB of 0.29 (i.e., accuracy of 99.71) with the initial values used. This is a satisfactory results and further tuning is unnecessary. Thus a random forest model with 53 predictors achieved the desired performance and we will use this model with the test dataset (predict(modFit, newdata = testd)) to answer the quiz.
