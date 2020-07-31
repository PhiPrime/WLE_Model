---
title: "Report"
author: "Luke Coughlin"
output: 
        html_document:
                keep_md: true
---




# Summary  
This report aims to create a model that can be used to accurately predict if a user is performing an exercise correctly from a given test set. We approach this by splitting our training set into 60% training and 40% validation. We explore the data by looking at the importance of variables then selecting several models: regularized regression, linear discriminant analysis, and random forest. We check the out of sample accuracy to find the random forest model greater than 99.9% out of sample accuracy, as such we use it to predict the `classe`s of the test set.  


# Source Data  


```r
trurl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
trdest <- paste0(getwd(), "/Data/training.csv")
teurl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
tedest <- paste0(getwd(), "/Data/testing.csv")
download.file(trurl, trdest)
download.file(teurl, tedest)
training <- read.csv(trdest)
testing <- read.csv(tedest)
```
The data that we are using comes from Velloso, E. et. al. and contains a training set of 19622 observations of six participants performing ten repetitions of the Unilateral Dumbbell Biceps Curl in five different ways, indicated by the `classe` variable:  
A: Correctly performing the exercise  
B: Mistakenly throwing their elbows to the front  
C: Mistakenly lifting the dumbbell only halfway  
D: Mistakenly lowering the dumbbell only halfway  
E: Mistakenly throwing their hips to the front  

Citation of paper data is from: 
Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. **Qualitative Activity Recognition of Weight Lifting Exercises**. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013. 

## Tidying Data  
The testing data that we are given don't have any of the summary statistic variables that are available at the end of each window in the training data, as such we'll be removing these variables from both sets before proceeding.  

```r
library(tidyverse)
summaryStats <- names(which(colSums(is.na(testing)) == 20))
rawTrain <- select(training, -all_of(summaryStats))
rawTest <- select(testing, -all_of(summaryStats))
```

In addition there are a few variables that are not helpful for making a prediction, such as the row number, so we will be removing those as well. The time stamp would be helpful if the test data we seek to predict on was also in a time series, however since this is not the case we'll be removing it as well. We are also going to mutate the `classe` variable into a factor to help with training the models.  

```r
extVar <- names(rawTrain)[c(1:7)]
tidyTrain <- select(rawTrain, -all_of(extVar))
tidyTest <- select(rawTest, -all_of(extVar))
tidyTrain <- mutate(tidyTrain, classe = as.factor(classe))
```

The current training set will be split into a 60/40 training & validation set, respectively.    

```r
library(caret)
inTrain <- createDataPartition(tidyTrain$classe, p = 0.6, list = FALSE)
tidyTrain <- tidyTrain[inTrain, ]
tidyValid <- tidyTrain[-inTrain,]
```

# Model Selection  

## Importance of Variables  

We are curious of the importance of the variables within a random forest for discerning between all `classe`s, as well as between each error type and the correct method. 

```r
library(randomForest)
set.seed(1618)
allMod <- randomForest(classe~., data = tidyTrain)
abMod <- randomForest(classe~., data = 
                              filter(tidyTrain,
                                        as.character(classe) == "A"|
                                        as.character(classe) == "B") %>%
                              mutate(classe = factor(classe)))
acMod <- randomForest(classe~., data = 
                              filter(tidyTrain,
                                        as.character(classe) == "A"|
                                        as.character(classe) == "C") %>%
                              mutate(classe = factor(classe)))
adMod <- randomForest(classe~., data = 
                              filter(tidyTrain,
                                        as.character(classe) == "A"|
                                        as.character(classe) == "D") %>%
                              mutate(classe = factor(classe)))
aeMod <- randomForest(classe~., data = 
                              filter(tidyTrain,
                                        as.character(classe) == "A"|
                                        as.character(classe) == "E") %>%
                              mutate(classe = factor(classe)))
```

To get an overview of the importance among the models that only look at the two `classe`s we're going to sum their values up and look at the highest values. To save on length from the output we're only going to display the highest 10 values for the summed importance and the importance from the model with all `classe`s present.


```r
allImp <-varImp(allMod)
head(arrange(allImp, desc(Overall)), 10)
```

```
##                    Overall
## roll_belt         749.1289
## yaw_belt          525.0582
## pitch_forearm     461.7223
## magnet_dumbbell_z 457.8742
## pitch_belt        406.6559
## magnet_dumbbell_y 390.4589
## roll_forearm      360.1755
## magnet_dumbbell_x 297.3519
## accel_dumbbell_y  244.8891
## roll_dumbbell     238.7349
```

```r
abImp <- varImp(abMod)
acImp <- varImp(acMod)
adImp <- varImp(adMod)
aeImp <- varImp(aeMod)

impSums <- data.frame(
        Variable = rownames(abImp),
        Overall = rowSums(cbind(abImp$Overall, 
                                acImp$Overall,
                                adImp$Overall, 
                                aeImp$Overall)))
head(arrange(impSums, desc(Overall)), 10)
```

```
##             Variable  Overall
## 1       roll_forearm 779.0289
## 2  magnet_dumbbell_z 648.8098
## 3          roll_belt 621.3672
## 4  magnet_dumbbell_y 612.5064
## 5      pitch_forearm 541.6621
## 6           yaw_belt 399.7962
## 7  magnet_dumbbell_x 353.2228
## 8   accel_dumbbell_y 331.4509
## 9         pitch_belt 307.8990
## 10     magnet_belt_y 279.6962
```

## Regularized Regression Model  
Seeing multiple readings from the same devices appearing in the top few variables we first considered using a Regularized Regression model. As this method would help restrict some varaibles so they don't overly factor into the model, such as readings for the belt which likely are only strong indicators of a `classe E` error.  

```r
fileLoc <- "./Data/rrMod.rds"
if (!file.exists(fileLoc)) { #If model's file not found
set.seed(1618)
tr <- as.data.frame(tidyTrain)
control <- trainControl(method = "repeatedcv",
                        number = 5,
                        repeats = 5,
                        search = "random")
rrMod <- train(classe ~., tr, method = "glmnet",
               preProc = c("center", "scale"),
               trControl = control)
saveRDS(rrMod, fileLoc)
} else {#Read from file
        rrMod <- readRDS(fileLoc)
}

confusionMatrix(predict(rrMod, tidyTrain), tidyTrain$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2617  528  454  173  155
##          B  153 1135  158  141  354
##          C  196  310 1220  212  237
##          D  339  150  168 1245  239
##          E   43  156   54  159 1180
## 
## Overall Statistics
##                                           
##                Accuracy : 0.6281          
##                  95% CI : (0.6193, 0.6369)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.5272          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.7817  0.49803   0.5940   0.6451   0.5450
## Specificity            0.8446  0.91513   0.9018   0.9090   0.9571
## Pos Pred Value         0.6664  0.58475   0.5609   0.5815   0.7412
## Neg Pred Value         0.9069  0.88368   0.9131   0.9289   0.9033
## Prevalence             0.2843  0.19353   0.1744   0.1639   0.1838
## Detection Rate         0.2222  0.09638   0.1036   0.1057   0.1002
## Detection Prevalence   0.3335  0.16483   0.1847   0.1818   0.1352
## Balanced Accuracy      0.8131  0.70658   0.7479   0.7770   0.7511
```

The low in-sample accuracy of this model convinced us to not use it any further, as accuracy is always best within the training data, and 60% is not acceptable.   


The next approach that we considered was to use Linear Discriminant Analysis, as it would create decision boundaries which may be good at determining when a reading is going outside of the acceptable method for lifting weights and entering an area that indicates an error is being performed.  

```r
ldamodel <- train(classe ~ ., data = tidyTrain, method = "lda")
confusionMatrix(predict(ldamodel, newdata = tidyTrain), tidyTrain$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2706  356  198  122   83
##          B   77 1455  224   78  355
##          C  282  272 1323  217  191
##          D  270   78  258 1445  202
##          E   13  118   51   68 1334
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7017          
##                  95% CI : (0.6933, 0.7099)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6226          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8082   0.6384   0.6441   0.7487   0.6162
## Specificity            0.9099   0.9227   0.9010   0.9179   0.9740
## Pos Pred Value         0.7810   0.6647   0.5790   0.6414   0.8422
## Neg Pred Value         0.9228   0.9141   0.9230   0.9491   0.9185
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2298   0.1236   0.1123   0.1227   0.1133
## Detection Prevalence   0.2942   0.1859   0.1940   0.1913   0.1345
## Balanced Accuracy      0.8591   0.7806   0.7726   0.8333   0.7951
```

Although the above results were better than our regularized regression model, it still had a pretty low in sample accuracy, which was concerning. So on the same idea of creating decision boundaries we decided to create a random forest model.

```r
fileLoc <- "./Data/rfMod.rds"
if (!file.exists(fileLoc)) { #If model's file not found
set.seed(1618033)
rfmodel <- train(classe~., data = tidyTrain, method = "rf")

saveRDS(rfmodel, fileLoc)
} else {#Read from file
        rfmodel <- readRDS(fileLoc)
}

confusionMatrix(predict(rfmodel, newdata = tidyTrain), tidyTrain$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3346    9    0    0    0
##          B    1 2267    6    0    0
##          C    0    3 2041   13    1
##          D    0    0    7 1915    1
##          E    1    0    0    2 2163
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9963         
##                  95% CI : (0.995, 0.9973)
##     No Information Rate : 0.2843         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9953         
##                                          
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9947   0.9937   0.9922   0.9991
## Specificity            0.9989   0.9993   0.9983   0.9992   0.9997
## Pos Pred Value         0.9973   0.9969   0.9917   0.9958   0.9986
## Neg Pred Value         0.9998   0.9987   0.9987   0.9985   0.9998
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2841   0.1925   0.1733   0.1626   0.1837
## Detection Prevalence   0.2849   0.1931   0.1748   0.1633   0.1839
## Balanced Accuracy      0.9992   0.9970   0.9960   0.9957   0.9994
```
This random forest model had a high in-sample accuracy, which is concerning as it may indicate overfitting. We will have to see if this is the case by using our validation set.


# Out of Sample Error  

We'll be testing the two models, linear discriminant analysis and random forest, with the validation subset that was taken from our training data previous to building the models.

```r
confusionMatrix(predict(ldamodel, newdata = tidyValid), tidyValid$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1066  151   91   50   32
##          B   31  584   84   34  146
##          C  106  117  553   81   71
##          D  114   32   92  562   84
##          E    5   41   20   33  525
## 
## Overall Statistics
##                                           
##                Accuracy : 0.6993          
##                  95% CI : (0.6859, 0.7123)
##     No Information Rate : 0.281           
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6196          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8064   0.6314   0.6583   0.7395   0.6119
## Specificity            0.9042   0.9220   0.9030   0.9184   0.9743
## Pos Pred Value         0.7669   0.6644   0.5959   0.6357   0.8413
## Neg Pred Value         0.9228   0.9109   0.9240   0.9482   0.9184
## Prevalence             0.2810   0.1966   0.1785   0.1615   0.1824
## Detection Rate         0.2266   0.1241   0.1175   0.1194   0.1116
## Detection Prevalence   0.2954   0.1868   0.1972   0.1879   0.1326
## Balanced Accuracy      0.8553   0.7767   0.7807   0.8289   0.7931
```



```r
confusionMatrix(predict(rfmodel, newdata = tidyValid), tidyValid$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1322    1    0    0    0
##          B    0  922    5    0    0
##          C    0    2  833    8    1
##          D    0    0    2  751    0
##          E    0    0    0    1  857
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9957          
##                  95% CI : (0.9934, 0.9974)
##     No Information Rate : 0.281           
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9946          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9968   0.9917   0.9882   0.9988
## Specificity            0.9997   0.9987   0.9972   0.9995   0.9997
## Pos Pred Value         0.9992   0.9946   0.9870   0.9973   0.9988
## Neg Pred Value         1.0000   0.9992   0.9982   0.9977   0.9997
## Prevalence             0.2810   0.1966   0.1785   0.1615   0.1824
## Detection Rate         0.2810   0.1960   0.1770   0.1596   0.1821
## Detection Prevalence   0.2812   0.1970   0.1794   0.1600   0.1824
## Balanced Accuracy      0.9999   0.9977   0.9944   0.9938   0.9993
```
This is great to see, our random forest model had 100% accuracy in the out of sample set. As such we'll be selecting this one to predict the `classe` values of the test set.  



# Conclusion
Since we transformed the testing data into `tidyTest` when we transformed the training data we can make our predictions with that variation of the data now.  

```r
testPred <- predict(rfmodel, newdata = tidyTest)
testPred
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
