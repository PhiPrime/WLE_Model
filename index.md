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
## roll_belt         746.4611
## yaw_belt          517.6173
## pitch_forearm     476.8380
## magnet_dumbbell_z 468.7751
## magnet_dumbbell_y 402.1081
## pitch_belt        401.8805
## roll_forearm      373.5585
## magnet_dumbbell_x 292.7264
## accel_dumbbell_y  256.9008
## roll_dumbbell     250.0916
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
## 1       roll_forearm 780.9628
## 2  magnet_dumbbell_z 642.2285
## 3          roll_belt 629.7583
## 4  magnet_dumbbell_y 600.2024
## 5      pitch_forearm 585.4433
## 6           yaw_belt 385.6272
## 7  magnet_dumbbell_x 349.7801
## 8   accel_dumbbell_y 333.3065
## 9         pitch_belt 304.0654
## 10     magnet_belt_y 295.6131
```

## Regularized Regression Model  
Seeing multiple readings from the same devices appearing in the top few variables we first considered using a Regularized Regression model. As this method would help restrict some variables so they don't overly factor into the model, such as readings for the belt which likely are only strong indicators of a `classe E` error.  

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
##          A 2621  511  429  188  155
##          B  162 1167  171  137  367
##          C  191  311 1228  202  254
##          D  335  150  170 1228  223
##          E   39  140   56  175 1166
## 
## Overall Statistics
##                                          
##                Accuracy : 0.6292         
##                  95% CI : (0.6204, 0.638)
##     No Information Rate : 0.2843         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.5286         
##                                          
##  Mcnemar's Test P-Value : < 2.2e-16      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.7829   0.5121   0.5979   0.6363  0.53857
## Specificity            0.8478   0.9119   0.9015   0.9108  0.95734
## Pos Pred Value         0.6714   0.5823   0.5618   0.5831  0.73985
## Neg Pred Value         0.9076   0.8862   0.9139   0.9274  0.90206
## Prevalence             0.2843   0.1935   0.1744   0.1639  0.18385
## Detection Rate         0.2226   0.0991   0.1043   0.1043  0.09901
## Detection Prevalence   0.3315   0.1702   0.1856   0.1788  0.13383
## Balanced Accuracy      0.8153   0.7120   0.7497   0.7735  0.74795
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
##          A 2758  354  227  114   73
##          B   65 1448  186   80  342
##          C  264  284 1348  228  198
##          D  251   84  233 1421  215
##          E   10  109   60   87 1337
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7058          
##                  95% CI : (0.6975, 0.7141)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6276          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8238   0.6354   0.6563   0.7363   0.6176
## Specificity            0.9089   0.9291   0.8998   0.9205   0.9723
## Pos Pred Value         0.7822   0.6827   0.5805   0.6447   0.8341
## Neg Pred Value         0.9285   0.9139   0.9253   0.9468   0.9186
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2342   0.1230   0.1145   0.1207   0.1135
## Detection Prevalence   0.2994   0.1801   0.1972   0.1872   0.1361
## Balanced Accuracy      0.8663   0.7823   0.7780   0.8284   0.7949
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
##          A 3347   14    0    0    0
##          B    1 2263    5    0    0
##          C    0    2 2045   12    1
##          D    0    0    4 1915    0
##          E    0    0    0    3 2164
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9964          
##                  95% CI : (0.9952, 0.9974)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9955          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9997   0.9930   0.9956   0.9922   0.9995
## Specificity            0.9983   0.9994   0.9985   0.9996   0.9997
## Pos Pred Value         0.9958   0.9974   0.9927   0.9979   0.9986
## Neg Pred Value         0.9999   0.9983   0.9991   0.9985   0.9999
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2842   0.1922   0.1737   0.1626   0.1838
## Detection Prevalence   0.2854   0.1927   0.1749   0.1630   0.1840
## Balanced Accuracy      0.9990   0.9962   0.9970   0.9959   0.9996
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
##          A 1106  149   97   49   31
##          B   30  582   84   38  125
##          C   97  114  525   97   85
##          D   97   34   84  578   92
##          E    3   36   18   26  535
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7059          
##                  95% CI : (0.6926, 0.7188)
##     No Information Rate : 0.2829          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6275          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8297   0.6361   0.6498   0.7335   0.6164
## Specificity            0.9035   0.9270   0.8993   0.9218   0.9784
## Pos Pred Value         0.7723   0.6775   0.5719   0.6531   0.8657
## Neg Pred Value         0.9308   0.9136   0.9254   0.9451   0.9187
## Prevalence             0.2829   0.1942   0.1715   0.1672   0.1842
## Detection Rate         0.2347   0.1235   0.1114   0.1227   0.1135
## Detection Prevalence   0.3039   0.1823   0.1948   0.1878   0.1312
## Balanced Accuracy      0.8666   0.7816   0.7745   0.8276   0.7974
```



```r
confusionMatrix(predict(rfmodel, newdata = tidyValid), tidyValid$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1332    5    0    0    0
##          B    1  910    2    0    0
##          C    0    0  804    7    0
##          D    0    0    2  780    0
##          E    0    0    0    1  868
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9962         
##                  95% CI : (0.994, 0.9977)
##     No Information Rate : 0.2829         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9952         
##                                          
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9992   0.9945   0.9950   0.9898   1.0000
## Specificity            0.9985   0.9992   0.9982   0.9995   0.9997
## Pos Pred Value         0.9963   0.9967   0.9914   0.9974   0.9988
## Neg Pred Value         0.9997   0.9987   0.9990   0.9980   1.0000
## Prevalence             0.2829   0.1942   0.1715   0.1672   0.1842
## Detection Rate         0.2827   0.1931   0.1706   0.1655   0.1842
## Detection Prevalence   0.2837   0.1938   0.1721   0.1660   0.1844
## Balanced Accuracy      0.9989   0.9969   0.9966   0.9947   0.9999
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
