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


In addition there are a few variables that are not helpful for making a prediction, such as the row number, so we will be removing those as well. The time stamp would be helpful if the test data we seek to predict on was also in a time series, however since this is not the case we'll be removing it as well. We are also going to mutate the `classe` variable into a factor to help with training the models.  


The current training set will be split into a 60/40 training & validation set, respectively.    


# Model Selection  

## Importance of Variables  

We are curious of the importance of the variables within a random forest for discerning between all `classe`s, as well as between each error type and the correct method. 


To get an overview of the importance among the models that only look at the two `classe`s we're going to sum their values up and look at the highest values. To save on length from the output we're only going to display the highest 10 values for the summed importance and the importance from the model with all `classe`s present.


```
##                    Overall
## roll_belt         749.0897
## yaw_belt          534.6641
## pitch_forearm     469.3488
## magnet_dumbbell_z 456.6335
## magnet_dumbbell_y 414.6818
## pitch_belt        397.3756
## roll_forearm      353.7688
## magnet_dumbbell_x 288.7125
## magnet_belt_z     251.3844
## accel_belt_z      245.3884
```

```
##             Variable  Overall
## 1       roll_forearm 771.1728
## 2          roll_belt 648.6632
## 3  magnet_dumbbell_y 639.4598
## 4  magnet_dumbbell_z 628.5421
## 5      pitch_forearm 572.3641
## 6           yaw_belt 397.1896
## 7  magnet_dumbbell_x 368.8236
## 8         pitch_belt 309.1935
## 9   accel_dumbbell_y 291.0834
## 10     magnet_belt_y 279.3876
```

## Regularized Regression Model  
Seeing multiple readings from the same devices appearing in the top few variables we first considered using a Regularized Regression model. As this method would help restrict some varaibles so they don't overly factor into the model, such as readings for the belt which likely are only strong indicators of a `classe E` error.  

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2624  523  435  174  160
##          B  159 1158  152  134  395
##          C  186  306 1236  218  226
##          D  344  152  183 1233  223
##          E   35  140   48  171 1161
## 
## Overall Statistics
##                                           
##                Accuracy : 0.6294          
##                  95% CI : (0.6206, 0.6381)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.5288          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.7838  0.50812   0.6018   0.6389  0.53626
## Specificity            0.8467  0.91155   0.9037   0.9084  0.95901
## Pos Pred Value         0.6701  0.57958   0.5691   0.5775  0.74662
## Neg Pred Value         0.9079  0.88535   0.9148   0.9277  0.90177
## Prevalence             0.2843  0.19353   0.1744   0.1639  0.18385
## Detection Rate         0.2228  0.09834   0.1050   0.1047  0.09859
## Detection Prevalence   0.3325  0.16967   0.1844   0.1813  0.13205
## Balanced Accuracy      0.8152  0.70983   0.7527   0.7736  0.74763
```

The low in-sample accuracy of this model convinced us to not use it any further, as accuracy is always best within the training data, and 60% is not acceptable.   


The next approach that we considered was to use Linear Discriminant Analysis, as it would create decision boundaries which may be good at determining when a reading is going outside of the acceptable method for lifting weights and entering an area that indicates an error is being performed.  

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2734  337  191  110   86
##          B   74 1468  192   80  366
##          C  253  272 1367  232  196
##          D  278   83  257 1426  205
##          E    9  119   47   82 1312
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7054          
##                  95% CI : (0.6971, 0.7136)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6274          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8166   0.6441   0.6655   0.7389   0.6060
## Specificity            0.9141   0.9250   0.9020   0.9164   0.9733
## Pos Pred Value         0.7906   0.6734   0.5892   0.6341   0.8362
## Neg Pred Value         0.9262   0.9155   0.9273   0.9471   0.9164
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2322   0.1247   0.1161   0.1211   0.1114
## Detection Prevalence   0.2936   0.1851   0.1970   0.1910   0.1332
## Balanced Accuracy      0.8654   0.7846   0.7838   0.8276   0.7896
```

Although the above results were better than our regularized regression model, it still had a pretty low in sample accuracy, which was concerning. So on the same idea of creating decision boundaries we decided to create a random forest model.

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3346   15    0    0    0
##          B    2 2262    7    0    0
##          C    0    2 2044   14    1
##          D    0    0    3 1915    1
##          E    0    0    0    1 2163
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9961          
##                  95% CI : (0.9948, 0.9971)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9951          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9925   0.9951   0.9922   0.9991
## Specificity            0.9982   0.9991   0.9983   0.9996   0.9999
## Pos Pred Value         0.9955   0.9960   0.9918   0.9979   0.9995
## Neg Pred Value         0.9998   0.9982   0.9990   0.9985   0.9998
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2841   0.1921   0.1736   0.1626   0.1837
## Detection Prevalence   0.2854   0.1928   0.1750   0.1630   0.1838
## Balanced Accuracy      0.9988   0.9958   0.9967   0.9959   0.9995
```
This random forest model had a high in-sample accuracy, which is concerning as it may indicate overfitting. We will have to see if this is the case by using our validation set.


# Out of Sample Error  

We'll be testing the two models, linear discriminant analysis and random forest, with the validation subset that was taken from our training data previous to building the models.

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1111  131   86   45   27
##          B   25  596   72   31  145
##          C   82  108  562   84   82
##          D  111   32   94  566   72
##          E    4   55   16   33  527
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7158          
##                  95% CI : (0.7026, 0.7286)
##     No Information Rate : 0.2838          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6402          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8335   0.6464   0.6771   0.7457   0.6178
## Specificity            0.9141   0.9277   0.9079   0.9215   0.9719
## Pos Pred Value         0.7936   0.6858   0.6122   0.6469   0.8299
## Neg Pred Value         0.9327   0.9148   0.9291   0.9495   0.9197
## Prevalence             0.2838   0.1963   0.1767   0.1616   0.1816
## Detection Rate         0.2365   0.1269   0.1197   0.1205   0.1122
## Detection Prevalence   0.2981   0.1850   0.1954   0.1863   0.1352
## Balanced Accuracy      0.8738   0.7871   0.7925   0.8336   0.7949
```



```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1332    6    0    0    0
##          B    1  916    2    0    0
##          C    0    0  827    5    0
##          D    0    0    1  754    1
##          E    0    0    0    0  852
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9966          
##                  95% CI : (0.9945, 0.9981)
##     No Information Rate : 0.2838          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9957          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9992   0.9935   0.9964   0.9934   0.9988
## Specificity            0.9982   0.9992   0.9987   0.9995   1.0000
## Pos Pred Value         0.9955   0.9967   0.9940   0.9974   1.0000
## Neg Pred Value         0.9997   0.9984   0.9992   0.9987   0.9997
## Prevalence             0.2838   0.1963   0.1767   0.1616   0.1816
## Detection Rate         0.2836   0.1950   0.1761   0.1605   0.1814
## Detection Prevalence   0.2849   0.1957   0.1771   0.1610   0.1814
## Balanced Accuracy      0.9987   0.9963   0.9975   0.9965   0.9994
```
This is great to see, our random forest model had 100% accuracy in the out of sample set. As such we'll be selecting this one to predict the `classe` values of the test set.  



# Conclusion
Since we transformed the testing data into `tidyTest` when we transformed the training data we can make our predictions with that variation of the data now.  

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
