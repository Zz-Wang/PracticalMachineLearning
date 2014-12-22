Practical Machine Learning
========================================================
## Zhizheng Wang

This is the R Markdown document for the Practical Machine Learning project.
This assignment makes use of data collected from armband device Fitbit, which makes it now possible to collect a large amount of data about personal activity relatively inexpensively. Fitbit is part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, I will use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. Then I will build random forest model to predict/classify 20 test individuals.


```r
opts_chunk$set(echo=TRUE, results = "asis", cache=TRUE, message=FALSE, warning=FALSE)
```

### Step.1 Loading the data and packages needed
First we need to load in the data and packages into R. 


```r
#load packages
library(lattice)
library(ggplot2)
library(caret)
library(rpart)
library(rattle)
library(Hmisc)
library(randomForest)
library(foreach)
library(doParallel)
library(xtable)
#load the dataset
rm(list = ls())
setwd("C:/Users/User/Documents/R/project")
train_full <- read.csv("pml-training.csv", header = TRUE, na.strings=c("#DIV/0!"))
test_full <- read.csv("pml-testing.csv", header = TRUE, na.strings=c("#DIV/0!"))
#set seed
set.seed(12345)
```
### Step.2 Looking at the data and Preprocessing
Then, we can look at all the variables in a little bit detail. We can find the y variable classe has five classes. From column 8th to the end the data looks numeric and we cast them to numeric form. Also, the first column is simply the row number, which would leave to deviation. We need to throw it away. Also we leave out the variables whose NA value are more than 70 percent. Besides, we check if the variable has any variation, in which here we don't throw any variable away. 


```r
table(train_full$classe)
```


   A    B    C    D    E 
5580 3797 3422 3216 3607 

```r
for(i in c(8:ncol(train_full)-1)) {train_full[,i] = as.numeric(as.character(train_full[,i]))}
train <- train_full[c(-1)]
train_copy <- train
#leave all variables with NA more than 70%
for(i in 1:dim(train_copy)[2]) {
        if( sum( is.na( train_copy[, i] ) ) / dim(train)[1] >= .7 ) {
                train_copy[, i] <- NA 
        }
}
train <- train_copy[,colSums(is.na(train_copy))<nrow(train_copy)]
#check if there exist any variance in all the variables
#count = 0
#compare <- function(v) all(sapply( as.list(v[-1]), 
#                                   FUN=function(z) {identical(z, v[1])}))
#for(i in 1:dim(train)[2]) {
#        if(compare(train[ ,i]) == TRUE) {
#                count = count+1
#        }
#}
#count = 0 no column is removed
```

### step.3 Splitting the data and Buiding the model
We split the dataset into training and testing(crossvalidation), and build 5 random forests with 150 trees each. We use parallel processing to speedup. We present error with confusion matrix for both training and testing.


```r
#Splicing
inTrain <- createDataPartition(y = train$classe, p = 0.7, list = FALSE)
training <- train[inTrain, ]
crossValidation <- train[-inTrain, ]
dim(training); dim(crossValidation)
```

[1] 13737    59
[1] 5885   59

```r
#Random Forest
registerDoParallel()
x <- training[-ncol(training)]
y <- training$classe

rf <- foreach(ntree=rep(150, 6), .combine=randomForest::combine, .packages='randomForest') %dopar% {
randomForest(x, y, ntree=ntree) 
}
#Results
predictions1 <- predict(rf, newdata=training)
xx1 <- xtable(confusionMatrix(predictions1,training$classe))
```

```
## Error in UseMethod("xtable"): no applicable method for 'xtable' applied to an object of class "confusionMatrix"
```

```r
predictions2 <- predict(rf, newdata=crossValidation)
confusionMatrix(predictions2,crossValidation$classe)
```

Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1674    1    0    0    0
         B    0 1138    2    0    0
         C    0    0 1021    1    0
         D    0    0    3  963    2
         E    0    0    0    0 1080

Overall Statistics
                                          
               Accuracy : 0.9985          
                 95% CI : (0.9971, 0.9993)
    No Information Rate : 0.2845          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.9981          
 Mcnemar's Test P-Value : NA              

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            1.0000   0.9991   0.9951   0.9990   0.9982
Specificity            0.9998   0.9996   0.9998   0.9990   1.0000
Pos Pred Value         0.9994   0.9982   0.9990   0.9948   1.0000
Neg Pred Value         1.0000   0.9998   0.9990   0.9998   0.9996
Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
Detection Rate         0.2845   0.1934   0.1735   0.1636   0.1835
Detection Prevalence   0.2846   0.1937   0.1737   0.1645   0.1835
Balanced Accuracy      0.9999   0.9994   0.9975   0.9990   0.9991

### step.4 Conclustion
As we see from the results above, prediction is pretty accurate and we use this model to predict the test set given and submit with the code given. All predictions are correct!

```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
```



