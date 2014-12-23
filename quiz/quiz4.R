### Practical ML Quiz 4
##Q1
#data preparation
library(ElemStatLearn)
library(caret)
library(randomForest)
library(glm)
train <- vowel.train
test <- vowel.test 
table(vowel.train$y)
train$y <-as.factor(train$y)
test$y <-as.factor(test$y)
set.seed(33833)
#Random Forest prediction
modRF <- train(y~., data = train, method = "rf", prox = TRUE)
predRF <- predict(modRF, test)
test$predRight <- predRF==test$y
#table(predRF, test$y)
sum(test$predRight)/dim(test)[1]
#boosting
modBoost <- train(y~., data = train, method = "gbm", verbose = FALSE)
#print(modBoost)
predBoost <- predict(modBoost, test)
test$predRight2 <- predBoost==test$y
sum(test$predRight2)/dim(test)[1]
#Mixed model not calculated

##Q2
library(caret)
library(gbm)
set.seed(3433)
library(AppliedPredictiveModeling)
library(MASS)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]
set.seed(62433)
#train rf
mod1 <- train(diagnosis~., data = training, method = "rf")
pred1 <- predict(mod1, testing)
predRight1 <- pred1==testing$diagnosis
sum(predRight1)/length(predRight1)
mod2 <- train(diagnosis~., data = training, method = "gbm")
pred2 <- predict(mod2, testing)
predRight2 <- pred2==testing$diagnosis
sum(predRight2)/length(predRight2)
mod3 <- train(diagnosis~., data = training, method = "lda")
pred3 <- predict(mod3, testing)
predRight3 <- pred3==testing$diagnosis
sum(predRight3)/length(predRight3)
predDF <- data.frame(pred1, pred2, pred3, diagnosis=testing$diagnosis)
combModFit <- train(diagnosis~., data = predDF, method = "rf")
combPred <- predict(combModFit, predDF)
predRight <- combPred==testing$diagnosis
sum(predRight)/length(predRight)
#RandFores, gbm,       lda,       combined
#0.7682927, 0.7926829, 0.7682927, 0.7926829

#Q3
set.seed(3523)
library(AppliedPredictiveModeling)
library(lars)
library(elasticnet)
data(concrete)
inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]
training = concrete[ inTrain,]
testing = concrete[-inTrain,]
set.seed(233)
#var <- names(training) %in% c("CompressiveStrength")
#x <- training[!var]
#object <- enet(as.matrix(x), as.vector(training$CompressiveStrength),lambda=1)
#png("now.png", width = 1280, height = 780)
#par(mar=c(1,1,1,1))
#plot(object, xvar = "step")
#dev.off()
lassoFit <- train( training$CompressiveStrength ~ ., method="lasso", data=training)
lassoPred <- predict(lassoFit,testing)
plot.enet(lassoFit$finalModel, xvar="penalty", use.color=T)

#Q4####
library(lubridate)
library(forecast)
dat = read.csv("./gaData.csv")
training = dat[year(dat$date)==2011,]
tstrain = ts(training$visitsTumblr)
testing = dat[year(dat$date)>2011,]
test = dat[year(dat$date) > 2011,]
pred <- forecast(fit, h=length(test$visitsTumblr),level=c(80,95))
fcast <- forecast(fit)
plot(fcast)
accuracy(fcast,test$visitsTumblr)
modBats <- bats(tstrain)
pred <- forecast(modBats, h=length(testing$visitsTumblr),level=c(80,95))
accuracy <- 1-sum(testing$visitsTumblr>pred$upper[,2])/length(testing$visitsTumblr)
accuracy <- 1-sum(test$visitsTumblr>pred$upper[,2])/length(test$visitsTumblr)


#Q5
library(e1071)
set.seed(3523)
library(AppliedPredictiveModeling)
library(caret)
data(concrete)
inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]
training = concrete[ inTrain,]
testing = concrete[-inTrain,]
set.seed(325)
svmFit <- svm(CompressiveStrength ~ ., data = training)
svmPred <- predict(svmFit,testing)
accuracy(svmPred, testing$CompressiveStrength)

