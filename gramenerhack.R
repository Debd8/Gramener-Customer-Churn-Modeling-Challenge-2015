## Reading the Training data into R

gram_train <- read.csv(".../Gramener_Ideatory/GramenerTraining.csv",header = T, stringsAsFactors = F)

## Converting some variables into categorical/factor

cols <- c("Int.l.Plan", "Message.Plan", "State", "Area.Code", "Churn")
gram_train[,cols] <- data.frame(apply(gram_train[cols], 2, as.factor))

## A hit and trial on feature selection algorithms
## Random forest algorithm

library(randomForest)
set.seed(123)
gram.rf <- randomForest(Churn~.-Phone, data = gram_train, mtry = 9, importance = T)
varImpPlot(gram.rf)

## Random forest with repeated cross validation

library(caret)
set.seed(123)
control <- trainControl(method="repeatedcv", number=10, repeats=3)
model1 <- train(Churn~.-Phone, data=gram_train, method="rf", preProcess="scale", trControl=control, importance = T)
importance1 <- varImp(model1, scale=FALSE)
plot(importance1)

## Recursive feature elimination with cross validation

set.seed(123)
control1 <- rfeControl(functions=rfFuncs, method="cv", number=10)
model2 <- rfe(gram_train[,1:19], gram_train[,21], sizes=c(1:19), rfeControl=control1)
plot(model2, type=c("g", "o"))

## Boruta algorithm

library(Boruta)
set.seed(123)
gram.boruta <- Boruta(Churn~.-Phone, data = gram_train)
plotImpHistory(gram.boruta)
boruta.final <- attStats(gram.boruta)

## Dropping variables based on the results of Boruta algorithm

drop <- c("Account.Length..Weeks.", "Day.Calls", "Eve.Calls", "Night.Calls", "State", "Area.Code", "Phone")
gram_train_final <- gram_train[,!names(gram_train) %in% drop]

## A hit and trial on model fitting
## Complementary log-log model with repeated cross validation

library(e1071)
set.seed(123)
tc <- trainControl(method="repeatedcv", number=7, repeats=10, savePredictions = T)
gram.cloglog <- train(Churn~., data=gram_train_final, method="glm", family = binomial(link = cloglog), trControl = tc)
gram.cloglog.pred <- gram.cloglog$finalModel$fitted.values
gram.cloglog.predt <- function(t) ifelse(gram.cloglog.pred > t ,1,0)
Accuracy <- function(t) confusionMatrix(gram.cloglog.predt(t),gram_train_final[,14])$overall[1]
interval <- seq(0.1,1,by=0.01)
funcs <- list()
funcs <- unlist(lapply(interval, Accuracy))

## Random forest model with repeated cross validation

gram.rf <- train(Churn~., data=gram_train_final, method="rf", trControl = tc, ntree = 5000, prox = T)

## Adaboost model with cross validation

tc <- trainControl(method="cv", number=7)
Grid <- expand.grid(maxdepth=25,nu=0.01,iter=100)
gram.ada <- train(Churn~.-Phone, data=gram_train, method="ada",trControl= tc,tuneGrid=Grid)

## Reading the Evaluation data into R and retaining the variables as in the training set

gram_test <- read.csv(".../Gramener_Ideatory/GramenerTesting.csv",header = T, stringsAsFactors = F)
gram_test$Churn <- NULL
gram_test_final <- gram_test[,!names(gram_test) %in% drop]
nocols <- c("Int.l.Plan", "Message.Plan")
gram_test_final[,nocols] <- data.frame(apply(gram_test_final[nocols], 2, as.factor))

## Predicting on the evaluation set

predict.rf <- predict(gram.rf, gram_test_final, type = "prob")
predict.ada <- predict(gram.ada, gram_test_final, type = "prob")

## Creating an ensemble model

predict.final <- (0.3 * predict.rf + 0.7 * predict.ada)

## Converting the class probabilities to class labels

predicted_Labels <- ifelse(predict.final > 0.3, 1, 0)
gram_test_final <- cbind(gram_test, predicted_Labels)

## Final data for submission

final_submission <- gram_test_final[,19:21]
