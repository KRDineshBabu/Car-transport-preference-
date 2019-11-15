setwd("C:\\Users\\asckrdb\\Dropbox\\Great Learning\\Predictive Modeling\\Week 5")
car_dataset <- read.csv("Cars-dataset.csv", header = TRUE)

#EDA on the data 

View(car_dataset)
str(car_dataset)
summary(car_dataset)
sum(is.na(car_dataset))
library("DataExplorer")
DataExplorer::create_report(car_dataset[,-9])
plot(car_dataset)
car_boxplot <- boxplot(car_dataset)
library("corrplot")
set.seed(1234)
cor(car_dataset[, c(1,3:8)])
corrplot(cor(car_dataset[, c(1,3:8)]), method = "number", type = "lower")

#data preparation

car_dataset <- na.omit(car_dataset)
sum(is.na(car_dataset))
car_boxplot <- boxplot(car_dataset)


# quantile(car_dataset$Salary, c(0.25,0.90))
# temp<-lapply(car_dataset[,-c(2:4,8,9 )],function(x) quantile(x,probs = c(0.0,0.25,0.50, 0.75,0.8, 0.85, 0.9, 0.95,0.97,0.98,0.99,1.00),na.rm = TRUE))
# temp<-as.data.frame(temp)
# View(temp)
# 
car_dataset_new <- car_dataset

capoutlier <- function(x) {
  qnt <- quantile(x, probs=c(.25, .75), na.rm = T)
  caps <- quantile(x, probs=c(.05, .95), na.rm = T)
  H <- 1.5 * IQR(x, na.rm = T)
  x[x < (qnt[1] - H)] <- caps[1]
  x[x > (qnt[2] + H)] <- caps[2]
  return(x)
}

car_dataset_new$Age <- capoutlier(car_dataset_new$Age)
car_dataset_new$Work.Exp <- capoutlier(car_dataset_new$Work.Exp)
car_dataset_new$Salary <- capoutlier(car_dataset_new$Salary)
car_dataset_new$Distance <- capoutlier(car_dataset_new$Distance)

car_dataset_new$Gender <- as.numeric(car_dataset_new$Gender)

str(car_dataset_new)
boxplot(car_dataset_new)
summary(car_dataset_new)
plot(car_dataset_new)

library(caTools)

set.seed(123)
car_dataset_new[,c(1:8)] <- scale(car_dataset_new[,c(1:8)])
split = sample.split(car_dataset_new$Transport, SplitRatio = 0.7)
training_set = subset(car_dataset_new, split == TRUE)
test_set = subset(car_dataset_new, split == FALSE)


#knn


library(caret)
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
set.seed(3333)
knn_fit <- train(Transport~., data = training_set, method = "knn", trControl=trctrl)
knn_fit
plot(knn_fit)
#test_knn
test_knn <- predict(knn_fit, newdata = test_set)
confusionMatrix(knn_fit)
confusionMatrix(test_knn, test_set[,9])


#naive bayes 
xtrain <- training_set[,-9]
ytrain <- training_set$Transport
xtest <- test_set[,-9]
ytest <- test_set$Transport

library(e1071)
nb <- naiveBayes(xtrain, ytrain)
nb_pred <- predict(nb, xtest)
table<- predict(nb, xtrain)

confusionMatrix(table, ytrain)
confusionMatrix(nb_pred, ytest)


#logistic regression
library(nnet)

mlr <- multinom(Transport~., training_set)
summary(mlr)
prob_mlr_train <- predict(mlr, training_set)
confusionMatrix(prob_mlr_train, ytrain)
prob_mlr <- predict(mlr, test_set)
confusionMatrix(prob_mlr, test_set$Transport)



#bagging for KNN model

set.seed(33)
library(ipred)

bag<- bagging(Transport~., data = training_set, nbagg = 500, coob = T)
pred_bag_train <- predict(bag, training_set)
confusionMatrix(pred_bag_train, ytrain)
pred_bag <- predict(bag, test_set)
cm_bag <- confusionMatrix(pred_bag, test_set$Transport)
cm_bag



#boosting
#install.packages('gbm')
library(gbm)          # basic implementation using AdaBoost
#install.packages('xgboost')
library(xgboost)

mod_gbm = gbm(Transport ~.,
              data = training_set,
              distribution = "multinomial",
              cv.folds = 10,
              shrinkage = .01,
              n.minobsinnode = 10,
              n.trees = 10000, 
              interaction.depth = 20)

print(mod_gbm)


pred_boost = predict.gbm(object = mod_gbm,
                         newdata = test_set[,-9],
                         type = "response")

labels = colnames(pred_boost)[apply(pred_boost, 1, which.max)]
confusionMatrix(test_set$Transport, as.factor(labels))

boxplot(car_dataset_new)

