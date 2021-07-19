################################
# Import Libraries and Dataset #
################################

setwd("~/NUS Studies/Year 3/Sem 2/ST4248/Group Project")
library(dplyr)
library(ggplot2)
library(randomForest)
library(gbm)
library(pROC)
library(rpart)
library(smotefamily)
library(DMwR)
library(tree)

empAttrition <- read.csv("employee_attrition.csv")

#################################
# Data Cleaning and Exploration #
#################################

dim(empAttrition)
glimpse(empAttrition)
summary(empAttrition)

# Check for missing values in each column:
sapply(empAttrition, function(x) sum(is.na(x)))

# Check for number of unique values in each column:
sapply(empAttrition, function(x) length(unique(x)))

### Initial observations: 

# 1. No missing values in any rows/columns.

# 2. Modification to some columns are needed.

# - i.e. "ï..Age" needs to be renamed to "Age"
colnames(empAttrition)[1] <- 'Age'

# - Redundant columns like "Over18", "EmployeeCount",
#   "StandardHours" and EmployeeNumber" should be removed.

droppedCols <- c("Over18", "EmployeeCount", "StandardHours",
                 "EmployeeNumber")
empAttrition <- empAttrition[,!colnames(empAttrition) %in% droppedCols]
colnames(empAttrition)

#######################
# Removal of Outliers #
#######################

# empAttrition$Attrition <- empAttrition$Attrition %>%
#   sapply(function(x) ifelse(x == "Yes", 1, 0))
# 
# mod <- lm(Attrition~., data=empAttrition)
# cooksd <- cooks.distance(mod) #metric to detect outlier
# 
# plot(cooksd, pch='*', cex=2,
#      main='Influential Points')
# abline(h=5*mean(cooksd, na.rm=T), col='red')
# text(x=1:length(cooksd)+1,y=cooksd,
#      labels=ifelse(cooksd>5*mean(cooksd, na.rm=T),
#                    names(cooksd),""), col='red')
# 
# influential <- as.numeric(names(cooksd)[(cooksd>5*mean(cooksd,na.rm=T))])
# length(influential) #64 rows identified, values change if 5 is changed to 4
# influential #rows identified 
# 
# empAttrition <- empAttrition[-influential,]
# 
# # Convert Attrition column back
# empAttrition$Attrition <- empAttrition$Attrition %>%
#   sapply(function(x) ifelse(x == 1, 'Yes', 'No'))

############################
# Convert Categorical Cols #
############################

# - Some columns need to be converted to categorical
#   i.e. Education, EnvironmentSatisfaction, JobInvolvement,
#   JobLevel, JobSatisfaction, PerformanceRating,
#   RelationshipSatisfaction, StockOptionLevel, WorkLifeBalance.

# StockOptionLevel has a scale of 0-3
# Education has a scale of 1-5
# The other columns have a scale of 1-4

catCols <- c("Education", "EnvironmentSatisfaction",
             "JobInvolvement", "JobLevel", "JobSatisfaction",
             "RelationshipSatisfaction", "StockOptionLevel",
             "WorkLifeBalance", "Attrition")

for (i in catCols) {
  empAttrition[[i]] <- factor(empAttrition[[i]])
}

# PerformanceSatisfaction has a scale of 1-4, but data only contains 3,4
empAttrition$PerformanceRating <- factor(empAttrition$PerformanceRating,
                                         levels = c(1,2,3,4))


# 3. Some column names/values in columns can be shortened.
# Replace then convert to factor.

# First perform name changes:
colnames(empAttrition) <- c("Age", "Attrition", "BizTravel",
                            "DailyRate", "Dept", "Distance",
                            "Education", "EduField", "EnvSatisfy",
                            "Gender", "HourlyRate", "JobInv", "JobLevel",
                            "JobRole", "JobSatisfy", "Marital",
                            "MonthlyInc", "MonthlyRate", "NumCompWorked",
                            "Overtime", "P.SalaryHike", "P.Rating",
                            "RSSatisfy", "StockOption", "WorkingYrs",
                            "NumTrainLY", "WLBalance", "YrsInComp",
                            "YrsInRole", "YrsLastPromo", "YrsWManager")

colValConvert <- function(x, y, z) {
  return (z[match(x, y)])
}

empAttrition$BizTravel <- sapply(empAttrition$BizTravel,
                                 function(x) colValConvert(x, levels(empAttrition$BizTravel), c("None", "Freq", "Rare")))

empAttrition$Dept <- sapply(empAttrition$Dept,
                            function(x) colValConvert(x, levels(empAttrition$Dept), c("HR", "R&D", "Sales")))

empAttrition$EduField <- sapply(empAttrition$EduField,
                                function(x) colValConvert(x, levels(empAttrition$EduField), c("HR", "LifeSci", "Marketing",
                                                                                              "Medical", "Other", "Technical")))

empAttrition$JobRole <- sapply(empAttrition$JobRole,
                               function(x) colValConvert(x, levels(empAttrition$JobRole), c("HealthRep", "HR", "LabTech",
                                                                                            "Manager", "ManuDir", "ResrDir",
                                                                                            "ResrSci", "SalesExe", "SalesRep")))

newCols <- c("JobRole", "EduField", "Dept", "BizTravel")
for (i in newCols) {
  empAttrition[[i]] <- factor(empAttrition[[i]])
}


#########
# SMOTE #
#########

# conNumCols <- c("BizTravel", "Dept", "EduField", "Gender", "JobRole",
#                 "Marital", "Overtime", "Attrition")
# 
# for (i in colnames(empAttrition)) {
#   empAttrition[[i]] <- as.numeric(empAttrition[[i]])
# }

#empAttrition <- smotefamily::SMOTE(X=empAttrition[,-2], target=empAttrition$Attrition,
#K=4)

#empAttrition <- DMwR::SMOTE(Attrition~.,empAttrition, perc.over=200, perc.under=200)

####################
# Machine Learning #
####################

# Aim:
# Try to predict employee attrition using rest of the columns. 

# Generate train and test sets
n = 1470
set.seed(77)
n <- dim(empAttrition)[1]
p <- dim(empAttrition)[2] - 1
train_index <- sample(1:n, 0.5*n)
test_index <- setdiff(1:n, train_index)

train <- empAttrition[train_index,]
test <- empAttrition[test_index,]
#train <- DMwR::SMOTE(Attrition~.,train, perc.over=200, perc.under=200)
#train <- smotefamily::SMOTE(X=train[,-2], target=train$Attrition,
#K=4)
y_train <- train$Attrition
y_test <- test$Attrition



############################
# Baseline Regression Tree #
############################

start_time <- Sys.time()
baselineTree <- tree(Attrition~., train)
end_time <- Sys.time()
end_time - start_time
# Time taken: 0.06855893 seconds

#testtree <- rpart(Attrition~., data=train, cp=.02)
#rpart.plot(testtree)  # this plots the decision tree

plot(baselineTree, type="uniform") # this plots the decision tree
text(baselineTree, cex=0.5)
#text(baselineTree, pretty=0,cex=0.6) # labels overlap
summary(baselineTree)

baselinePred <- predict(baselineTree, test, type="class")   # predictions for test set
tableBaseline <- table(baselinePred, y_test)                # confusion matrix
tableBaseline

mean(baselinePred!=y_test)                                     # test error rate
singletesterr <- mean(baselinePred!=y_test)        
tableBaseline[2,2] / (tableBaseline[2,2] + tableBaseline[1,2]) # Sensitivity
tableBaseline[1,1] / (tableBaseline[1,1] + tableBaseline[2,1]) # Specificity

baselineTree.probs <- predict(baselineTree, test)
baselineAUC <- auc(y_test, baselineTree.probs[,2])
baselineAUC
par(pty = "s")
plot(roc(y_test,baselineTree.probs[,2]), xlim=c(1,0))
baselineROC <- roc(y_test,baselineTree.probs[,2])

# Test error: 0.152381
# Sensitivity: 0.287037
# Specificity: 0.9441786
# F1 Score: 
# AUC: 0.6476

##########################################
# Cost-Complexity Pruning and 10-fold CV #
##########################################

set.seed(77)
CV <- cv.tree(baselineTree, FUN=prune.misclass)
CV
plot(CV$size, CV$dev, type="b", xlab="size of tree", ylab="CV error")

prunedTree <- prune.misclass(baselineTree, best=4)
plot(prunedTree, type="uniform")
text(prunedTree, pretty=3)
summary(prunedTree)

prunedPred <- predict(prunedTree,test,type="class")
tablePruned <- table(prunedPred, y_test)           # confusion matrix
tablePruned

mean(prunedPred!=y_test)                                     # test error rate
tablePruned[2,2] / (tablePruned[2,2] + tablePruned[1,2]) # Sensitivity
tablePruned[1,1] / (tablePruned[1,1] + tablePruned[2,1]) # Specificity

prunedTree.probs <- predict(prunedTree,test)
prunedAUC <- auc(y_test, prunedTree.probs[,2])
prunedAUC
par(pty = "s")
plot(roc(y_test,prunedTree.probs[,2]), xlim=c(1,0))
prunedROC <- roc(y_test,prunedTree.probs[,2])

# Test error: 0.1442177
# Sensitivity: 0.1666667
# Specificity: 0.9744817
# AUC: 0.64


###########
# Bagging #
###########

set.seed(77)
Bval <- seq(from=100, to=4000, by=100)
T <- length(Bval)
testerr_bag <- rep(0, T)  # store test set error
OOBerr_bag <- rep(0, T)   # store OOB error
for (t in 1:T){
  B <- Bval[t]
  start_time <- Sys.time()
  bag <- randomForest(Attrition~., data=train, mtry=p, ntree=B) #mtry=p for bagging
  end_time <- Sys.time()
  ypred <- predict(bag, newdata=test, type="class")
  testerr_bag[t] <- mean(y_test!=ypred)
  OOBpred <- bag$predicted
  OOBerr_bag[t] <- mean(y_train!=OOBpred)
  cat("Time taken for B =",B, "and m =",p, "vars:",end_time - start_time,"\n")
}

which.min(testerr_bag) # 6
min(testerr_bag)
which.min(OOBerr_bag) # 6
min(OOBerr_bag)

plot(Bval, testerr_bag, type="l", ylim = c(0.12, 0.2),
     xlab="B", ylab="error", main="Bagging", lwd=2)
points(Bval, OOBerr_bag, type="l", col="blue", lwd=2)
abline(h=singletesterr, col="seagreen", lwd=2, lty=2) # Test error by single decision tree
legend("topright", legend=c("single tree", "test error", "OOB error"),
       col=c("seagreen", "black", "blue"), lty=c(2,1,1), cex=0.8, lwd=2, y.intersp = 0.3)

# Test error for bagging is slightly worse than single pruned tree.
# But this is expected, because pruning is meant to 
# lower classification error rate. 
# Whereas bagging might fare better in other metrics,
# i.e. sensitivity or specificity etc. 

# Bagging: B=4000
bagging4k <- randomForest(Attrition~., data=train, mtry=p, ntree=4000)
test_pred <- predict(bagging4k, test, type="class")
mean(test_pred!=y_test) # test error

baggingTable <- table(test_pred, y_test)
baggingTable
baggingTable[2,2] / (baggingTable[2,2] + baggingTable[1,2]) # Sensitivity
baggingTable[1,1] / (baggingTable[1,1] + baggingTable[2,1]) # Specificity
bagging4k.probs <- predict(bagging4k, test, type = "prob")
baggingAUC <- auc(y_test, bagging4k.probs[,2])
baggingAUC
par(pty = "s")
plot(roc(y_test,bagging4k.probs[,2]), xlim=c(1,0))
baselineROC <- roc(y_test,bagging4k.probs[,2])
impBagging <- data.frame(importance(bagging4k,type="2"))
impBagging <- data.frame(variable = rownames(impBagging),
                         MeanDecreaseGini = impBagging$MeanDecreaseGini)
impBagging[order(impBagging$MeanDecreaseGini, decreasing=T)[1:10],]

# Variable importance
importance(bagging4k)
varImpPlot(bagging4k)

# Test error: 0.1360544
# Sensitivity: 0.2314815
# Specificity: 0.9728868
# AUC: 0.8119


##################
# Random Forests #
##################

# For m = floor(sqrt(p)) and m = p/2
set.seed(77)
mval <- c(floor(sqrt(p)),p/2)
Bval <- seq(from=100, to=4000, by=100)
T <- length(Bval)
testerr_rf <- matrix(0, length(Bval), length(mval)) # store test set error
OOBerr_rf <- matrix(0, length(Bval), length(mval)) # store OOB error
for (i in 1:length(mval)){
  m <- mval[i]
  for (t in 1:T){
    B <- Bval[t]
    start_time <- Sys.time()
    bag <- randomForest(Attrition~., data=train, mtry=m, ntree=B)
    end_time <- Sys.time()
    ypred <- predict(bag, newdata=test, type="class")
    testerr_rf[t,i] <- mean(y_test!=ypred)
    OOBpred <- bag$predicted
    OOBerr_rf[t,i] <- mean(y_train!=OOBpred)
    cat("Time taken for B =",B, "and m =",m, "vars:",end_time - start_time,"\n")
  }
}

# For m=5
which.min(testerr_rf[,1]) # 9 
which.min(OOBerr_rf[,1]) # 11
min(testerr_rf[,1])
min(OOBerr_rf[,1])

# For m=15
which.min(testerr_rf[,2]) # 3
which.min(OOBerr_rf[,2]) # 2
min(testerr_rf[,1])
min(OOBerr_rf[,1])


plot(Bval, testerr_rf[,1], type="l", ylim=c(0.12, 0.18),
     xlab="B", ylab="error", main="Random forests", col = "blue", lwd=2)
points(Bval, testerr_rf[,2], type="l", col="red", lwd=2)
points(Bval, testerr_bag, type="l", col="black", lwd=2)
abline(h=singletesterr, col="seagreen", lwd=2, lty=2)
legend("topright", legend=c("RF(m=5)", "RF(m=15)", "Bagging", "single Tree"),
       col=c("blue", "red", "black", "seagreen"), lty=c(2,2,2,1), cex=0.8, lwd=2, y.intersp=0.2)


# For m=5,B=4000 RF
rf5 <- randomForest(Attrition~., data=train, mtry=5, ntree=4000)
# Variable importance
importance(rf5)
varImpPlot(rf5)

test_pred <- predict(rf5, test, type="class")
mean(test_pred!=y_test) # test error

rf5Table <- table(test_pred, y_test)
rf5Table
rf5Table[2,2] / (rf5Table[2,2] + rf5Table[1,2]) # Sensitivity
rf5Table[1,1] / (rf5Table[1,1] + rf5Table[2,1]) # Specificity
rf5.probs <- predict(rf5, test, type = "prob")
rf5AUC <- auc(y_test, rf5.probs[,2])
rf5AUC
par(pty = "s")
plot(roc(y_test,rf5.probs[,2]), xlim=c(1,0))
rf5ROC <- roc(y_test,rf5.probs[,2])
imprf5 <- data.frame(importance(rf5,type="2"))
imprf5 <- data.frame(variable = rownames(imprf5),
                     MeanDecreaseGini = imprf5$MeanDecreaseGini)
imprf5[order(imprf5$MeanDecreaseGini, decreasing=T)[1:10],]

# Test error: 0.1346939
# Sensitivity: 0.1388889
# Specificity: 0.9904306
# AUC: 0.825


# For m=15,B=4000 RF
rf15 <- randomForest(Attrition~., data=train, mtry=15, ntree=4000)
# Variable importance
importance(rf15)
varImpPlot(rf15)

test_pred <- predict(rf15, test, type="class")
mean(test_pred!=y_test) # test error

rf15Table <- table(test_pred, y_test)
rf15Table
rf15Table[2,2] / (rf15Table[2,2] + rf15Table[1,2]) # Sensitivity
rf15Table[1,1] / (rf15Table[1,1] + rf15Table[2,1]) # Specificity
rf15.probs <- predict(rf15, test, type = "prob")
rf15AUC <- auc(y_test, rf15.probs[,2])
rf15AUC
par(pty = "s")
plot(roc(y_test,rf15.probs[,2]), xlim=c(1,0))
rf15ROC <- roc(y_test,rf15.probs[,2])
imprf15 <- data.frame(importance(rf15,type="2"))
imprf15 <- data.frame(variable = rownames(imprf15),
                      MeanDecreaseGini = imprf15$MeanDecreaseGini)
imprf15[order(imprf15$MeanDecreaseGini, decreasing=T)[1:10],]

# Test error: 0.1360544
# Sensitivity: 0.2222222
# Specificity: 0.9760766
# AUC: 0.8164

############
# Boosting #
############

# Exclusively for boosting
train2 <- data.frame(train)
train2$Attrition <- as.numeric(train2$Attrition)-1
y_train2 <- train2$Attrition

test2 <- data.frame(test)
test2$Attrition <- as.numeric(test2$Attrition)-1
y_test2 <- test2$Attrition

# for shrinkage = 0.01
set.seed(77)
Bval <- seq(from=100, to=4000, by=100)
T <- length(Bval)
testerr_boost <- matrix(0, 2, T)  # store test set error
for (d in 1:2) {
  for (t in 1:T) {
    B <- Bval[t]
    start_time <- Sys.time()
    boost <- gbm(Attrition~., data=train2, distribution="bernoulli",
                 n.trees=B, interaction.depth=d, shrinkage=0.01)
    end_time <- Sys.time()
    ypred <- predict(boost, newdata=test2, type="response", n.trees=B)
    testerr_boost[d,t] <- mean(round(ypred)!=y_test2)
    cat("Time taken for B =",B, "and d =",d,":",end_time - start_time,"\n")
  }
}

plot(Bval, testerr_boost[1,], type="l",
     xlab="B", ylab="error", main="Boosted Trees, shrinkage = 0.01", ylim=c(0.075, 0.20), lwd=2)
points(Bval, testerr_boost[2,], type="l", col="red", lwd=2)
abline(h=singletesterr, col="seagreen", lwd=2, lty=2)
legend("topright", legend=c("single tree", "d=1", "d=2"),
       col=c("seagreen", "black", "red"), lty=c(2,1,1), cex=0.6, y.intersp=0.5)


# for shrinkage = 0.1
set.seed(77)
Bval <- seq(from=100, to=4000, by=100)
T <- length(Bval)
testerr_boost2 <- matrix(0, 2, T)  # store test set error
for (d in 1:2) {
  for (t in 1:T) {
    B <- Bval[t]
    start_time <- Sys.time()
    boost <- gbm(Attrition~., data=train2, distribution="bernoulli",
                 n.trees=B, interaction.depth=d, shrinkage=0.1)
    end_time <- Sys.time()
    ypred <- predict(boost, newdata=test2, type="response", n.trees=B)
    testerr_boost2[d,t] <- mean(round(ypred)!=y_test2)
    cat("Time taken for B =",B, "and d =",d,":",end_time - start_time,"\n")
  }
}

plot(Bval, testerr_boost2[1,], type="l",
     xlab="B", ylab="error", main="Boosted Trees, shrinkage = 0.1", ylim=c(0.075, 0.20), lwd=2)
points(Bval, testerr_boost2[2,], type="l", col="red", lwd=2)
abline(h=singletesterr, col="seagreen", lwd=2, lty=2)
legend("topright", legend=c("single tree", "d=1", "d=2"),
       col=c("seagreen", "black", "red"), lty=c(2,1,1), cex=0.6, y.intersp=0.5)


# For shrinkage = 0.01
which.min(testerr_boost[1,]) # 23
which.min(testerr_boost[2,]) # 23
min(testerr_boost[1,])
min(testerr_boost[2,])

# For shrinkage = 0.1
which.min(testerr_boost2[1,]) # 4
which.min(testerr_boost2[2,]) # 4
min(testerr_boost2[1,])
min(testerr_boost2[2,])

############################
# Boosting: Shinkrage 0.01 #
############################

# For B=2300, d=1, shrinkage=0.01
boostd1s001 <- gbm(Attrition~., data=train2, distribution="bernoulli",
                   n.trees=2300, interaction.depth=1, shrinkage=0.01)
# Variable importance
summary.gbm(boostd1s001)

# For B=2300, d=2, shrinkage=0.01
boostd2s001 <- gbm(Attrition~., data=train2, distribution="bernoulli",
                   n.trees=2300, interaction.depth=2, shrinkage=0.01)
# Variable importance
summary.gbm(boostd2s001)

# Metrics of Boosting: B=2300, d=1, shrinkage=0.01

test_pred <- predict(boostd1s001, test2, type="response")
mean(round(test_pred)!=y_test2) # test error

boostd1s001Table <- table(round(test_pred), y_test2)
boostd1s001Table
boostd1s001Table[2,2] / (boostd1s001Table[2,2] + boostd1s001Table[1,2]) # Sensitivity
boostd1s001Table[1,1] / (boostd1s001Table[1,1] + boostd1s001Table[2,1]) # Specificity
boostd1s001.probs <- predict(boostd1s001, test2, type = "response")
boostd1s001AUC <- auc(y_test2, boostd1s001.probs)
boostd1s001AUC
par(pty = "s")
plot(roc(y_test2,boostd1s001.probs), xlim=c(1,0))
boostd1s001ROC <- roc(y_test2,boostd1s001.probs)
summary.gbm(boostd1s001)[1:10,]

# Test error: 0.09795918
# Sensitivity: 0.462963
# Specificity: 0.9776715
# AUC: 0.8636


# Metrics of Boosting: B=2300, d=2, shrinkage=0.01

test_pred <- predict(boostd2s001, test2, type="response")
mean(round(test_pred)!=y_test2) # test error

boostd2s001Table <- table(round(test_pred), y_test2)
boostd2s001Table
boostd2s001Table[2,2] / (boostd2s001Table[2,2] + boostd2s001Table[1,2]) # Sensitivity
boostd2s001Table[1,1] / (boostd2s001Table[1,1] + boostd2s001Table[2,1]) # Specificity
boostd2s001.probs <- predict(boostd2s001, test2, type = "response")
boostd2s001AUC <- auc(y_test2, boostd2s001.probs)
boostd2s001AUC
par(pty = "s")
plot(roc(y_test2,boostd2s001.probs), xlim=c(1,0))
boostd2s001ROC <- roc(y_test2,boostd2s001.probs)
summary.gbm(boostd2s001)[1:10,]

# Test error: 0.1222826
# Sensitivity: 0.3934426
# Specificity: 0.9739414
# AUC: 0.8233

###########################
# Boosting: Shinkrage 0.1 #
###########################

# For B=400, d=1, shrinkage=0.1
boostd1s01 <- gbm(Attrition~., data=train2, distribution="bernoulli",
                  n.trees=400, interaction.depth=1, shrinkage=0.1)
# Variable importance
summary.gbm(boostd1s01)

# For B=400, d=2, shrinkage=0.1
boostd2s01 <- gbm(Attrition~., data=train2, distribution="bernoulli",
                  n.trees=400, interaction.depth=2, shrinkage=0.1)
# Variable importance
summary.gbm(boostd2s01)


# Metrics of Boosting: B=400, d=1, shrinkage=0.1
test_pred <- predict(boostd1s01, test2, type="response")
mean(round(test_pred)!=y_test2) # test error

boostd1s01Table <- table(round(test_pred), y_test2)
boostd1s01Table
boostd1s01Table[2,2] / (boostd1s01Table[2,2] + boostd1s01Table[1,2]) # Sensitivity
boostd1s01Table[1,1] / (boostd1s01Table[1,1] + boostd1s01Table[2,1]) # Specificity
boostd1s01.probs <- predict(boostd1s01, test2, type = "response")
boostd1s01AUC <- auc(y_test2, boostd1s01.probs)
boostd1s01AUC
par(pty = "s")
plot(roc(y_test2,boostd1s01.probs), xlim=c(1,0))
boostd1s01ROC <- roc(y_test2,boostd1s01.probs)
summary.gbm(boostd1s01)[1:10,]

# Test error: 0.1020408
# Sensitivity: 0.4907407
# Specificity: 0.9681021
# AUC: 0.8569


# Metrics of Boosting: B=400, d=2, shrinkage=0.1
test_pred <- predict(boostd2s01, test2, type="response")
mean(round(test_pred)!=y_test2) # test error

boostd2s01Table <- table(round(test_pred), y_test2)
boostd2s01Table
boostd2s01Table[2,2] / (boostd2s01Table[2,2] + boostd2s01Table[1,2]) # Sensitivity
boostd2s01Table[1,1] / (boostd2s01Table[1,1] + boostd2s01Table[2,1]) # Specificity
boostd2s01.probs <- predict(boostd2s01, test2, type = "response")
boostd2s01AUC <- auc(y_test2, boostd2s01.probs)
boostd2s01AUC
par(pty = "s")
plot(roc(y_test2,boostd2s01.probs), xlim=c(1,0))
boostd2s01ROC <- roc(y_test2,boostd2s01.probs)
summary.gbm(boostd2s01)[1:10,]

# Test error: 0.1059783
# Sensitivity: 0.4918033
# Specificity: 0.9739414
# AUC: 0.8228

###############
# Final Plots #
###############

# Boosting and Single Pruned Tree
plot(Bval, testerr_boost[1,], type="l", ylim=c(0.08, 0.20),
     xlab="B", ylab="error", main="Boosting", lwd=2)
points(Bval, testerr_boost[2,], type="l", col="red", lwd=2)
points(Bval, testerr_boost2[1,], type="l", col="blue", lwd=2)
points(Bval, testerr_boost2[2,], type="l", col="brown", lwd=2)
abline(h=singletesterr, col="seagreen", lty=2, lwd=2)
legend("topright", legend=c("d=1,shrinkage=0.01", "d=2,shrinkage=0.01", "d=1,shrinkage=0.1", "d=2,shrinkage=0.1", "Single Tree"),
       col=c("black", "red", "blue", "brown", "seagreen"), lty=c(1,1,1,1,2), cex=0.6, lwd=2, y.intersp=0.5)


# Comparison of Methods
plot(Bval, testerr_boost[1,], type="l", ylim=c(0.08, 0.20),
     xlab="B", ylab="error", main="Comparison of Test Errors", lwd=2)
points(Bval, testerr_rf[,1], type="l", col="red", lwd=2)
points(Bval, testerr_rf[,2], type="l", col="blue", lwd=2)
points(Bval, testerr_bag, type="l", col="brown", lwd=2)
abline(h=singletesterr, col="seagreen", lwd=2, lty=2)
#legend("topright", legend=c("Boosting (d=1, s=0.01)", "RF(m=5)", "Bagging", "Single Tree"),
      #col=c("black", "red", "blue", "seagreen"), lty=c(1,1,1,2), cex=0.6, y.intersp=0.5)
legend("topright", legend=c("Boosting (d=1, s=0.01)", "RF(m=5)", "RF(m=15)", "Bagging", "Single Tree"),
       col=c("black", "red", "blue", "brown", "seagreen"), lty=c(1,1,1,1,2), cex=0.6, y.intersp=0.5)

# Comparison of ROC curves
par(pty = "s")
final_roc <- plot(roc(y_test,baselineTree.probs[,2]), xlim=c(1,0),col="seagreen")
final_roc <- plot(roc(y_test,bagging4k.probs[,2]), xlim=c(1,0), col="brown",print.auc.y=.4, add=TRUE)
final_roc <- plot(roc(y_test,rf5.probs[,2]), xlim=c(1,0), col="red",print.auc.y=.4, add=TRUE)
final_roc <- plot(roc(y_test,rf15.probs[,2]), xlim=c(1,0), col="blue",print.auc.y=.4, add=TRUE)
final_roc <- plot(roc(y_test2,boostd1s001.probs), xlim=c(1,0), col="black",print.auc.y=.4, add=TRUE)
legend("bottomright", legend=c("Boosting (d=1, s=0.01)", "RF(m=5)", "RF(m=15)", "Bagging", "Single Tree"),
       col=c("black", "red", "blue", "brown", "seagreen"), lty=c(1,1,1,1,2), cex=0.6, y.intersp=0.2, text.width=0.3)
#legend("bottomright", legend=c("Boosting (d=1, s=0.01)", "RF(m=5)", "Bagging", "Single Tree"),
       #col=c("black", "red", "blue", "seagreen"), lty=c(1,1,1,2), cex=0.6, y.intersp=0.2, text.width=0.3)
