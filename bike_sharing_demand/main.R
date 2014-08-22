setwd("~/workspace/kaggle_competitions/bike_sharing_demand/")

library(lubridate)
library(e1071)
library(caret)
library(doMC)

preprocess.data <- function() {
  # laod the raw training set
  train <- read.table("data//train.csv", header=T, sep=",")
  
  # convert categorical variables into factors
  train$season <- factor(train$season)
  train$holiday <- factor(train$holiday)
  train$workingday <- factor(train$workingday)
  train$weather <- factor(train$weather)
  
  # The count of bike rented exhibits strong daily effect,
  # so the hour of day is extracted from datetime column as a new feature
  # and is treated as numerical value instead of categorical value.
  train$datetime <- ymd_hms(train$datetime)
  train$year <- year(train$datetime)
  train$year <- factor(train$year)
  train$hday <- hour(train$datetime)
  train <- train[, !names(train) %in% c("datetime", "registered", "casual", "atemp")]
  
  # also drop categorical 
  #train <- train[, !names(train) %in% c("holiday", "workingday", "dmon")]
  
  # create dummy variables for categorical variables
  train <- predict(dummyVars( ~ ., data=train), train)
  train <- as.data.frame(train)
  
  # variable registered is highly correlated with variable count
  # weather.4 has too few data points
  train <- train[, !names(train) %in% c("weather.4")]
  
  return(train)
}

m <- count ~ . + hday:temp + hday:humidity + hday:windspeed + 
  hday:season.1 + hday:season.2 + hday:season.3 + hday:season.4 +
  hday:weather.1 + hday:weather.2 + hday:weather.3 +
  temp:season.1 + temp:season.2 + temp:season.3 + temp:season.4 +
  temp:weather.1 + temp:weather.2 + temp:weather.3 +
  humidity:season.1 + humidity:season.2 + humidity:season.3 + humidity:season.4 +
  humidity:weather.1 + humidity:weather.2 + humidity:weather.3 +
  windspeed:season.1 + windspeed:season.2 + windspeed:season.3 + windspeed:season.4 +
  windspeed:weather.1 + windspeed:weather.2 + windspeed:weather.3 +
  temp:year.2011 + temp:year.2012 + humidity:year.2011 + humidity:year.2012 + windspeed:year.2011 + windspeed:year.2012


cvControl <- trainControl(method = "cv",
                          number = 10)


trainGbm <- function(train.df, model) {
  registerDoMC(cores=4)
  
  # stochastic boosted machine
  gbmGrid <-  expand.grid(interaction.depth = seq(1, 25),
                          n.trees = (1:100) * 10,
                          shrinkage = c(0.1, 0.5, 1.0))
  
  gbmTune <- train(model, data=train.df, method="gbm",
                   trControl=cvControl,
                   preProc=c("center", "scale"),
                   tuneGrid=gbmGrid)
  saveRDS(gbmTune, "gbmTune.R")
  return(gbmTune)
}

trainKNN <- function(train.df, model) {
  registerDoMC(cores=4)
  knnTune <- train(model, data=train.df, method="knn",
                   preProc=c("center", "scale"),
                   tuneGrid=data.frame(.k=1:30),
                   trControl=cvControl)

  saveRDS(knnTune, "knnTune")
  return(knnTune)
}
  

trainNnet <- function(train.df, model) {
  registerDoMC(cores=4)

  nnetGrid <- expand.grid(.decay=c(0, 0.01, 0.1), .size=c(1:15), .bag=F)
  nnetTune <- train(model, data=train.df, method="avNNet", tuneGrid=nnetGrid,
                    preProc=c("center", "scale"),
                    linout=F,
                    trace=F,
                    MaxNWts=10000,
                    maxit=500)
  
  saveRDS(nnetTune, "nnetTune")
  return(nnetTune)
}


trainMARS <- function(train.df, model) {
  registerDoMC(cores=1)

  marsGrid <- expand.grid(.degree = 1:5, .nprune = 2:100)
  marsTune <- train(model, data=train.df, method="earth", tuneGrid=marsGrid,
                    preProc=c("center", "scale"))

  saveRDS(marsTune, "marsTune")
  return(marsTune)
}


trainPls <- function(train.df, model) {
  registerDoMC(cores=1)
  plsTune <- train(model, data=train.df, method="pls", tuneLength=50,
                    preProc=c("center", "scale"))
  saveRDS(plsTune, "plsTune")
  return(plsTune)
}

trainLm <- function(train.df, model) {
  registerDoMC(cores=1)
  lmTune <- train(model, data=train.df, method="lm",
                   preProc=c("center", "scale"))
  saveRDS(lmTune, "lmTune")
  return(lmTune)
}
