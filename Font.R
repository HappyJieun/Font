###########################
## JIEUN LEE             ##  
###########################

# set Repositories
setRepositories(ind = 1:7)

#library
library(C50) 
library(caret) #modleing
library(caTools)
library(data.table)
library(doParallel)
library(dplyr)
library(gganimate) #animated plot
library(ggplot2) #visualization
library(ggpubr) #visualization
library(gifski) #gif endcoding
library(glue) #handling of string
library(grid)
library(gridExtra)
library(httr)
library(janitor) #data cleansing
library(kernlab)
library(lubridate) #date, time
library(netstat)
library(plyr)
library(rJava)
library(rlang)
library(RSelenium)
library(rvest) #scraping
library(sda)
library(seleniumPipes)
library(stringr)
library(tibble)
library(tictoc)
library(tidyverse) #tibble
library(tidyr)
library(XML)
library(xml2)

## set work. dir
WORK_DIR <- "C:\\Users\\ABC\\Desktop\\2023_Midterm_v2"
DATA_DIR <- "C:\\Users\\ABC\\Desktop\\2023_Midterm_v2\\Q4_Data"
setwd(WORK_DIR)


#sets the system locale to "English"
Sys.setlocale("LC_ALL", "English")


setwd(DATA_DIR)
##Data loading
ARIAL <- data.frame(fread("ARIAL.csv"))
CREDITCARD <- data.frame(fread("CREDITCARD.csv"))
HANDPRINT <- data.frame(fread("HANDPRINT.csv"))
ITALIC <- data.frame(fread("ITALIC.csv"))
TIMES <- data.frame(fread("TIMES.csv"))

##check shape
dim(ARIAL)
dim(CREDITCARD)
dim(HANDPRINT)
dim(ITALIC)
dim(TIMES)

## find common col names
common_cols <- Reduce(intersect, lapply(list(ARIAL, CREDITCARD, HANDPRINT, ITALIC, TIMES), colnames))
length(common_cols) 

## remove quality variables
quality <- c("m_label", "strength", "italic", "orientation", "m_top", "m_left", "originalH", "originalW", "h", "w")
extract_cols <- common_cols[!common_cols %in% quality]
length(extract_cols) # extract_cols들만 모델링에 사

# Check which column is cleansed
ARIAL_removeCols <- ARIAL %>% 
  select(-extract_cols) %>% 
  colnames()
CREDITCARD_removeCols <- CREDITCARD %>% 
  select(-extract_cols) %>% 
  colnames()
HANDPRINT_removeCols <-HANDPRINT %>% 
  select(-extract_cols) %>% 
  colnames()
ITALIC_removeCols <-ITALIC %>% 
  select(-extract_cols) %>% 
  colnames()
TIMES_removeCols <-TIMES %>% 
  select(-extract_cols) %>% 
  colnames()

ARIAL_removeCols
CREDITCARD_removeCols
HANDPRINT_removeCols
ITALIC_removeCols
TIMES_removeCols
union(union(union(union(ARIAL_removeCols, CREDITCARD_removeCols), HANDPRINT_removeCols), ITALIC_removeCols), TIMES_removeCols)

# Select extract columns & generate label column
ARIAL_clean <- ARIAL %>% 
  select(extract_cols) %>% 
  cbind(group = rep(x=c("ARIAL"), times = nrow(ARIAL)))
CREDITCARD_clean <- CREDITCARD %>% 
  select(extract_cols) %>% 
  cbind(group = rep(x=c("CREDITCARD"), times = nrow(CREDITCARD)))
HANDPRINT_clean <- HANDPRINT %>% 
  select(extract_cols) %>% 
  cbind(group = rep(x=c("HANDPRINT"), times = nrow(HANDPRINT)))
ITALIC_clean <- ITALIC %>% 
  select(extract_cols) %>% 
  cbind(group = rep(x=c("ITALIC"), times = nrow(ITALIC)))
TIMES_clean <- TIMES %>% 
  select(extract_cols) %>% 
  cbind(group = rep(x=c("TIMES"), times = nrow(TIMES)))

##check shape
dim(ARIAL_clean)
dim(CREDITCARD_clean)
dim(HANDPRINT_clean)
dim(ITALIC_clean)
dim(TIMES_clean)

# merge
mergedData <- rbind(ARIAL_clean, CREDITCARD_clean, HANDPRINT_clean, ITALIC_clean, TIMES_clean)
mergedData$group <- as.character(mergedData$group)
dim(mergedData)
glimpse(mergedData)

# Find the column containing NA in data
na_cols <- mergedData %>% 
  is.na() %>% 
  colSums()
na_cols[na_cols>0]
na_colName <- names(na_cols[na_cols > 0])
na_colName

## Remove column which contains NA
cleanedData <- mergedData %>% 
  select(-na_colName) 

dim(cleanedData)

### Step 2 - Computer vs. Handprint
### Change computer writing's label
# cleanedData_binary
cleanedData_bi <- cleanedData
cleanedData_bi$group[cleanedData_bi$group != "HANDPRINT"] <- "COMP"

## check shape
dim(cleanedData_bi)
length(which(cleanedData_bi$group == "HANDPRINT"))
length(which(cleanedData_bi$group == "COMP"))

# train, test로 나눔
index_bi <- createDataPartition(cleanedData_bi$group, p = 0.7, list = FALSE)
trainData_bi <- cleanedData_bi[index_bi, ]
testData_bi <- cleanedData_bi[-index_bi, ]

### Define training control 
trainControl_bi <- trainControl(method = "cv", number = 10)

### Modeling - cleanedData_binary

## For parallel processing
cl <- makePSOCKcluster(6)
registerDoParallel(cl)

#KNN Model
tic("LR(knn) Modelling") 
model_knn_bi <- train(group~., data = trainData_bi, method = "knn", trControl = trainControl_bi)
predictionResult_knn_bi <- predict(model_knn_bi, newdata = testData_bi)
toc()
table(predictionResult_knn_bi, testData_bi$group)
model_knn_bi$results

## For stop parallel processing
stopCluster(cl)

## For parallel processing
cl <- makePSOCKcluster(6)
registerDoParallel(cl)

#C5.0
tic("LR(C5.0) Modelling") 
model_C5_bi <- train(group~., data = trainData_bi, method = "C5.0", trControl = trainControl_bi)
predictionResult_C5_bi <- predict(model_C5_bi, newdata = testData_bi)
toc()
table(predictionResult_C5_bi, testData_bi$group)
model_C5_bi$results

## For stop parallel processing
stopCluster(cl)

## For parallel processing
cl <- makePSOCKcluster(6)
registerDoParallel(cl)

#LogitBoost
tic("LR(LogitBoost) Modelling") 
model_lb_bi <- train(group~., data = trainData_bi, method = "LogitBoost", trControl = trainControl_bi)
predictionResult_lb_bi <- predict(model_lb_bi, newdata = testData_bi)
toc()
table(predictionResult_lb_bi, testData_bi$group)
model_lb_bi$results

## For stop parallel processing
stopCluster(cl)

## For parallel processing
cl <- makePSOCKcluster(6)
registerDoParallel(cl)

#sda
tic("LR(sda) Modelling") 
model_sda_bi <- train(group~., data = trainData_bi, method = "sda", trControl = trainControl_bi)
predictionResult_sda_bi <- predict(model_sda_bi, newdata = testData_bi)
toc()
table(predictionResult_sda_bi, testData_bi$group)
model_sda_bi$results

## For stop parallel processing
stopCluster(cl)

## For parallel processing
cl <- makePSOCKcluster(6)
registerDoParallel(cl)

#rpart
tic("LR(rpart) Modelling")
model_rpart_bi <- train(group~., data = trainData_bi, method = "rpart", trControl = trainControl_bi)
predictionResult_rpart_bi <- predict(model_rpart_bi, newdata = testData_bi)
toc()
table(predictionResult_rpart_bi, testData_bi$group)
model_rpart_bi$results

## For stop parallel processing
stopCluster(cl)

## For parallel processing
cl <- makePSOCKcluster(6)
registerDoParallel(cl)

#gbm
tic("LR(gbm) Modelling")
model_gbm_bi <- train(group~., data = trainData_bi, method = "gbm", trControl = trainControl_bi)
predictionResult_gbm_bi <- predict(model_gbm_bi, newdata = testData_bi)
toc()
table(predictionResult_gbm_bi, testData_bi$group)
model_gbm_bi$results

## For stop parallel processing
stopCluster(cl)

## For parallel processing
cl <- makePSOCKcluster(6)
registerDoParallel(cl)

#svmRadial
tic("LR(svmRadial) Modelling")
model_svmRadial_bi <- train(group~., data = trainData_bi, method = "svmRadial", trControl = trainControl_bi)
predictionResult_svm_bi <- predict(model_svmRadial_bi, newdata = testData_bi)
toc()
table(predictionResult_svm_bi, testData_bi$group)
model_svmRadial_bi$results

## For stop parallel processing
stopCluster(cl)

## For parallel processing
cl <- makePSOCKcluster(6)
registerDoParallel(cl)

#glm
tic("LR(glm) Modelling")
model_glm_bi <- train(group~., data = trainData_bi, method = "glm", trControl = trainControl_bi)
predictionResult_glm_bi <- predict(model_glm_bi, newdata = testData_bi)
toc()
table(predictionResult_glm_bi, testData_bi$group)
model_glm_bi$results

## For stop parallel processing
stopCluster(cl)

## Select Model
table(predictionResult_svm_bi, testData_bi$group)
model_svmRadial_bi$results

################################################################################
### Step 3 - 5 different fonts

# train, test로 나눔
index <- createDataPartition(cleanedData$group, p = 0.8, list = FALSE)
trainData <- cleanedData[index, ]
testData <- cleanedData[-index, ]

### Define training control 
trainControl <- trainControl(method = "cv", number = 10)

### Modeling - cleanedData
## For parallel processing
cl <- makePSOCKcluster(6)
registerDoParallel(cl)

#KNN Model
tic("LR(knn) Modelling") 
model_knn <- train(group~., data = trainData, method = "knn", trControl = trainControl)
predictionResult_knn <- predict(model_knn, newdata = testData)
toc()
table(predictionResult_knn, testData$group)
model_knn$results

## For stop parallel processing
stopCluster(cl)

## For parallel processing
cl <- makePSOCKcluster(6)
registerDoParallel(cl)


#C5.0
tic("LR(C5.0) Modelling") 
model_C5 <- train(group~., data = trainData, method = "C5.0", trControl = trainControl)
predictionResult_C5 <- predict(model_C5, newdata = testData)
toc()
table(predictionResult_C5, testData$group)
model_C5$results

## For stop parallel processing
stopCluster(cl)

## For parallel processing
cl <- makePSOCKcluster(6)
registerDoParallel(cl)

#LogitBoost
tic("LR(LogitBoost) Modelling") 
model_lb <- train(group~., data = trainData, method = "LogitBoost", trControl = trainControl)
predictionResult_lb <- predict(model_lb, newdata = testData)
toc()
table(predictionResult_lb, testData$group)
model_lb$results

## For stop parallel processing
stopCluster(cl)

## For parallel processing
cl <- makePSOCKcluster(6)
registerDoParallel(cl)

#sda
tic("LR(sda) Modelling") 
model_sda <- train(group~., data = trainData, method = "sda", trControl = trainControl)
predictionResult_sda <- predict(model_sda, newdata = testData)
toc()
table(predictionResult_sda, testData$group)
model_sda$results

## For stop parallel processing
stopCluster(cl)

## For parallel processing
cl <- makePSOCKcluster(6)
registerDoParallel(cl)

#rpart
tic("LR(rpart) Modelling")
model_rpart <- train(group~., data = trainData, method = "rpart", trControl = trainControl)
predictionResult_rpart <- predict(model_rpart, newdata = testData)
toc()
table(predictionResult_rpart, testData$group)
model_rpart$results

## For stop parallel processing
stopCluster(cl)

## For parallel processing
cl <- makePSOCKcluster(6)
registerDoParallel(cl)

#rf
tic("LR(rf) Modelling")
model_rf <- train(group~., data = trainData, method = "rf", trControl = trainControl)
predictionResult_rf <- predict(model_rf, newdata = testData)
toc()
table(predictionResult_rf, testData$group)
model_rf$results

## For stop parallel processing
stopCluster(cl)

## For parallel processing
cl <- makePSOCKcluster(6)
registerDoParallel(cl)

#gbm
tic("LR(gbm) Modelling")
model_gbm <- train(group~., data = trainData, method = "gbm", trControl = trainControl)
predictionResult_gbm <- predict(model_gbm, newdata = testData)
toc()
table(predictionResult_gbm, testData$group)
model_gbm$results

## For stop parallel processing
stopCluster(cl)

## For parallel processing
cl <- makePSOCKcluster(6)
registerDoParallel(cl)

#svmRadial
tic("LR(svmRadial) Modelling")
model_svmRadial <- train(group~., data = trainData, method = "svmRadial", trControl = trainControl)
predictionResult_svm <- predict(model_svmRadial, newdata = testData)
toc()
table(predictionResult_svm, testData$group)
model_svmRadial$results

## For stop parallel processing
stopCluster(cl)

## For parallel processing
cl <- makePSOCKcluster(6)
registerDoParallel(cl)

#glm
tic("LR(glm) Modelling")
model_glm <- train(group~., data = trainData, method = "glm", trControl = trainControl)
predictionResult_glm <- predict(model_glm, newdata = testData)
toc()
table(predictionResult_glm, testData$group)
model_glm$results

## For stop parallel processing
stopCluster(cl)

# save Q4 RData
save.image("Q4.RData")



