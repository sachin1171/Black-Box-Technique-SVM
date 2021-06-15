####################problem 1####################
library(readr)
# Partition Data into train and test data
salary_train <-read.csv("C:/Users/usach/Desktop/Black box technquie-SVM/SalaryData_Train (1).csv") 
salary_test  <-read.csv("C:/Users/usach/Desktop/Black box technquie-SVM/SalaryData_Test (1).csv")

# Training a model on the data
str(salary_train)
summary(salary_train)
sum(is.na(salary_train))
sum(is.na(salary_test))

salary_train$workclass=as.factor(salary_train$workclass)
salary_train$education=as.factor(salary_train$education)
salary_train$educationno=NULL

salary_train$maritalstatus=as.factor(salary_train$maritalstatus)
salary_train$occupation=as.factor(salary_train$occupation)
salary_train$relationship=NULL
salary_train$race=NULL

salary_train$native=as.factor(salary_train$native)
salary_train$sex=as.factor(salary_train$sex)
salary_train$Salary=as.factor(salary_train$Salary)

library(kernlab)
str(salary_train)
salary_classifier <- ksvm(Salary ~ ., data = salary_train, kernel = "vanilladot")
?ksvm
summary(salary_classifier)

salary_test_pred <- predict(salary_classifier, salary_test)

library(gmodels)
CrossTable(salary_test_pred, salary_test$Salary,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))

test_acc = mean(salary_test_pred == salary_test$Salary)
test_acc

# predictions on testing dataset
letter_predictions <- predict(salary_classifier, salary_test)

table(letter_predictions, salary_test$Salary)
agreement <- letter_predictions == salary_test$Salary
table(agreement)
prop.table(table(agreement))

# On Training Data
sms_train_pred <- predict(salary_classifier, salary_train)

train_acc = mean(sms_train_pred == salary_train$Salary)
train_acc
#####################problem 2 ####################
# Load the Dataset
fire <- read.csv(file.choose())

summary(fire)
fire=fire[1:517,3:31]
for(unique_value in unique(fire$size_category)){
  
  
  fire[paste("Size_cateogary", unique_value, sep = ".")] <- ifelse(fire$size_category== unique_value, 1, 0)
}
fire$size_category=NULL
summary(fire)
str(fire)
# Partition Data into train and test data
fire_train <- fire[1:413, ]
fire_test  <- fire[414:517, ]

# Training a model on the data 
# Begin by training a simple linear SVM

library(kernlab)

fire_classifier <- ksvm(area ~ ., scaled = TRUE,data = fire_train, kernel = "vanilladot")
?ksvm

## Evaluating model performance 
# predictions on testing dataset
fire_predictions<- predict(fire_classifier,fire_test)
table(fire_predictions, fire_test$area)
agreement <- fire_predictions == fire_test$area
table(agreement)
prop.table(table(agreement))

test_acc = mean(fire_predictions == fire_test$area)
test_acc
## Improving model performance 
fire_classifier_rbf <- ksvm(area ~ ., data = fire_train, kernel = "rbfdot")
fire_predictions_rbf <- predict(fire_classifier_rbf, fire_test)
agreement_rbf <- fire_predictions_rbf == fire_test$area
table(agreement_rbf)
prop.table(table(agreement_rbf))
rbf = mean(fire_predictions_rbf == fire_test$area)
rbf
