rm(list=ls(all=T))
setwd("/home/zozo/Documents/edwisor/Project/Santander/")
getwd()
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees',"ROCR","Matrix")
lapply(x, require, character.only = TRUE)
rm(x)

train=read.csv("train.csv",header = TRUE)
head(train)
summary(train)
dim(train)
sapply(train, typeof)

## Convert type of target variable
lapply(train$target,as.factor)
train=train[,-c(1)]

### Missing value Analysis 

missing_val = data.frame(apply(train,2,function(x){sum(is.na(x))}))
missing_val$Columns = row.names(missing_val)
names(missing_val)[1] =  "Missing_percentage"
missing_val$Missing_percentage = (missing_val$Missing_percentage/nrow(train)) * 100
missing_val = missing_val[order(-missing_val$Missing_percentage),]
row.names(missing_val) = NULL
missing_val = missing_val[,c(2,1)]

## we can see that there are no missing values in train dataset

### Important Features
train$target=as.factor(train$target)
rf=randomForest(target~.,ntree=10,importance=TRUE)
important_variables=importance(rf,type=2)
important_variables

### we can see that var_81,var_26,var_53,var_12 and var_139 are most important variables

### Lets take a look on correlation between independent variables

corr=cor(train[,2:201])
view(corr)

### we can observe that there is very less correlation among independent variables

### Lets check distribution of classes in target variable\
count=table(train$target)
count
summary(train$target)
## lets visualize count of the target class

ggplot(train, aes_string(x = train$target)) +
  geom_bar(stat="count",fill="green") + theme_bw() +
  xlab("target") + ylab('Count') +
  ggtitle("Marketing Campaign Analysis") +  theme(text=element_text(size=10))

### we can see that only 10.05% positive class is present in target variable which means target 
### class imbalance problem exist here.

### Let's divide dataset into train and test
set.seed(1234)
train.index = createDataPartition(train$target, p = .80, list = FALSE)
train1 = train[ train.index,]
test  = train[-train.index,]

### Let's perform Logistic Regression on imbalanced data
logit_model = glm(target ~ ., data = train1, family = "binomial")
logit_Predictions = predict(logit_model, newdata = test[,-1], type = "response")
logit_Predictions = ifelse(logit_Predictions > 0.5, 1, 0)

ConfMatrix_LR = table(test$target, logit_Predictions)
ConfMatrix_LR 
Accuracy=((35499+1062)*100)/(35499+481+2957+1062)
Accuracy
library(ROCR)
pred=prediction(logit_Predictions,test$target)
perf = performance(pred,"tpr","fpr")
plot(perf,colorize=TRUE)

accuracy.meas(test$target,logit_Predictions)
### precision: 0.688
### recall: 0.264
### F: 0.191

### Even though our model is giving accuracy of 91.42% but from ROC curve and f1-score we can clearly
### see that this model is not performing well on imbalanced data.

### Lets balance the data using smote
train_rose <- ROSE(target ~ ., data = train1, seed = 1)$data
test_rose <- ROSE(target ~ ., data = test, seed = 1)$data
table(train_rose$target)
table(test_rose$target)

### Lets apply logistic regression on synthetic data
logit_model = glm(target ~ ., data=train_rose, family = "binomial")
logit_Predictions = predict(logit_model, newdata = test_rose[,-1], type = "response")
logit_Predictions = ifelse(logit_Predictions > 0.5, 1, 0)

ConfMatrix_LR = table(test_rose$target, logit_Predictions)
ConfMatrix_LR 
Accuracy=((14278+13912)*100)/(14278+13912+6002+5807)
Accuracy

library(ROCR)
pred=prediction(logit_Predictions,test_rose$target)
perf = performance(pred,"tpr","fpr")
plot(perf,colorize=TRUE)

accuracy.meas(logit_Predictions,test_rose$target)

### Accuracy : 70.4768


### From roc curve and f-1 score we can say that logistic regression is performing better on synthetic
### data than original data

### Lets build Random Forest model
RF_model = randomForest(target ~ ., train1, importance = TRUE, ntree = 5)
RF_Predictions = predict(RF_model, test[,-1])
ConfMatrix_RF=table(test$target,RF_Predictions)
ConfMatrix_RF
Accuracy=(35469+259)/(35469+259+3760+511)
Accuracy

RF_Predictions=as.double(RF_Predictions)
pred=prediction(RF_Predictions,test$target)
perf = performance(pred,"tpr","fpr")
plot(perf,colorize=TRUE)

accuracy.meas(RF_Predictions,test$target)

### Look at the roc curve and f1-score. Random Forest is performing horrible on imbalanced data

### Let's use Random Forest on synthetic data 

RF_model = randomForest(target ~ ., train_rose, importance = TRUE, ntree = 100,seed=2)
RF_Predictions = predict(RF_model, test_rose[,-1])

ConfMatrix_RF=table(test_rose$target,RF_Predictions)
ConfMatrix_RF
Accuracy=(13606+13840)/(6479+13606+13840+6074)
Accuracy

RF_Predictions=as.double(RF_Predictions)
pred=prediction(RF_Predictions,test_rose$target)
perf = ROCR::performance(pred,"tpr","fpr")
plot(perf,colorize=TRUE)

accuracy.meas(RF_Predictions,test_rose$target)

### Accuracy=68.59171

### Even though RandoForest is giving better roc curve and f-1 score but it is giving accuracy of
### Just 68.59%

##### Let's implement lightgbm model ########
library(lightgbm)

X_train=as.matrix(train1[,-c(1)])
y_train=as.matrix(train1$target)
X_test=as.matrix(test[,-c(1)])
y_test=as.matrix(test$target)
test_lgb=as.matrix(test[,-c(1)])

#training data
lgb_train= lgb.Dataset(data=X_train, label=y_train)
#test data
lgb_test=lgb.Dataset(data=X_test,label=y_test)

## Hyperparameters
set.seed(653)
param = list(objective = "binary",
                metric = "auc",
                boost='gbdt',
                max_depth=-1,
                boost_from_average='false',
                min_sum_hessian_in_leaf = 12,
                feature_fraction = 0.05,
                bagging_fraction = 0.45,
                bagging_freq = 5,
                learning_rate=0.02,
                tree_learner='serial',
                num_leaves=20,
                num_threads=5,
                min_data_in_bin=150,
                min_gain_to_split = 30,
                min_data_in_leaf = 90,
                verbosity=-1,
                is_unbalance = TRUE)

set.seed(7663)
lgbm.model=lgb.train(params = param, data = lgb_train, nrounds =20000,eval_freq =1000,
           valids=list(val1=lgb_train,val2=lgb_test),early_stopping_rounds = 5000)


### Based on auc score of valid1 and valid2 we can say that the lightgbm is performing best 

#### predicting test data using lightgbm model
test_df=read.csv("test.csv")
dim(test_df)
test_df=test_df[,-1]

test_df1=as.matrix(test_df)
lgbm_pred_prob=predict(lgbm.model,test_df1)
lgbm_pred=ifelse(lgbm_pred_prob>0.5,1,0)

test_df$target=lgbm_pred
table(test_df$target)
