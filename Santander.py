#!/usr/bin/env python
# coding: utf-8

# In[20]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,auc,roc_auc_score,classification_report
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn import tree
import lightgbm as lgb


# In[22]:


os.chdir("/home/zozo/Documents/edwisor/Project/Santander")


# In[23]:


os.getcwd()


# In[24]:


train=pd.read_csv("train.csv")


# In[25]:


train.shape


# In[6]:


train.head()


# In[31]:


train.describe()


# look at the summary of variables from var_0 to var_199. If you will observe summary of these variables then you will finnd out that the mean value of these variables is not very small when compared to the max value of these variables that means outliers does not exist in dataset.

# In[8]:


train.dtypes


# #### Missing value analysis

# In[11]:


train.isnull().sum().sum()


# as we can see that there are no missing values in the data

# ##### Important Features

# 

# In[46]:


X = train.iloc[:,2:202]  #independent columns
y = train.iloc[:,1]    #target column i.e price range
model = ExtraTreesClassifier(random_state=42)
model.fit(X,y)
print(model.feature_importances_) 
#plot graph of feature importances for better visualization
plt.subplots(figsize=(40,45))
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(200).plot(kind='barh')
plt.show()


# we can see that some of the most important variables are var_81,var_53,var_26 and var_139
# and some of the least important variables in 200 variables are var_20,var_14,var_98 and var_120
# 

# ##### Let's plot histogram of some of the variables

# In[281]:


train['var_81'].hist()


# In[282]:


train['var_53'].hist()


# In[283]:


train['var_20'].hist()


# In[284]:


train['var_14'].hist()


# if you will see all the histograms you can find out histogram of two of most important features are similar and histogram of least 2 important feature are similar

# #### Let's count and visualize the labels of target variable

# In[15]:


train['target'].value_counts()


# In[16]:


sns.countplot(train['target'],palette="Set2")


# In[17]:


train['target'].describe()


# we can see that there are only 10.049% positivie labels present in target variable. There is target class imbalance problem

# #### Let's look at the correlation between variables

# In[18]:


df_corr=train.iloc[:,2:202]


# In[19]:


df_corr.corr()


# as we can see that there is very less correlation among independent variables

# In[ ]:





# ###### Let's build a logistic regression model 

# In[26]:


train=train.drop(columns="ID_code", axis=1)


# In[27]:


X = train.drop(columns="target")
y = train["target"]

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2,stratify=y)
print(X_train.shape)
print(X_test.shape)


# In[28]:


#Logistic Regression
logit=LogisticRegression(random_state=42)
logit_model=logit.fit(X_train,y_train)


# In[29]:


logit_pred=logit_model.predict(X_test) 


# In[33]:


CM = pd.crosstab(y_test,logit_pred)


# In[34]:


CM


# In[35]:


TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]
Accuracy=((TN+TP)/(TN+FP+TP+FN))*100
print(Accuracy)


# even though we are getting high accuracy but sometimes accuracy is not the best measure to evaluate a model specially if we have imbalanced data. So we will be using other measures like roc,auc and f1-scores

# In[32]:


FPR,TPR,thresholds=roc_curve(y_test,logit_pred)


# In[33]:


print(auc(FPR,TPR))


# In[38]:


plt.plot(FPR,TPR,'g')


# In[39]:


print(classification_report(y_test,logit_pred))


# as we can see that f1-score is very low for positive class.
# so by looking at the roc_curve and f1-score we can say that this model is not performing well on imbalanced data
# and we will try other models.

# In[ ]:





# ##### SO let's use SMOTE to handle target class imbalance problem

# In[34]:


sm = SMOTE(random_state=42)
X_train_smote,y_train_smote=sm.fit_sample(X_train,y_train)
X_test_smote,y_test_smote=sm.fit_sample(X_test,y_test)
print(X_train_smote.shape)
print(X_test_smote.shape)


# ###### Applying logistics regression on synthetic data

# In[35]:


smote=LogisticRegression(random_state=42)
smote_model=smote.fit(X_train_smote,y_train_smote)


# In[36]:


smote_pred=smote_model.predict(X_test_smote)


# In[37]:


CM_smote=pd.crosstab(y_test_smote,smote_pred)


# In[38]:


CM_smote


# In[39]:


TN = CM_smote.iloc[0,0]
FN = CM_smote.iloc[1,0]
TP = CM_smote.iloc[1,1]
FP = CM_smote.iloc[0,1]
Accuracy=((TN+TP)/(TN+FP+TP+FN))*100
print(Accuracy)


# In[41]:


FPR,TPR,thresholds=roc_curve(y_test_smote,smote_pred)


# In[44]:


print(auc(FPR,TPR))


# In[55]:


plt.plot(FPR,TPR,'g')


# In[56]:


print(classification_report(y_test_smote,smote_pred))


# we can see from roc_curve and f1-score that logistic regression is performing well on synthetic data 

# In[ ]:





# ##### Decision Tree

# In[29]:





# In[45]:


C50_model = tree.DecisionTreeClassifier(criterion='entropy').fit(X_train, y_train)
#predict new test cases
C50_Predictions = C50_model.predict(X_test)


# In[46]:


CM_dec=pd.crosstab(y_test,C50_Predictions)


# In[47]:


CM_dec


# In[48]:


TN = CM_dec.iloc[0,0]
FN = CM_dec.iloc[1,0]
TP = CM_dec.iloc[1,1]
FP = CM_dec.iloc[0,1]
Accuracy=((TN+TP)/(TN+FP+TP+FN))*100
print(Accuracy)


# In[50]:


FPR,TPR,thresholds=roc_curve(y_test,C50_Predictions)


# In[51]:


print(auc(FPR,TPR))


# In[64]:


plt.plot(FPR,TPR,'g')


# In[65]:


print(classification_report(y_test,C50_Predictions))


# So, by looking at the roc_curve and f1-score we can say that decision tree also not performing well on imbalanced target
# class data

# In[ ]:





# ###### Let's apply decision tree on synthetic data

# In[52]:


C50_model = tree.DecisionTreeClassifier(criterion='entropy').fit(X_train_smote, y_train_smote)
#predict new test cases
C50_Predictions = C50_model.predict(X_test_smote)


# In[53]:


CM_dec=pd.crosstab(y_test_smote,C50_Predictions)


# In[54]:


CM_dec


# In[55]:


TN = CM_dec.iloc[0,0]
FN = CM_dec.iloc[1,0]
TP = CM_dec.iloc[1,1]
FP = CM_dec.iloc[0,1]
Accuracy=((TN+TP)/(TN+FP+TP+FN))*100
print(Accuracy)


# In[57]:


FPR,TPR,thresholds=roc_curve(y_test_smote,C50_Predictions)


# In[58]:


print(auc(FPR,TPR))


# In[71]:


print(classification_report(y_test_smote,C50_Predictions))


# as we can see accuracy is very low when we applied decision tree model on synthetic data that means decision tree is not performing good on synthetic data as well

# In[ ]:





# In[59]:


#training data
lgb_train=lgb.Dataset(X_train_smote,label=y_train_smote)
#test data
lgb_test=lgb.Dataset(X_test_smote,label=y_test_smote)


# In[60]:


param = {'objective' : "binary", 
               'boost':"gbdt",
               'metric':"auc",
               'boost_from_average':"false",
               'num_threads':8,
               'learning_rate' : 0.01,
               'num_leaves' : 13,
               'max_depth':-1,
               'tree_learner' : "serial",
               'feature_fraction' : 0.05,
               'bagging_freq' : 5,
               'bagging_fraction' : 0.4,
               'min_data_in_leaf' : 80,
               'min_sum_hessian_in_leaf' : 10.0,
               'verbosity' : 1}


# In[61]:


num_rounds=20000
lgbm= lgb.train(param,lgb_train,num_rounds,valid_sets=[lgb_train,lgb_test],verbose_eval=400,early_stopping_rounds = 3000)
lgbm


# In[62]:


lgbm_predict_prob=lgbm.predict(X_test_smote,random_state=42,num_iteration=lgbm.best_iteration)
lgbm_predict=np.where(lgbm_predict_prob>=0.5,1,0)


# In[63]:


CM_lgbm=pd.crosstab(y_test_smote,lgbm_predict)


# In[64]:


CM_lgbm


# In[65]:


TN = CM_lgbm.iloc[0,0]
FN = CM_lgbm.iloc[1,0]
TP = CM_lgbm.iloc[1,1]
FP = CM_lgbm.iloc[0,1]
Accuracy=((TN+TP)/(TN+FP+TP+FN))*100
print(Accuracy)


# In[67]:


FPR,TPR,thresholds=roc_curve(y_test_smote,lgbm_predict)


# In[68]:


print(auc(FPR,TPR))


# In[69]:


plt.plot(FPR,TPR,'g')


# In[70]:


print(classification_report(y_test_smote,lgbm_predict))


# from the roc_curve and f1-score we can conclude that Lightgbm model is performing well 

# In[ ]:





# ##### As we got best accuracy,precision,recall and f1-score from lightgbm model we used it predict the test data 

# In[11]:


test=pd.read_csv("test.csv")


# In[287]:


test.shape


# In[288]:


test.head()


# In[12]:


test=test.drop("ID_code",axis=1)


# In[13]:


test.shape


# In[14]:


lgbm_predict_prob=lgbm.predict(test,random_state=42,num_iteration=lgbm.best_iteration)
lgbm_predict=np.where(lgbm_predict_prob>=0.5,1,0)


# In[15]:


test['target']=lgbm_predict


# In[16]:


test.shape


# In[299]:


test.to_csv("predicted_test.csv",index=False)


# In[ ]:




