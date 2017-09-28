#encoding=utf-8
import pandas as pd
titanic=pd.read_csv("data/train.csv");
titanic.head(3);

titanic["Age"]=titanic["Age"].fillna(titanic["Age"].median());      #中间数
#print(titanic.describe());
#print(titanic["Sex"].unique());             # unique和unique()不一样
titanic.loc[titanic["Sex"]== "male","Sex"]=0;
titanic.loc[titanic["Sex"]=="female","Sex"]=1;  #数值转化
#print(titanic["Sex"].unique());


titanic["Embarked"]=titanic["Embarked"].fillna("S");
titanic.loc[titanic["Embarked"]=="S","Embarked"]=0;
titanic.loc[titanic["Embarked"]=="C","Embarked"]=1;
titanic.loc[titanic["Embarked"]=="Q","Embarked"]=2;

#titanic["Embarked"]=titanic["Embarked"].fillna(titanic["Embarked"].mean());  #均值

#print(titanic["Embarked"].unique());

from sklearn.linear_model import LinearRegression;
from sklearn.cross_validation import KFold;

predictors=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"];

alg=LinearRegression();
#kf=KFold(n_splits=3,random_state=1);
kf=KFold(titanic.shape[0],n_folds=3,random_state=1);
predictions=[];                                                          #每个测试集的测试结果集合
for train,test in kf:                                    #每次切分过后
    train_predictors=(titanic[predictors].iloc[train,:]);
    train_target=titanic["Survived"].iloc[train];
    alg.fit(train_predictors,train_target);
    test_predicions=alg.predict(titanic[predictors].iloc[test,:]);
    predictions.append(test_predicions);

import numpy as np;
predictions=np.concatenate(predictions,axis=0);

predictions[predictions>0.5]=1;
predictions[predictions<=0.5]=0;

accuracy=sum(predictions[predictions==titanic["Survived"]])/len(predictions);
print(accuracy);

# for i in predictions:
#     print(i);





