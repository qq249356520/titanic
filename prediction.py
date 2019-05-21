# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

data_train = pd.read_csv('./dataset/train.csv')
#print(data_train)
#查看列信息状态
#data_train.info()
#查看信息分布
#print(data_train.describe())
plt.rcParams[u'font.sans-serif'] = ['simhei'] #用simhei 字体显示中文
plt.rcParams['axes.unicode_minus'] = False  #这个用来正常显示负号

#数据预处理
from sklearn.ensemble import RandomForestRegressor
#使用随机森林拟合缺失的年龄数据， 使用RandomForestRegressor填补缺失的年龄属性
def set_missing_ages(df):
    #把已有的数值特征取出来丢进RandomForestRegressor中
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    #乘客分为已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].values
    unknown_age = age_df[age_df.Age.isnull()].values

    y = known_age[:, 0] #y即目标年龄
    x = known_age[:, 1:] #x为特征属性值
    #将其fit到rfr中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(x, y)
    #用得到的模型进行为止年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1:])
    #用得到的结果填补原缺失的数据df.loc[(df.Cabin.notnull()), 'Cabin'] = 'Yes'

    df.loc[df.Age.isnull(), 'Age'] = predictedAges
    return df, rfr
def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = 'Yes'
    df.loc[(df.Cabin.isnull()), 'Cabin'] = 'No'
    return df

data_train, rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)

#print(data_train)


#特征因子化
dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')
dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix='Embark')
dummies_Sex = pd.get_dummies(data_train['Sex'], prefix='Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')
df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
print(df)

#数据标准化(高斯分布)
import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(df['Age'].values.reshape(-1, 1))
df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1, 1), age_scale_param)
face_scale_param = scaler.fit(df['Fare'].values.reshape(-1, 1))
df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1, 1), face_scale_param)
print(df)

#逻辑回归建模（sklearn）
from sklearn import linear_model
from sklearn import model_selection  #交叉验证
#用正则取出所需的属性
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.values
y = train_np[:, 0] #Survival结果
X = train_np[:, 1:] #属性特征值
#将其fit到LR中
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(X, y)
print(clf)
print(model_selection.cross_val_score(clf, X, y, cv=5))

#对测试数据进行预处理，过程与训练集相同
data_test = pd.read_csv('./dataset/test.csv')
data_test.loc[(data_test.Fare.isnull()), 'Fare'] = 0
#对test做和train一致的特征变换，还是首先用RF填上丢失的年龄
tmp_df = data_test[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].values
#根据X预测年龄并补上
X = null_age[:, 1:]
predictedAges = rfr.predict(X) #rfr代表与训练集采用相同的分布
data_test.loc[(data_test.Age.isnull()), 'Age'] = predictedAges
data_test = set_Cabin_type(data_test)

dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix='Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix='Embark')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix='Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix='Pclass')
df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'].values.reshape(-1, 1), age_scale_param)
df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'].values.reshape(-1, 1), face_scale_param)
print(df_test)


test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(test)
result = pd.DataFrame({'PassengerId' : data_test['PassengerId'].values, 'Survived' : predictions.astype('int32')})
result.to_csv('predict_res.csv', index=False)
print(result)

