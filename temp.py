# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np
from sklearn.linear_model import LinearRegression


df = pd.read_csv("D:/2020秋季学期/机器学习/医疗花费预测/train.csv")
#print(df.head(5))



df["region"] = df["region"].replace("northwest",0)
df["region"] = df["region"].replace("southwest",1)
df["region"] = df["region"].replace("northeast",2)
df["region"] = df["region"].replace("southeast",3)

df["smoker"] = df ["smoker"].replace("yes",1)
df["smoker"] = df ["smoker"].replace("no",0)

df["sex"] = df ["sex"].replace("male",1)
df["sex"] = df ["sex"].replace("female",0)

dummy_ranks_a = pd.get_dummies(df['children'],prefix = 'children')
dummy_ranks_b = pd.get_dummies(df['region'],prefix = 'region')


'''
cols_to_keep = ['age', 'sex', 'bmi','smoker']
data = df[cols_to_keep].join(dummy_ranks.ix[:, 'children_2':])

dummy_ranks2 = pd.get_dummies(df['region'],prefix = 'region')
data = data.join(dummy_ranks2.ix[:, 'region_2':])

print(data.head(5))

'''
'''
df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
'''
train= pd.concat([df,dummy_ranks_a,dummy_ranks_b],axis = 1)
#train.drop(['children','region'], axis=1, inplace=True)
train = df

df2 = pd.read_csv("D:/2020秋季学期/机器学习/医疗花费预测/test_sample.csv")
#print(df.head(5))



df2["region"] = df2["region"].replace("northwest",0)
df2["region"] = df2["region"].replace("southwest",1)
df2["region"] = df2["region"].replace("northeast",2)
df2["region"] = df2["region"].replace("southeast",3)

df2["smoker"] = df2 ["smoker"].replace("yes",1)
df2["smoker"] = df2 ["smoker"].replace("no",0)

df2["sex"] = df2 ["sex"].replace("male",1)
df2["sex"] = df2 ["sex"].replace("female",0)

dummy_ranks_a2 = pd.get_dummies(df2['children'],prefix = 'children')
dummy_ranks_b2 = pd.get_dummies(df2['region'],prefix = 'region')

test= pd.concat([df2,dummy_ranks_a2,dummy_ranks_b2],axis = 1)
#test.drop(['children','region'], axis=1, inplace=True)
test = df2


order = ['charges','age','sex','bmi','smoker','children','region']
train=  train[order]
test = test[order]

orderr = ['age','sex','bmi','smoker','children','region','charges']

train_cols = train.columns[1:]

model = LinearRegression()

model.fit(train[train_cols],train['charges'])


#combos['predict'] = result.predict(combos[predict_cols])

test_cols = test.columns[1:]
test['charges'] = model.predict(test[test_cols])

test = test[orderr]


pd_data = pd.DataFrame(test)
pd_data.to_csv(r"D:/2020秋季学期/机器学习/医疗花费预测/submission.csv",encoding='utf_8_sig',index = False)









