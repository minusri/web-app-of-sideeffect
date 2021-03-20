# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 19:56:45 2021

@author: DHYAN
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
df=pd.read_csv("minu.csv")
LE=LabelEncoder()
df=df.drop(['Date','DrugId','Satisfaction','Reviews','UsefulCount'],axis=1)
print(df.head())
df['Age'] = df['Age'].fillna( df['Age'].dropna().mode().values[0] )
df['Condition'] = df['Condition'].fillna(df['Condition'].mode()[0])
df['Sex'] = df['Sex'].fillna( df['Sex'].dropna().mode().values[0] )
mappings=list()
cols=['Age','Sex','Condition','Drug','sideeffect']
LE=LabelEncoder()
for x in cols:
    df[x]=LE.fit_transform(df[x])
    mappings_dict={index:label for index, label in enumerate(LE.classes_)}
    mappings.append(mappings_dict)
print(mappings)
x=df.drop('sideeffect',axis=1)
y=df['sideeffect']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()

#from sklearn.ensemble import GradientBoostingClassifier
#classifier=GradientBoostingClassifier(n_estimators=58,random_state=123,min_samples_split=0.07)
m=rf.fit(x_train,y_train)
pickle.dump(rf,open('model.pkl','wb'))

example_dict={1: 'Mild Side Effects', 2: 'Moderate Side Effects', 3: 'No Side Effects', 4: 'Severe Side Effects',0:'Extremely severe side effects'}

pickle_out=open("dict_pickle.pkl","wb")
pickle.dump(example_dict,pickle_out)
pickle_out.close()