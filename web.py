# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 19:38:13 2021

@author: DHYAN
"""

from flask import Flask,render_template,request
import pickle
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
PEOPLE_FOLDER=os.path.join('static')
LE=LabelEncoder()

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
app.config['UPLOAD_FOLDER']=PEOPLE_FOLDER
@app.route('/')
def home():
    return render_template('home.html')

#taking the input values from the home.html
@app.route('/predict',methods=['POST'])

def predict():
    Age=str(request.form['Age'])
    Condition=str(request.form['Condition'])
    Drug=str(request.form['Drug'])
    EaseofUse=float(request.form['EaseofUse'])
    effectiveness=int(request.form['Effectiveness'])
    Sex=str(request.form['Sex'])
    predictinput=['Age','Condition','Drug','EaseofUse','effectiveness','Sex']
   # pickle_in=open("dict_pickle.pkl","rb")
    test=predictinput
    test = LE.fit_transform(test)
    test = np.reshape(test, (1,-1))
    out_test = model.predict(test)
    arr_out = out_test.item()
    
    pickle_in=open("dict_pickle.pkl","rb")
    example_dict=pickle.load(pickle_in)
    output=example_dict[arr_out]
    print(output)
    return render_template ('result.html',prediction_text="This Drug may cause {}".format(output))
def results():
    oip = os.path.join(app.config['UPLOAD_FOLDER'], 'result.jpeg')
    return render_template('result.html', user_image=oip)


if __name__=='__main__':
  app.run(port=8000)
    