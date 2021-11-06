from flask import Flask,render_template,url_for,request
import numpy as np
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from sklearn.metrics.pairwise import cosine_similarity

app=Flask(__name__)

mnb = pickle.load(open('Naive_Bayes_model_imdb.pkl','rb'))
countVect = pickle.load(open('countVect_imdb.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

    if request.method == 'POST':
        Reviews = request.form["review"]
        data = [Reviews]
        vect = countVect.transform(data).toarray()
        my_prediction = mnb.predict(vect)
        neg=mnb.predict_proba(vect)[0][0]*100
        pos=mnb.predict_proba(vect)[0][1]*100
    
    return render_template('result.html',prediction = my_prediction, neg="{:.2f}".format(neg),pos="{:.2f}".format(pos))



if __name__ == '__main__':
    app.run(debug=True)
    
