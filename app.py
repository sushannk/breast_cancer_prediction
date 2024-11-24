from distutils.log import warn
import string
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify, redirect, url_for
import pickle

app = Flask(__name__, template_folder='templates')
model = pickle.load(open('Breast_Cancer_Detector.pickle','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/prediction_form')
def preiction_form():
    return render_template('prediction_form.html')

@app.route('/index')
def home_return():
    return redirect(url_for('home'))
    
@app.route('/predict',methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    
    features_name = ["radius mean", "radius se", "radius worst", "texture mean", "texture se", "texture worst",
    "perimeter mean", "perimeter se", "perimeter worst", "area mean", "area se", "area worst", "smoothness mean",
    "smoothness se", "smoothness worst", "compactness mean", "compactness se", "compactness worst", "concavity mean",
    "concavity se", "concavity worst", "concave points mean", "concave points se", "concave points worst",
    "symmetry mean", "symmetry se", "symmetry worst", "fractal dimension mean", "fractal dimension se", "fractal dimension worst"]
         
    df = pd.DataFrame(features_value, columns=features_name)
    
    output = model.predict(df)
        
    if output == 1:
        res_val = " breast cancer i.e., Malignant Tumour "
    else:
        res_val = " no breast cancer i.e., Benign Tumour"
        

    return render_template('prediction_form.html', prediction_text='Patient has probable chances of {}'.format(res_val))

if __name__ == "__main__":
    app.run(debug=True)
