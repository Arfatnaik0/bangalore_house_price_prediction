import pandas as pd
from flask import Flask, request, render_template
import joblib


model=joblib.load('../model/bangalore_house_price_model.pkl')
loc_name=joblib.load('../model/loc_name.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data=request.form.to_dict()

    # preprocess input data
    data['location']=data['location'].lower()
    data['size']=int(data['size'])
    data['total_sqft']=float(data['total_sqft'])
    data['bath']=int(data['bath'])
    data['balcony']=int(data['balcony'])

    # handle unseen locations
    if data['location'] not in loc_name:
        data['location']='other'

    input_data = pd.DataFrame([data])
    prediction = model.predict(input_data)
    return render_template('index.html', prediction=f'Predicted House Price: {prediction[0]:.2f} lakhs')

if __name__ == '__main__':
    app.run(debug=True)