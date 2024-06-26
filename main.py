from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('heart.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

app = Flask(__name__)


@ app.route('/')

def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['age']
    data2 = request.form['sex']
    data3 = request.form['chest pain type']
    data4 = request.form['resting bp s']
    data5 = request.form['cholesterol']
    data6 = request.form['fasting blood sugar']
    data7 = request.form['resting ecg']
    data8 = request.form['max heart rate']
    data9 = request.form['exercise angina']
    data10 = request.form['old peak']
    data11 = request.form['ST slope']
    arr = np.array([[data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11]])
    test_x = scaler.transform(arr)
    pred = model.predict(test_x)
    
    return render_template('preds.html', data=pred)
if __name__ == "__main__":
    app.run(debug=True)