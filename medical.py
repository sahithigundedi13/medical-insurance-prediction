from flask import Flask, request, render_template, url_for
import pickle
import numpy as np
import json
import requests



app = Flask(__name__)
with open('pymodel.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route("/")
def f():
    return render_template("index.html")

@app.route("/inspect")
def inspect():
    return render_template("inspect.html")


@app.route("/output", methods=["GET", "POST"])
def output():
    if request.method == 'POST':
        var1 = request.form["AGE"]
        var2 = request.form["GENDER"]
        var3 = request.form["BMI"]
        var4 = request.form["CHILDREN"]
        var5 = request.form["SMOKER"]
        var6 = request.form["REGION"]
        

        # Convert the input data into a numpy array
        predict_data = np.array([var1, var2, var3, var4, var5, var6]).reshape(1, -1)

        # Use the loaded model to make predictions
        predict = model.predict(predict_data)
        return render_template('output.html', predict=predict)
    return render_template("output.html")

if __name__ == "__main__":
    app.run(debug=False)
