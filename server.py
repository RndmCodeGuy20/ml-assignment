from flask import Flask, render_template, request
import pickle
import numpy as np

# from tensorflow.keras.models import load_model

app = Flask(__name__)

classification = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}


def SVM_Model(sl, sw, pl, pw):
    savedModel = pickle.load(open("./static/saved_models/SVM_Model.sav", "rb"))
    y_hat = savedModel.predict([[sl, sw, pl, pw]])
    return y_hat[0]


def DecisionTree(sl, sw, pl, pw):
    savedModel = pickle.load(open("./static/saved_models/DT_Model.sav", "rb"))
    y_hat = savedModel.predict([[sl, sw, pl, pw]])
    return y_hat[0]


def NeuralNetwork(sl, sw, pl, pw):
    savedModel = load_model("./static/saved_models/NN_Model.h5")
    return np.argmax(savedModel.predict([[sl, sw, pl, pw]]))


@app.route("/")
def home():
    result = ""
    return render_template("index.html", **locals())


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.form["model"] == "Support Vector Machines":
        result = classification[
            SVM_Model(
                float(request.form["sepal_length"]),
                float(request.form["sepal_width"]),
                float(request.form["petal_length"]),
                float(request.form["petal_width"]),
            )
        ]

    if request.form["model"] == "Decision Tree":
        result = classification[
            DecisionTree(
                float(request.form["sepal_length"]),
                float(request.form["sepal_width"]),
                float(request.form["petal_length"]),
                float(request.form["petal_width"]),
            )
        ]

    if request.form["model"] == "Neural Networks":
        result = classification[
            NeuralNetwork(
                float(request.form["sepal_length"]),
                float(request.form["sepal_width"]),
                float(request.form["petal_length"]),
                float(request.form["petal_width"]),
            )
        ]

    return render_template("index.html", **locals())


if __name__ == "__main__":
    app.run(debug=True)
