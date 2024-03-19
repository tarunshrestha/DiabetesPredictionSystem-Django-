from django.shortcuts import render, redirect

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score




# Create your views here.
def index(request):
    return render(request, 'index.html', {})


def PredictForm(request):
    return render(request, 'form.html', {})


def result(request):
    if request.method == 'POST':
        data = pd.read_csv('diabetes.csv')

        X = data.drop('Outcome', axis=1)
        y = data['Outcome']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = LogisticRegression()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        val1 = float(request.POST['Pregnancies'])
        val2 = float(request.POST['Glucose'])
        val3 = float(request.POST['BloodPressure'])
        val4 = float(request.POST['SkinThickness'])
        val5 = float(request.POST['Insulin'])
        val6 = float(request.POST['BMI'])
        val7 = float(request.POST['DiabetesPedigreeFunction'])
        val8 = float(request.POST['Age'])

        pred = model.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])
        if pred[0] == 1:
            result = "Diabetic"
        else:
            result = "Not Diabetic"

    return render(request, 'form.html', {'result':result})
