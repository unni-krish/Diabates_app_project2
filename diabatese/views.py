from django.shortcuts import render,redirect
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.decomposition import PCA
# from django.http import HttpResponse


def index(request):
    input_data = None
    k=0
    if request.method == "POST":
        Pregnancies = request.POST.get('Pregnancies')
        Glucose = request.POST.get('Glucose')
        BloodPressure = request.POST.get('BloodPressure')
        SkinThickness = request.POST.get('SkinThickness')
        Insulin = request.POST.get('Insulin')
        BMI = request.POST.get('BMI')
        DiabetesPedigree = request.POST.get('DiabetesPedigree')
        Age = request.POST.get('Age')
        if Pregnancies and Glucose and BloodPressure and SkinThickness and Insulin and BMI and DiabetesPedigree and Age:
            input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigree, Age]])
    df = pd.read_csv('C:\\Users\\ukrs2\\Downloads\\diabetes.csv')
    sc = StandardScaler()
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_new = sc.fit_transform(X)
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_new, y)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=10)
    rf=RandomForestClassifier(criterion= 'entropy', n_estimators= 20,random_state=2)
    rf.fit(X_train, y_train)

    if input_data is not None:  # check if input_data has been updated
        result = rf.predict(sc.transform(input_data))
        k=1
    else:
        result = None
        print(result)

    return render(request, 'index.html', {'result': result,'k':k})

# Create your views here.
