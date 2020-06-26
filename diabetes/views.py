from django.shortcuts import render
from django.http import HttpResponse
import numpy as np
import joblib


# Create your views here.

def index(request):
    return render(request, 'index.html')

def eda(request):
    return render(request, 'eda.html')

def result(request):
    if request.method == "POST":
        clf = joblib.load('./staticfiles/finalized_model.sav')

        pregnant = request.POST.get('pregnant')
        glucose = request.POST.get('glucose')
        bloodp = request.POST.get('bloodpressure')
        skin = request.POST.get('skin')
        insulin = request.POST.get('insulin')
        bmi = request.POST.get('bmi')
        pedigree = request.POST.get('pedigree')
        age = request.POST.get('age')

        var = np.array([pregnant,glucose,bloodp,skin,insulin,bmi,pedigree,age])
        var = var.reshape(1,-1)
        ans  = clf.predict(var)
        prob = clf.predict_proba(var)
        
        if(ans[0]==0):
            context = {'ans':ans, 'prob':prob[0][0], 'prob1' : prob[0][1] , 'color':'green', 'note': 'Hey, You are fine!!' }
            return render(request, 'result.html', context)
            
        else:
            context = {'ans':ans, 'prob':prob[0][0], 'prob1' : prob[0][1], 'color':'red', 'note': 'Oops! It seems you are suffering from diabetes, Please consult your doctor.'}
            return render(request, 'result.html', context)
        

    