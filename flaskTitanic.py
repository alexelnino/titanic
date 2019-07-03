from flask import Flask, render_template, request, jsonify
from sklearn.externals import joblib

app = Flask(__name__)

# @app.route('/')
# def home():
#     return ('<h1>'
#     +str(model.predict([[0, 1, 0, 1, 0, 2, 23, 1, 2, 200, 1, 0]]))+
#     '</h1>')

# [ 0.    1.    0.    1.    0.    3.   22.    1.    0.    7.25  1.    0.  ]   --> the first number (0 1) is the sex column, meanwhile the next numbers ( 0 1 0 ) is the who columns transformed
# read:                fm    male  chld  man   wmn  pclass age  sibsp  parch  fare adultM alone

@app.route('/')
def welcome():
    return '<h1>Welcome</h1>'

@app.route('/postTitanic', methods = ['POST'])
def postTitanic():
    if request.method == 'POST' :
        body = request.json
        sex=body['sex']     # female = 0, male = 1
        age=body['age']
        sibsp=body['sibsp']
        parch=body['parch']
        pclass=body['pclass']
        fare=body['fare']

        female=body['female']
        male=body['male']
        child=body['child']
        man=body['man']


        # [ 0.    1.    0.    1.    0.    3.   22.    1.    0.    7.25    1.    0.  ]   --> the first number (0 1) is the sex column, meanwhile the next numbers ( 0 1 0 ) is the who columns transformed
# read     fm    male  chld  man   wmn  pclass age  sibsp  parch  fare adultM alone

        if int(sex)==0:
            if int(age) <15 :
                female = 1; male = 0; child=1; man=0; woman= 1; adultman= 0
            else:
                female = 1; male = 0; child=0; man=0; woman= 1; adultman= 0 
        else:
            if int(age) <15 :
                female = 0; male = 1; child=1; man=0; woman= 0; adultman= 0
            else:
                female = 0; male = 1; child=0; man=1; woman= 0; adultman= 1
        
        if int(sibsp)==0 and int(parch) ==0:
            alone=1
        woman=body['woman']
        pclass=body['pclass']
        age=body['age']
        sibsp=body['sibsp']
        parch=body['parch']
        fare=body['fare']
        adultman=body['adultman']
        alone=body['alone']
        prediction=model.predict([[
            female, male, child, man, woman, pclass, age, sibsp, parch, fare, adultman, alone
            ]])
        print(female)
        return jsonify({
            '0response' : 'POST succesfull',
            'female':body['female'],
            'male':body['male'],
            'child':body['child'],
            'man':body['man'],
            'woman':body['woman'],
            'pclass':body['pclass'],
            'age':body['age'],
            'sibsp':body['sibsp'],
            'parch':body['parch'],
            'fare':body['fare'],
            'adultman':body['adultman'],
            'alone':body['alone']
            'zPREDIKSI':int(prediksi)
        })

if __name__ == '__main__' :
    model= joblib.load('modelTitanic')
    app.run(debug=True)