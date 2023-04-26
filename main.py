import pandas as pd
from flask import Flask, render_template, request, url_for, flash, redirect
from werkzeug.utils import secure_filename
import os
from sklearn.model_selection import train_test_split
import shutil
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from oversampling import X,y,X_test,y_test
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score  
import pygal
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score


app=Flask(__name__)
app.config['UPLOAD_FOLDER']="C:\\Users\\kireeti\\Downloads\\brain_stroke_main_1\\brain_stroke_main\\uploadcsv"
app.config['SECRET_KEY']='b0b4fbefdc48be27a6123605f02b6b86'

@app.route("/")
def main():
    return render_template("index.html")

@app.route("/upload",methods=['POST','GET'])
def upload():
    if request.method=='POST':
        myfile = request.files['filename']
        ext = os.path.splitext(myfile.filename)[1]
        print("1111!!!!!!")
        print(ext)
        if ext.lower() == ".csv":
            shutil.rmtree(app.config['UPLOAD_FOLDER'])
            os.mkdir(app.config['UPLOAD_FOLDER'])
            myfile.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(myfile.filename)))
            flash('The data is loaded successfully', 'success')
            return render_template('upload.html')
        else:
            flash('Please upload a CSV type document only', 'warning')
            return render_template('upload.html')
    return render_template("upload.html")

@app.route('/view')
def view():
    #dataset
    myfile=os.listdir(app.config['UPLOAD_FOLDER'])
    global full_data
    full_data=pd.read_csv(os.path.join(app.config["UPLOAD_FOLDER"],myfile[0]))
    return render_template('view_dataset.html', col=full_data.columns.values, df=list(full_data.values.tolist()))

@app.route('/split', methods=['POST','GET'])
def split():
    if request.method=="POST":
        test_size=float(request.form['size'])
        global X_train_SMOTE, y_train_SMOTE
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        SMOT = SMOTE()
        X_train_SMOTE, y_train_SMOTE = SMOT.fit_resample(X_train, y_train)
        flash('The dataset is transformed and split successfully','success')
        return redirect(url_for('model_performance'))
    return render_template('split_dataset.html')

@app.route('/model_performance', methods=['GET','POST'])
def model_performance():
    if request.method=="POST":
        model_no=int(request.form['algo'])
        if model_no==0:
            print("U have not selected any model")
        elif model_no==1:
            model = SVC()
            model.fit(X_train_SMOTE, y_train_SMOTE)
            y_pred = model.predict(X_test)
            import numpy as np
            print(np.unique(y_pred))
            # re1 = recall_score(y_test, y_pred, average='micro')
            # pr1 = precision_score(y_test, y_pred, average='micro')
            # roc1 = roc_auc_score(y_true=y_test, y_score=y_pred, average='micro')
            accuracyscore = accuracy_score(y_test, y_pred)
            return render_template('train_model.html', acc=accuracyscore, model=model_no)

        elif model_no == 2:
            model = RandomForestClassifier(n_estimators=20)
            model.fit(X_train_SMOTE, y_train_SMOTE)
            y_pred = model.predict(X_test)
            global re2,pr2
            # re2 = recall_score(y_test, y_pred, average='micro')
            # pr2 = precision_score(y_test, y_pred, average='micro')
            # roc2 = roc_auc_score(y_true=y_test, y_score=y_pred, average='micro')
            accuracyscore = accuracy_score(y_test, y_pred)
            return render_template('train_model.html', acc=accuracyscore, model=model_no)

        elif model_no == 3:
            model = DecisionTreeClassifier(criterion='entropy', random_state=0)
            model.fit(X_train_SMOTE, y_train_SMOTE)
            y_pred = model.predict(X_test)
            global re3, pr3
            # re3 = recall_score(y_test, y_pred, average='micro')
            # pr3 = precision_score(y_test, y_pred, average='micro')
            # roc3 = roc_auc_score(y_true=y_test, y_score=y_pred, average='micro')
            accuracyscore = accuracy_score(y_test, y_pred)
            return render_template('train_model.html', acc=accuracyscore, model=model_no)
        elif model_no == 4:
            model = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
            model.fit(X_train_SMOTE, y_train_SMOTE)
            y_pred = model.predict(X_test)
            global re4, pr4
            # re4 = recall_score(y_test, y_pred, average='micro')
            # pr4 = precision_score(y_test, y_pred, average='micro')
            # roc4 = roc_auc_score(y_true=y_test, y_score=y_pred, average='micro')
            accuracyscore = accuracy_score(y_test, y_pred)
            return render_template('train_model.html', acc=accuracyscore, model=model_no)
        elif model_no == 5:
            model = XGBClassifier()
            model.fit(X_train_SMOTE, y_train_SMOTE)
            y_pred = model.predict(X_test)
            global re5, pr5
            # re5 = recall_score(y_test, y_pred, average='micro')
            # pr5 = precision_score(y_test, y_pred, average='micro')
            # roc5 = roc_auc_score(y_true=y_test, y_score=y_pred, average='micro')
            accuracyscore = accuracy_score(y_test, y_pred)
            return render_template('train_model.html', acc=accuracyscore, model=model_no)
    return render_template('train_model.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method=='POST':
        f1 = request.form['f1']
        f2 = request.form['f2']
        f3 = request.form['f3']
        f4 = request.form['f4']
        f5 = request.form['f5']
        f6 = request.form['f6']
        f7 = request.form['f7']
        f8 = request.form['f8']
        f9 = request.form['f9']
        f10 = request.form['f10']
        print("11111111")
        all_obj_vals=[[float(f1),float(f2),float(f3),float(f4),float(f5),float(f6),float(f7),float(f8),float(f9),float(f10)]]
        model = RandomForestClassifier(n_estimators=20)
        model.fit(X_train_SMOTE, y_train_SMOTE)
        pred=model.predict(all_obj_vals)
        return render_template('predict.html',pred=pred)
    return render_template('predict.html')

@app.route("/bar_chart")
def bar_chart():
    model1 = SVC()
    model1.fit(X_train_SMOTE, y_train_SMOTE)
    y_pred1 = model1.predict(X_test)
    accuracyscore1 = accuracy_score(y_test, y_pred1)

    model2 = RandomForestClassifier(n_estimators=20)
    model2.fit(X_train_SMOTE, y_train_SMOTE)
    y_pred2 = model2.predict(X_test)
    accuracyscore2 = accuracy_score(y_test, y_pred2)

    model3 = DecisionTreeClassifier(criterion='entropy', random_state=0)
    model3.fit(X_train_SMOTE, y_train_SMOTE)
    y_pred3 = model3.predict(X_test)
    accuracyscore3 = accuracy_score(y_test, y_pred3)

    model4 = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
    model4.fit(X_train_SMOTE, y_train_SMOTE)
    y_pred4 = model4.predict(X_test)
    accuracyscore4 = accuracy_score(y_test, y_pred4)

    model5 = XGBClassifier()
    model5.fit(X_train_SMOTE, y_train_SMOTE)
    y_pred5 = model5.predict(X_test)
    accuracyscore5 = accuracy_score(y_test, y_pred5)

    line_chart = pygal.Bar()
    line_chart.title = 'Classification of stroke disease using machine learning algorithms and their accuracy scores'
    line_chart.add('SVM', [accuracyscore1])
    line_chart.add('Random Forest', [accuracyscore2])
    line_chart.add('Decision Tree', [accuracyscore3])
    line_chart.add('SGD', [accuracyscore4])
    line_chart.add('XGBoost', [accuracyscore5])
    graph_data = line_chart.render()
    return render_template('bar_chart.html', graph_data=graph_data)

@app.route("/recall")
def recall():
    model1 = SVC()
    model1.fit(X_train_SMOTE, y_train_SMOTE)
    y_pred = model1.predict(X_test)
    import numpy as np
    print(np.unique(y_pred))
    global pr1
    global re1
    global roc1
    re1 = recall_score(y_test, y_pred, average='micro')
    pr1 = precision_score(y_test, y_pred, average='micro')
    roc1 = roc_auc_score(y_true=y_test, y_score=y_pred, average='micro')

    model2 = RandomForestClassifier(n_estimators=20)
    model2.fit(X_train_SMOTE, y_train_SMOTE)
    y_pred = model2.predict(X_test)
    global re2,pr2,roc2
    re2 = recall_score(y_test, y_pred, average='micro')
    pr2 = precision_score(y_test, y_pred, average='micro')
    roc2 = roc_auc_score(y_true=y_test, y_score=y_pred, average='micro')

    model3 = DecisionTreeClassifier(criterion='entropy', random_state=0)
    model3.fit(X_train_SMOTE, y_train_SMOTE)
    y_pred = model3.predict(X_test)
    global re3, pr3,roc3
    re3 = recall_score(y_test, y_pred, average='micro')
    pr3 = precision_score(y_test, y_pred, average='micro')
    roc3 = roc_auc_score(y_true=y_test, y_score=y_pred, average='micro')

    model4 = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
    model4.fit(X_train_SMOTE, y_train_SMOTE)
    y_pred = model4.predict(X_test)
    global re4, pr4, roc4
    re4 = recall_score(y_test, y_pred, average='micro')
    pr4 = precision_score(y_test, y_pred, average='micro')
    roc4 = roc_auc_score(y_true=y_test, y_score=y_pred, average='micro')

    model5 = XGBClassifier()
    model5.fit(X_train_SMOTE, y_train_SMOTE)
    y_pred = model5.predict(X_test)
    global re5, pr5, roc5
    re5 = recall_score(y_test, y_pred, average='micro')
    pr5 = precision_score(y_test, y_pred, average='micro')
    roc5 = roc_auc_score(y_true=y_test, y_score=y_pred, average='micro')

    line_chart = pygal.Bar()
    line_chart.title = 'Recall Score'
    line_chart.add('SVM', [re1])
    line_chart.add('Random Forest', [re2])
    line_chart.add('Decision Tree', [re3])
    line_chart.add('SGD', [re4])
    line_chart.add('XGBoost', [re5])
    graph_data = line_chart.render()
    return render_template('recall.html', recall_graph=graph_data)


@app.route("/precision")
def precision():
    line_chart = pygal.Bar()
    line_chart.title = 'Precision Score'
    line_chart.add('SVM', [pr1])
    line_chart.add('Random Forest', [pr2])
    line_chart.add('Decision Tree', [pr3])
    line_chart.add('SGD', [pr4])
    line_chart.add('XGBoost', [pr5])
    graph_data = line_chart.render()
    return render_template('precision.html', precision_graph=graph_data)

@app.route("/rocauc")
def rocauc():
    line_chart = pygal.Bar()
    line_chart.title = 'ROC AUC Score'
    line_chart.add('SVM', [roc1])
    line_chart.add('Random Forest', [roc2])
    line_chart.add('Decision Tree', [roc3])
    line_chart.add('SGD', [roc4])
    line_chart.add('XGBoost', [roc5])
    graph_data = line_chart.render()
    return render_template('rocauc.html', rocauc_graph=graph_data)

if(__name__)==("__main__"):
    app.run(debug=True)

