from flask import Flask
from flask import *
import pandas as pd
from DataHandling import missing_values
from DataHandling import texthandling
from DataHandling import categorical_cols_test
import pickle

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report


from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

def handle(e):
    print("Exception raised in " + e)

def read_csv(path):
    try:
        if('csv' == path.split(".")[-1]):
            data = pd.read_csv(path)
        else:
            print("The files is not a CSV file")
    except Exception as e:
        handle('file reading')
    return data

def score_and_save(y_pred):
    try:
        data = read_csv('data/fake_job_test.csv')
        columns = ['telecommuting','has_company_logo','has_questions','salary_range','employment_type']
        # columns = ['company_profile', 'requirements','benefits', 'telecommuting','has_company_logo', 'has_questions',
        #                           'required_experience', 'required_education','industry', 'function']

        # data = data.drop(columns=['company_profile', 'requirements','benefits', 'telecommuting','has_company_logo', 'has_questions',
        #                           'required_experience', 'required_education','industry', 'function'])
        for col in columns:
            del data[col]
            
        y_test = data['fraudulent']
        cm = confusion_matrix(y_test, y_pred)
        print("\n"+"SCORES")
        print("confusion matrix")
        print(cm)
        print(classification_report(y_test,y_pred))
        print('F1-Score'+' = '+str(round(f1_score(y_test, y_pred),4)))
        print('Precision'+' = '+str(round(precision_score(y_test, y_pred),4)))
        print('Recall'+' = '+str(round(recall_score(y_test, y_pred),4)))
        print('Accuracy'+' = '+str(round(accuracy_score(y_test,y_pred),4)))

        data['fraud_prediction'] = y_pred
        return cm
        # data.to_csv('predictionoutput/testsetprediction.csv')
    except Exception as e:
        print(e)
        handle('scoring and saving process')

def load_model_predict(data):
    print(data.shape)
    try:
        X_test = data.drop('fraudulent',axis = 1)
        y_test = data['fraudulent']
        # print("X_test")
        # print(X_test)
        scaler = pickle.load( open( "model/scaler.p", "rb" ) )
        X_test = scaler.transform(X_test)
        print("X_test 2")
        print(X_test)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_test)
        print("X_test 3")
        print(X_train)
        filename = 'model/finalized_model.p'
        model = pickle.load(open(filename, 'rb'))

        y_pred = model.predict(X_test)
        print("y_pred")
        print(y_pred)
        return y_pred
        # return score_and_save(y_pred)
    except Exception as e:
        print(e)
        handle('prediction process')


def singlePredict(single_data_point):
    print("single data point: ", single_data_point)
    print("\n\n\n*****n\n\n")
    print(single_data_point)
    print("\n\n\n*****n\n\n")
    single_data_point = single_data_point.drop('fraudulent',axis = 1)
    print(single_data_point)
    # Reshape the data point
    # single_data_point = single_data_point.to_numpy().reshape(1, -1)

    # Scale the data point
    scaler = pickle.load( open( "model/scaler.p", "rb" ) )
    single_data_point = scaler.transform(single_data_point)
    print(single_data_point)
    # Make a prediction
    filename = 'model/finalized_model.p'
    model = pickle.load(open(filename, 'rb'))
    y_pred = model.predict(single_data_point)

    # Print the prediction
    print(y_pred)
    return y_pred[0]


@app.route('/')
# ‘/’ URL is bound with hello_world() function.
def index():
    data = read_csv('data/fake_job_test.csv')
    filtered_data = data[data['fraudulent']==1]
    filtered_data = filtered_data[["job_id","title","location"]]
    return render_template("index.html",headings=filtered_data.columns.values, data=filtered_data.to_numpy())

@app.route("/process", methods=["POST"])
def process():
    # Your Python code here
    data = request.get_json() # Get data sent from the client
    # Process the data
    print(data)
    print(data['jobId'])
    print(type(data['jobId']))

    csv_data = read_csv('data/fake_job_test.csv')

    # idx = csv_data['job_id'].values.tolist().index(data['jobId'])
    # print(type(csv_data.sample().to_numpy[0]))
    fraudIdx = csv_data.index[csv_data["fraudulent"] == 1].tolist()[0]
    print("fraudIdx : ",fraudIdx)
    frow = csv_data.iloc[fraudIdx].to_frame().T
    print(frow)
    idx = csv_data.index[csv_data["job_id"] == int(data['jobId'])].tolist()[0]
    print("index : ", idx)
    # row = csv_data.iloc[idx,:].to_frame()
    row = csv_data.iloc[idx].to_frame().T
    sam = row.sample()
    print(row)
    # print("cols",row.columns.values)
    # print("new DF",newDf)
    columns = ['telecommuting','has_company_logo','has_questions','salary_range','employment_type']
    for col in columns:
        del row[col]
    # newDf = pd.DataFrame(row)

    # data.head()

    print("row ",row)
    # print("newDf ",row)
    fixedMis = missing_values(row)
    # print(fixedMis)
    handledText =texthandling(fixedMis,idx)
    # print(handledText)
    catecols= categorical_cols_test(handledText)
    # print(catecols)
    confusion_matrix = load_model_predict(catecols)
    # print("last data",confusion_matrix)
    resp = confusion_matrix[0]
    print(resp)
    # (row.pipe(missing_values).pipe(texthandling)
    #             .pipe(categorical_cols_test).pipe(singlePredict))

    return str(resp)