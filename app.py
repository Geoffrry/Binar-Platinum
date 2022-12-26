import pandas as pd
import pickle, os, json, csv
import numpy as np
from module.Cleansing import cleansing

from flask import Flask, jsonify
from flask import request
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from

from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

app = Flask(__name__)
app.config['DEBUG'] = True
UPLOAD_FOLDER = 'static/files'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.json_encoder = LazyJSONEncoder
swagger_template = dict(
    info =  {
        'title' : LazyString(lambda: 'API Documentation for Sentiment Classification Using Neural Network and LSTM Model'),
        'version' : LazyString(lambda: '1.0.0'),
        'description' : LazyString(lambda: 'Dokumentasi API untuk Klasifikasi Sentiment dengan Model Neural Network dan LSTM')
    },
    host = LazyString(lambda: request.host)
)

swagger_config =    {
    'headers' : [],
    'specs':    [
        {
            'endpoint' : 'docs',
            'route' : '/docs.json'
        }
    ],
    'static_url_path' : '/flassger_static',
    'swagger_ui' : True,
    'specs_route' : '/docs/'
}

swagger = Swagger(app, template = swagger_template, config = swagger_config)

# Load Pickle and model - Neural Network
cv = pickle.load(open('pickle/feature_cv.pkl', 'rb'))
model_neural = loaded_model = pickle.load(open('pickle/model_neural.pkl', 'rb'))

# Load Pickle and model - LSTM
sentiment = ['negatif', 'netral', 'positif']
load_tokenizer = pickle.load(open('pickle/tokenizer.pickle', 'rb'))
load_sequencer = pickle.load(open('pickle/x_pad_sequences.pickle', 'rb'))
model_lstm = load_model('pickle//model.h5')


# First Route for Neural Network - Text
@swag_from("docs/nn-text.yml", methods=['POST'])
@app.route('/nn-text', methods=['POST'])
def nn_text():

    text = request.form.get('text')
    cleanse_text = [cleansing(text)]
    
    feature = cv.transform(cleanse_text)
    prediction = model_neural.predict(feature)

    json_response = {
        'status_code': 200,
        'description': text,
        'sentiment': str(prediction)
    }

    response_data = jsonify(json_response)
    return response_data

# Second Route for Neural Network - File(csv)
@swag_from('docs/nn-file.yml', methods= ['POST'])
@app.route('/nn-file', methods = ['POST'])
def nn_file():
    data = []
    uploaded_file = request.files['file']
    if uploaded_file.filename !=  '':
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        uploaded_file.save(file_path)
        with open(file_path, 'r') as file:
            csv_file = csv.reader(file)
            for row in csv_file:
                data.append(row)

    csvData = pd.DataFrame(data)
    csvData.columns = csvData.iloc[0]
    csvData = csvData[1:]
    csvData.reset_index(drop=True, inplace=True)
    csvData = csvData[['Tweet']]
    csvData = csvData.rename(columns = {'Tweet' :'text'})
    csvData.drop_duplicates(inplace = True, ignore_index = True)
    csvData = csvData['text'].astype(str)
    csvData = csvData.apply(cleansing) # Clean the csv file
            
    #Feature Extraction
    temp = csvData.tolist()
    
    # CV and Neural Network
    X_test = cv.transform(temp) 
    predict_test = model_neural.predict(X_test)
 

    json_response = {
        'status_code': 200,
        'description': 'Kumpulan Tweet dengan Sentimen',
        'sentiment':  dict(zip(list(csvData), list(predict_test)))
    }

    response_data = jsonify(json_response)
    return response_data

# Third Route for LSTM - Text
@swag_from("docs/lstm-text.yml", methods=['POST'])
@app.route('/lstm-text', methods=['POST'])
def lstm_text():

    text = request.form.get('text')
    cleanse_text = [cleansing(text)]

    # feature = tokenizer.texts_to_sequences(cleanse_text)
    feature = load_tokenizer.texts_to_sequences(cleanse_text)
    feature = pad_sequences(feature, maxlen=load_sequencer.shape[1])

    prediction = model_lstm.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]

    json_response = {
        'status_code': 200,
        'description': text,
        'sentiment': get_sentiment
    }

    response_data = jsonify(json_response)
    return response_data

# Fourth Route for LSTM - File
@swag_from("docs/lstm-file.yml", methods=['POST'])
@app.route('/lstm-file', methods=['POST'])
def lstm_file():

    data = []
    uploaded_file = request.files['file']
    if uploaded_file.filename !=  '':
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        uploaded_file.save(file_path)
        with open(file_path, 'r') as file:
            csv_file = csv.reader(file)
            for row in csv_file:
                data.append(row)

    csvData = pd.DataFrame(data)
    csvData.columns = csvData.iloc[0]
    csvData = csvData[1:]
    csvData.reset_index(drop=True, inplace=True)
    csvData = csvData[['Tweet']]
    csvData = csvData.rename(columns = {'Tweet' :'text'})
    csvData.drop_duplicates(inplace = True, ignore_index = True)
    csvData = csvData['text'].astype(str)
    csvData = csvData.apply(cleansing)

    #Feature Extraction
    temp = csvData.tolist()
    feature = load_tokenizer.texts_to_sequences(temp)
    feature = pad_sequences(feature, maxlen=load_sequencer.shape[1])

    #Make prediction
    prediction = model_lstm.predict(feature)
    holder = [ ]
    for i in range(0,len(prediction)):
      polarity = np.argmax(prediction[i])
      holder.append(sentiment[polarity])
    
    json_response = {
        'status_code': 200,
        'description': "Kumpulan Tweet dengan Sentimen",
        'sentiment': dict(zip(list(csvData), holder))
    }

    response_data = jsonify(json_response)
    return response_data

if __name__ == '__main__':
    app.run(debug=True)