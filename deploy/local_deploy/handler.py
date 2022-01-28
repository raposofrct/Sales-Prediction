from flask import Flask,request,Response
from rossmann import rossmann
import pickle
import pandas as pd

app = Flask(__name__)

@app.route('/predict',methods=['POST'])

def predict():
    
    json = request.get_json() # Request json

    if json: # Foram de fato passados dados
        if isinstance( json, dict ): # um único exemplo
            df_json = pd.DataFrame( json, index=[0] ) 
        else:
            df_json = pd.DataFrame( json, columns=json[0].keys() ) # múltiplos examplos
            
        dados = df_json.copy() # Data copy (starting data malipulation)
        dados = rossmann().data_cleaning(dados) # Data cleaning
        dados = rossmann().feature_engineering(dados) # Data Feature Engineering
        dados = rossmann().data_preprocessing_feature_selection(dados) # Data Preprocessing and Feature Selection
        json_pred = rossmann().predict(dados) # Model prediction

        return json_pred # predictions (already formatted  to json in rossmann class)

    else:
        return Response('{}',status=200,mimetype='application/json') # retorna vazio
    
if __name__=='__main__':
    app.run(host='0.0.0.0',port=8080,debug=False)