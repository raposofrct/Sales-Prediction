from flask import Flask,request,Response
from Rossmann import rossmann
import pickle
import pandas as pd

pipeline = rossmann()        
model = pickle.load(open('/Users/nando/Comunidade DS/ds_em_producao/deploy/local_deploy/model.pickle', 'rb'))

app = Flask(__name__)

@app.route('/predict',methods=['POST'])
def predict():
    
    json = request.get_json() # Get the data
    
    if json: # Foram de fato passados dados
        if isinstance( json, dict ): # um único exemplo
            df_json = pd.DataFrame( json, index=[0] ) 
        else:
            df_json = pd.DataFrame( json, columns=json[0].keys() ) # múltiplos examplos
        
        # data modeling
        dados = df_json.copy()
        dados = pipeline.data_cleaning(dados)
        dados = pipeline.feature_engineering(dados)
        dados = pipeline.data_preprocessing(dados)
        dados = pipeline.feature_selection(dados)
        
        # predict
        dados_response = pipeline.get_prediction(model,df_json,dados)
        
        return dados_response # retorna um df com os dados sem modeling e a prediction sem modeling também
        
            
    else: # não foram passados os dados
        return Response('{}',status=200,mimetype='application/json') # retorna vazio
    
    
if __name__=='__main__':
    app.run(host='0.0.0.0',port=8080,debug=False)