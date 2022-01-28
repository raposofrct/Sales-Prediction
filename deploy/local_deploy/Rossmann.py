import pandas as pd
import numpy as np
import inflection
import datetime
import pickle

class rossmann(object):
    def __init__(self):
        self.ordinal_encoder = pickle.load(open('/Users/nando/Comunidade DS/ds_em_producao/deploy/local_deploy/ordinal_encoder.pickle','rb'))
        self.target_encoder = pickle.load(open('/Users/nando/Comunidade DS/ds_em_producao/deploy/local_deploy/target_encoder.pickle','rb'))
        self.one_hot_encoder = pickle.load(open('/Users/nando/Comunidade DS/ds_em_producao/deploy/local_deploy/one_hot_encoder.pickle','rb'))
        self.scalers = pickle.load(open('/Users/nando/Comunidade DS/ds_em_producao/deploy/local_deploy/scalers.pickle','rb'))
        self.model = pickle.load(open('/Users/nando/Comunidade DS/ds_em_producao/deploy/local_deploy/model.pickle','rb'))
        
    def data_cleaning(self,dados):
        
        ## Rename Columns and Values
        for c in range(len(dados.columns)):
            dados.rename(columns={dados.columns.values[c]:inflection.underscore(dados.columns.values[c])},inplace=True)

        dados['state_holiday'] = dados['state_holiday'].map({'a':'public holiday', 'b':'Easter holiday', 'c':'Christmas', '0':'None'})
        dados['assortment'] = dados['assortment'].map({'a':'basic','b':'extra','c':'extended'})

        ## Missing Data
        for index in dados[dados['competition_distance'].isna()].index:
            dados.loc[index,'competition_distance'] = 999999 # High values suggests that the competitors are so far that there's no competition
            dados.loc[index,'competition_open_since_month'] = -1 # It suggets that it has never been opened
            dados.loc[index,'competition_open_since_year'] = -1 # It suggets that it has never been opened

        for index in dados[dados['competition_open_since_month'].isna()].index:
            dados.loc[index,'competition_open_since_month'] = 999999 
            dados.loc[index,'competition_open_since_year'] = 999999 

        for index in dados[dados['promo2_since_week'].isna()].index:
            dados.loc[index, 'promo2_since_week'] = -1 # It suggets that there's no promo2 start
            dados.loc[index, 'promo2_since_year'] = -1 # It suggets that there's no promo2 start
            dados.loc[index, 'promo_interval'] = 'No promo2' # It suggets that there's no promo2 start

        ## Data Types
        dados['date'] = pd.to_datetime(dados['date'])
        dados['competition_open_since_month'] = dados['competition_open_since_month'].astype(int)
        dados['competition_open_since_year'] = dados['competition_open_since_year'].astype(int)
        dados['promo2_since_week'] = dados['promo2_since_week'].astype(int)
        dados['promo2_since_year'] = dados['promo2_since_year'].astype(int)
        
        return dados

    def feature_engineering(self,dados):
        # Day, Month, Week and Year Variables
        dados['day'] = pd.to_datetime(dados['date']).dt.day
        dados['month'] = pd.to_datetime(dados['date']).dt.month
        dados['year'] = pd.to_datetime(dados['date']).dt.year
        dados['week'] = pd.to_datetime(dados['date']).dt.week

        # Semester and Quarter
        dados['quarter'] = pd.to_datetime(dados['date']).dt.quarter
        dados['semester'] = pd.to_datetime(dados['date']).dt.quarter.apply(lambda x: 1 if x == 1 or x == 2 else 2)

        # Season
        def season_of_date(date):
            year = str(date.year)
            seasons = {'spring': pd.date_range(start='21/03/'+year, end='20/06/'+year),
                       'summer': pd.date_range(start='21/06/'+year, end='22/09/'+year),
                       'autumn': pd.date_range(start='23/09/'+year, end='20/12/'+year)}
            if date in seasons['spring']:
                return 'spring'
            if date in seasons['summer']:
                return 'summer'
            if date in seasons['autumn']:
                return 'autumn'
            else:
                return 'winter'

        dados['season'] = pd.to_datetime(dados['date']).map(season_of_date)

        # Weeks for timeseries analyses
        dados['week_bin'] = np.nan
        for index in range(dados.shape[0]):
            if dados.loc[index,'year']==2013:
                dados.loc[index,'week_bin'] = dados.loc[index,'week']
            if dados.loc[index,'year']==2014:
                dados.loc[index,'week_bin'] = dados.loc[index,'week'] + 52
            if dados.loc[index,'year']==2015:
                dados.loc[index,'week_bin'] = dados.loc[index,'week'] + 104
        dados['week_bin'] = dados['week_bin'].astype('int')

        # Competition Lifetime
        dados['competition_lifetime'] = np.nan
        for index in range(dados.shape[0]):
            try:
                data = datetime.datetime(dados.loc[index,'competition_open_since_year'],dados.loc[index,'competition_open_since_month'],1)
                dados.loc[index, 'competition_lifetime'] = (pd.to_datetime(dados.loc[index,'date']) - data).days

            except:
                dados.loc[index,'competition_lifetime'] = -999 # Never opened
        dados['competition_lifetime'] = dados['competition_lifetime'].astype('int')

        # Competition
        dados['competition'] = dados['competition_lifetime'].apply(lambda x: 0 if x<0 else 1) # Negative values suggests that this competition has never been started yet or we don't even have competitors (we chose -999 above)

        # Promo2 Lifetime
        dados['promo2_lifetime'] = np.nan
        for index in range(dados.shape[0]):
            if dados.loc[index,'promo_interval'] != 'No promo2':
                year = dados.loc[index,'promo2_since_year']
                week = dados.loc[index,'promo2_since_week']
                date = "{}-W{}".format(year,week)
                date_datetime = datetime.datetime.strptime(date + '-1', "%Y-W%W-%w")
                dados.loc[index,'promo2_lifetime'] = (pd.to_datetime(dados.loc[index,'date']) - date_datetime).days
            else:
                dados.loc[index,'promo2_lifetime'] = -999
        dados['promo2_lifetime'] = dados['promo2_lifetime'].astype(int)

        # Promo Count per Week
        dados['promo_count_per_week'] = np.nan
        for index in range(dados.shape[0]):
            week = dados.loc[index,'week_bin']
            store = dados.loc[index,'store']
            try:
                dados.loc[index,'promo_count_per_week'] = dados[dados['store']==store].groupby('week_bin').sum()['promo'][week]
            except:
                dados.loc[index,'promo_count_per_week'] = 0
        dados['promo_count_per_week'] = dados['promo_count_per_week'].astype('int')

        # Enumerate per week each store promotion
        dados['promo_n'] = np.nan
        for c in range(dados.shape[0]):
            if dados.loc[c,'promo_count_per_week']>0: # os que tem promoção na semana: Enumerate the promotion per store on each week
                week = dados.loc[c,'week_bin']
                store = dados.loc[c,'store']
                df_aux = pd.DataFrame(dados[(dados['store']==store)&(dados['week_bin']==week)].sort_values(by='date')['promo'])[dados['promo']==1].reset_index().reset_index().set_index('index')
                df_aux['level_0'] = df_aux['level_0'] + 1
                for index in df_aux.index:
                    dados.loc[index,'promo_n'] = df_aux.loc[index,'level_0']
            elif dados.loc[c,'promo_count_per_week']==0: # os que não tem promoção na semana: iguala a 0
                dados.loc[c,'promo_n'] = 0
        dados['promo_n'].fillna(-1,inplace=True) # São dias sem promoção em semanas com promoção: iguala a -1
        dados['promo_n'] = dados['promo_n'].astype('int')
        
        dados.drop('date',1,inplace=True) # drop date
        
        return dados    
    
    def data_preprocessing_feature_selection(self,dados):

        # Target Encoding
        dados = self.target_encoder.transform(dados)

        # Ordinal Encoding
        dados = self.ordinal_encoder.transform(dados)

        # One Hot Encoding
        dados = self.one_hot_encoder.transform(dados)

        # Cyclic Encoding
        def cyclic_transform(x, cols, n):
            n_cols = len(cols)
            for c in range(n_cols):
                x[cols[c]+'_sin'] = x[cols[c]].apply(lambda x: np.sin(2*np.pi*x/n[c]))
                x[cols[c]+'_cos'] = x[cols[c]].apply(lambda x: np.cos(2*np.pi*x/n[c]))
                x.drop(cols[c],axis=1,inplace=True)
            return x
        dados = cyclic_transform(dados,['day_of_week','competition_open_since_month', 'promo2_since_week', 'day', 'month', 'week', 'quarter', 'semester', 'season'],[7,12,52,30,12,52,4,2,4])
        
        # Feature Selection
        cols = ['school_holiday','day_sin','day_cos','month_sin','month_cos','year','promo','promo_interval','store','assortment','store_type','promo_n','promo2','promo2_lifetime']
        dados = dados[cols]
        
        # Rescaling
        dados = self.scalers.transform(dados)
        
        return dados

    def predict( self, dados ):

        pred = pd.Series(self.model.predict( dados ))

        return pred.to_json( orient='records', date_format='iso' ) # convert to json