import pandas as pd
import numpy as np
import inflection
import datetime
import pickle

initial_data = pd.read_csv('/Users/nando/Comunidade DS/ds_em_producao/data/inital_data.csv') # i will use later

class rossmann(object):
    
    def __init__(self):
        self.store_type_label_encoder = pickle.load(open('/Users/nando/Comunidade DS/ds_em_producao/deploy/store_type_label_encoder.pickle','rb'))
        self.promo_interval_target_encoder = pickle.load(open('/Users/nando/Comunidade DS/ds_em_producao/deploy/promo_interval_target_encoder.pickle','rb'))
        self.min_max_scaler = pickle.load(open('/Users/nando/Comunidade DS/ds_em_producao/deploy/min_max_scaler.pickle','rb'))
        self.robust_scaler = pickle.load(open('/Users/nando/Comunidade DS/ds_em_producao/deploy/robust_scaler.pickle','rb'))

    def data_cleaning(self,dados):
        ## 1.1 Rename Columns and Values

        # CamelCase to snake_case
        for c in range(len(dados.columns)):
            dados.rename(columns={dados.columns.values[c]:inflection.underscore(dados.columns.values[c])},inplace=True)

        dados['state_holiday'] = dados['state_holiday'].map({'a':'public holiday', 'b':'Easter holiday', 'c':'Christmas', '0':'None'})
        dados['assortment'] = dados['assortment'].map({'a':'basic','b':'extra','c':'extended'})

        ## 1.3 Data Types

        # date to datetime64
        dados['date'] = pd.to_datetime(dados['date'])

        ### 1.4 Dealing With NAN

        ## competition_distance, competition_open_since_month, competition_open_since_year
        # I suppose that the competition_distance == nan refers that there's no competition!
        for index in dados[dados['competition_distance'].isna()].index:
            dados.loc[index,'competition_distance'] = 999999 # High values suggests that the competitors are so far that there's no competition
            dados.loc[index,'competition_open_since_month'] = 0 # It suggets that it has never been opened
            dados.loc[index,'competition_open_since_year'] = 0 # It suggets that it has never been opened

        ## competition_open_since_month, competition_open_since_year
        # I suppose that if there's no year, there's no month and vice-versa
        month_median = round(initial_data['CompetitionOpenSinceMonth'].median(),0) # get the median of the whole dataset
        year_median = round(initial_data['CompetitionOpenSinceYear'].median(),0) # get the median of the whole dataset
        for index in dados[dados['competition_open_since_month'].isna()].index:
            dados.loc[index,'competition_open_since_month'] = month_median # Im gonna use median and round to have discret values
            dados.loc[index,'competition_open_since_year'] = year_median # Im gonna use median and round to have discret values

        ## promo_interval, promo2_since_week, promo2_since_year
        # Because if there's no promo2, theres no since date or interval
        for index in dados[dados['promo2_since_week'].isna()].index:
            dados.loc[index, 'promo2_since_week'] = 0 # It suggets that there's no promo2 week start
            dados.loc[index, 'promo2_since_year'] = 0 # It suggets that there's no promo2 year start
            dados.loc[index, 'promo_interval'] = 'No promo2' # For now, im going to substitute with this

        ## 1.5 Data Types (after NaN)

        # float to int in columns related to dates
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
        for index in dados.index:
            if dados.loc[index,'year']==2013:
                dados.loc[index,'week_bin'] = dados.loc[index,'week']
            if dados.loc[index,'year']==2014:
                dados.loc[index,'week_bin'] = dados.loc[index,'week'] + 52
            if dados.loc[index,'year']==2015:
                dados.loc[index,'week_bin'] = dados.loc[index,'week'] + 104
        dados['week_bin'] = dados['week_bin'].astype('int')

        # Competition Lifetime
        dados['competition_lifetime'] = np.nan
        for index in dados.index:
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
        for index in dados.index:
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
        for index in dados.index:
            week = dados.loc[index,'week_bin']
            store = dados.loc[index,'store']
            try:
                dados.loc[index,'promo_count_per_week'] = dados[dados['store']==store].groupby('week_bin').sum()['promo'][week]
            except:
                dados.loc[index,'promo_count_per_week'] = 0
        dados['promo_count_per_week'] = dados['promo_count_per_week'].astype('int')

        # Enumerate per week each store promotion
        dados['promo_n'] = np.nan
        for c in dados.index:
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
        
        return dados
    
    def data_preprocessing(self,dados):
        ## 3.2 Encoding

        ## One Hot Encoding (state_holiday)
        dados['easter'] = dados['state_holiday'].apply(lambda x: 1 if x=='Easter holiday' else 0)
        dados['public_holiday'] = dados['state_holiday'].apply(lambda x: 1 if x=='public holiday' else 0)
        dados['christmas'] = dados['state_holiday'].apply(lambda x: 1 if x=='Christmas' else 0)
        dados.drop('state_holiday',1,inplace=True)

        ## Label Encoding (store_type)
        dados['store_type'] = self.store_type_label_encoder.transform(dados[['store_type']])

        ## Ordinal Encoding (assortment, season)
        dados['assortment'] = dados['assortment'].map({'basic':0,'extended':1,'extra':2})
        dados['season'] = dados['season'].map({'winter':1,'spring':2,'summer':3,'autumn':4})

        ## Target Encoding (promo_interval)
        dados['promo_interval'] = self.promo_interval_target_encoder.transform(dados['promo_interval'])

        ## Cyclic Encoding (day_of_week, competition_open_since_month, promo2_since_week, day, month, week, quarter, semester, season)
        dados['day_of_week_sin'] = dados['day_of_week'].apply(lambda x: np.sin(2*np.pi*x/7))
        dados['day_of_week_cos'] = dados['day_of_week'].apply(lambda x: np.cos(2*np.pi*x/7))
        dados.drop('day_of_week',1,inplace=True)

        dados['competition_open_since_month_sin'] = dados['competition_open_since_month'].apply(lambda x: np.sin(2*np.pi*x/12))
        dados['competition_open_since_month_cos'] = dados['competition_open_since_month'].apply(lambda x: np.cos(2*np.pi*x/12))
        dados.drop('competition_open_since_month',1,inplace=True)

        dados['promo2_since_week_sin'] = dados['promo2_since_week'].apply(lambda x: np.sin(2*np.pi*x/52))
        dados['promo2_since_week_cos'] = dados['promo2_since_week'].apply(lambda x: np.cos(2*np.pi*x/52))
        dados.drop('promo2_since_week',1,inplace=True)

        dados['day_sin'] = dados['day'].apply(lambda x: np.sin(2*np.pi*x/30))
        dados['day_cos'] = dados['day'].apply(lambda x: np.cos(2*np.pi*x/30))
        dados.drop('day',1,inplace=True)

        dados['month_sin'] = dados['month'].apply(lambda x: np.sin(2*np.pi*x/12))
        dados['month_cos'] = dados['month'].apply(lambda x: np.cos(2*np.pi*x/12))
        dados.drop('month',1,inplace=True)

        dados['week_sin'] = dados['week'].apply(lambda x: np.sin(2*np.pi*x/52))
        dados['week_cos'] = dados['week'].apply(lambda x: np.cos(2*np.pi*x/52))
        dados.drop('week',1,inplace=True)

        dados['quarter_sin'] = dados['quarter'].apply(lambda x: np.sin(2*np.pi*x/4))
        dados['quarter_cos'] = dados['quarter'].apply(lambda x: np.cos(2*np.pi*x/4))
        dados.drop('quarter',1,inplace=True)

        dados['semester_sin'] = dados['semester'].apply(lambda x: np.sin(2*np.pi*x/2))
        dados['semester_cos'] = dados['semester'].apply(lambda x: np.cos(2*np.pi*x/2))
        dados.drop('semester',1,inplace=True)

        dados['season_sin'] = dados['season'].apply(lambda x: np.sin(2*np.pi*x/4))
        dados['season_cos'] = dados['season'].apply(lambda x: np.cos(2*np.pi*x/4))
        dados.drop('season',1,inplace=True)

        ## 3.3 Rescaling

        ## Min Max Scaler (store,week_bin, promo_count_per_week, promo_n, year, store_type, assortment, promo_interval, day_of_week_sin/cos, competition_open_since_month_sin/cos, promo2_since_week_sin/cos, day_sin/cos, month_sin/cos, week_sin/cos, quarter_sin/cos, semester_sin/cos, season_sin/cos)

        mms_columns_lst = ['store','week_bin', 'promo_count_per_week', 'promo_n', 'year', 'store_type', 'assortment',
        'promo_interval', 'day_of_week_sin','day_of_week_cos', 'competition_open_since_month_sin',
        'competition_open_since_month_cos', 'promo2_since_week_sin','promo2_since_week_cos', 
        'day_sin','day_cos', 'month_sin','month_cos', 'week_sin','week_cos', 'quarter_sin',
        'quarter_cos', 'semester_sin','semester_cos', 'season_sin','season_cos']

        dados[mms_columns_lst] = self.min_max_scaler.transform(dados[mms_columns_lst])

        ## Robust Scaler (competition_distance, competition_open_since_year, promo2_since_year, competition_lifetime, promo2_lifetime)

        rs_columns_lst = ['competition_distance', 'competition_open_since_year', 'promo2_since_year',
                          'competition_lifetime', 'promo2_lifetime']

        dados[rs_columns_lst] = self.robust_scaler.transform(dados[rs_columns_lst])

        return dados
    
    def feature_selection(self,dados):
        cols_selected = ['store',
         'promo',
         'store_type',
         'assortment',
         'competition_distance',
         'promo_interval',
         'competition_lifetime',
         'promo2_lifetime',
         'promo_n',
         'day_of_week_sin',
         'day_of_week_cos',
         'day_sin',
         'day_cos',
         'month_sin',
         'month_cos']

        dados = dados[cols_selected]
        
        return dados
    
    def get_prediction( self, model, original_data, input_data ):
        
        pred = model.predict( input_data )
        
        original_data['prediction'] = np.expm1( pred )
        
        return original_data.to_json( orient='records', date_format='iso' ) # return a df with the original imput and it's prediction