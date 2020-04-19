import pandas as pd
import numpy as np 
#from .train_preprocessing import *
'''
prepara daframe o conjunto de dados para treino, teste e validacao
'''

def X_structure(df):
    '''
    Retorna X e Y a partir do dataframe df
    input:
        df: pandas dataframe com as variaveis de interesse '''
    try:
        X = df[[ 'lu_index', 'slope', 'topo', 'xlat','xlong','aspect'
           ]].values
        Y = df.t2.values
        return X , Y
    
    except:
        X = df[[ 'lu_index', 'slope', 'topo', 'xlat','xlong','aspect']].values
        return X
        

        #funcoes
def time_pass(index, start='2017-12-17  00:00:00', periods=67):
    '''
    Define o Dataindex para o passo X, considerando o intervalo do estudo\n
    necessario ajustar o inioio para i periodo desejado
    
    input:
        index: posicao no idex desejada
        start: data e hora de inicio das mensuracoes
        periodos: periodo total da modelagem
        
    output:
        time index para o passo desejado
    
    '''
    time = pd.date_range(start = start,
                  periods=periods,freq='H')
    return time[index]

# Tratamento de x
#padronizando processando variáveis dummys
from sklearn.preprocessing import StandardScaler, LabelEncoder , OneHotEncoder, MinMaxScaler
def PreProcessingX(X):
    '''
    padroniza os valores de X e prepara as variasveis dummy
    input:
        X: dataframe em que X[:,0] seja a variavel dummy
    
    output:
        dataframe com variavel dummy processada e demais variaveis padronizadas
    '''
    stander = StandardScaler()
    X[:,1:] = stander.fit(X[:,1:]).transform(X[:,1:])
    Label = LabelEncoder()
    X[:,0] = Label.fit_transform(X[:,0])
    onehotenconder = OneHotEncoder(categories='auto')
    dummy = onehotenconder.fit_transform(X[:,0].reshape(-1,1)).toarray()
    X = np.concatenate((dummy,X), axis=1)
    return X