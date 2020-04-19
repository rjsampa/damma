import pandas as pd
import numpy as np
from .train_preprocessing import *

'''
Algoritmos de treino e de teste 
'''


# fit model Random forest
def fit_RandomForest(X,Y):
    """
    Implementa o algoritmo Random forest
    input:
         X: variaveis independente (numpy array)
         Y: Variavel depende (numpy array)
    
    output:
        regressor treinado 
    """
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(bootstrap= True, max_depth=10,
                                     max_features='auto',min_samples_leaf= 4,
                                     min_samples_split=2, n_estimators=20)
    regressor.fit(X,Y)
    
    return regressor

# otimizando RF
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor


parametro = {'bootstrap': [True, False],
                 'max_depth': [10,  None],
                 'max_features': ['auto'],
                 #'min_samples_leaf': [1, 2, 4],
                 'min_samples_split': [2, 10],
                 'n_estimators': [200,    1000,     2000]}

def fit_GridCVRandomforest(X,Y,parametros = parametro, cv = 5, n_jobs = 2):
    '''
    Seleciona a melhor combinacao de parametros utilizando GridSearchCV
    input:
         X: variaveis independente (numpy array)
         Y: Variavel depende (numpy array)
         parametros: dicionario com os hiperparametros a serem analisados
         cv = numero de partes da validacao cruzada
         n_jobs = nucleos de processamento
    
    output:
        regressor treinado
    '''
    estimador = RandomForestRegressor()
    regressor = GridSearchCV(estimador,
                             param_grid=parametros, 
                             cv = cv, n_jobs = n_jobs)
    regressor.fit(X,Y)
    return regressor


# fit SVr
def fit_SVR(X,Y):
    '''
    Implementa o algoritmo máquina de vetor de suporte 
    input:
         X: variaveis independente (numpy array)
         Y: Variavel depende (numpy array)
    
    output:
        regressor treinado
    '''
    
    from sklearn.svm import SVR
    regressor = SVR(C=10, coef0=0.0001, gamma=0.2, kernel='rbf')
    regressor.fit(X,Y)
    
    return regressor


# create grade:
def GradeCreatorDom_f(Y):
    t2 = Y.reshape(85,195)
    return t2

# aplica regressor desejado
def train(X, dom_i,dom_f, model = 1, regressor = None):
    import time
    '''
    Aplica regressor desejado retornando o campo de temperatura modelado para o dom_f e o modelo treinado
    input:
        X: array 3d com o campo de temperatura do dominio inicial \n
        dom_i: dados do terreno relativo ao dominio inicial \n
        dom_f: dados do terreno relativo ao dominio final\n
        model : 1 - Random forest, 2 - SVR, 3 - regressor
        regressor: valido apenas para model 3
    
    output:
        result: array 3d com os campos de temperatura para o intervalo de tempo desejado
        regressor
        
    exemplo:
        def main():
            # carregando dados
            X = np.load('T2_dom_i.npy')
            dom_i = pd.read_json('dom_i_dados_terrain.json')
            dom_f = pd.read_json('teste.json')
            
            # aplicando o modelo
            Y_predic, model = train(X, dom_i, dom_f)    

            # salvando 
            np.save('rf_sumulacoes_dom_f.npy', Y_predict)
        
    '''
    print('Iniciando... ' , end='')
    print()
    t1 = time.time()
    # criando saidas
    result = []
    #preparando dados dom_f
    x_test = dom_f
    
    #Tempo de processamento
    TempoProce = []
    for i in range(X.shape[0]):
        if model==1:            
            ti = time.time()            
            X_train ,Y_train = dom_i , np.array(X[i].flatten())            
            regressor = fit_RandomForest(X_train,Y_train)#model.fit(X_train,Y_train)
            Y_predict = regressor.predict(x_test)
            result.append(GradeCreatorDom_f(Y_predict))
            tii = time.time()
            TT = tii - ti
            TempoProce.append(TT)
            kk = i + 1
            print('Step %i completo no tempo de %.3f segundos' % (kk,  TT) )
        
        elif model==2:          
            
            ti = time.time()
            
            X_train ,Y_train = dom_i , np.array(X[i].flatten())
            regressor =fit_SVR(X_train, Y_train) #model1.fit(X_train,Y_train)
            Y_predict = regressor.predict(x_test)
            result.append(GradeCreatorDom_f(Y_predict))
            tii = time.time()
            TT = tii - ti
            TempoProce.append(TT)
            kk = i + 1
            print('Step %i completo no tempo de %.3f segundos' % (kk,  TT)  )
            
        elif model==3:
            ti = time.time()            
            X_train ,Y_train = dom_i , np.array(X[i].flatten())            
            regressor = regressor.fit(X_train,Y_train)
            Y_predict = regressor.predict(x_test)
            result.append(GradeCreatorDom_f(Y_predict))
            tii = time.time()
            TT = tii - ti
            TempoProce.append(TT)
            kk = i + 1
            print('Step %i completo no tempo de %.3f segundos' % (kk,  TT)  )
        
        else:
            print('Escolha um modelo válido!!')
            break 


    result = np.array(result)
    TempoProce = np.array(TempoProce)
    TempoMed = TempoProce.mean()
    t2 = time.time()
    tf = t2 -t1
    
    print('-'*82)
    print()    
    ff = ' Finalizado '    
    print('*'*82)
    print(ff.center( 82, '*'))
    print('*'*82)
    print()
    print('-'*82)
    print('Tempo Médio: %.4f' %TempoMed)
    print('Tempo total: %.4f' %tf)
    return result ,regressor


