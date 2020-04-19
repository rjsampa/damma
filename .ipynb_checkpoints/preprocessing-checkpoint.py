from netCDF4 import Dataset 
import numpy as np
import pandas as pd


#------------------------------------------------------------------------
"""
Isola variaveis dependentes e independentes do arquivo wrfoutput

"""

def rotT2(T2):
    """Rotaciona a imagem T2 do wrf para um o padrao Norte superior"""
    Rep = np.rot90(T2,2)
    Rep = np.flip(Rep,1)
    return Rep

class dataset:
    def __init__(self,wrfoutput):
        """
        Abre o arquivo netcdf wrfout(..) e extrai as variaveis de interese
        input: 
            wrfouput: nome ou diretorio para o arquvio wrfout(..) entre aspas
        """
        self.wrfoutput = Dataset(wrfoutput)
      
    def Temperatura(self):
        """
        Retorna a temperatura a 2m referente ao aquivo wrfouput 
        """        
        self.T2m = self.wrfoutput.variables['T2'][:]
        t2rot = []
        for i , n in enumerate(self.T2m):
            campo = rotT2(n)
            t2rot.append(campo)
        self.T2m = np.array(t2rot)
        return self.T2m
    
    def getVar(self, name):
        """
        Retorna a temperatura a 2m referente ao aquivo wrfouput 
        """        
        self.var = self.wrfoutput.variables[name][:]
        varRot = []
        for i , n in enumerate(self.var):
            campo = rotT2(n)
            varRot.append(campo)
        self.varGet = np.array(varRot)
        return self.varGet
    
    def xlat(self):
        """
        Retorna campo com as LATitudes referente ao aquivo wrfouput 
        """
        self.lat = self.wrfoutput.variables['XLAT'][1]
        return rotT2(self.lat)
    
    def xlong(self):
        """
        Retorna com  as longitudes referente ao aquivo wrfouput 
        """
        
        self.long = self.wrfoutput.variables['XLONG'][1]
        return rotT2(self.long)
    
    def mde(self):
        """
        Retorna modelo digital de elevacao referente ao aquivo wrfouput 
        """
        
        self.hgt = self.wrfoutput.variables['HGT'][1]
        return rotT2(self.hgt)
    
    def landUse(self):
        """
        Retorna o mapa de uso e cobertura do terreno referente ao aquivo wrfouput 
        """
        
        self.lulc = self.wrfoutput.variables['LU_INDEX'][1]
        return rotT2(self.lulc)
    

    
####  variaveis do terreno
def slope_aspect(mde):    
    '''
    Retorno a declividade e aspecto do terreno
    input:
        mde: array do modelo digital de elevacao 
        
    output:
        slope: 2D array 
        aspect: 2D array  
    '''
    x, y = np.gradient(mde)
    slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)
    return np.array(slope) , aspect

def hillshade(array, azimuth, angle_altitude):

    x, y = np.gradient(array)
    slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)
    azimuthrad = azimuth*np.pi / 180.
    altituderad = angle_altitude*np.pi / 180.


    shaded = np.sin(altituderad) * np.sin(slope) \
     + np.cos(altituderad) * np.cos(slope) \
     * np.cos(azimuthrad - aspect)
    return 255*(shaded + 1)/2


# ------------------ Criando dataframe Test e treino -------------------
#funcoes
def flatten(array):
    '''
    traforma uma array nD numa array 1D 
    input: 
        array: numpy array nDimensional
    output:
        array numpy 1D'''
    
    return array.flatten()
    

def time_index(start='2017-12-17  00:00:00', periods=67):    
    """ 
    Define o Dataindex para o passo X, considerando o intervalo do estudo
    necessario ajustar o inicio para i periodo desejado
    input:
        start: data e hora de inicio
        periods = numero de periodos desejados 
        
    output:
        time serie
    """
    time = pd.date_range(start = start,
                  periods=periods,freq='H')
    return time

def pandasDF(lista, nomes,start='2017-12-17  00:00:00'):
    '''
    Cria um pandas daframe a partir de uma lista com as variaveis de interesse
    input:
        lista: lista/tupla com variaveis de interesse em 2d
        nomes: lista com nomes das variaveis de interesse
    
    output:
        Pandas Daframe com variaveis de interesse     
    '''
    df = pd.DataFrame()
    for i , var in enumerate(lista):
        df[nomes[i]] = var
        
    
    return df
        
        