from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate
from keras.layers import Bidirectional

import dateutil.relativedelta as reldelta

from datetime import datetime

import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt


def set_index(dataset):
    dataset['date'] = dataset['date_recorded'] + ' ' + dataset['time_recorded']
    
    dataset.set_index("date", inplace = True)
    dataset = dataset.drop(columns = ['date_recorded','time_recorded'])
    
    date_format = "%d/%m/%Y %H:%M:%S"
    dataset.index = pd.to_datetime(dataset.index, format = date_format)
    
    dataset["second"] = 0
    dataset["minute"] = pd.DatetimeIndex(dataset.index).minute
    dataset["hour"] = pd.DatetimeIndex(dataset.index).hour
    dataset["day"]= pd.DatetimeIndex(dataset.index).day
    dataset["month"] = pd.DatetimeIndex(dataset.index).month
    dataset["year"] = pd.DatetimeIndex(dataset.index).year
    dataset["Date"] = pd.to_datetime(dataset[["year","month","day","hour","minute","second"]])
    dataset.set_index("Date", inplace = True)
    dataset["Date"] = pd.to_datetime(dataset[["year","month","day","hour","minute","second"]])
    dataset = dataset.drop(columns = ["year","month","day","hour","minute","second"])
    
    ini_date = dataset["Date"].iloc[0]
    fin_date = dataset["Date"].iloc[-1]
    
    dataset = dataset.drop(columns = "Date")
    
    return dataset, ini_date, fin_date

def create_date_df(ini_date, fin_date):
        '''
        INPUTS:
            -ini_date: Fecha de inicio.
            -fin_date: Fecha de fin.
        OUTPUT:
            - date_df: Pandas DataFrame.

        Funcionamiento:
        Esta función crea, y devuelve, un dataframe con todas las entradas cuarthorarias en el intervalo [ini_date, fin_date].
        '''

        ini_year = ini_date.year
        fin_year = fin_date.year

        ini_month = ini_date.month
        fin_month = fin_date.month+1

        data_index = []

        if ini_year == fin_year:
            year = ini_year
            for month in range(ini_month, fin_month + 1):
                if month == 2:
                    if int(year) % 4 == 0:
                        for day in range(1,30):
                            for hour in range(0,24):
                                for minute in [0,15,30,45]:
                                    data_index.append(datetime(year,month,day,hour,minute))
                    else:
                        for day in range(1,29):
                            for hour in range(0,24):
                                for minute in [0,15,30,45]:
                                    data_index.append(datetime(year,month,day,hour,minute))

                elif month in [4,6,9,11]:
                    for day in range(1,31):
                        for hour in range(0,24):
                            for minute in [0,15,30,45]:
                                data_index.append(datetime(year,month,day,hour,minute))

                elif month in [1,3,5,7,8,10,12]:
                    for day in range(1,32):
                        for hour in range(0,24):
                            for minute in [0,15,30,45]:
                                data_index.append(datetime(year,month,day,hour,minute))

        elif (fin_year - ini_year) == 1:
            year = ini_year
            for month in range(ini_month,13):
                if month == 2:
                    if int(year) % 4 == 0:
                        for day in range(1,30):
                            for hour in range(0,24):
                                for minute in [0,15,30,45]:
                                    data_index.append(datetime(year,month,day,hour,minute))
                    else:
                        for day in range(1,29):
                            for hour in range(0,24):
                                for minute in [0,15,30,45]:
                                    data_index.append(datetime(year,month,day,hour,minute))

                elif month in [4,6,9,11]:
                    for day in range(1,31):
                        for hour in range(0,24):
                            for minute in [0,15,30,45]:
                                data_index.append(datetime(year,month,day,hour,minute))

                elif month in [1,3,5,7,8,10,12]:
                    for day in range(1,32):
                        for hour in range(0,24):
                            for minute in [0,15,30,45]:
                                data_index.append(datetime(year,month,day,hour,minute))

            year = fin_year
            for month in range(1,fin_month + 1):
                if month == 2:
                    if int(year) % 4 == 0:
                        for day in range(1,30):
                            for hour in range(0,24):
                                for minute in [0,15,30,45]:
                                    data_index.append(datetime(year,month,day,hour,minute))
                    else:
                        for day in range(1,29):
                            for hour in range(0,24):
                                for minute in [0,15,30,45]:
                                    data_index.append(datetime(year,month,day,hour,minute))

                elif month in [4,6,9,11]:
                    for day in range(1,31):
                        for hour in range(0,24):
                            for minute in [0,15,30,45]:
                                data_index.append(datetime(year,month,day,hour,minute))

                elif month in [1,3,5,7,8,10,12]:
                    for day in range(1,32):
                        for hour in range(0,24):
                            for minute in [0,15,30,45]:
                                data_index.append(datetime(year,month,day,hour,minute))

        elif (fin_year - ini_year) > 2:
            year = ini_year
            for month in range(ini_month,13):
                if month == 2:
                    if int(year) % 4 == 0:
                        for day in range(1,30):
                            for hour in range(0,24):
                                for minute in [0,15,30,45]:
                                    data_index.append(datetime(year,month,day,hour,minute))
                    else:
                        for day in range(1,29):
                            for hour in range(0,24):
                                for minute in [0,15,30,45]:
                                    data_index.append(datetime(year,month,day,hour,minute))

                elif month in [4,6,9,11]:
                    for day in range(1,31):
                        for hour in range(0,24):
                            for minute in [0,15,30,45]:
                                data_index.append(datetime(year,month,day,hour,minute))

                elif month in [1,3,5,7,8,10,12]:
                    for day in range(1,32):
                        for hour in range(0,24):
                            for minute in [0,15,30,45]:
                                data_index.append(datetime(year,month,day,hour,minute))

            for year in range(ini_year + 1, fin_year):
                for month in range(1,13):
                    if month == 2:
                        if int(year) % 4 == 0:
                            for day in range(1,30):
                                for hour in range(0,24):
                                    for minute in [0,15,30,45]:
                                        data_index.append(datetime(year,month,day,hour,minute))
                        else:
                            for day in range(1,29):
                                for hour in range(0,24):
                                    for minute in [0,15,30,45]:
                                        data_index.append(datetime(year,month,day,hour,minute))

                    elif month in [4,6,9,11]:
                        for day in range(1,31):
                            for hour in range(0,24):
                                for minute in [0,15,30,45]:
                                    data_index.append(datetime(year,month,day,hour,minute))

                    elif month in [1,3,5,7,8,10,12]:
                        for day in range(1,32):
                            for hour in range(0,24):
                                for minute in [0,15,30,45]:
                                    data_index.append(datetime(year,month,day,hour,minute))
            year = fin_year
            for month in range(1,fin_month + 1):
                if month == 2:
                    if int(year) % 4 == 0:
                        for day in range(1,30):
                            for hour in range(0,24):
                                for minute in [0,15,30,45]:
                                    data_index.append(datetime(year,month,day,hour,minute))
                    else:
                        for day in range(1,29):
                            for hour in range(0,24):
                                for minute in [0,15,30,45]:
                                    data_index.append(datetime(year,month,day,hour,minute))

                elif month in [4,6,9,11]:
                    for day in range(1,31):
                        for hour in range(0,24):
                            for minute in [0,15,30,45]:
                                data_index.append(datetime(year,month,day,hour,minute))

                elif month in [1,3,5,7,8,10,12]:
                    for day in range(1,32):
                        for hour in range(0,24):
                            for minute in [0,15,30,45]:
                                data_index.append(datetime(year,month,day,hour,minute))
        else:
            print(f"Función create_date_df. Ini_date = {ini_date},  Fin_date = {fin_date}.")

        df = pd.DataFrame(data_index, columns = ['Date'])
        df.set_index('Date', inplace = True)

        return df
    
def missing_value( row, *tuple_column):
        '''
        Función apply (más información en la documentación de pandas.).
        Detecta si la columna (tuple_column[0]) está vacía en la fila "row" de un pandas Dataframe.
        Analiza un elemento y devuelve True, si es vacío, o False, en caso contrario.
        '''
        column = tuple_column[0]

        if pd.isnull(row[column]) == True:
            return True

        else:
            return False
        
def clean_quarters(df, date_df, columns_train):
        '''
        INPUTS:
            - df: Pandas DataFrame, sobre el que se quiere hacer el limpiado. Debe cumplir:
            - El índice (df.index) debe estar formado por fechas en formato datetime.
            - date_df: Pandas Dataframe con todas los registros cuart-horarios, durante las fechas que el dataframe "df" tiene datos.
        OUTPUT:
            - df: Mismo Pandas DataFrame con los valores rellenados.

        Funcionamiento:

        Lo hace para las variables "Temperatura" e "Irradiación", en caso de que estén entre las columnas especificadas en 
        "columns_train", en el archivo de configuración.

        Esta función completa valores que falten (dónde hay nulos) que se tengan que aproximar mediante un "average". Lo podemos 
        aplicar para la Temperatura y la Radiación.

        Por ejemplo,

        12.00 = 20      -->     12.00 = 20
        12.15 = -       -->     12.15 = 22.5
        12.30 = -       -->     12.30 = 25
        12.45 = -       -->     12.45 = 27.5
        13.00 = 30      -->     13.00 = 30

        Si faltan más de 4 valores consecutivos, esto se traslada a más de una hora, no hacemos nada. Esto se puede cambiar 
        modificando la variable "max_number_consecutive_missing_values"
        '''

        df['Date2'] = df.index

        max_number_consecutive_missing_values = 4 # 4 == una hora

        start_date  = min(df.index)
        end_date    = max(df.index)

        date_df         = date_df[(date_df.index >= start_date) & (date_df.index <= end_date)]
        
        
        df              = pd.merge(date_df, df, how = 'left', left_index = True, right_index = True)
        df              = df.reset_index()
        
        max_index       = max(df.index)

        end_index       = -1
        numeric_columns = []

        for column in columns_train:
        
            df[column]          = df[column].astype('float')
            missing_column      = column +'_Missing'
            list_column         = []
            list_column.append(column)
            tuple_column        = tuple(list_column)
            df[missing_column]  = df.apply(missing_value, axis = 1, args = tuple_column)
        
        for column in columns_train:

            # Check distinto para cada columna

            end_index       = -1
            list_indexes    = []
            missing_column  = column + '_Missing'

            # Creamos lista de índices que indican que entre estos dos índices, los valores faltan (para esa columna en cuestión).

            for index,row in df.iterrows():

                if index <= end_index:

                    continue

                elif row[missing_column] == True:
                    end_index = index

                    while end_index <= max_index:

                        if df[missing_column][end_index] == True:

                            end_index += 1

                        else:

                            break
                    list_indexes.append([index, end_index - 1])

            # Ahora, miramos si podemos hacer average de estos valores que faltan.
            
            
            for group in list_indexes:
                N = group[1] - group[0] + 1

                if group[0] == 0 and N > max_number_consecutive_missing_values :
                    continue

                if group[0] == 0 and N <= max_number_consecutive_missing_values :

                    for index in range(group[0], group[1] + 1):

                        new_value           = df[column][group[1] + 1]
                        df.at[index,column] = new_value

                elif group[1] == max_index and N <= max_number_consecutive_missing_values :

                    for index in range(group[0], group[1] + 1):

                        df.at[index,column] = new_value

                elif group[0] != 0 and N <= max_number_consecutive_missing_values :

                    multiplier = 1
                    difference = df[column][group[1] + 1] - df[column][group[0] - 1]

                    for index in range(group[0], group[1] + 1):

                        new_value           = df[column][group[0] - 1] + difference*multiplier/(N + 1)
                        df.at[index,column] = new_value
                        multiplier          += 1
                else:
                    #  Estar aquí significa que falta más de una hora.
                    continue
                
        df.set_index('Date', inplace = True)
        
        return df


def defining_df(df):
        df, ini_date, end_date= set_index(df)
        date_df = create_date_df(ini_date,end_date)
        n = len(df.columns)
        df = clean_quarters(
            df      = df,
            date_df = date_df,
            columns_train = df.columns
            )
        df = df.iloc[:,:n]
        return df,date_df

def fill_dti(df,dTi_col,Ti_col):
    '''
    df: pandas dataframe
    '''
    df[dTi_col] = 0

    for index,row in df.iterrows():

        index_15 = index - reldelta.relativedelta(minutes = 15)

        try:
            df.at[index, dTi_col] = row[Ti_col] - df.at[index_15, Ti_col]

        except:
            # In this case, we didn't found a row with 15 minutes less
            df.at[index, dTi_col] = None
            
    return df