import pandas as pd
import numpy as np
import re
import os

from numpy import array

from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM, RepeatVector, TimeDistributed
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D
from keras.models import Model
from keras import optimizers
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate
from keras.layers import Bidirectional

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from datetime import datetime

import matplotlib.pyplot as plt


def Ph0(dataset,column):
    df = dataset
    cont = 0
    for i in range(0,len(dataset[column])):
        if i == 0:
            if dataset[column][i]<0.15*max(dataset[column]) and dataset[column][i+1]<0.15*max(dataset[column]):
                df[column][i] = 0
                cont +=1
        elif i == len(dataset[column])-1:
            if dataset[column][i]<0.15*max(dataset[column]) and dataset[column][i-1]<0.15*max(dataset[column]):
                df[column][i] = 0
                cont +=1
        else:
            if dataset[column][i]<0.15*max(dataset[column]) and dataset[column][i-1]<0.15*max(dataset[column]) and dataset[column][i+1]<0.15*max(dataset[column]):
                df[column][i] = 0
                cont +=1
    return df,cont

# convert series to supervised learning
def series_to_supervised(data, columns, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1] #data.shape[1] devuelve el número de columnas
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i)) #df.shift() para desplazar el indice el número de periodos deseado 
        names += [str(columns[j]) + '(t-' + str(i) + ")" for j in range(n_vars)] # le pone el nombre correspondiente a la columna
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [str(columns[j]) + '(t)' for j in range(n_vars)]
        else:
            names += [str(columns[j]) + '(t+' + str(i) + ")" for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def preparing_df(data,n_pred,n_values): 

    #first = data.index[1] #when calculating the deltaT the first row is eliminated
    first = data.index[0]
    df = pd.DataFrame()
    reframed = pd.DataFrame()
    
    i = 0
    dic_ind_date = {}
    
    for index,row in data.iterrows():
        if (row['Delta']!=15.0) and np.isnan(row['Delta']) != True:
            df = data.loc[first:index_1]
            #df.iloc[-5:,:].to_csv(os.getcwd()+'df_salto_prev.csv')
            columns = df.columns[0:n_values+1]
            for j in range(0,len(df.index.array)-4):
                dic_ind_date[i] = df.index.array[j]
                i+=1
            if n_pred == 4:
                values = df.iloc[:,0:n_values+1]
            elif n_pred == 1:
                values = df.iloc[:-3,0:n_values+1]
            values = values.to_numpy()
            values = values.astype('float32')
            reframed = reframed.append(series_to_supervised(values, columns, 0, n_pred+1))
            reframed.iloc[-5:,:].to_csv('ref_prev.csv')
            first = index
            print(index)
        index_1 = index
        
    df = data.loc[first:index_1]
    #df.iloc[0:5,:].to_csv(os.getcwd()+'df_salto.csv')
    for j in range(0,len(df.index.array)-4):
        dic_ind_date[i] = df.index.array[j]
        i+=1
    
    columns = df.columns[0:n_values+1]
    values = df.iloc[0:,0:n_values+1]
    values = values.to_numpy()
    values = values.astype('float32')
    #series_to_supervised(values, columns, 0, n_pred+1).iloc[0:5,:].to_csv('ref_post.csv')
    reframed = reframed.append(series_to_supervised(values, columns, 0, n_pred+1))
    reframed = reframed.reset_index(drop = True)
    values = reframed.iloc[:,:n_values]
    values = values.to_numpy()
    values = values.astype('float32')
    
    return reframed, values, dic_ind_date


def moving(df, variables_training, variables_prediction, n_values, method_data):
    '''
    The moving function moves the output columns to the end of the DataFrame.

    INPUT:
    - df: The dataframe we want to modify.
    - n_values: How many variables we have per timestep.
    
    OUTPUT:
    - df: DataFrame with the output variables at the end.
    '''
    if method_data == True:
        df1 = df[variables_training]
        df2 = df[variables_prediction]
        df = pd.concat([df1,df2],axis=1)
    
    
    elif method_data == False:
        columns = df.columns[n_values:len(df.columns):n_values]
        for i in columns:
            first_col = df.pop(i)
            df.insert(len(df.columns),i,first_col)

    return df

def scaling(df):
    '''
    The scaling function transforms features by scaling each feature to a given range.

    INPUT: 
    - df: Dataframe to be scaled.

    OUTPUT:
    - scaler: The scaling function.
    - scaled: The scaled data in a numpyarray.
    '''
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df)
    return scaler,scaled

def training(data_array,n_pred,n_train):
    train = data_array[:n_train, :]
    test = data_array[n_train:, :]
    
    # split into input and outputs
    train_X, train_y = train[:, :-n_pred], train[:, -n_pred:] 
    test_X, test_y = test[:, :-n_pred], test[:, -n_pred:]
    
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape(train_X.shape[0], 1, train_X.shape[1])
    test_X = test_X.reshape(test_X.shape[0], 1, test_X.shape[1])
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    
    # design network
    model = Sequential()
    model.add(Bidirectional(LSTM(50,return_sequences=True,input_shape=(train_X.shape[1], train_X.shape[2]))))
    #model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(50,return_sequences=True,input_shape=(train_X.shape[1], train_X.shape[2]))))
    model.add(Dropout(0.2))
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Dense(n_pred))
    model.compile(loss='mae', optimizer='adam')
    
    # fit network
    history = model.fit(train_X, train_y, epochs=200, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
    return test_X, test_y, model

def predicting_blocks(test_X,test_y,model,n_pred):
    # make a prediction
    inv_yhat = model.predict(test_X)
    
    test_y = test_y.reshape((len(test_y), n_pred)) #uste dot hau ez dala beharrezkoa
    inv_y = test_y
    
    return inv_yhat,inv_y

def predicting_blocks_1(test_X,test_y,model,scaler_Ti,n_pred):
    all_predictions = [[]]*(len(test_y))
    inv_y = test_y
    for index in range(0,len(test_y)):
        prediction = []
        if index == 0:
            dato_a_predecir = test_X[index,:,:]
            dato_a_predecir = dato_a_predecir.reshape(1, 1, test_X.shape[2])
            Ti_hat = model.predict(dato_a_predecir)
            prediction = np.append(prediction,Ti_hat)
            
        else: 
            dato_a_predecir = test_X[index,:,1:]
            Ti_hat_scaled = scaler_Ti.transform(Ti_hat[0,0].reshape(1, -1))
            dato_a_predecir = np.concatenate((Ti_hat_scaled,dato_a_predecir), axis=1)
            dato_a_predecir = dato_a_predecir.reshape(1, 1, test_X.shape[2])
            Ti_hat = model.predict(dato_a_predecir)
            prediction = np.append(prediction,Ti_hat)
            
        all_predictions[index] = prediction
            
    print(all_predictions)
            
    return all_predictions,inv_y

def predicting_step_Ti(test_X,test_y,model,scaler_Ti,n_pred,n_values):
    all_predictions = [[]]*(len(test_y)-3)

    for index in range(0,len(test_y)-3):
        prediction = []
        for i in range(0,n_values):
            if i == 0:
                dato_a_predecir = test_X[index+i,:,:]
                dato_a_predecir = dato_a_predecir.reshape(1, 1, test_X.shape[2])
                Ti_hat = model.predict(dato_a_predecir)
                prediction = np.append(prediction,Ti_hat)
                

            else:
                dato_a_predecir = test_X[index+i,:,1:]
                Ti_hat_scaled = scaler_Ti.transform(Ti_hat)
                dato_a_predecir = np.concatenate((Ti_hat_scaled,dato_a_predecir), axis=1)
                dato_a_predecir = dato_a_predecir.reshape(1, 1, test_X.shape[2])
                Ti_hat = model.predict(dato_a_predecir)
                prediction = np.append(prediction,Ti_hat)

            all_predictions[index] = prediction
      
    return all_predictions

def predicting_step_dTi(test_X,test_y,model,scaler_Ti,n_pred):
    all_predictions = [[]]*(len(test_y)-3)

    for index in range(0,len(test_y)-3):
        prediction = []
        for i in range(0,4):
            if i == 0:
                dato_a_predecir = test_X[index+i,:,:]
                dato_a_predecir = dato_a_predecir.reshape(1, 1, test_X.shape[2])
                dTi_hat = model.predict(dato_a_predecir)
                Ti_scaled = test_X[index+i,:,0]
                Ti = scaler_Ti.inverse_transform(Ti_scaled.reshape(1, -1))
                Ti_hat = Ti + dTi_hat
                prediction = np.append(prediction,Ti_hat)
                

            else:
                dato_a_predecir = test_X[index+i,:,1:]
                Ti_hat_scaled = scaler_Ti.transform(Ti_hat)
                dato_a_predecir = np.concatenate((Ti_hat_scaled,dato_a_predecir), axis=1)
                dato_a_predecir = dato_a_predecir.reshape(1, 1, test_X.shape[2])
                dTi_hat = model.predict(dato_a_predecir)
                Ti_scaled = test_X[index+i,:,0]
                Ti = scaler_Ti.inverse_transform(Ti_scaled.reshape(1, -1))
                #Ti_hat = Ti + dTi_hat
                Ti_hat = Ti_hat + dTi_hat
                prediction = np.append(prediction,Ti_hat)

            all_predictions[index] = prediction
      
    return all_predictions


def Ph0_input_step_Ti(test_X,test_y,model,scaler_Ti,scaler_Ta,scaler_Ps,dic_ind_date,Ti_mean,date_1,date_2,n_train,R,C,Aw,input_position_Ti,input_position_Ta,input_position_Ps,input_position_Ph):
    
    count_vector = []
    count_vector_RC = []
    
    key1 = get_key(date_1,dic_ind_date)
    key2 = get_key(date_2,dic_ind_date)
    
    print(Ti_mean,key1,key2)
    df_events = pd.DataFrame()
    df_RC = pd.DataFrame()
    
    for index in range(key1-n_train,key2-n_train):
        dato_a_predecir = test_X[index,:,:]
        dato_a_predecir_RC = test_X[index-1,:,:]
        dato_a_predecir = dato_a_predecir.reshape(1, 1, test_X.shape[2])
        dato_a_predecir_RC = dato_a_predecir_RC.reshape(1, 1, test_X.shape[2])
        if dato_a_predecir[0,0,input_position_Ph] == 0:
            continue
        Ti_scaled = dato_a_predecir[0,0,input_position_Ti]
        Ta_scaled = dato_a_predecir_RC[0,0,input_position_Ta]
        Ps_scaled = dato_a_predecir_RC[0,0,input_position_Ps]
        Ti = scaler_Ti.inverse_transform(Ti_scaled.reshape(1, -1))
        Ta_RC = scaler_Ta.inverse_transform(Ta_scaled.reshape(1, -1))
        Ps_RC = scaler_Ps.inverse_transform(Ps_scaled.reshape(1, -1))
        Ti_RC = Ti
        cont = 0
        cont_RC = 0
        i = 1
        j = 1
        
        df_events.loc[cont,dic_ind_date[int(index)+n_train]] = Ti
        df_RC.loc[cont_RC,dic_ind_date[int(index)+n_train]] = Ti
        
        
        #FLEX MODEL
        while Ti_mean-0.5<=Ti<=Ti_mean+0.5 and index+i<key2-n_train:
            
            Ti_hat = model.predict(dato_a_predecir)
            dato_a_predecir = test_X[index+i,:,1:]
            Ti_hat_scaled = scaler_Ti.transform(Ti_hat)
            dato_a_predecir = np.concatenate((Ti_hat_scaled,dato_a_predecir), axis=1)
            dato_a_predecir = dato_a_predecir.reshape(1, 1, test_X.shape[2])
            dato_a_predecir[0,0,input_position_Ph] = 0
            cont+=1
            df_events.loc[cont,dic_ind_date[int(index)+n_train]] = Ti_hat
            Ti = Ti_hat
            i+=1
        
        #FLEX RC
        while Ti_mean-0.5<=Ti_RC<=Ti_mean+0.5 and index+j<key2-n_train: 
            Ti_hat_RC = Ti_RC+((Ta_RC-Ti_RC)/(R*C)+(Ps_RC/1000)*Aw/C)*0.25
            cont_RC+=1
            df_RC.loc[cont_RC,dic_ind_date[int(index)+n_train]] = Ti_hat_RC
                        
            dato_a_predecir_RC = test_X[index+j-1,:,:]
            dato_a_predecir_RC = dato_a_predecir_RC.reshape(1, 1, test_X.shape[2])
            Ta_scaled = dato_a_predecir_RC[0,0,input_position_Ta]
            Ps_scaled = dato_a_predecir_RC[0,0,input_position_Ps]
            Ta_RC = scaler_Ta.inverse_transform(Ta_scaled.reshape(1, -1))
            Ps_RC = scaler_Ps.inverse_transform(Ps_scaled.reshape(1, -1))
            
            Ti_RC = Ti_hat_RC
            j+=1
            
        count_vector= np.append(count_vector, [index,cont], axis = 0)
        count_vector_RC= np.append(count_vector_RC, [index,cont_RC], axis = 0)
    count_vector = np.reshape(count_vector,(int(len(count_vector)/2),2))
    count_vector_RC = np.reshape(count_vector_RC,(int(len(count_vector_RC)/2),2))
    return count_vector,count_vector_RC,df_events,df_RC


def Ph0_input_step_dTi(test_X,test_y,model,scaler_Ti,scaler_Ta,scaler_Ps,dic_ind_date,Ti_mean,date_1,date_2,n_train,R,C,Aw,input_position_Ti,input_position_Ta,input_position_Ps,input_position_Ph):
    
    count_vector = []
    count_vector_RC = []
    
    key1 = get_key(date_1,dic_ind_date)
    key2 = get_key(date_2,dic_ind_date)
    print(Ti_mean,key1,key2)
    
    df_events = pd.DataFrame()
    df_RC = pd.DataFrame()
    
    for index in range(key1-n_train,key2-n_train):
        i = 0
        dato_a_predecir = test_X[index,:,:]
        dato_a_predecir_RC = test_X[index-1,:,:]
        dato_a_predecir = dato_a_predecir.reshape(1, 1, test_X.shape[2])
        dato_a_predecir_RC = dato_a_predecir_RC.reshape(1, 1, test_X.shape[2])
        
        if dato_a_predecir[0,0,input_position_Ph] == 0:
            continue
            
        Ti_scaled = dato_a_predecir[0,0,input_position_Ti]
        Ta_scaled = dato_a_predecir_RC[0,0,input_position_Ta]
        Ps_scaled = dato_a_predecir_RC[0,0,input_position_Ps]
        Ti = scaler_Ti.inverse_transform(Ti_scaled.reshape(1, -1))
        Ta_RC = scaler_Ta.inverse_transform(Ta_scaled.reshape(1, -1))
        Ps_RC = scaler_Ps.inverse_transform(Ps_scaled.reshape(1, -1))
        Ti_RC = Ti
        #cont = -1
        #cont_RC = -1
        cont = 0
        cont_RC = 0
        i = 1
        j = 1
        
        df_events.loc[cont,dic_ind_date[int(index)+n_train]] = Ti
        df_RC.loc[cont_RC,dic_ind_date[int(index)+n_train]] = Ti
        
        
        #FLEX MODEL
        while Ti_mean-0.5<=Ti<=Ti_mean+0.5 and index+i<key2-n_train:
            dTi_hat = model.predict(dato_a_predecir)
            #print(dTi_hat)
            Ti_hat = Ti + dTi_hat
            dato_a_predecir = test_X[index+i,:,1:]
            Ti_hat_scaled = scaler_Ti.transform(Ti_hat)
            dato_a_predecir = np.concatenate((Ti_hat_scaled,dato_a_predecir), axis=1)
            dato_a_predecir = dato_a_predecir.reshape(1, 1, test_X.shape[2])
            dato_a_predecir[0,0,input_position_Ph] = 0
            cont+=1
            df_events.loc[cont,dic_ind_date[int(index)+n_train]] = Ti_hat
            Ti = Ti_hat
            i+=1
            
        #FLEX RC
        while Ti_mean-0.5<=Ti_RC<=Ti_mean+0.5 and index+j<key2-n_train: 
            Ti_hat_RC = Ti_RC+((Ta_RC-Ti_RC)/(R*C)+(Ps_RC/1000)*Aw/C)*0.25
            cont_RC+=1
            df_RC.loc[cont_RC,dic_ind_date[int(index)+n_train]] = Ti_hat_RC
            
            dato_a_predecir_RC = test_X[index+j-1,:,:]
            dato_a_predecir_RC = dato_a_predecir_RC.reshape(1, 1, test_X.shape[2])
            Ta_scaled = dato_a_predecir_RC[0,0,input_position_Ta]
            Ps_scaled = dato_a_predecir_RC[0,0,input_position_Ps]
            Ta_RC = scaler_Ta.inverse_transform(Ta_scaled.reshape(1, -1))
            Ps_RC = scaler_Ps.inverse_transform(Ps_scaled.reshape(1, -1))
            
            Ti_RC = Ti_hat_RC
            j+=1 
            
        count_vector= np.append(count_vector, [index,cont], axis = 0)
        count_vector_RC= np.append(count_vector_RC, [index,cont_RC], axis = 0)
    count_vector = np.reshape(count_vector,(int(len(count_vector)/2),2))
    count_vector_RC = np.reshape(count_vector_RC,(int(len(count_vector_RC)/2),2))
    return count_vector,count_vector_RC,df_events,df_RC



def ml_algorithm(building_name, floor, month, method, vars_ini, variables_training, variables_prediction,data,n_pred,n_values, percentage_train_data, results_path, results_path_method,date_1,date_2,R,C,Aw):
    '''
    The ml_algorithm function collects the distribution of all above functions to fit the block and the step methos depending on the 
    value of the n_pred we give. 
    It collects in three csv archives the results: one for collecting the original test data, another for collecting the results of 
    the block method and the last one for collecting the results of the step method.

    INPUT:
    - data: The dataset with all the relevant information for the model.
    - n_pred: Number of output variables.
    - n_values: Number of input variables for each timestep.

    OUTPUT:
    - dic_ind_date: index date dictionary.
    '''
         
    reframed_0,values_0,dic_ind_date_0 = preparing_df(data,4,len(vars_ini))
    filter_col = [col for col in reframed_0 if col.startswith('Ti(t+')]
    reframed_0 = reframed_0[filter_col]
    print('reframed_0')
    print(reframed_0)
    
    
    reframed,values,dic_ind_date = preparing_df(data,n_pred,n_values)
    reframed = moving(reframed,variables_training, variables_prediction, n_values, method_data = True )
    variables_to_drop = []
    for column in reframed.columns:
        if column not in variables_training + variables_prediction:
            variables_to_drop.append(column)
    reframed = reframed.drop(columns = variables_to_drop)
    print('reframed')
    print(reframed)
    reframed.loc[0:5].to_csv(results_path + 'r_head.csv')
    
    scale = reframed.iloc[:,:-n_pred]
    scaler,scaled = scaling(scale)
    scaled = np.append(scaled, reframed.iloc[:,-n_pred:].to_numpy(), axis=1)
    print(scaled)

    scaler_Ti,scaled_Ti = scaling(data['Ti'].to_numpy().reshape(-1, 1))
    scaler_Ta,scaled_Ta = scaling(data['Ta'].to_numpy().reshape(-1, 1))
    scaler_Ps,scaled_Ps = scaling(data['Ps'].to_numpy().reshape(-1, 1))

    n_train = int(percentage_train_data*values_0.shape[0])

    print(values.shape)
    test_X, test_y, model = training(scaled,n_pred,n_train)  
    
    test_y_aux = reframed_0[filter_col]
    
    df_data = pd.DataFrame(test_y_aux, columns = ['Ti(t+1)','Ti(t+2)','Ti(t+3)','Ti(t+4)'])
    df_data = df_data.reset_index(drop=True)
    
    df_data.to_csv(results_path+'data_konp_'+ building_name + '.csv') 
    
    #block method
    if method == 'block':
        inv_yhat,inv_y = predicting_blocks(test_X,test_y,model,n_pred)
        #inv_yhat,inv_y = predicting_blocks_1(test_X,test_y,model,scaler_Ti,n_pred)
        df = pd.DataFrame(inv_yhat, columns = ['Ti(t+1)','Ti(t+2)','Ti(t+3)','Ti(t+4)'])
        df.to_csv(results_path_method + 'block_pred_' + building_name + '.csv')
        return dic_ind_date, n_train
        

    #step Ti method
    elif method == 'step_Ti':
        all_Ti = predicting_step_Ti(test_X,test_y,model,scaler_Ti,n_pred,n_values)
        all_Ti = np.array(all_Ti)
        df_Ti = pd.DataFrame(all_Ti, columns = ['Ti(t+1)','Ti(t+2)','Ti(t+3)','Ti(t+4)'])
        df_Ti.to_csv(results_path_method + 'step_Ti_pred_' + building_name + '.csv')
        
        if month == 'February' or month == 'July':
            Ti_mean = np.mean(reframed['Ti(t)'][reframed['Ph(t+1)']>0].to_numpy())
            count_array,count_array_RC,df_events,df_RC = Ph0_input_step_Ti(test_X,test_y,model,scaler_Ti,scaler_Ta,scaler_Ps,dic_ind_date,Ti_mean,date_1,date_2,n_train,R,C,Aw,input_position_Ti=0,input_position_Ta=1,input_position_Ps=2,input_position_Ph=3)

            df_count,df_count_RC = storing_flex(reframed,count_array,count_array_RC,dic_ind_date,results_path,n_train)

            df_events.to_csv(results_path_method + 'flex_Ti.csv')
            df_RC.to_csv(results_path + 'flex_RC.csv')

            return dic_ind_date, n_train, Ti_mean, df_count, df_count_RC, df_events, df_RC
        else:
            return dic_ind_date, n_train
        
    #step dTi method
    elif method == 'step_dTi' :
        all_Ti = predicting_step_dTi(test_X,test_y,model,scaler_Ti,n_pred)
        all_Ti = np.array(all_Ti)
        df_Ti = pd.DataFrame(all_Ti, columns = ['Ti(t+1)','Ti(t+2)','Ti(t+3)','Ti(t+4)'])
        df_Ti.to_csv(results_path_method + 'step_dTi_pred_' + building_name + '.csv')
              
        if month == 'February' or month == 'July':
            Ti_mean = np.mean(reframed['Ti(t)'][reframed['Ph(t+1)']>0].to_numpy())
            count_array,count_array_RC,df_events,df_RC = Ph0_input_step_dTi(test_X,test_y,model,scaler_Ti,scaler_Ta,scaler_Ps,dic_ind_date,Ti_mean,date_1,date_2,n_train,R,C,Aw,input_position_Ti=0,input_position_Ta=1,input_position_Ps=2,input_position_Ph=3)
        
            df_count,df_count_RC = storing_flex(reframed,count_array,count_array_RC,dic_ind_date,results_path,n_train)
        
            df_events.to_csv(results_path_method + 'flex_dTi.csv')
            df_RC.to_csv(results_path + 'flex_RC.csv')
        
    
            return dic_ind_date, n_train, Ti_mean, df_count, df_count_RC, df_events, df_RC
        else:
            return dic_ind_date, n_train

def get_key(val,dic):
    '''
    The get_key function returns a key of a given value in a dictionary.

    INPUT:
    - val: Value to find in the dictionary.
    - dic: The dictionary in which the information is collected.

    OUTPUT:
    - key: key associated to the value in the dictionary.
    '''
    for key, value in dic.items():
         if val == value:
            return key

def visualization(method, building_name, floor, month, date_1, date_2, dic_ind_date, n_train, results_path, results_path_method):
    
    key1 = get_key(date_1,dic_ind_date)
    key2 = get_key(date_2,dic_ind_date)
    
    data_vis = pd.read_csv(results_path + "data_konp_" + building_name + '.csv',index_col = 0)
    Ti_data = data_vis.iloc[n_train:,:]
    Ti_data = Ti_data.to_numpy()
    Ti_data = Ti_data.astype('float32')
    
    method_vis = pd.read_csv(results_path_method + method + '_pred_' + building_name + '.csv',index_col = 0)
    Ti_method = method_vis.iloc[:,:]
    Ti_method = Ti_method.to_numpy()
    Ti_method = Ti_method.astype('float32')
    
    n_pred = 4
    err_method = []
    err_method_vis = []
    '''
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
    '''
    for i in range(0,n_pred):
        rmse_1 = np.sqrt(mean_squared_error(Ti_data[:,i], Ti_method[:,i]))
        err_method = np.append(err_method,rmse_1)
        
        rmse_vis = np.sqrt(mean_squared_error(Ti_data[key1-n_train:key2-n_train,i], Ti_method[key1-n_train:key2-n_train,i]))
        err_method_vis = np.append(err_method_vis,rmse_vis)
        
    for i in range(0,4):
        date = []
        #x = [k for k in range (0,key2-key1+4,4)]
        x = [k for k in range (0,key2-key1+4,8)]
        #for j in range(key1+i,key2+i+4,4):
        for j in range(key1+i,key2+i+4,8):
            date = np.append(date,np.array(dic_ind_date[j]))
        plt.xticks(x, date, rotation=90)
        
        plt.plot(Ti_data[key1-n_train:key2-n_train,i],label='Measured temperature')
        if method == 'block':
            plt.plot(Ti_method[key1-n_train:key2-n_train,i],label='Forecast',color="orange")
            plt.title(label = month +' Ti(t+'+str(i+1)+')', fontsize=14, pad = 20)
          
        elif method == 'step_Ti':
            plt.plot(Ti_method[key1-n_train:key2-n_train,i],label= 'Forecast',color="green")
            plt.title(month + ' Ti(t+'+str(i+1)+')', fontsize=14, pad = 20)
        
        elif method == 'step_dTi':
            plt.plot(Ti_method[key1-n_train:key2-n_train,i],label='Forecast',color="deeppink")
            plt.title(month + ' Ti(t+'+str(i+1)+')', fontsize=14, pad = 20)
        
        plt.subplots_adjust(left=None, bottom=None, right=1.4, top=1.2, wspace=None, hspace=None)
        plt.legend()
        plt.grid()
        name = method + building_name +'Ti(t+'+str(i+1)+')'+'.png'
        plt.tight_layout()
        plt.savefig(results_path_method + name, bbox_inches='tight', dpi = 400)
        plt.show()    
        
    return err_method, err_method_vis

def storing_flex(reframed,count_array,count_array_RC,dic_ind_date,results_path,n_train):
    
    df_count = pd.DataFrame(count_array, columns = ['Date','Flexibility'])
    df_count.set_index("Date", inplace = True)
    
    df_count_RC = pd.DataFrame(count_array_RC, columns = ['Date','Flexibility'])
    df_count_RC.set_index("Date", inplace = True)
    
    for index,row in df_count.iterrows():
        df_count.loc[index,'Date'] = dic_ind_date[int(index)+n_train]
        
    df_count.set_index("Date", inplace = True)
   
    
    for index,row in df_count_RC.iterrows():
        df_count_RC.loc[index,'Date'] = dic_ind_date[int(index)+n_train]
    df_count_RC.set_index("Date", inplace = True)
    
    
    df_count['Ph moved'] = 0
    df_count_RC['Ph moved'] = 0
    
    for index,row in df_count.iterrows():
        key = get_key(index,dic_ind_date)
        for i in range (int(df_count.loc[index,'Flexibility'])):
            df_count.loc[index,'Ph moved'] += reframed.loc[key+i,'Ph(t+1)']
    
    for index,row in df_count_RC.iterrows():
        key = get_key(index,dic_ind_date)
        for i in range (int(df_count_RC.loc[index,'Flexibility'])):
            df_count_RC.loc[index,'Ph moved'] += reframed.loc[key+i,'Ph(t+1)']
    
    df_count.to_csv(results_path  + 'flexibility.csv')
    df_count_RC.to_csv(results_path  + 'flexibility_RC.csv')
    
    return df_count,df_count_RC

'''
def visualization_flex(date, method_Ti, method_dTi, building_name, floor, month, dic_ind_date, n_train, Ti_mean, location, results_path_Ti, results_path_dTi,results_path):
    
    data_vis = pd.read_csv(results_path + 'data_konp_' + building_name + '.csv',index_col = 0)
    Ti_data = data_vis.iloc[n_train:,:]
    Ti_data = Ti_data.to_numpy()
    Ti_data = Ti_data.astype('float32')
    
    method_Ti_vis = pd.read_csv(results_path_Ti + method_Ti + '_pred_' + building_name + '.csv',index_col = 0)
    Ti_method = method_Ti_vis.iloc[:,:]
    Ti_method = Ti_method.to_numpy()
    Ti_method = Ti_method.astype('float32')

    method_dTi_vis = pd.read_csv(results_path_dTi + method_dTi + '_pred_' + building_name + '.csv',index_col = 0)
    dTi_method = method_dTi_vis.iloc[:,:]
    dTi_method = dTi_method.to_numpy()
    dTi_method = dTi_method.astype('float32')
    
    RC_vis= pd.read_csv(results_path + 'flex_RC.csv',index_col = 0)
    RC_flex = RC_vis.iloc[:,:]
    RC_flex = RC_flex.to_numpy()
    RC_flex = RC_flex.astype('float32')
    
    flex_Ti_vis = pd.read_csv(results_path_Ti + 'flex_Ti.csv',index_col = 0)
    Ti_flex = flex_Ti_vis.iloc[:,:]
    Ti_flex = Ti_flex.to_numpy()
    Ti_flex = Ti_flex.astype('float32')
    
    flex_dTi_vis = pd.read_csv(results_path_dTi + 'flex_dTi.csv',index_col = 0)
    dTi_flex = flex_dTi_vis.iloc[:,:]
    dTi_flex = dTi_flex.to_numpy()
    dTi_flex = dTi_flex.astype('float32')
    
    index_Ti = flex_Ti_vis.columns.get_loc(date)
    index_dTi = flex_dTi_vis.columns.get_loc(date)
    index_RC = RC_vis.columns.get_loc(date)
    key1 = get_key(date,dic_ind_date)
    key2 = key1 + len(flex_Ti_vis[date].dropna())

    
    print(key1,key2)
    date = []
    x = [k for k in range (0,key2-key1+4,4)]
    for j in range(key1,key2+4,4):
        date = np.append(date,np.array(dic_ind_date[j]))
    plt.xticks(x, date, rotation=90)

    plt.plot(Ti_data[key1-n_train:key2-n_train,0],label='Ti_data',color="blue")
    
    plt.plot(Ti_method[key1-n_train:key2-n_train,0],label='Ti_method',color="green")
    plt.plot(dTi_method[key1-n_train:key2-n_train,0],label='dTi_method',color="deeppink")

    plt.plot(Ti_flex[:key2-key1,index_Ti],label='Ti_flex',color="red")
    print(len(Ti_flex[:,index_Ti]))
    plt.plot(RC_flex[:key2-key1,index_RC],label='RC_flex',color="purple")
    plt.plot(dTi_flex[:key2-key1,index_dTi],label='dTi_flex',color="brown")
    
    Ti_plus= [] 
    Ti_minus = []
    for i in range(key2-key1):
        Ti_plus.append(Ti_mean+0.5)
        Ti_minus.append(Ti_mean-0.5)
    plt.plot(Ti_plus,'--',color = 'grey')
    plt.plot(Ti_minus,'--',color = 'grey')
    

    plt.subplots_adjust(left=None, bottom=None, right=1.4, top=1.2, wspace=None, hspace=None)
    plt.legend(loc = location)
    plt.grid()
    name = 'flex.png'
    plt.tight_layout()
    plt.savefig(results_path + name, bbox_inches='tight', dpi = 400)
    plt.show()    
    
'''