import numpy as np
import pandas as pd
import math
import tensorflow.keras as keras
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K
import scipy.stats
from keras.layers import Input,Dense,Conv1D,Flatten
from sklearn import metrics
import properscoring as ps
from scipy.stats import norm

def get_mixture_coef(output, Training=True):   # out_mu.shape=[len(x_data),KMIX]
    KMIX = 2
    out_pct = output[:, :KMIX]
    out_mu = output[:, KMIX:2*KMIX]
    out_std = K.exp(output[:, 2*KMIX:]) if Training else np.exp(output[:, 2*KMIX:])
    return out_pct, out_mu, out_std

def get_loss(pct, mu, std, y):
    factors = 1 / math.sqrt(2*math.pi) / std
    exponent = K.exp(-1/2*K.square((y-mu)/std))
    GMM_likelihood = K.sum(pct*factors*exponent, axis=1);
    log_likelihood = -K.log(GMM_likelihood)
    return K.mean(log_likelihood)

def loss_func(y_true, y_pred):
    out_pct, out_mu, out_std = get_mixture_coef(y_pred)
    result = get_loss(out_pct, out_mu, out_std, y_true)
    return result

def mdgru_matrix(X_train,X_test,y_train,y_test,NHIDDEN1,NHIDDEN2,NHIDDEN3,EP):
    from sklearn.preprocessing import StandardScaler
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train1 = x_scaler.fit_transform(X_train)
    #X_calib = x_scaler.transform(X_calib)
    X_test1 = x_scaler.transform(X_test)

    y_train = np.array(y_train).reshape(1, -1).T
    y_test = np.array(y_test).reshape(1, -1).T
    y_train1 = y_scaler.fit_transform(y_train)
    #y_calib = y_scaler.transform(y_calib)
    y_test1 = y_scaler.transform(y_test)
    
    X_train1 = np.expand_dims(X_train1, axis=2)
    X_test1 = np.expand_dims(X_test1, axis=2)
    y_train1 = np.expand_dims(y_train1, axis=2)
    y_test1 = np.expand_dims(y_test1, axis=2)

    KMIX = 2
    NOUT = KMIX * 3          # number of pct, mu, std

    Input = keras.Input(shape=(12,1))
    hidden = layers.GRU(NHIDDEN1,activation ='relu')(Input)
    hidden = layers.Dropout(0.5)(hidden)
    hidden = layers.Dense(NHIDDEN2,activation ='relu')(hidden)
    hidden = Flatten()(hidden)
    hidden = layers.Dense(NHIDDEN3,activation ='relu')(hidden)
    hidden = Flatten()(hidden)
#     op = layers.GRU(NHIDDEN2,name='op',return_sequences=True)(hidden)
#     op = layers.GRU(NHIDDEN3)(op)
    op = layers.Dense(KMIX,activation='linear')(hidden)
    op = layers.Softmax()(op)

#     ou = layers.GRU(NHIDDEN2,name='ou',return_sequences=True)(hidden)
#     ou = layers.GRU(NHIDDEN3)(ou)
    ou = layers.Dense(KMIX,activation='linear')(hidden)

#     os = layers.GRU(NHIDDEN2,name='os',return_sequences=True)(hidden)
#     os = layers.GRU(NHIDDEN3)(os)
    os = layers.Dense(KMIX,activation='linear')(hidden)

    Output = layers.Concatenate()([op,ou,os])
    model = keras.Model(Input,Output)
    model.compile(optimizer='adam',loss=loss_func,metrics=[loss_func])
    history=model.fit(X_train1,y_train1,epochs=EP,batch_size=16,verbose=2)

    out_pct_test, out_mu_test, out_std_test = get_mixture_coef(model.predict(X_test1), Training=False)
    out_new = model.predict(np.array(X_test1))
    pct_new, mu_new, std_new = get_mixture_coef(out_new, Training=False)  # (*_new)都是行向量
    r =  pct_new*mu_new
    rr = r.sum(1)
    y_pred =  y_scaler.inverse_transform(rr)
    r_std =  pct_new*std_new
    std_pred = r_std.sum(1)
    
    q=0.95
    upper = mu_new + std_new * scipy.stats.norm.ppf((1 + q) / 2)
    lower = mu_new - std_new * scipy.stats.norm.ppf((1 + q) / 2)
    r_upper =  pct_new*upper
    rr_upper = r_upper.sum(1)
    y_pred_upper =  y_scaler.inverse_transform(rr_upper)
    r_lower =  pct_new*lower
    rr_lower = r_lower.sum(1)
    y_pred_lower =  y_scaler.inverse_transform(rr_lower)
    from sklearn import metrics
    MSE = metrics.mean_squared_error(y_test,y_pred)
    RMSE = metrics.mean_squared_error(y_test, y_pred)**0.5
    MAE = metrics.mean_absolute_error(y_test,y_pred)
    MAPE = metrics.mean_absolute_percentage_error(y_test, y_pred)
    import properscoring as ps
    from scipy.stats import norm
    s = pct_new*std_new
    crps = np.zeros(len(y_test))
    for i in range(len(y_test)):
        crps[i] = ps.crps_gaussian(y_test1[i], mu=rr[i], sig=s.sum(1)[i])
    CRPS = np.mean(crps)
    MA = [RMSE, MAPE, CRPS]
    return MA, y_pred, std_pred

def mdgru_values(X_train,X_test,y_train,y_test,NHIDDEN1,NHIDDEN2,NHIDDEN3,EP):
    from sklearn.preprocessing import StandardScaler
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train1 = x_scaler.fit_transform(X_train)
    #X_calib = x_scaler.transform(X_calib)
    X_test1 = x_scaler.transform(X_test)

    y_train = np.array(y_train).reshape(1, -1).T
    y_test = np.array(y_test).reshape(1, -1).T
    y_train1 = y_scaler.fit_transform(y_train)
    #y_calib = y_scaler.transform(y_calib)
    y_test1 = y_scaler.transform(y_test)
    
    X_train1 = np.expand_dims(X_train1, axis=2)
    X_test1 = np.expand_dims(X_test1, axis=2)
    y_train1 = np.expand_dims(y_train1, axis=2)
    y_test1 = np.expand_dims(y_test1, axis=2)

    KMIX = 6
    NOUT = KMIX * 3          # number of pct, mu, std

    Input = keras.Input(shape=(6,1))
    hidden = layers.GRU(NHIDDEN1,activation ='relu')(Input)
    hidden = layers.Dropout(0.5)(hidden)
    hidden = layers.Dense(NHIDDEN2,activation ='relu')(hidden)
    hidden = Flatten()(hidden)
    hidden = layers.Dense(NHIDDEN3,activation ='relu')(hidden)
    hidden = Flatten()(hidden)
#     op = layers.GRU(NHIDDEN2,name='op',return_sequences=True)(hidden)
#     op = layers.GRU(NHIDDEN3)(op)
    op = layers.Dense(KMIX,activation='linear')(hidden)
    op = layers.Softmax()(op)

#     ou = layers.GRU(NHIDDEN2,name='ou',return_sequences=True)(hidden)
#     ou = layers.GRU(NHIDDEN3)(ou)
    ou = layers.Dense(KMIX,activation='linear')(hidden)

#     os = layers.GRU(NHIDDEN2,name='os',return_sequences=True)(hidden)
#     os = layers.GRU(NHIDDEN3)(os)
    os = layers.Dense(KMIX,activation='linear')(hidden)

    Output = layers.Concatenate()([op,ou,os])
    model = keras.Model(Input,Output)
    model.compile(optimizer='adam',loss=loss_func,metrics=[loss_func])
    history=model.fit(X_train1,y_train1,epochs=EP,batch_size=16,verbose=2)
    out_pct_test, out_mu_test, out_std_test = get_mixture_coef(model.predict(X_test1), Training=False)
    out_new = model.predict(np.array(X_test1))
    pct_new, mu_new, std_new = get_mixture_coef(out_new, Training=False) 
    r_mu =  pct_new*mu_new
    rr_mu = r_mu.sum(1)
    mu_pred =  y_scaler.inverse_transform(rr_mu)
    r_std =  pct_new*std_new
    rr_std = r_std.sum(1)
    std_pred =  y_scaler.inverse_transform(rr_std)
    return mu_pred, std_pred

