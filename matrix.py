import scipy.stats
import numpy as np
import pandas as pd
def mdn_matrix(y_test,y_pred,mu,std):

    y_test = np.array(y_test).reshape(1, -1).T
    y_pred = np.array(y_pred).reshape(1, -1).T
    mu = np.array(mu).reshape(1, -1).T
    std = np.array(std).reshape(1, -1).T

    from sklearn import metrics
    MSE = metrics.mean_squared_error(y_test,y_pred)
    RMSE = metrics.mean_squared_error(y_test, y_pred)**0.5
    MAE = metrics.mean_absolute_error(y_test,y_pred)
    MAPE = metrics.mean_absolute_percentage_error(y_test, y_pred)
    import properscoring as ps
    from scipy.stats import norm
    crps = np.zeros(len(y_test))
    for i in range(len(y_test)):
        crps[i] = ps.crps_gaussian(y_test[i], mu=mu[i], sig=std[i])
    CRPS = np.mean(crps)
    return RMSE, MAPE, CRPS

