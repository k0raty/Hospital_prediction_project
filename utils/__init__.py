import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#from utils.clean_serie import *
from datetime import datetime
import os
import numpy as np
if os.getcwd() == r'C:\Users\anton\Documents\Helean\Hospital-AI\utils':
    os.chdir(r'C:\Users\anton\Documents\Helean\Hospital-AI')
from data.header import HEADER, KNOWN_HEADER, DAYS, MONTHS

PATH = 'C:\\Users\\anton\\Documents\\Helean\\database\\'


def NumpyStats(arr):
    """
    Prints all of the statistics of a given numpy array 
    """

    df_describe = pd.DataFrame(arr).describe()
    print(df_describe)

    return df_describe


def NumpyHistogram(arr):
    plt.hist(arr, bins='auto')
    plt.show()


def ModelLauncher(model, **kwargs):
    print('\nStart of the model training...\n')
    try:
        model.predict(**kwargs)
    except KeyboardInterrupt:
        try:
            save = input("\n\nSave ?(yes/n): ")
            if save == "yes":
                model.save()
        except (KeyboardInterrupt, EOFError) as err:
            print('\nShutting down...')
            return


def days_distance(d0, d1):
    """
    Return d1 - d0 in days 
    """
    date_format = "%Y-%m-%d"
    a = datetime.strptime(d0, date_format)
    b = datetime.strptime(d1, date_format)
    delta = b - a
    return delta.days


def df_converter(df: pd.DataFrame, T: int, y_dim: int):
    """
    Create X the data and Y the labelisation of an unique UM
    """
    X, Y = [], []
    df = df[HEADER]
    for k in range(df.shape[0] - T):
        x = df.iloc[k:k+T].to_numpy().tolist()

        # getting the value to predict
        # y_dim is fixed, it is the last two values of each row of df (each date)
        y = x[T-1][-y_dim:]

        # putting 0 in place of the value we want to forecast (in the x single data entry)
        for k in range(1, y_dim+1):
            x[T-1][-k] = 0
        X.append(x)
        Y.append(y)
    return X, Y


def clean_df_converter(df: pd.DataFrame, T: int):
    """
    Create X the data and Y the labelisation of an unique UM
    You can choose features to conserve by manipulating HEADER and KNOWN_HEADER
    
    KNOWN_HEADER : Corresponds to columns that we already know for each day 
    HEADER : Corresponds to columns that we have to predict/suppose
    
    """
    UM = df['UM'].iloc[0]  # On récupère l'um évaluée
    assert(T < len(df)), "L'UM %s ne contient pas assez d'élèment , seulement %d quand T est %d" % (
        UM, len(df), T)
    # Be careful to not put the mean of known values at the end of x !
    predictable_df = df[HEADER]
    X, Y = [], []
    for k in range(df.shape[0] - T):
        x = predictable_df.iloc[k:k+T].copy()  # Important to use copy !
        
        # getting the value to predict
        # y_dim is fixed, it is the last two values of each row of df (each date)
        y = x.iloc[-1][['nb_lits', 'nb_entry']]
        y['UM']=UM
        # be careful to not consider the last element in the mean !
        x_to_predict = x.iloc[:-1].mean()
        x.iloc[-1] = x_to_predict
        # putting 0 in place of the value we want to forecast (in the x single data entry)
        # You need to use .at to indeed modify the value of the dateframe
        x.at[x.index[-1], 'nb_lits'] = 0
        x.at[x.index[-1], 'nb_entry'] = 0
        assert(x['nb_lits'][:-1].to_list() ==
               df.iloc[k:k+T]['nb_lits'][:-1].to_list()), k
        assert(int(x.iloc[-1]['nb_lits']) == 0 &
               int(x.iloc[-1]['nb_entry']) == 0), k
        assert(y[['nb_entry', 'nb_lits']].to_list() == df.iloc[k:k+T]
               [['nb_entry', 'nb_lits']].iloc[-1].to_list()), k
        x[KNOWN_HEADER] = df[KNOWN_HEADER].iloc[k:k+T].copy()
        x["sum_days"] = x[DAYS].sum(axis=1)
        x["sum_months"] = x[MONTHS].sum(axis=1)
        # On vérifie qu'on ne prend pas la moyenne dans ce cas précis.
        assert(len(x) == len(x[x["sum_days"] == 1])
               ), "Days aren't well defined !"
        # On vérifie qu'on ne prend pas la moyenne dans ce cas précis.
        assert(len(x) == len(x[x["sum_months"] == 1])
               ), "Months aren't well defined !"
        x = x.drop(columns=['sum_days', 'sum_months','day','month'])
        order_columns=list(x.columns)
        index_lit=order_columns.index('nb_lits')
        order_columns.pop(index_lit)
        index_entry=order_columns.index('nb_entry')
        order_columns.pop(index_entry)
        order_columns+=['nb_lits','nb_entry']
        x=x.reindex(columns=order_columns)
        X.append(x)
        Y.append(y)
    for x in X:
        assert(('UM' in x.columns) ==
               True), "L'UM n'est pas renseigné dans l'un des x de X"
    return X, Y


def split_data(X, Y, test_split=0.1, validation_split=0.2):
    _X_train, X_test, _Y_train, Y_test = train_test_split(
        X, Y, shuffle=True, test_size=test_split, random_state=None)

    X_train, X_val, Y_train, Y_val = train_test_split(
        _X_train, _Y_train, shuffle=True, test_size=validation_split, random_state=None)

    return X_train, X_val, Y_train, Y_val, X_test, Y_test


# X_test_df and Y_test_df are for plotting expectation
def splitting_data(X, Y, test_split=0.1, validation_split=0.2,LSTM=False):
    
    X_train_df, X_val, Y_train_df, Y_val, X_test, Y_test = split_data(
        X, Y, test_split=test_split, validation_split=validation_split)

    X_train, X_val = np.array([X_train_df[i].loc[:, X_train_df[i].columns != 'UM'].to_numpy() for i in range(0, len(
        X_train_df))]), np.array([X_val[i].loc[:, X_val[i].columns != 'UM'].to_numpy() for i in range(0, len(X_val))])
    Y_train, Y_val = np.array([Y_train_df[i][Y_train_df[i].index != 'UM'].to_numpy() for i in range(0, len(
        Y_train_df))]), np.array([Y_val[i][Y_val[i].index != 'UM'].to_numpy() for i in range(0, len(Y_val))])
    X_test, Y_test = np.array([X_test[i].loc[:, X_test[i].columns != 'UM'].to_numpy() for i in range(
        0, len(X_test))]), np.array([Y_test[i][Y_test[i].index != 'UM'].to_numpy() for i in range(0, len(Y_test))])
    Y_train, Y_val,Y_test=Y_train.astype(int),Y_val.astype(int),Y_test.astype(int)
    return X_train, X_val, Y_train, Y_val, X_test, Y_test, X_train_df, Y_train_df


def mean(x: list):
    return sum(x)/len(x)
