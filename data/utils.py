from typing import List
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os
os.chdir(r'C:\Users\anton\Documents\Helean\Hospital-AI')
import data.data as data

def severity_score(x: str):
    try:
        # return the severity if x is a number
        return int(x)/5
    except ValueError:
        if x == "J":
            return 0.4
        elif x == "T":
            return 0.3
        elif x == "E":
            return 1
        elif x == "Z":
            return 0.1
        elif x in ["A", "B", "C", "D"]:
            return 0.2*(1+["A", "B", "C", "D"].index(x))
        return np.nan


def get_grippe():
    grippe_df = pd.read_csv(data.PATH+'grippe.csv')

    grippe_df['week'] = grippe_df['week'].astype(str)

    grippe_df['date'] = pd.to_datetime(
        grippe_df['week'].str[:4] + '-' + grippe_df['week'].str[4:] + '-1', format="%Y-%W-%w")

    grippe_df = grippe_df.resample('D', on='date').max().fillna(
        method='ffill')

    grippe_df = grippe_df[grippe_df['date'] > '2017-01-01']

    return grippe_df['inc100'].sort_index().rename('grippe')


def get_calendar():
    """
    Renseigne sur les vacances et jours fériés, ou jours particulier.
    

    Returns
    -------
    calendar_df : dataframe de la sorte :
                special_day  vac_a  vac_b  vac_c
    date                                        
    2015-01-01            0      1      1      1
    2015-01-02            0      1      1      1
    2015-01-03            0      1      1      1
    2015-01-04            0      0      0      0 

    """
    
    calendar_df = pd.read_csv(data.PATH + 'calendar.csv', sep=';')

    calendar_df.drop(['Dimanche', 'Lundi', 'Mardi', 'Mercredi',
                      'Jeudi', 'Vendredi', 'Samedi'], axis=1, inplace=True)

    calendar_df['vac_a'] = calendar_df['holidays'].str.contains(
        'A').astype(int)
    calendar_df['vac_b'] = calendar_df['holidays'].str.contains(
        'B').astype(int)
    calendar_df['vac_c'] = calendar_df['holidays'].str.contains(
        'C').astype(int)

    
    calendar_df= calendar_df.drop(['holidays','Ferie'], axis=1).set_index('date').rename(columns={
        'SpecialDays': 'special_day'
    })
    calendar_df.reset_index(drop=False, inplace=True) #add a column date
    
    calendar_df['date']=pd.to_datetime(calendar_df['date']) #Changing index type to date-time with rearranging 
    calendar_df=calendar_df.set_index('date')
    return calendar_df

def get_covid():
    covid = pd.read_csv(data.PATH+'covid.csv')

    covid['week'] = covid['week'].astype(str)

    covid['date'] = pd.to_datetime(
        covid['week'].str[:4] + '-' + covid['week'].str[4:] + '-1', format="%Y-%W-%w")

    covid = covid.resample('D', on='date').max().fillna(
        method='ffill')

    return covid['inc100'].sort_index().rename('covid')


def get_date_df(date_index: List[str]):

    df = pd.DataFrame(index=date_index, columns={
                      'lun': []*len(date_index)})
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d')

    df[['lun', 'mar', 'mer', 'jeu', 'ven', 'sam', 'dim']] = df.apply(
        lambda x: [1 if i == x.name.weekday() else 0 for i in range(7)],
        axis=1,
        result_type='expand'
    ) # Associe le jour avec chaque date passée en index de date_index

    df['day']=df.apply(lambda x: give_day(x.name.weekday()),axis=1,result_type='expand')
    
    
    df[["janv", "fev", "mars", "avr", "mai", "juin",
        "juil", "aout", "sept", "oct", "nov", "dec"
        ]] = df.apply(
        lambda x: [1 if i == x.name.month-1 else 0 for i in range(12)],
        axis=1,
        result_type='expand'
    )

    df['month']=df.apply(lambda x: give_month(x.name.month),axis=1,result_type='expand')

    df.index = df.index.strftime('%Y-%m-%d') #Important d'avoir un type d'index unique n ici c'est du string

    return df

def give_day(i):
    """
    

    Parameters
    ----------
    i : x.name.weekday()

    Returns
    -------
    String corresponding to the current day  .

    """
    days =['lun', 'mar', 'mer', 'jeu', 'ven', 'sam', 'dim']
    return days[i]

def give_month(i):
    """
    

    Parameters
    ----------
    i : x.name.month

    Returns
    -------
    String corresponding to the current month  .

    """
    months =["janv", "fev", "mars", "avr", "mai", "juin",
        "juil", "aout", "sept", "oct", "nov", "dec"
        ]
    return months[i-1]

def mean(x: list):
    return sum(x)/len(x)

def percent_error(y_true,y_pred):
    """
    Erreur relative proportionnelle qui prend en compte le nombre de lit dans chaque unité :
        -Une erreur de un lit pour une prediction de 10 lit est finalement égale à une erreur de 10 lits pour une prédiction de 100.
        -On aspire donc à une performance en proportion plutôt qu'en nombre de lit absolu.
        -Si jamais y_true fait 0, on le remplace par 1
    Parameters
    ----------
    y_true : np.array()
        .
    y_pred : np.array()
        .

    Returns
    -------
   Proportionnal error in % ,ratio -> float quantifying the loss of data in percent %

    """
    n=len(y_true)
    L=[abs(y_true[i]-y_pred[i])/max(1,y_true[i])*100 for i in range(0,len(y_true))]
    
    #Removing outliers#
    q75,q25 = np.percentile(L,[75,25])
    intr_qr = q75-q25
    maximum = q75+(1.5*intr_qr)
    M=[i for i in L if i < maximum]
    ratio=(1- len(M)/len(L))*100 #Taux de perte de données
    return (1/n)*sum(M), ratio