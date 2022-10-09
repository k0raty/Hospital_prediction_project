"""
Class to create the dateframe

"""
from typing import List
from matplotlib import pyplot as plt
import os
import pandas as pd
import numpy as np
from datetime import datetime as dt

os.chdir(r'C:\Users\anton\Documents\Helean\Hospital-AI')

if os.getcwd() == r'C:\Users\anton\Documents\Helean\Hospital-AI\data':
    from utils import get_calendar, get_covid, get_date_df, get_grippe, severity_score, mean #In data folder
    from data import CMD, SPECIAL_CMD #In data folder
    from header import HEADER #In data folder
else:
    from data.data import CMD, SPECIAL_CMD #In data folder
    from data.header import HEADER #In data folder
    from data.utils import get_calendar, get_covid, get_date_df, get_grippe, severity_score, mean #In data folder

PATH = 'C:\\Users\\anton\\Documents\\Helean\\database\\'
#PATH="/home/k0raty/Documents/Helean/database/"

class HospitalDataHandler:

    def __init__(self):
        self.df: pd.DataFrame = pd.read_csv(PATH + 'RSA_1_sorted.csv', sep=';')
        for k in range(2, 6):
            self.df = self.df.append(
                pd.read_csv(f'{PATH}RSA_{k}_sorted.csv', sep=';')
            )#
        self._buff_df = pd.DataFrame()

    def get_columns_values(self):
        for col in self.df.columns:
            print(self.df[col].value_counts())

    @property
    def UM_LIST(self):
        try:
            return self._UM_LIST
        except AttributeError:
            self.build_df()
            return self._UM_LIST
   
    def build_df(self, finess=1):
        if not self._buff_df.empty:
            return self._buff_df.copy()

        df = self.df.copy()

        # keeping the data of the first hospital
        df = df[df['Finess'] == finess]
        
        
        # removing rows where we don't have an end date value
        df = df[df['end_date'] != '-1']
        df = df[df['start_date'] != '-1']

        # removing special CMD
        df = df[~df['CMD'].isin(SPECIAL_CMD)]

        # entrée en urgence uniquement
        df=df[df["Mode_Entree"]=='8']
        
        #You must change the type of this data right now in order to create df['date'] without issue .
        #df['start_date']=df.agg({'start_date' : lambda x : dt.strptime(str(x),'%Y-%m-%d')}) #On change les type des dates du fichier csv qui sont des string 
        df['start_date']=pd.to_datetime(df['start_date'],dayfirst=True) #dayfirst = True is essential
        #df['end_date']=df.agg({'end_date' : lambda x : dt.strptime(str(x),'%Y-%m-%d') }) #Si ce sont des string , la creation de df_dates est erronée ! min et max sont pris aléatoirement ......
        df['end_date']=pd.to_datetime(df['end_date'],dayfirst=True)
        # creating a rows with all of the days where the patient was in the specific hospital region
        df['date'] = df.apply(
            lambda x: pd.date_range(x['start_date'], x['end_date'], freq='d').strftime(
                '%Y-%m-%d').tolist(),
            axis=1
        )
        df['date']=df.agg({'date' : lambda x : [dt.strptime(str(i),'%Y-%m-%d') for i in x]}) #Change the type of date in datetime (to order after otherwise it is impossible)
        # the id of the location in the hospital
        
        #df['UM'] = df['CMD']+df['Num_Unite'].astype(str)
        df['UM'] = df['Num_Unite'].astype(str)
        UM_list = df['UM'].value_counts()

        # removing rows with a UM that isn't popular enough
        UM_list = UM_list[UM_list > 400].index.tolist()
        df = df[df['UM'].isin(UM_list)]

        # saving the list of UMs
        self._UM_LIST: list[str] = UM_list
        
       
        
        # getting the entry and the exit of the person in the hospital
        df_dates = df.filter(['Num_Index', 'start_date', 'end_date'], axis=1 #filter keeps this columns
                             ).groupby('Num_Index' #To apply an operation based on raws with the same index
                                       ).agg({'start_date': 'min', 'end_date': 'max'} #Give the min and the max of the entire stay for a particular patient
                                             ).rename(columns={'start_date': 'entry_date', 'end_date': 'exit_date'}
                                                      ).reset_index(drop=False)
              
        ###Checking if the min and max were well defined###OK
        for i in range(0,len(df_dates)):
            assert(df_dates['entry_date'].iloc[i]<=df_dates['exit_date'].iloc[i]),"Problème au niveau du min et max des dates à la ligne %d"%i
            
        df = df.merge(df_dates, how='left', on='Num_Index') # It add the three other columns of df_dates but identically for each same num_index 
        
        for i in range(0,len(df_dates)):
            assert(df_dates['entry_date'].iloc[i]<=df_dates['exit_date'].iloc[i]),"Problème au niveau du min et max des dates à la ligne %d"%i
            
        #On retire les informations inutiles , maintenant qu'on a la date notamment. 
        df.drop(['Mois_Entree',
                 'Annee_Entree',
                 'Annee_Sortie',
                 'Mois_Sortie',
                 # 'CMD',
                 'Type_GHM',
                 'DP',
                 'Mode_Sortie'
                 ], axis=1, inplace=True)

        self._buff_df = df.copy()

        return df


    def UM_dataframe(self, UM: str):
        df: pd.DataFrame = self.build_df()

        # selecting rows with the UM code
        #df = df[df['UM'] == UM].drop(['UM'], axis=1)
        df = df[df['UM'] == UM] #not dropping the  column UM

        if df.empty:
            return df

        # getting the CMD info
       # CMD_list = df['CMD'].value_counts().index.tolist()
        #for c in CMD_list:
           # print(CMD[c])
   #     df.drop('CMD', axis=1, inplace=True)

        # exploding according to the date column and
        # groupping by this column
        df_groupby = df.explode('date').groupby('date') #Explode aims at creating a row for each element of the lists in "date" column.

        df = df_groupby.agg({
            'Sexe': 'mean',
            'Complexite_GHM': lambda x: mean([severity_score(i) for i in x]),
            'Age_Annees': 'mean',
            'Nbr_Actes': 'mean',
            'Num_Index': lambda x: len(x),
            'start_date': lambda x: list(x),
            'end_date': lambda x: list(x),
            'entry_date': lambda x: list(x),
            'exit_date': lambda x: list(x)
        }).rename(columns={
            'Num_Index': 'nb_lits',
            'Age_Annees': 'age',
            'Complexite_GHM': 'severity',
            'Sexe': 'gender_avg',
            'Nbr_Actes': 'nb_actions'
        }) #At  this point , each row is a day in the UM , we got plenty of means relatively.
        
        df['nb_entry'] = df.apply( #df is a multilevel index, x.name corresponds to the date (so the index)
             lambda x: sum([
                 1 for i in x['start_date'] if  i == x.name
             ]),
             axis=1
         )
       ###Sorting by dates###
        df=df.sort_index(axis = 0)
        for i in range(0,len(df)-1):
            assert(df.index[i]<df.index[i+1]),"Problème au niveau de l'ordre des dates à l'index %d"%i
            assert(min(df['start_date'].iloc[i])<=df.index[i]<=min(df['end_date'].iloc[i])),"Problème au niveau de start_date et end_date à l'index %d"%i
        # quantifying dates columns
        for key in ['start_date', 'end_date', 'entry_date', 'exit_date']:
            df[key] = df.apply(
                lambda x: mean([
                    abs((x.name - i).days) for i in x[key]
                ]),
                axis=1
            )
            
        
        # adding a column for the entrance of the day

       
        # adding missing dates with mean of the previous 20 days to any of the attributes
        df.reset_index(drop=False, inplace=True) #add a column date
        
        df = df.resample('D', on='date').max() #Add missing dates between the known ones
        df2 = df[df.isna().any(axis=1)] #Keeping missing dates
        for i in range(0,len(df2.index)):
            index=df2.index[i]
            dp=df.loc[:index]
            dp=dp.drop(dp.tail(1).index)
            df.loc[index]=dp.iloc[-20:].mean() #Filling values with the mean of the previous 20 dates
        df['UM']=df.apply(lambda x: UM,axis=1)
        ###Rounding values of missing dates###
        df[['nb_lits',
               'start_date', 'end_date', 'entry_date', 'exit_date', 'nb_entry']]=df[['nb_lits',
               'start_date', 'end_date', 'entry_date', 'exit_date', 'nb_entry']].round()
        df=df.drop(["date"], axis=1)  
        
        # converting index to string for better merging later on
                                                                          
        df.index = df.index.strftime('%Y-%m-%d')
        # adding calendar data

        calendar = get_calendar()
        calendar.index = calendar.index.strftime('%Y-%m-%d')

        df = df.merge(calendar, how='left', left_index=True, right_index=True)
        
        ###Checking if merging was ok###
        for i in range(0,len(df)-1):
            date=df.index[i]
            ligne_df=df.loc[date]
            ligne_calendar=calendar.loc[date]
            assert(ligne_df['vac_a']==ligne_calendar['vac_a']),"Probleme au niveau du merging entre calendrier et dataframe à la ligne %d"%i
            assert(ligne_df['vac_b']==ligne_calendar['vac_b']),"Probleme au niveau du merging entre calendrier et dataframe à la ligne %d"%i
            assert(ligne_df['vac_c']==ligne_calendar['vac_c']),"Probleme au niveau du merging entre calendrier et dataframe à la ligne %d"%i

        # adding grippe data
        grippe = get_grippe()
        grippe.index = grippe.index.strftime('%Y-%m-%d')
        #df = df.merge(grippe, how='left', left_index=True, right_index=True)

        # adding covid data
        covid = get_covid()
        covid.index = covid.index.strftime('%Y-%m-%d')
        #df = df.merge(covid, how='left', left_index=True, right_index=True)

        # dates
        weeknmonth = get_date_df(df.index.tolist()) 
        df = df.merge(weeknmonth, how='left',
                      left_index=True, right_index=True) ###Adding name of the days and month for each of the dates.
        
        if False:
            df['nb_lits'].plot()
            (df['vac_b']*40).plot()
            (df['grippe']/10).plot()
            (df['covid']/10).plot()
            plt.show()

        # df[['nb_lits', 'nb_entry']].plot()

        # sns.pairplot(df[['nb_lits', 'grippe', 'covid']], kind="kde")
        # plt.show()
        index=df.index #On verifie que chaque date est bien unique 
        assert((True in index.duplicated())== False),"Il y a des dates qui revienne plusieurs fois dans l'UM %s"%UM
        return df


    
