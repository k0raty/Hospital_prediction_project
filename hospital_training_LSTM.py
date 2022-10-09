# -*- coding: utf-8 -*-


"""
File to generate the model from the scratch.
"""
import numpy as np
from numpy import array
import os
os.chdir(r'C:\Users\anton\Documents\Helean\Hospital-AI')
from numpy import mean 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_absolute_error
from data.data_handler import HospitalDataHandler
from data.data import PATH
from utils import clean_df_converter,split_data
from tqdm.auto import tqdm, trange
from utils.data_plotting import hist_plot
from data.header import HEADER,KNOWN_HEADER,TOTAL_COLUMNS,NO_UM_KNOWN_HEADER,MONTHS,DAYS,UM_LIST,STRING_HEADER
from data.utils import percent_error,get_calendar,get_date_df
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from data.utils import give_day,give_month
import warnings
import logging
import sys
import pandas as pd
import datetime as dt
import shutil

warnings.simplefilter(action='ignore', category=FutureWarning)

class HospitalPredictor:

    def __init__(self):  
        self.data = HospitalDataHandler()
        self.data.build_df() #Build the dataframe with all the UM
        self.params = {
            'T': 20, #Number of days to forcast
            'test_split': 0.15,
            'validation_split': 0.15,
            'verbose': 1,
            'epoch': 10,
            'UM_tot':15

        }
        self.history= {} # History of the model
        self.dict_df_temp= {}#List of dataframes predicting for each prediction 
        self.df_prediction_lit=pd.DataFrame(columns=MONTHS+DAYS+['UM','projection','taux_perte']) # keep performances data (mean_abs_err) for each prediction query (UM,prediction)
        self.normalize={}#Dictionary of constants of normalization {'inf':,'max':}
        self.X_prediction, self.Y_prediction=[],[] #equivalent to X and Y but in oder to make predictions instead of training the model. 
        self._model={}
        self.stop_date={}
    def train(self,in_advance,UM_train=None,stop_date=None):#UM_train is a list of UM we want to use to train the model
        """
        On impose une date butoir pour l'entraînement , le set sera conservé.
        Train the model to build it
        It's essential to use many other functions such as plot_prediction ...
        """
        
       

        if 'verbose' in self.params.keys():
            if self.params['verbose'] == 0:
                logging.basicConfig(stream=sys.stderr, level=logging.INFO)
            elif self.params['verbose'] == 1:
                logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
        
                
        #On identifie le nombre d'UM à intégrer dans le set d'entraînement#
        
        if UM_train == None :
            self.UM_train= self.data.UM_LIST[1:self.params['UM_tot']]
        else : 
            self.UM_train= UM_train
        
        self.stop_date[in_advance]=stop_date #Remembering the stopping date for your model
        self.set=self.create_set(stop_date=stop_date) #Creating the set based on concatenation of previous UM's
        self.set=self.normalize_model()  #Normalizing the set
        self.build_set(in_advance) #Filling X and Y 
        
        n_features = self.X.shape[2]
        
        # define model
        model = Sequential()
        model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(self.params['T'], n_features)))
        model.add(LSTM(100, activation='relu')) #reccurent_activation ??? sigmoid/tanh ect...
        model.add(Dense(in_advance))
        model.compile(optimizer='adam', loss='mse',metrics=[
            tf.metrics.MeanAbsolutePercentageError(),
            tf.metrics.MeanAbsoluteError()
        ])
        
        self._model[in_advance] = model
        # train and test sets
        self.X_train, self.X_val, self.Y_train, self.Y_val, self.X_test, self.Y_test= split_data(
            self.X, self.Y,
            self.params['test_split'],
            self.params['validation_split']
        )
             
        try:
            self.history[in_advance]=self._model[in_advance].fit(
                x=self.X_train, y=self.Y_train,
                epochs=self.params['epoch'],
                batch_size=30,
                validation_data=(self.X_val, self.Y_val),
            )
        except KeyboardInterrupt:
            save = input('\nSave Model ?(yes/): ')
            if save == 'yes':
                self.evaluate(self.X_test, self.Y_test)
                self.save()
            return self

        self.evaluate(self.X_test, self.Y_test)

        return self

    def save(self,in_advance):
        try:
            self._model[in_advance].save(
                PATH + "model",
                include_optimizer=False,
                save_format='tf'
            )
        except:
            logging.error('Cannot save model')

        return self
    
    def create_set(self,stop_date:str=None):
        """
        

        Parameters
        ----------
        stop_date : str, optional
            Datetime until we stop taking the data to train the model. The default is None.

        Returns
        -------
        df : TYPE
            DESCRIPTION.

        """
        if stop_date == None :
            df=self.data.UM_dataframe(self.UM_train[0])
            if len(self.UM_train)>1:
                df_2=df.copy()
                for UM in self.UM_train[1:]:
                    df_2=self.data.UM_dataframe(UM)
                    df=pd.concat([df,df_2])
            return df
        else : 
            df=self.data.UM_dataframe(self.UM_train[0]).loc[:stop_date]
            if len(self.UM_train)>1:
                df_2=df.copy()
                for UM in self.UM_train[1:]:
                    df_2=self.data.UM_dataframe(UM).loc[:stop_date]
                    df=pd.concat([df,df_2])
            return df
    def build_set(self,in_advance,df=pd.DataFrame()):
        """
        Place les colonnes à prédire en dernier.

        Parameters
        ----------
        in_advance : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        if len(df)==0:
            df=self.set
        
        #Ordering columns#
        order_columns=list(df.columns)
        index_lit=order_columns.index('nb_lits') #Columns to forcast are at the end.

        order_columns.pop(index_lit)
        #index_entry=order_columns.index('nb_entry')
        #order_columns.pop(index_entry)
        #order_columns=order_columns+['nb_lits','nb_entry']
        order_columns=order_columns+['nb_lits']
        df=df.reindex(columns=order_columns)
       # df=df.drop(columns=['nb_entry'])
        dataset=df.drop(columns=STRING_HEADER).to_numpy().copy()
        X, y = self.split_sequences(dataset, in_advance)

        self.X=X #Each element of X is a dataframe with T-1 prevision and the raw to predict
        self.Y=y

    def split_sequences(self, dataset, n_steps_out):
        """
        
        Parameters
        ----------
        sequences : Dataset to rearrange to train the model.
        n_steps_in : Number of days to take into account before the day to forcast.
        n_steps_out : Number of days to forcast.

        Returns
        -------
        X and Y , data set and label set.

        """
        
        n_steps_in=self.params['T']
       	X, y = list(), list()
       	for i in range(len(dataset)):
       		# find the end of this pattern
       		end_ix = i + n_steps_in
       		out_end_ix = end_ix + n_steps_out-1
       		# check if we are beyond the dataset
       		if out_end_ix > len(dataset):
       			break
       		# gather input and output parts of the pattern
       		seq_x, seq_y = dataset[i:end_ix, :-1], dataset[end_ix-1:out_end_ix, -1]
       		X.append(seq_x)
       		y.append(seq_y)
       	return array(X), array(y)
   
    def reboot_constant(self):
        """
        Rebooting attributes that are used to make predictions (after training the model).

        Returns
        -------
        Empty attributes.(lists...)

        """
        self.X_prediction,self.Y_prediction=[],[]
        
    def normalize_model(self):
        """
        First normalization , before training the model !

        Returns
        -------
        normalized_df : dataframe normalized

        """
        df=self.set
        int_df = df[HEADER].drop(columns=['nb_entry','nb_lits']) ###Keeping usefull columns
        
        #Keeping constants of normalization#
        
        self.normalize['inf'],self.normalize['max']=int_df.min(),int_df.max()

        #Normalizing#
        
        df["nb_entry"] = pd.to_numeric(df["nb_entry"])#Changing type to int for labelisation. 
        df["nb_lits"] = pd.to_numeric(df["nb_lits"])
        normalized_df = (int_df-int_df.min())/(int_df.max()-int_df.min())
        normalized_df[['nb_lits', 'nb_entry']] = df[['nb_lits', 'nb_entry']]
        normalized_df = normalized_df.fillna(0) #Important , il arrive que des moyennes soient nulles  ! Ne pas les oublier dans la normalisation. 
        normalized_df[KNOWN_HEADER]=df[KNOWN_HEADER]
        
        return normalized_df
    
    def normalize_df(self,df_UM):
        """
        To normalize dataframes of UM after the model was trained

        Parameters
        ----------
        df_UM : dataframe of UM.

        Returns
        -------
        Normalized_df , df normalized with normalization constants.
        
        Be carfull : Sometimes constants are lower than the greatest constant of the set , in that case, the set isn't fully normalized. 
        """
        int_df = df_UM[HEADER].drop(columns=['nb_entry','nb_lits']) ###Keeping usefull columns
        
        #Normalizing#
        
        df_UM["nb_entry"] = pd.to_numeric(df_UM["nb_entry"])#Changing type to int for labelisation. 
        df_UM["nb_lits"] = pd.to_numeric(df_UM["nb_lits"])
        normalized_df = (int_df-self.normalize['inf'])/(self.normalize['max']-self.normalize['inf'])
        normalized_df[['nb_lits', 'nb_entry']] = df_UM[['nb_lits', 'nb_entry']]
        normalized_df = normalized_df.fillna(0) #Important , il arrive que des moyennes soient nulles  ! Ne pas les oublier dans la normalisation. 
        normalized_df[KNOWN_HEADER]=df_UM[KNOWN_HEADER]
        return normalized_df
    
    def classes(self):
        """
        Plot the set of all classes
        """
        
        Y_tot=np.concatenate((self.Y_train,self.Y_test,self.Y_val),axis=0)

        dataset=pd.DataFrame(Y_tot , columns=['nb_lits','nb_entry'])
        fig, ax_class = plt.subplots(1, 1)
        # we will only keep pertinent classes
        keep_classes = {}
        for i in dataset['nb_lits'].drop_duplicates().sort_values():
            keep_classes[i] = len(dataset[dataset['nb_lits'] == i])
        ax_class.plot(keep_classes.keys(), keep_classes.values(), 'ro')
    
    def heatmaps(self,in_advance):
        """
        Plot heatmaps relatively to X_train and X_test
        It concerns number of beds and entries
        """
        
        model,X_test,Y_test,X_train,Y_train=self._model[in_advance],self.X_test,self.Y_test,self.X_train,self.Y_train

        ####Number of beds###
        ##Test set##
        Y_test=Y_test.astype(int)
        y_pred_test = model(X_test).numpy().round().astype(int) #round is important 
        y_pred_test=np.clip(y_pred_test,0,None) #Removing negative values
        nb_lits=Y_test[:,0:1]
        nb_lits_pred=y_pred_test[:,0:1]
        nb_lits_pred[nb_lits_pred<0]=0
        
        #Defining the figure#
        fig, ax1 = plt.subplots(1, 1,figsize=(30,20))
        fig.suptitle('Heatmaps for number of bed , test_set')
        
        #Heatmaps and classification report #
        g1 = sns.heatmap(confusion_matrix(nb_lits_pred, nb_lits), annot=False, ax=ax1,cmap="YlGnBu",linewidths=.5)
        g1.set_ylabel('y_test')
        g1.set_xlabel('y_pred')
        print("Report for test of number of beds : \n",
              classification_report(nb_lits_pred, nb_lits))
        ##Train set##
        y_pred_train = model(X_train).numpy().round().astype(int)
        y_pred_train=np.clip(y_pred_train,0,None) #Removing negative values

        nb_lits=Y_train[:,0:1]
        nb_lits_pred=y_pred_train[:,0:1]
        nb_lits_pred[nb_lits_pred<0]=0
        
        #Defining the figure#

        fig, ax2 = plt.subplots(1, 1,figsize=(30,20))
        fig.suptitle('Heatmaps for number of bed , training_set')
        
        #Heatmaps and classification report #

        g2 = sns.heatmap(confusion_matrix(nb_lits_pred, nb_lits), annot=False, ax=ax2,cmap="YlGnBu",linewidths=.5)
        g2.set_ylabel('y_train')
        g2.set_xlabel('y_pred')
        print("Report for train of number of beds : \n",
              classification_report(nb_lits_pred, nb_lits))
        
        ####Numbers of entries###
        
        ##Test set##

        nb_entry=Y_test[:,1:2]
        nb_entry_pred=y_pred_test[:,1:2]  
        nb_entry_pred[nb_entry_pred<0]=0
        
        #Defining the figure#

        fig, ax1 = plt.subplots(1, 1,figsize=(30,20))
        fig.suptitle('Heatmaps for number of entry, test_set')
        
        #Heatmaps and classification report #

        g1 = sns.heatmap(confusion_matrix(nb_entry_pred, nb_entry), annot=False, ax=ax1,cmap="YlGnBu",linewidths=.5)
        g1.set_ylabel('y_test')
        g1.set_xlabel('y_pred')
        print("Report for test of number of entries : \n",
              classification_report(nb_entry_pred, nb_entry))
        
        ##Train set##

        nb_entry=Y_train[:,1:2]
        nb_entry_pred=y_pred_train[:,1:2]
        nb_entry_pred[nb_entry_pred<0]=0
        
        #Defining the figure#

        fig, ax2 = plt.subplots(1, 1,figsize=(30,20))
        fig.suptitle('Heatmaps for number of entries , training_set')
        
        #Heatmaps and classification report #

        g2 = sns.heatmap(confusion_matrix(nb_entry_pred, nb_entry), annot=False, ax=ax2,cmap="YlGnBu",linewidths=.5)
        g2.set_ylabel('y_train')
        g2.set_xlabel('y_pred')
        print("Report for train of number of entries : \n",
              classification_report(nb_lits_pred, nb_lits))
    
    def performance(self,in_advance):
        
        """
        plot curves concerning the performance/history of the models which are : 
            -loss
            -mean_absolute_percentage_error
            -mean_absolute_error
            -val_loss
            -val_mean_absolute_percentage_error
            -val_mean_absolute_error
        
        """
        
        history=self.history[in_advance] #All the data were stored in  history
        
        ###Creating 3 figures which fit each of the parameters to plot###
        
        fig2, (ax3,ax4,ax5) = plt.subplots(1, 3,figsize=(20,10))
        fig2.suptitle("Overview of the performances")
        
        ##Plotting##
        
        ax5.plot(history.history['loss'], label='loss')
        ax4.plot(history.history['mean_absolute_percentage_error'], label='mean_absolute_percentage_error')
        ax3.plot(history.history['mean_absolute_error'], label='mean_absolute_error')
        ax5.plot(history.history['val_loss'], label='val_loss')
        ax4.plot(history.history['val_mean_absolute_percentage_error'], label='val_m_abs_%')
        ax3.plot(history.history['val_mean_absolute_error'], label='val_m_abs')
        
        ##Adding legend##
        
        ax3.legend()
        ax4.legend()
        ax5.legend()
    

 

    def prediction(self,in_advance,UM,test:bool,stop_date:str,take_all:bool): #Be careful to not shuffle those data, order must be preserved.
        """
        Help to plot the curve of prediction given an UM. 
        
        If the UM asked isn't in the test set, you get a warning.
        
        in_advance : Int which gives the number of day to forecast. 
        
        return : df_temp the dataframe to plot 
        """
        
        n_steps_in=self.params['T']
        n_steps_out=in_advance
        
        if test == True :
            print('Entraînement du modèle avec une date butoire, les performances ne seront pas enregistrées')
            #On réentraîne le model selon ces nouvelles conditions
            if take_all == False :
                self.train(in_advance,UM_train=[UM],stop_date=stop_date)
            else :
                self.train(in_advance,stop_date=stop_date)

        else :
            if in_advance not in self._model.keys():
                print("Aucun modèle n'existe pour ce nombre de jour , création et entraînement du modèle sans date butoire, les performances seront enregistrées ... \n")
            if take_all == True : 
                    self.train(in_advance)
            else : 
                    self.train(in_advance,UM_train=[UM])

               
            
        model=self._model[in_advance]
        
       
        ###Creating vectors and dataframes we need ###
        
        #Selection de l'UM en question #
        
        df_UM=self.data.UM_dataframe(UM)
        
        df_UM=self.normalize_df(df_UM)

        self.build_set(in_advance=in_advance,df=df_UM)
        #Normalization#
        
       
        df_temp=df_UM[['nb_lits','day','month','UM']].copy()    
        df_temp=df_temp.iloc[n_steps_in-1:] #We need to avoid the T-1 first dates
        df_pred=pd.DataFrame(columns=['nb_lits_pred'],index=df_temp.index) #df_pred stores the predictions concerning df_temp
        df_temp['Début de prédiction']=-1 #Initialize values with-1

        ###Filling df_temp and df_pred###
        #Selecting right predictible columns#
        columns=df_UM[df_UM['UM']==UM].drop(columns=STRING_HEADER).columns
        
        
        day=0
        with tqdm(total=len(self.Y),desc="Making predictions") as pbar:
            
            ##Running through each row of the UM##
            while day + n_steps_out < len(self.Y): 
                
                current_day=df_pred.index[day]
                until_day=df_pred.index[day+n_steps_out-1]
                X_forcast= np.array([self.X[day]])
                y_pred_test = model(X_forcast).numpy().round().astype(int) #round is important 
                y_pred_test=np.clip(y_pred_test,0,None) #Removing negative values

                #Making projection#
                df_pred.loc[current_day:until_day]=y_pred_test.T
                df_temp['Début de prédiction'].loc[current_day]=y_pred_test[0][0]
                day+=n_steps_out
                pbar.update(day) #Updating the progression bar
                
        ##Merging with the truth##
        df_temp=df_temp.merge(df_pred,how='left',left_index=True, right_index=True)
        df_temp=df_temp.dropna() #Dropping nan values in case T isn't a multiple of len(df_temp)
        
        #Put nan values for dates which aren't checkpoints#
        temp=df_temp['Début de prédiction'].replace(-1,np.nan)
        df_temp['Début de prédiction']=temp
        
        #Setting date in a columns for plotting#
        df_temp.reset_index(drop=False, inplace=True) #add a column date
        df_temp.date=pd.to_datetime(df_temp.date, format='%Y-%m-%d')
        df_temp.index = df_temp.date
        df_temp.index=df_temp.index.strftime('%Y-%m-%d')  
        
        ##Saving performances only when test_mode is off otherwise it doesn't have any sense##
        if test == False :
            line_to_append_lit={}
            line_to_append_lit['projection']=in_advance
            line_to_append_lit['UM']=UM
            line_to_append_lit['taux_perte']={}
            #About months#
            
            for columns in MONTHS:
                line_to_append_lit[columns],line_to_append_lit['taux_perte'][columns]=percent_error(df_temp['nb_lits'][df_temp['month']==columns].to_numpy(),df_temp['nb_lits_pred'][df_temp['month']==columns].to_numpy())
    
            #About days #
            
            for columns in DAYS:
                line_to_append_lit[columns],line_to_append_lit['taux_perte'][columns]=percent_error(df_temp['nb_lits'][df_temp['day']==columns].to_numpy(),df_temp['nb_lits_pred'][df_temp['day']==columns].to_numpy())
            
            #Adding to the dataframe of performances#
            self.df_prediction_lit=self.df_prediction_lit.append(line_to_append_lit, ignore_index=True)
            self.df_prediction_lit=self.df_prediction_lit.reset_index().drop(columns='index')
            self.df_prediction_lit['average_error']= self.df_prediction_lit[MONTHS+DAYS].mean(axis=1)
            
            
            ##Saving in a dictionnary##
            if in_advance in self.dict_df_temp.keys(): #Adding to a dictionnary in order to keep predictions 
                self.dict_df_temp[in_advance]=pd.concat([self.dict_df_temp[in_advance],df_temp]) #Concatenation des deux df_temp pour une projection du même nombre de jour
            else:
                self.dict_df_temp[in_advance]=df_temp
        
        return df_temp
            
    
    def plot_prediction(self,UM,in_advance,month=None,day=None,from_date:str=None,to_date:str=None,test:bool=False,take_all:bool=False):
        """
        Plot the prediction of a requested UM  and compare it to reality.
        The tuple( from_date , to_date) need to be in the database.
        
        If the dataframe of prediction wasn't build for the query requested , it creates it.
        -test : Whether , those data have to be taken into account by the model during training
        -UM : A string of the UM (ex : '0036')
        -in_advance : A int corresponding to the range of forcasting
        -month , day are string which will display the months and days requested
        -from_date and to_date are str shaped like : '  2018-11-14 '... , it will display the projection between this dates
        -take_all : Do we have to consider all UM's to train the model ?
        Return a plot
        
        """
        ##Verifying that the query does exist##
        
        if test == False:
            if in_advance not in self.dict_df_temp.keys(): #On effectue toujours une prédiction lorsque test est vrai sinon trop galère
                print("L'UM %s n'a pas été analysé , on procède tout d'abord à l'analyse." %UM)
                self.prediction(in_advance,UM,test,from_date,take_all) 
            
            df_temp=self.dict_df_temp[in_advance]
            
            if len(df_temp[df_temp['UM']==UM])==0 :
                print("L'UM %s n'a pas été analysé , on procède tout d'abord à l'analyse." %UM)
                self.prediction(in_advance,UM,test,from_date,take_all) 
                df_temp=self.dict_df_temp[in_advance]
        
        else : 
            df_temp=self.prediction(in_advance,UM,test,from_date,take_all) 

        
        ###Plotting ###
        
        ##Selecting the correct UM##
        
        df_temp=df_temp[df_temp['UM']==UM]
        
        if from_date==None:
            from_date=df_temp.index[0]
        if to_date==None : 
            to_date=df_temp.index[-1]
            
        #Taking the values of the previous year#
     #   to_date_minus_year=dt.datetime.strptime(str(to_date),'%Y-%m-%d')#Converting to_date to datetime  type.
      #  to_date_minus_year=to_date_minus_year.replace(year = to_date_minus_year.year -2)
                
        from_date_minus_year=dt.datetime.strptime(str(from_date),'%Y-%m-%d')#Converting to_date to datetime  type.
        from_date_minus_year=from_date_minus_year.replace(year = from_date_minus_year.year -2)
        
       # delta = to_date_minus_year-from_date_minus_year
 #       nb_days=delta.days+1
        from_date_minus_year=dt.datetime.strftime(from_date_minus_year,'%Y-%m-%d') #converting to str
        

        df_previous_year=df_temp.loc[from_date_minus_year:]
        
        df_temp=df_temp.loc[from_date:to_date] #To define the scale of study (from_date -> to_date)
        nb_days=len(df_temp.index)

        #Recalibrage selon les même noms de jours#
        L_days=list(df_previous_year['day'])
        same_day=df_temp.iloc[0]['day']
        index_same_day=L_days.index(same_day)
        df_previous_year=df_previous_year.iloc[index_same_day:index_same_day+nb_days]
         
        #Récupération du nombre de lits#
        
        df_previous_year=df_previous_year['nb_lits']
        df_temp['annee_precedente']=df_previous_year.to_numpy()
        
        #Calculating error#
        
        y_true=df_temp['nb_lits'].to_numpy()
        y_pred=df_temp['nb_lits_pred'].to_numpy()
        
        y_true_previous_year=df_temp['annee_precedente'].to_numpy()
        
        error,ratio=percent_error(y_true,y_pred)
        MAE=mean_absolute_error(y_true,y_pred)
        error_previous_year,ratio_previous_year=percent_error(y_true,y_true_previous_year)
        MAE_previous_year=mean_absolute_error(y_true,y_true_previous_year)
        
        #Renaming columns for labeling on the plot#
        column_previous_year='annee_precedente | ER: %d '%error_previous_year+'%'+ ' privé de %d'%ratio_previous_year+'%'+' du set | MAE :%d lits'%MAE_previous_year
        column_lits_pred= 'nb_lits_pred | ER: %d' %error+'%'+ ' privé de %d'%ratio+'%'+' du set | MAE :%d lits'%MAE
        df_temp.rename(columns = {'annee_precedente' : column_previous_year , 'nb_lits_pred' : column_lits_pred}, inplace = True)
        fig, ax = plt.subplots(1, 1,figsize=(20,10))
      
        UM=df_temp['UM'].iloc[0]
        if month!=None :
            df_temp[df_temp['month']==month].plot(kind='line',x='date',y='nb_lits',ax=ax)
            df_temp[df_temp['month']==month].plot(kind='line',x='date',y=column_lits_pred, color='red',ax=ax)
            df_temp[df_temp['month']==month].plot(kind='line',x='date',y=column_previous_year, color='green',ax=ax)
            df_temp[df_temp['month']==month].plot.scatter(x='date',y='Début de prédiction', s=80, c='purple',ax=ax, label="Début de la prédiction")
        elif day != None :
            df_temp[df_temp['day']==day].plot(kind='line',x='date',y='nb_lits',ax=ax)
            df_temp[df_temp['day']==day].plot(kind='line',x='date',y=column_lits_pred, color='red',ax=ax)
            df_temp[df_temp['day']==day].plot(kind='line',x='date',y=column_previous_year, color='green',ax=ax)
            df_temp[df_temp['day']==day].plot.scatter(x='date',y='Début de prédiction', s=80, c='purple',ax=ax, label="Début de la prédiction")
        else:
            df_temp.plot(kind='line',x='date',y='nb_lits',ax=ax)
            df_temp.plot(kind='line',x='date',y=column_lits_pred, color='red',ax=ax)
            df_temp.plot(kind='line',x='date',y=column_previous_year, color='green',ax=ax)
            df_temp.plot.scatter(x='date',y='Début de prédiction', s=80, c='purple',ax=ax, label="Début de la prédiction")
        
        ax.set_ylabel("Nombre de lit")
        fig.suptitle("Evolution de la prediction des lits avec une avance de %d pour l'UM %s"%(in_advance,UM))
     
        
        print('Une erreur de %d' %error+'%'+'est commise en excluant %d'%ratio +'%'+ 'du set')
        print('Une erreur de %d' %error_previous_year+'%'+'est commise en excluant %d'%ratio +'%'+ 'du set en prenant l année précédente')
        return error,error_previous_year

    
    def plot_hist_pred(self,in_advance,UM=None):
        
        """
        After having generated predictions

        Parameters
        ----------
        projection : int about the number of days forcasted

        Returns
        -------
        A histogram of percent_error concerning all UM for a particular projection

        """
       
            
        df_prediction_lit=self.df_prediction_lit[self.df_prediction_lit['projection']==in_advance]
        if UM != None :
            df_prediction_lit=df_prediction_lit[df_prediction_lit['UM']==UM]

        df_hist_tot=pd.DataFrame(columns=['calendar','error_lit','error_entry'])
        for columns in MONTHS+DAYS:
            df_hist=pd.DataFrame(columns=['calendar','error_lit','error_entry'])
            list_hist_lit=list(df_prediction_lit[columns])

            df_hist['error_lit']=list_hist_lit

            df_hist['calendar'] = columns
            df_hist_tot=pd.concat([df_hist_tot,df_hist])
        
        df_hist_tot=df_hist_tot.reset_index()
        
        ###Separating months and days###
        
        fig,(ax_day_lit,ax_month_lit)=plt.subplots(2,1,figsize=(20,10))

        ##months##
        
        #Lits#
        filter=df_hist_tot['calendar'].isin(MONTHS)
        filter=filter[filter==True]
        sns.barplot(
            ax=ax_month_lit,
            y="error_lit", 
            x="calendar", 
            data=df_hist_tot.loc[filter.index], 
            estimator=mean, 
            ci=None, 
            color='#69b3a2')
        ax_month_lit.set_title("Erreur selon les mois pour les lits")
        
        ##days##
        
        #Lits#
        
        filter=df_hist_tot['calendar'].isin(DAYS)
        filter=filter[filter==True]
        sns.barplot(
            ax=ax_day_lit,
            y="error_lit", 
            x="calendar", 
            data=df_hist_tot.loc[filter.index], 
            estimator=mean, 
            ci=None, 
            color='#69b3a2')
        ax_day_lit.set_title("Erreur selon les jours pour les lits")
        
        ##Adding titles##
        plt.tight_layout()
        if UM!= None :
            fig.suptitle("Erreur moyenne absolue pour une projection de %d jours conçernant l'UM %s "% (in_advance,UM))
        else:
            fig.suptitle("Erreur moyenne absolue pour une projection de %d jours conçernant tout les UM  ")
        return fig

        
        
    def save_prediction(self):
        ##After having making predictions stored in dict_df_temp##
        ##Creating directories for each projection ##
        
        # Parent Directory path
        parent_dir = os.getcwd()
    

        for projection in self.dict_df_temp.keys():
            folder="projection_%d" %projection
            # Path
            path= os.path.join(parent_dir, folder)
            if os.path.exists(path):
            # removing the file using the os.remove() method
                shutil.rmtree(path)
                
            os.mkdir(path)
            print("Directory '% s' created" % folder)
            
            #Creating the folder prediction#
            
            folder="prediction"
            path2= os.path.join(path, folder)
            if os.path.exists(path2):
            # removing the file using the os.remove() method
                shutil.rmtree(path2)
            
            os.mkdir(path2)
            
            #Creating the folder histogramme#
            folder="histogramme"
            path3= os.path.join(path, folder)
            if os.path.exists(path3):
            # removing the file using the os.remove() method
                shutil.rmtree(path3)
            
            os.mkdir(path3)
            
            ##Selecting each UM computed for this prediction and creating its directory##
            
            df_prediction_lit=self.df_prediction_lit
            df_prediction_lit=df_prediction_lit.sort_values(by='average_error')
            list_UM=list(df_prediction_lit['UM'][df_prediction_lit['projection']==projection].unique()) #List of UM which were evaluated
            for i in tqdm(range(0,len(list_UM)),desc = 'Affichage des courbes pour chaque UM '):
                UM=list_UM[i]
                df_UM=self.data.UM_dataframe(UM)
                fig_hist=self.plot_hist_pred(projection,UM)
                fig_hist.savefig(path2 + "\prediction_%s_%d"%(UM,i)) #%d position
                fig_UM=hist_plot(df=self.normalize_df(df_UM),UM=UM)
                fig_UM.savefig(path3 + "\histogramme_%s_%d"%(UM,i))
            
            ##Plotting performance depending on the carateristics##
            
            print("Affichage des performances pour la projection %d"%projection)
            fig,fig_var=self.plot_prediction_performance(projection)
            fig.savefig(path+"\moyennes_erreurs")
            fig_var.savefig(path+"variances_erreurs")

    #Checkpoint#
    def plot_prediction_performance(self,projection):
        """
        
        Parameters
        ----------
        projection :  int about the number of days forcasted


        Returns
        -------
        
        Curves about error depending on the following features :
            ...
        """
        
        df_data_var=pd.DataFrame() #dataframe with all the variances values for each UM in df_prediction
        df_data_mean=pd.DataFrame() #dataframe with all the mean values for each UM in df_prediction
        df_prediction_lit=self.df_prediction_lit
        list_UM=list(df_prediction_lit['UM'][df_prediction_lit['projection']==projection].unique()) #List of UM which were evaluated
        for i in range (0,len(list_UM)):
            
            
            UM=list_UM[i]
            df_UM=self.data.UM_dataframe(UM)
            df_UM=self.normalize_df(df_UM)
            #Mean#
            line_to_append_mean=df_UM.mean()
            line_to_append_mean['UM']=UM
            df_data_mean=df_data_mean.append(line_to_append_mean,ignore_index=True)
            #Var#
            line_to_append_var=df_UM.var()
            line_to_append_var['UM']=UM
            df_data_var=df_data_var.append(line_to_append_var,ignore_index=True)
            
        fig,((ax_little_lit,ax_great_lit),(ax_little_entry,ax_great_entry))=plt.subplots(2,2,figsize=(20,10))

        discrete_features=['special_day', 'vac_a', 'vac_b', 'vac_c',
        'lun', 'mar', 'mer', 'jeu', 'ven', 'sam', 'dim', 'janv', 'fev', 'mars',
        'avr', 'mai', 'juin', 'juil', 'aout', 'sept', 'oct', 'nov', 'dec','day','month']
        continuous_features=list(set(df_UM.columns)-set(discrete_features))
        
        ###Mean###

        df_data_mean=df_data_mean[continuous_features]
        df_data_mean=df_data_mean.merge(df_prediction_lit[['average_error','UM']], how='left', on='UM')#Merging with UM's
        df_data_mean.rename(columns = {'average_error':'average_error_lit'}, inplace = True)
        df_data_mean=df_data_mean.merge(self.df_prediction_entry[['average_error','UM']], how='left', on='UM')
        df_data_mean.rename(columns = {'average_error':'average_error_entry'}, inplace = True)
        df_data_mean=df_data_mean.drop(columns='UM')
        
        ##lits##
        #Caractéristiques normalisées#
        
        little_filter=df_data_mean.columns.difference(['average_error_lit','average_error_entry','nb_lits','nb_entry'])
        df_data_mean.plot(x='average_error_lit',y=little_filter,style='-o',ax=ax_little_lit)
        ax_little_lit.set_title("moyenne des caratéristiques / erreur sur le nombre de lit")
        ax_little_lit.yaxis.set_label_position("right")
        ax_little_lit.yaxis.tick_right()
        
        #Nombre de lits#
        
        df_data_mean.plot(x='average_error_lit',y='nb_lits',style='-o',ax=ax_great_lit)
        ax_great_lit.set_title("moyenne du nombre de lits/ erreur sur le nombre de lit")
        ax_great_lit.yaxis.set_label_position("right")
        ax_great_lit.yaxis.tick_right()
        
        ##Entry##
        #Caractéristiques normalisées#
        
        df_data_mean.plot(x='average_error_entry',y=little_filter,style='-o',ax=ax_little_entry)
        ax_little_entry.set_title("moyenne des caratéristiques / erreur sur le nombre de d'entrées")
        ax_little_entry.yaxis.set_label_position("right")
        ax_little_entry.yaxis.tick_right()
        
        #Nombre d'entrée#
        
        df_data_mean.plot(x='average_error_entry',y='nb_entry',style='-o',ax=ax_great_entry)
        ax_great_lit.set_title("moyenne du nombre d'entrée/ erreur sur le nombre d'entrée")
        ax_great_lit.yaxis.set_label_position("right")
        ax_great_lit.yaxis.tick_right()
        
        fig.suptitle("Evolutions des moyennes des caractéristiques de chaque unité /erreur ")
        
        ###Variance###
        
        fig_var,((ax_little_lit_var,ax_great_lit_var),(ax_little_entry_var,ax_great_entry_var))=plt.subplots(2,2,figsize=(20,10))
        
        df_data_var=df_data_var[continuous_features]
        df_data_var=df_data_var.merge(df_prediction_lit[['average_error','UM']], how='left', on='UM')#Merging with UM's
        df_data_var.rename(columns = {'average_error':'average_error_lit'}, inplace = True)
        df_data_var=df_data_var.merge(self.df_prediction_entry[['average_error','UM']], how='left', on='UM')
        df_data_var.rename(columns = {'average_error':'average_error_entry'}, inplace = True)
        df_data_var=df_data_var.drop(columns='UM')
        
        ##lits##
        #Caractéristiques normalisées#
        
        little_filter=df_data_var.columns.difference(['average_error_lit','average_error_entry','nb_lits','nb_entry'])
        df_data_var.plot(x='average_error_lit',y=little_filter,style='-o',ax=ax_little_lit_var)
        ax_little_lit_var.set_title("moyenne des caratéristiques / erreur sur le nombre de lit")
        ax_little_lit_var.yaxis.set_label_position("right")
        ax_little_lit_var.yaxis.tick_right()
        
        #Nombre de lits#
        
        df_data_var.plot(x='average_error_lit',y='nb_lits',style='-o',ax=ax_great_lit_var)
        ax_great_lit_var.set_title("moyenne du nombre de lits/ erreur sur le nombre de lit")
        ax_great_lit_var.yaxis.set_label_position("right")
        ax_great_lit_var.yaxis.tick_right()
        
        ##Entry##
        #Caractéristiques normalisées#
        
        df_data_var.plot(x='average_error_entry',y=little_filter,style='-o',ax=ax_little_entry_var)
        ax_little_entry_var.set_title("moyenne des caratéristiques / erreur sur le nombre de d'entrées")
        ax_little_entry_var.yaxis.set_label_position("right")
        ax_little_entry_var.yaxis.tick_right()
        
        #Nombre d'entrée#
        
        df_data_var.plot(x='average_error_entry',y='nb_entry',style='-o',ax=ax_great_entry_var)
        ax_great_lit_var.set_title("moyenne du nombre d'entrée/ erreur sur le nombre d'entrée")
        ax_great_lit_var.yaxis.set_label_position("right")
        ax_great_lit_var.yaxis.tick_right()
        
        fig_var.suptitle("Evolutions des variances des caractéristiques de chaque unité /erreur ")
        
        return fig,fig_var
    def evaluate(self, X_test, Y_test):
        try:
            ev = self._model.evaluate(X_test, Y_test) #[loss,mean_percentage_error,mean_error]
            logging.info(f'Eval result : {ev}')
            plt.plot(self.history[in_advance].history['loss'], label='train')
            plt.plot(self.history[in_advance].history['val_loss'], label='test')
            plt.legend()
            plt.show()
            return ev
        except:

            logging.error('Cannot evaluate model')
          
if __name__ == "__main__": 
   model = HospitalPredictor()
   in_advance=5
   model.plot_prediction('0042',3,from_date='2021-01-01',to_date='2021-02-30',test=True,take_all = False)
  