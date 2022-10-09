# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 22:20:18 2022

@author: anton
"""
PATH = 'C:\\Users\\anton\\Documents\\Helean\\database\\RSA_1.csv'
import os
os. chdir('C:\\Users\\anton\\Documents\\Helean\\Hospital-AI')  #Changing current working directory to permit the imports
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from data.data_handler import HospitalDataHandler
from tqdm import tqdm
#board=HospitalDataHandler()

def boxplot(board=None,UM=None,df=None):
    """
    Boxplotting of several features concerning the dataframe, it tries to adapt to scales of values. 

    Parameters
    ----------
    board : HospitalDataHandler

    Returns
    -------
    Boxplots.

    """
    if len(df) == 0 :
        df=board.UM_dataframe(UM)
    discrete_features=['special_day', 'vac_a', 'vac_b', 'vac_c',
    'lun', 'mar', 'mer', 'jeu', 'ven', 'sam', 'dim', 'janv', 'fev', 'mars',
    'avr', 'mai', 'juin', 'juil', 'aout', 'sept', 'oct', 'nov', 'dec','day','month','UM']
    continuous_features=list(set(df.columns)-set(discrete_features))
    continuous_features_0=[i for i in continuous_features if df[i].max()<=1]
    
    ##Plotting boxplots of severals features depending on their range of application ###
    if (len(continuous_features_0) == len(df.columns)-len(discrete_features)-2):
        fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(20,10))
        continuous_features_1=[i for i in continuous_features if 1 <df[i].max()]
        to_plot=[(ax1,continuous_features_0),(ax2,continuous_features_1)]
    else:  
        continuous_features_1=[i for i in continuous_features if 1 <df[i].max()<10]
        continuous_features_2=[i for i in continuous_features if 10<= df[i].max()<100]
        continuous_features_3=[i for i in continuous_features if df[i].max()>=100]
    ##Plotting boxplots of severals features depending on their range of application ###
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,figsize=(20,10))
        to_plot=[(ax1,continuous_features_1),(ax2,continuous_features_2),(ax3,continuous_features_3),(ax4,continuous_features_0)]
    i=1
    for plot in to_plot:
        ax=plot[0]
        cont_features=plot[1]
        fig.canvas.set_window_title('A Boxplot Example')
        fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)
    
        bp = sns.boxplot(ax=ax,data=df[cont_features]) 
    
        # Add a horizontal grid to the plot, but make it very light in color
        # so we can use it for reading data values but not be distracting
        ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                       alpha=0.5)
    
        # Hide these grid behind plot objects
        ax.set_axisbelow(True)
        ax.set_title('Comparison Resampling Distributions with values between %s and %s'%(i,10*i))
        ax.set_xlabel('Distribution')
        ax.set_ylabel('Value')
        i=10*i
    plt.tight_layout()
    fig.suptitle("Boxplots des caratéristiques du tableur pour l'UM %s "%UM)

#####Violin plot###
def violin_plot(columns,board=None,UM=None,df=None):
    
  
    fig,axs=plt.subplots(1,2,figsize=(30, 15))
    if len(df) == 0 :
        df=board.UM_dataframe(UM)
        fig.suptitle("Violinplot en fonction des jours et des mois pour l'UM %s"%UM)
    else : fig.suptitle("Violinplot en fonction des jours et des mois pour le set d'entrée")

    day_month=['day','month']

    i=0
    for ax in axs:
        #sns.violinplot(ax=ax,x=continuous_features_0[i],y='vac_b',data=df)
        #sns.violinplot(ax=ax,x=day_month[i],y='nb_entry',hue='vac_b',data=df)
        sns.violinplot(ax=ax,x=day_month[i],y=columns,data=df,hue='vac_b',split=True)
        i+=1
    plt.tight_layout()

def hist_plot(board=None,UM=None,df=pd.DataFrame()):
    
    if df.empty == True :
        df=board.UM_dataframe(UM)
    discrete_features=['special_day', 'vac_a', 'vac_b', 'vac_c',
    'lun', 'mar', 'mer', 'jeu', 'ven', 'sam', 'dim', 'janv', 'fev', 'mars',
    'avr', 'mai', 'juin', 'juil', 'aout', 'sept', 'oct', 'nov', 'dec','day','month','UM']
    continuous_features=list(set(df.columns)-set(discrete_features))
    continuous_features_0=[i for i in continuous_features if df[i].max()<=1]
    
     ##Plotting boxplots of severals features depending on their range of application ###
    if (len(continuous_features_0) == len(df.columns)-len(discrete_features)-2):
        fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(20,10))
        continuous_features_1=[i for i in continuous_features if 1 <df[i].max()]
        to_plot=[(ax1,continuous_features_0),(ax2,continuous_features_1)]
        features=[continuous_features_0,continuous_features_1]
    
    else:  
        continuous_features_1=[i for i in continuous_features if 1 <df[i].max()<10]
        continuous_features_2=[i for i in continuous_features if 10<= df[i].max()<100]
        continuous_features_3=[i for i in continuous_features if df[i].max()>=100]
        features=[continuous_features_0,continuous_features_1,continuous_features_2,continuous_features_3]
    outer = gridspec.GridSpec(3, 2)
    fig = plt.figure(figsize=(20, 15))
    
    for i in range(len(features)):
        if len(features[i])>=1:
            inner = gridspec.GridSpecFromSubplotSpec(1, len(features[i]),subplot_spec=outer[i], wspace=0.1, hspace=0.1)
            for j in range(len(features[i])):
                    axs = plt.Subplot(fig, inner[j])
                    sns.histplot(ax=axs,x=features[i][j],data=df,kde=True)
                    fig.add_subplot(axs)
        
    outer.tight_layout(fig)   
    
    
    fig.suptitle("Histogramme des caratéristiques du tableur pour l'UM %s "%UM)
    
    return fig
