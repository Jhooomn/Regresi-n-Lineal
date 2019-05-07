# -*- coding: utf-8 -*-
"""
Created on Mon May  6 18:59:56 2019

@author: Jhon Baron
"""
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing

data = pd.read_csv('movies.csv', header=0);
#print(data)
df = pd.DataFrame(data);
#print(df)

scaler = preprocessing.Normalizer(norm='l2', copy=True)
df[['imdb_score','movie_facebook_likes']]=scaler.fit_transform(df[['imdb_score','movie_facebook_likes']])

def linear_reg(x,y,n):
    sumx = sum(x);
    sumy = sum(y);
    sumx2 = sum(x*x);
    sumy2 = sum(y*y);
    sumxy = sum(x*y);
    promx = sumx/n; #Promedio de score
    promy = sumy/n; #Promedio de likes
    
    m = (sumx*sumy - n*sumxy)/(sumx**2 - n*sumx2);#Pendiente
    b = promy - m*promx;#Intercepto
    
    sigmax = math.sqrt(sumx2/n - promx**2); 
    sigmay = math.sqrt(sumy2/n - promy**2);
    sigmaxy = sumxy/n - promx*promy;
    
    R2 = (sigmaxy/(sigmax*sigmay))**2; #Coeficiente de correlacion
    
    print('Coeficiente de correlacion cuadrado:', R2);
    print('Pendiente: ', m);
    print('Intercepto: ', b);
    
    plt.plot(x,y,'o', label = 'Datos');
    plt.plot(x, m*x + b, label = 'Ajuste');
    plt.ylabel('X Values');
    plt.xlabel('Y Values');
    plt.title('Linear Regresion');
    plt.grid();
    plt.legend();

score = df['imdb_score'];
likes= df['movie_facebook_likes'];
n=5043; 
#No. datos/filas
#print(score) #x
#print(likes) #y
linear_reg(score, likes, n); #Implementando la funcion

