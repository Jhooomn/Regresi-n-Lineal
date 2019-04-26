# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 09:40:10 2019
@author: Jhon Baron
"""
import pandas as pd
import matplotlib.pyplot as plt
import math

data = pd.read_csv('movies.csv', header=0);
#print(data)
df = pd.DataFrame(data);
#print(df)
score = df['imdb_score'];
likes= df['movie_facebook_likes'];

#print(score) #x
#print(likes) #y

n=5043; #No. datos/filas
sumx = sum(score);
sumy = sum(likes);
sumx2 = sum(score*score);
sumy2 = sum(likes*likes);
sumxy = sum(score*likes);
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

plt.plot(score,likes,'o', label = 'Datos');
plt.plot(score, m*score + b, label = 'Ajuste');
plt.ylabel('Likes');
plt.xlabel('Score');
plt.title('Regresion lineal');
plt.grid();
plt.legend();
