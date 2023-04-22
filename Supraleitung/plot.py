import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
import json
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties.unumpy import uarray
from uncertainties import unumpy as unp
from uncertainties.unumpy import (nominal_values as noms,std_devs as stds)
from scipy.stats import sem

def abw(exact,approx):
    return (exact-approx)*100/exact  #Abweichnung



#####
#Temperaturmessgerät von R in T (-200 bis 0 Celsius):
A = 3.9083e-3 #°C^-1
B = -5.7750e-7 #°C^-2
C = -4.1830e-12 #°C^-4

def R_func(T):
    return (1000*(1+A*T+B*T**2+C*(T-100)*T**3))



##Kalibrationskurve
#T_lin = np.linspace(-200,0)
#plt.plot(T_lin,R_func(T_lin), label='Kalibrationskurve')
#plt.grid()
#plt.legend()
#plt.xlabel(r'$T \mathbin{/} \si{\celsius}$')
#plt.ylabel(r'$R \mathbin{/} \si{\ohm}$')
#plt.savefig('build/Kalibrationskurve.pdf')
#plt.clf()





########
#ii)
#Bestimmung der kritischen Temperatur durch Meißner-Ochsenfeld-Effekt
#Zeit, Spannung, Widerstand, Strom
#Zeit wird jeweils auf 0 gesetzt
#Messwert, bei dem der Magnet fällt: kritX
#Messung 1
t1, V1 ,  R1 , I1 = np.genfromtxt('2023/Krit_Meiß/M1.txt', delimiter=',', unpack=True)
t1 = t1-np.min(t1)
krit1 = 51
#Messung 2
t2, V2 ,  R2 , I2 = np.genfromtxt('2023/Krit_Meiß/M2.txt', delimiter=',', unpack=True)
t2 = t2-np.min(t2)
krit2 = 38
#Messung 3
t3, V3 ,  R3 , I3 = np.genfromtxt('2023/Krit_Meiß/M3.txt', delimiter=',', unpack=True)
t3 = t3-np.min(t3)
krit3 = 66

####
#plots der Messungen
#M1
plt.plot(t1,R1,'o',label='Messung 1')
plt.plot(t1[krit1],R1[krit1], 'kx' ,label='Magnet fällt')
plt.grid()
plt.xlabel(r'$t \mathbin{/} \si{\s}$')
plt.ylabel(r'$R \mathbin{/} \si{\ohm}$')
plt.grid()
plt.legend()
plt.savefig('build/M_Krit1.pdf')
plt.clf()

#M2
plt.plot(t2,R2,'o',label='Messung 2')
plt.plot(t2[krit2],R2[krit2], 'kx' ,label='Magnet fällt')
plt.grid()
plt.xlabel(r'$t \mathbin{/} \si{\s}$')
plt.ylabel(r'$R \mathbin{/} \si{\ohm}$')
plt.grid()
plt.legend()
plt.savefig('build/M_Krit2.pdf')
plt.clf()

#M3
plt.plot(t3,R3,'o',label='Messung 3')
plt.plot(t3[krit3],R3[krit3], 'kx' ,label='Magnet fällt')
plt.grid()
plt.xlabel(r'$t \mathbin{/} \si{\s}$')
plt.ylabel(r'$R \mathbin{/} \si{\ohm}$')
plt.grid()
plt.legend()
plt.savefig('build/M_Krit3.pdf')
plt.clf()

##Curvefit
#def BeispielFunktion(x,a,b):
#    return a*x+b 
#params, cov = curve_fit(BeispielFunktion, x-Werte, y-Werte,sigma=fehler_der_y_werte,p0=[schätzwert_1,#schätzwert_2])
#a = ufloat(params[0],np.absolute(cov[0][0])**0.5)
#b = ufloat(params[1],np.absolute(cov[1][1])**0.5)
#
#
##Json
#Ergebnisse = json.load(open('data/Ergebnisse.json','r'))
#if not 'Name' in Ergebnisse:
#    Ergebnisse['Name'] = {}
#Ergebnisse['Name']['Name des Wertes']=Wert
#
#json.dump(Ergebnisse,open('data/Ergebnisse.json','w'),indent=4)
