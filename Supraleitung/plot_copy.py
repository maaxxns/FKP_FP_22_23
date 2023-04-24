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
from scipy.constants import mu_0

def abw(exact,approx):
    return (exact-approx)*100/exact  #Abweichnung

#####
#Temperaturmessgerät von T in R (-200 bis 0 Celsius):
#A = 3.9083e-3 #°C^-1
#B = -5.7750e-7 #°C^-2
#C = -4.1830e-12 #°C^-4

def R_func(T, A = 3.9083e-3,B = -5.7750e-7,C = -4.1830e-12,D=0):
    return (1000*(1+A*T+B*T**2+C*(T-100)*T**3)) + D
def T_func(R, A, B, C, D):
    return (A*R**3+B*R**2+C*R**1+D)

#Temperaturmessgerät von R in T (-200 bis 0 Celsius):
T = np.linspace(-200,0,100)
R = R_func(T)

prms, cov = curve_fit(T_func, R, T)


######Daten einlesen (ohne Magnet)
t1, R1, U1, I1 = np.genfromtxt('2023/4_Punkt/I_crit/M1_09A.txt', delimiter=' ', unpack=True)
t1 = t1-np.min(t1)
t2, R2, U2, I2 = np.genfromtxt('2023/4_Punkt/I_crit/M2_07A.txt', delimiter=' ', unpack=True)
t2 = t2-np.min(t2)
t3, R3, U3, I3 = np.genfromtxt('2023/4_Punkt/I_crit/M3_05A.txt', delimiter=' ', unpack=True)
t3 = t3-np.min(t3)
t4, R4, U4, I4 = np.genfromtxt('2023/4_Punkt/I_crit/M4_03A.txt', delimiter=' ', unpack=True)
t4 = t4-np.min(t4)
t5, R5, U5, I5 = np.genfromtxt('2023/4_Punkt/I_crit/M5_06A.txt', delimiter=' ', unpack=True)
t5 = t5-np.min(t5)


#Daten einlesen der Messung mit Magnet
### s entspricht starken Magnet (M5)
### h entspricht Magnet in der Halterung (M3)
ts, Rs, Us, Is = np.genfromtxt('2023/4_Punkt/M1_SMag.txt', delimiter=' ', unpack=True)
ts = ts-np.min(ts)
th, Rh, Uh, Ih = np.genfromtxt('2023/4_Punkt/M2_HMag.txt', delimiter=' ', unpack=True)
th = th-np.min(th)

#Bestimme die Temperaturen aus den gemessenen Widerständen:
T1 = T_func(R1,*prms)
T2 = T_func(R2,*prms)
T3 = T_func(R3,*prms)
T4 = T_func(R4,*prms)
T5 = T_func(R5,*prms)

Ts = T_func(Rs,*prms)
Th = T_func(Rh,*prms)

#########
#iii)
#a)



#Erstelle Plots für die verschiedenen Strom Werte
plt.plot(T1, U1/I1, 'x', label = 'gem. Widerstand 0.9A')
#plt.xlabel(r'$T \mathbin{/} \si{\celsius}$')
#plt.ylabel(r'$R \mathbin{/} \si{\ohm} $')
plt.grid()
plt.legend()
plt.savefig('build/4_Punkt_09A.pdf')
plt.clf()

plt.plot(T2, U2/I2, 'x', label = 'gem. Widerstand 0.7A')
#plt.xlabel(r'$T \mathbin{/} \si{\celsius}$')
#plt.ylabel(r'$R \mathbin{/} \si{\ohm} $')
plt.grid()
plt.legend()
plt.savefig('build/4_Punkt_07A.pdf')
plt.clf()

plt.plot(T3, U3/I3, 'x', label = 'gem. Widerstand 0.5A')
#plt.xlabel(r'$T \mathbin{/} \si{\celsius}$')
#plt.ylabel(r'$R \mathbin{/} \si{\ohm} $')
plt.grid()
plt.legend()
plt.savefig('build/4_Punkt_05A.pdf')
plt.clf()

plt.plot(T4, U4/I4, 'x', label = 'gem. Widerstand 0.3A')
#plt.xlabel(r'$T \mathbin{/} \si{\celsius}$')
#plt.ylabel(r'$R \mathbin{/} \si{\ohm} $')
plt.grid()
plt.savefig('build/4_Punkt_03A.pdf')
plt.clf()

plt.plot(T5, U5/I5, 'x', label = 'gem. Widerstand 0.6A')
#plt.xlabel(r'$T \mathbin{/} \si{\celsius}$')
#plt.ylabel(r'$R \mathbin{/} \si{\ohm} $')
plt.grid()
plt.legend()
plt.savefig('build/4_Punkt_06A.pdf')
plt.clf()


###b)

plt.plot(Ts, Us/Is, 'x', label = '0.6A mit M5')
#plt.xlabel(r'$T \mathbin{/} \si{\celsius}$')
#plt.ylabel(r'$R \mathbin{/} \si{\ohm} $')
plt.grid()
plt.legend()
plt.savefig('build/krit_SMag.pdf')
plt.clf()

plt.plot(Th, Uh/Ih, 'x', label = '0.6A mit M3')
#plt.xlabel(r'$T \mathbin{/} \si{\celsius}$')
#plt.ylabel(r'$R \mathbin{/} \si{\ohm} $')
plt.grid()
plt.legend()
plt.savefig('build/krit_HMag.pdf')
plt.clf()



#B-Felder der Magneten
#Messdaten einlesen in mm und mT
rh, Bh = np.genfromtxt('2023/Hall_H.txt', delimiter=' ', unpack=True)
rs, Bs = np.genfromtxt('2023/Hall_S.txt', delimiter=' ', unpack=True)

#Strom einer Leiterschleife bei bekanntem B-Feld
def I_func(B,r,R):
    return (B*2/mu_0*(R**2+r**2)**(3/2)/R**2)

#B Feld einer Leiterschleife
def B_func(r,A,R,B_0):
    return (A*R**2/((R**2+r**2)**(3/2)))+B_0

h_prms ,h_cov = curve_fit(B_func, rh, Bh)
s_prms ,s_cov = curve_fit(B_func, rs, Bs)

r_lin = np.linspace(0,25)

plt.plot(rh, Bh, 'x', label = 'B-Feld M3')
plt.plot(r_lin, B_func(r_lin,*h_prms))
#plt.xlabel(r'$r \mathbin{/} \si{\milli\m}$')
#plt.ylabel(r'$B \mathbin{/} \si{\milli\tesla} $')
plt.grid()
plt.legend()
plt.savefig('build/HMag.pdf')
plt.clf()

plt.plot(rs, Bs, 'x', label = 'B-Feld M5')
plt.plot(r_lin, B_func(r_lin,*s_prms))
#plt.xlabel(r'$r \mathbin{/} \si{\milli\m}$')
#plt.ylabel(r'$B \mathbin{/} \si{\milli\tesla} $')
plt.grid()
plt.legend()
plt.savefig('build/SMag.pdf')
plt.clf()



####
#v)
#Messwert
B = 0.000777 #T
r = 0.001 #m ()geschätze Höhe der Messung
R = 0.015/2 #m
I = I_func(B,r,R)
print('I in Ring:', I)


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
