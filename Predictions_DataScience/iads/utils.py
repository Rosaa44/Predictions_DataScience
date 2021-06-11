# -*- coding: utf-8 -*-

"""
Package: iads
Fichier: utils.py
Année: semestre 2 - 2019-2020, Sorbonne Université
"""

# ---------------------------
# -*- coding: utf-8 -*-

"""
Package: iads
File: utils.py
Année: LU3IN026 - semestre 2 - 2020-2021, Sorbonne Université
"""


# Fonctions utiles pour les TDTME de LU3IN026
# Version de départ : Février 2021

# import externe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------ 
def plot2DSet(data_desc,data_label):
    data_negatifs = data_desc[data_label == -1]
    data_positifs = data_desc[data_label == +1]
    plt.scatter(data_negatifs[:,0],data_negatifs[:,1],marker='o',color='red') # 'o' pour la classe -1
    plt.scatter(data_positifs[:,0],data_positifs[:,1],marker='x',color='blue') # 'x' pour la classe +1

    
def plot_frontiere(desc_set, label_set, classifier, step=30):
    """ desc_set * label_set * Classifier * int -> NoneType
        Remarque: le 4e argument est optionnel et donne la "résolution" du tracé
        affiche la frontière de décision associée au classifieur
    """
    mmax=desc_set.max(0)
    mmin=desc_set.min(0)
    x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],step),np.linspace(mmin[1],mmax[1],step))
    grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))
    
    # calcul de la prediction pour chaque point de la grille
    res=np.array([classifier.predict(grid[i,:]) for i in range(len(grid)) ])
    res=res.reshape(x1grid.shape)
    # tracer des frontieres
    # colors[0] est la couleur des -1 et colors[1] est la couleur des +1
    plt.contourf(x1grid,x2grid,res,colors=["darksalmon","skyblue"],levels=[-1000,0,1000])    
# ------------------------ 
def genere_dataset_uniform(p, n, binf=-1, bsup=1):
    vecteurs=np.random.uniform(binf,bsup,(n,p))
    labels=np.asarray([-1 for i in range(0,int(n/2))] + [+1 for i in range(0,int(n/2))])
    return(vecteurs,labels)
    
def genere_dataset_gaussian(positive_center,positive_sigma,negative_center,negative_sigma,nb_points):
    negative_normale=np.random.multivariate_normal(negative_center,negative_sigma,nb_points)
    positive_normale=np.random.multivariate_normal(positive_center,positive_sigma,nb_points)
    vecteurs=np.vstack((negative_normale,positive_normale))
    labels=np.asarray([-1 for i in range(0,nb_points)] + [+1 for i in range(0,nb_points)])
    return (vecteurs,labels)
# ------------------------ 
def create_XOR(n,sigma):
    tableau_sigma=np.array([[sigma,0],[0,sigma]])
    positive_normale=np.vstack((np.random.multivariate_normal(np.array([0,1]),tableau_sigma,int(n/2)),np.random.multivariate_normal(np.array([1,0]),tableau_sigma,int(n/2))))
    negative_normale=np.vstack((np.random.multivariate_normal(np.array([0,0]),tableau_sigma,int(n/2)),np.random.multivariate_normal(np.array([1,1]),tableau_sigma,int(n/2))))
    vecteurs=np.vstack((negative_normale,positive_normale))
    labels=np.asarray([-1 for i in range(0,n)] + [+1 for i in range(0,n)])
    return (vecteurs,labels)
 # ------------------------ 
def crossval(X, Y, n_iterations, iteration):
    taille=int(len(X)/n_iterations)
    n1=iteration*taille
    n2=iteration*taille+taille
    Xtest=np.array(X[n1:n2])
    Ytest=np.array(Y[n1:n2]) 
    Xapp=np.concatenate((X[:n1],X[n2:]))
    Yapp=np.append(Y[:n1],Y[n2:])
    return (Xapp,Yapp,Xtest,Ytest)
 # ------------------------ 
def crossval_strat(X, Y, n_iterations, iteration):
    taille=int(len(X)/(2*n_iterations))
    coupure=int(len(X)/2)
    n1_1=iteration*taille
    n1_2=iteration*taille+taille
    n2_1=coupure+iteration*taille
    n2_2=coupure+iteration*taille+taille
    Xtest=np.concatenate((X[n1_1:n1_2],X[n2_1:n2_2]))
    Ytest=np.append(Y[n1_1:n1_2],Y[n2_1:n2_2])
    Xapp=np.concatenate((np.concatenate((X[:n1_1],X[n1_2:n2_1])),X[n2_2:]))
    Yapp=np.append(np.append(Y[:n1_1],Y[n1_2:n2_1]),Y[n2_2:])
    return (Xapp,Yapp,Xtest,Ytest)
 # ------------------------ 
def plot2DSetMulticlass(data_desc,data_label):
    data_negatifs_1 = data_desc[data_label == 0]
    data_positifs_1 = data_desc[data_label == +1]
    data_negatifs_2 = data_desc[data_label == +2]
    data_positifs_2 = data_desc[data_label == +3]
    plt.scatter(data_negatifs_1[:,0],data_negatifs_1[:,1],marker='o',color='red') 
    plt.scatter(data_positifs_1[:,0],data_positifs_1[:,1],marker='o',color='blue') 
    plt.scatter(data_negatifs_2[:,0],data_negatifs_2[:,1],marker='o',color='yellow') 
    plt.scatter(data_positifs_2[:,0],data_positifs_2[:,1],marker='o',color='green')
    
#------------------

def classe_majoritaire(Y):
    """ Y : (array) : array de labels
        rend la classe majoritaire ()
    """
    liste = {}
    for i in Y :
        if i not in liste:
            liste[i]=0
        liste[i]+=1
    return max(liste)
    
#--------------------------   
import math
def shannon(P):
    """ list[Number] -> float
        Hypothèse: la somme des nombres de P vaut 1
        P correspond à une distribution de probabilité
        rend la valeur de l'entropie de Shannon correspondante
    """
    somme = 0
    if len(P)==1:
        return somme
    for pi in P:
        if pi != 0:
            somme -= pi*math.log(pi,len(P))
    return somme  
    
#------------------
def entropie(Y):
    """ Y : (array) : ensemble de labels de classe
        rend l'entropie de l'ensemble Y
    """
    liste = {}
    P = []
    for c in Y :
        if c not in liste:
            liste[c]=0
        liste[c]+=1
    for c in liste:
        P.append(liste[c]/len(Y))
        
    return shannon(P)
    
#------------------

def normalisation(A):
    return (A - A.min())/(A.max() - A.min())
    
    
def dist_vect(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)
    
def centroide(ex) :
    return [np.mean(ex[:,i]) for i in range(len(ex[0]))]
    
import math
def inertie_cluster(A) :
    centro = centroide(A)
    return sum([math.pow(np.linalg.norm(A[i] - centro), 2) for i in range(len(A))])
    
def initialisation(K,data):
    liste = [i for i in range(len(data))]
    y = np.random.choice(liste,K) 
    return data[y]
    
def plus_proche(ex, centro) :
    x = 0
    for i in range(len(centro)) :
        a = np.linalg.norm(ex - centro[x])
        b = np.linalg.norm(ex - centro[i])
        if  a > b :
            x = i 
    return x
    
def affecte_cluster(data, kcentro) :
    
    mat_aff = {}
    for i in range(len(data)):
        x = plus_proche(data[i],kcentro)
        if x not in mat_aff:
            mat_aff[x]=[]
        mat_aff[x].append(i) #Kindices des clusters
    
    return mat_aff
        
def nouveaux_centroides(data,mat_aff):
    return [(centroide(data[mat_aff[elem]])) for elem in mat_aff]


def inertie_globale(data,mat_aff):
    sum = 0
    for elem in mat_aff:
        sum+= inertie_cluster(data[mat_aff[elem]])
    return sum
    
def kmoyennes(K,data,epsilon,iter_max):
    
    centro = initialisation(K,data)
    affect = affecte_cluster(data, centro)
    centro = nouveaux_centroides(data, affect)
    inert = inertie_globale(data, affect)
    
    for i in range(iter_max) :
        
        affect = affecte_cluster(data, centro)
        centro = nouveaux_centroides(data, affect)
        inert2 = inertie_globale(data, affect)
        
        if ((inert2 - inert) < epsilon) :
            print("Fini")
            return centro, affect
    inert = inert2  
    
    return np.asarray(centro), affect
  
import colorsys
import matplotlib
def affiche_resultat(data,centro,affect,K):
    colorNames = list(matplotlib.colors.cnames.keys())
    for i in range(len(centro)):
        couleur = np.random.choice(colorNames)
        colorNames.remove(couleur)
        plt.scatter(data[affect[i]][:,0],data[affect[i]][:,1],color=couleur)
        
    centro = np.array(centro)
    plt.scatter(centro[:,0],centro[:,1],color='r',marker='x')
  

