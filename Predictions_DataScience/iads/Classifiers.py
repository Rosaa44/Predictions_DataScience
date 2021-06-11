# -*- coding: utf-8 -*-

"""
Package: iads
File: Classifiers.py
Année: LU3IN026 - semestre 2 - 2020-2021, Sorbonne Université
"""

# Classfieurs implémentés en LU3IN026
# Version de départ : Février 2021

# Import de packages externes
import numpy as np
import pandas as pd
import math,random


# ---------------------------
class Classifier:
    """ Classe pour représenter un classifieur
        Attention: cette classe est une classe abstraite, elle ne peut pas être
        instanciée.
    """
    #TODO: Classe à Compléter
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        raise NotImplementedError("Please Implement this method")
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        raise NotImplementedError("Please Implement this method")
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")
    def getW(self):
        return self.w
    def accuracy(self, desc_set, label_set):
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        res=0
        n=len(desc_set)
        if(n==0):
            return
        for i in range(n):
            if(self.predict(desc_set[i])==label_set[i]):
                res+=1
            #else:
                #print(desc_set[i])
        return res/n
# ---------------------------
class ClassifierLineaireRandom(Classifier):
    """ Classe pour représenter un classifieur linéaire aléatoire
        Cette classe hérite de la classe Classifier
    """
    
    def __init__(self, input_dimension):
        self.w=np.random.uniform(-1,1,input_dimension)
        
    def train(self, desc_set, label_set):
        print("Pas d'apprentissage pour ce classifieur")
    
    def score(self,x):
        score=0
        for i in range(len(x)):
            score+=x[i]*self.w[i]
        return score
    
    def predict(self, x):
        if(self.score(x)<0):
            return -1
        return 1    
# ---------------------------
class ClassifierKNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """
    
    def __init__(self, input_dimension, k):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        self.k=k
        self.dimension=input_dimension
        self.desc_ref=None
        self.label_ref=None
        
    def score(self,x):
        """ rend la proportion de +1 parmi les k ppv de x (valeur réelle)
            x: une description : un ndarray
        """
        def distance(elem,x):
            res=0
            for i in range(len(x)):
                res+=(elem[i]-x[i])**2
            return math.sqrt(res)
        def equ(elem,x):
            for i in range(len(x)):
                if(elem[i]!=x[i]):
                    return False
            return True
        tableau=[None for _ in range(self.k)]
        tableau_indice=[-1 for _ in range(self.k)]
        i=0
        for elem in self.desc_ref:
            if(not equ(elem,x)):
                b=True
                d=distance(elem,x)
                j=0
                while(b and j<self.k):
                    if(type(tableau[j])==type(None) or distance(tableau[j],x)>d):
                        tableau[j]=elem
                        tableau_indice[j]=i
                        b=False
                    j+=1
                i+=1
        proportion=0
        for i_indice in range(self.k):
            if(self.label_ref[tableau_indice[i_indice]]==1):
                proportion+=1  
        return proportion/self.k
    
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
            x: une description : un ndarray
        """
        if(self.score(x)>0.5):
            return 1
        return -1

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.desc_ref=desc_set
        self.label_ref=label_set
    def copy(self):
        return ClassifierKNN(self.dimension,self.k)
 # ---------------------------
class ClassifierPerceptron(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self, input_dimension,learning_rate,history=False):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples
                - learning_rate : epsilon
            Hypothèse : input_dimension > 0
        """
        self.history=history
        self.learning_rate=learning_rate
        self.input_dimension=input_dimension
        self.w=[0 for _ in range(input_dimension)]
        self.allw=[self.w.copy()]        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        sur_place=0
        n=len(desc_set)
        cpt=0
        while(sur_place<n and cpt<2*n):
            i=random.randrange(n)
            xi=desc_set[i]
            yi=label_set[i]
            if(self.score(xi)*yi<=0):
                sur_place=0
                for k in range(len(self.w)):
                    self.w[k]+=self.learning_rate*xi[k]*yi
                if(self.history):
                    self.allw.append(self.w.copy())
            else:
                sur_place+=1
            
            cpt+=1
 
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        res=0
        for i in range(len(self.w)):
            res+=x[i]*self.w[i]
        return res
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        if(self.score(x)<0):
            return -1
        return 1
    def copy(self):
        return ClassifierPerceptron(self.input_dimension,self.learning_rate,self.history)
# ---------------------------

class ClassifierPerceptronBiais(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self, input_dimension,learning_rate,history=False):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples
                - learning_rate : epsilon
            Hypothèse : input_dimension > 0
        """
        self.history=history
        self.learning_rate=learning_rate
        self.input_dimension=input_dimension
        self.w=[0 for _ in range(input_dimension)]
        self.allw=[self.w.copy()]
        self.allc=[]
    def c(self,x,y):
        res=0
        for i in range(len(x)):
            t=(1-self.score(x[i])*y[i])
            if(t>0):
                res+=t
        return res
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        sur_place=0
        n=len(desc_set)
        cpt=0
        while(sur_place<n and cpt<2*n):
            i=random.randrange(n)
            xi=desc_set[i]
            yi=label_set[i]
            if(self.score(xi)*yi<1):
                sur_place=0
                for k in range(len(self.w)):
                    self.w[k]+=self.learning_rate*xi[k]*yi
                if(self.history):
                    self.allw.append(self.w.copy())
                    self.allc.append(self.c(desc_set,label_set))
            else:
                sur_place+=1
            
            cpt+=1
    def copy(self):
        return ClassifierPerceptronBiais(self.input_dimension,self.learning_rate,self.history)
            
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        res=0
        for i in range(len(self.w)):
            res+=x[i]*self.w[i]
        return res
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        if(self.score(x)<0):
            return -1
        return 1
 # ---------------------------
class Perceptron_MC(Classifier):
    def __init__(self, input_dimension, learning_rate, classes):
        """
            classes=[label_1,label_2,...] représente les valeurs associées aux différentes classes
        """
        self.input_dimension=input_dimension
        self.learning_rate=learning_rate
        self.classes=classes
        self.perceptrons=[None for _ in range(len(classes))]
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        res=[]
        for perceptron in self.perceptrons:
            res.append(perceptron.score(x))
        return res
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        liste=self.score(x)
        i=liste.index(max(liste))
        return self.classes[i]
  
    def train(self, desc_set, label_set):
        def annexe(elem,c):
            if(elem==c):
                return 1
            else:
                return -1
        i=0
        for c in self.classes:
            self.perceptrons[i]=ClassifierPerceptron(self.input_dimension,self.learning_rate)
            label_set_modifie = np.array([annexe(elem,c) for elem in label_set])
            self.perceptrons[i].train(desc_set,label_set_modifie)
            i+=1
 # ---------------------------
class Kernel():
    """ Classe pour représenter des fonctions noyau
    """
    def __init__(self, dim_in, dim_out):
        """ Constructeur de Kernel
            Argument:
                - dim_in : dimension de l'espace de départ (entrée du noyau)
                - dim_out: dimension de l'espace de d'arrivée (sortie du noyau)
        """
        self.input_dim = dim_in
        self.output_dim = dim_out
        
    def get_input_dim(self):
        """ rend la dimension de l'espace de départ
        """
        return self.input_dim

    def get_output_dim(self):
        """ rend la dimension de l'espace d'arrivée
        """
        return self.output_dim
    
    def transform(self, V):
        """ ndarray -> ndarray
            fonction pour transformer V dans le nouvel espace de représentation
        """        
        raise NotImplementedError("Please Implement this method")
        
class KernelBias(Kernel):
    """ Classe pour un noyau simple 2D -> 3D
    """
    def transform(self, V):
        """ ndarray de dim 2 -> ndarray de dim 3            
            rajoute une 3e dimension au vecteur donné
        """
        V_proj = np.append(V,np.ones((len(V),1)),axis=1)
        return V_proj
    
class ClassifierPerceptronKernel(Classifier):
    def __init__(self, input_dimension, learning_rate, noyau):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (espace originel)
                - learning_rate : 
                - noyau : Kernel à utiliser
            Hypothèse : input_dimension > 0
        """
        self.noyau=noyau
        self.input_dimension=input_dimension
        self.learning_rate=learning_rate
        self.w=[0 for _ in range(noyau.get_output_dim())]
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        x=self.noyau.transform(np.array([x]))[0]
        res=0
        for i in range(len(self.w)):
            res+=x[i]*self.w[i]
        return res
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        if(self.score(x)<0):
            return -1
        return 1
  
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        desc_set=self.noyau.transform(desc_set)
        sur_place=0
        n=len(desc_set)
        cpt=0
        while(sur_place<n and cpt<10*n):
            i=random.randrange(n)
            xi=desc_set[i]
            yi=label_set[i]
            if(self.score(xi)*yi>0):
                sur_place+=1
            else:
                sur_place=0
                for k in range(len(self.w)):
                    self.w[k]+=self.learning_rate*xi[k]*yi
            cpt+=1
    def copy(self):
        return ClassifierPerceptronKernel(self.input_dimension,self.learning_rate,self.noyau)
# ------------------------ 
# code de la classe pour le classifieur ADALINE

class ClassifierADALINE(Classifier):
    """ Perceptron de ADALINE
    """
    def __init__(self, input_dimension, learning_rate, history=False, niter_max=1000):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples
                - learning_rate : epsilon
                - history : stockage des poids w en cours d'apprentissage
                - niter_max : borne sur les iterations
            Hypothèse : input_dimension > 0
        """
        
        self.learning_rate=learning_rate
        self.input_dimension=input_dimension
        self.w=[0 for _ in range(input_dimension)]
        self.history=history
        self.niter_max=niter_max
        
        self.history=history
        self.allw=[self.w.copy()]
        self.allc=[]
    def c(self,x,y):
        res=0
        for i in range(len(x)):
            res+=(1-self.score(x[i])*y[i])**2
        return res
    def gradient(self,xd,yd):
            res=xd.copy()
            r=0
            for i in range(len(xd)):
                r+=(xd[i]*self.w[i])
            for i in range(len(res)):
                res[i]*=(r-yd)
            return res
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        
        cpt=0
        n=len(desc_set)
        while(cpt<self.niter_max):
            i=random.randrange(n)
            xi=desc_set[i]
            yi=label_set[i]
            
            if(self.score(xi)*yi<1):
                m=self.score(xi)-yi
                grad=self.gradient(xi,yi)
                for j in range(self.input_dimension):
                    self.w[j]-=self.learning_rate*grad[j]
                if(self.history):
                    self.allw.append(self.w.copy())
                    self.allc.append(self.c(desc_set,label_set))
            cpt+=1
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        res=0
        for i in range(self.input_dimension):
            res+=x[i]*self.w[i]
        return res
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        if(self.score(x)<0):
            return -1
        return 1
    def copy(self):
        return ClassifierADALINE(self.input_dimension,self.learning_rate,self.history,self.niter_max)
# ------------------------ 
class ClassifierMultiOAA(Classifier):
    def __init__(self,perceptron_init,classes=[0,1,2,3]):
        """
            classes=[label_1,label_2,...] représente les valeurs associées aux différentes classes
        """
        self.perceptron_init=perceptron_init
        self.classes=classes
        self.perceptrons=[self.perceptron_init.copy() for _ in range(len(self.classes))]
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        res=[]
        for perceptron in self.perceptrons:
            res.append(perceptron.score(x))
        return res
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        liste=self.score(x)
        i=liste.index(max(liste))
        return self.classes[i]
  
    def train(self, desc_set, label_set):
        def annexe(elem,c):
            if(elem==c):
                return 1
            else:
                return -1
        i=0
        for c in self.classes:
            label_set_modifie = np.array([annexe(elem,c) for elem in label_set])
            self.perceptrons[i].train(desc_set,label_set_modifie)
            i+=1
    

       
