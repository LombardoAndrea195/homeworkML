from operator import itemgetter

import pandas as pd

from preprocessing import from_jsonl_to_csv
import numpy as np

from preprocessingForBlind import from_jsonl_to_csv2, buildthecsv, obtainInstructions
from textToVector import ManageContentToVector, ManageContentForBlindSetToVectBinary, \
    ManageContentForBlindSetToVectMulticlass
from classification import classificator,classificator2,classificator3
number=3
from_jsonl_to_csv2('test_dataset_blind.jsonl','train-blind-dataset.csv') #costruisco i csv dai jsonl
from_jsonl_to_csv('train_dataset.jsonl','train-dataset.csv') #costruisco i csv dai jsonl
dataset_dictonary2=pd.read_csv('train-blind-dataset.csv')
dataset_dictonary=pd.read_csv('train-dataset.csv') #sono interessato a leggere i file che ho costruito precedentemente

targetBinario=dataset_dictonary['opt']
targetMulticlasse=dataset_dictonary['compiler']
dataset_dictonary['opt']=dataset_dictonary.opt.map({'L': 0, 'H': 1})#invece che avere L o H preferisco costruirmi una leggenda: L->0 e H->1
dataset_dictonary['compiler']=dataset_dictonary.compiler.map({'gcc': 0, 'clang': 1, 'icc':2})#costruisco una mappa anche con compiler per gestirmi il problema di multiclasse

    #se inserisco number=1 applico HashingToVect, oppure number=2 applicoTF-IDF approach,altriemnti numero=3 applico TextToVect
vector= ManageContentToVector(number,dataset_dictonary['instructions']) #per classificare decido di utilizzare una codifica textToVector(395)
vector2= ManageContentToVector(number-1,dataset_dictonary['instructions']) #per classificare decido di utilizzare una codifica textToVector
vector3= ManageContentToVector(number-2,dataset_dictonary['instructions']) #per classificare decido di utilizzare una codifica textToVector



print("Primo test basato su test Binari")

classificator(vector,dataset_dictonary['opt'],0.30,10,"BernulliNB","Binario")
classificator(vector,dataset_dictonary['opt'],0.30,10,"DecisionTreeClassifier","Binario")
classificator(vector,dataset_dictonary['opt'],0.30,10,"GaussianNB","Binario")

print("Secondo test basato su test Binari")
classificator(vector,dataset_dictonary['opt'],0.80,50,"BernulliNB","Binario")
classificator(vector,dataset_dictonary['opt'],0.80,50,"DecisionTreeClassifier","Binario")
classificator(vector,dataset_dictonary['opt'],0.80,50,"GaussianNB","Binario")

print("Terzo test basato su test Binari")
classificator(vector,dataset_dictonary['opt'],0.55,0,"BernulliNB","Binario")
classificator(vector,dataset_dictonary['opt'],0.55,0,"DecisionTreeClassifier","Binario")
classificator(vector,dataset_dictonary['opt'],0.55,0,"GaussianNB","Binario")

print("Primo test basato su test Multiclasse")
classificator2(vector,dataset_dictonary['compiler'],0.30,10,"MultinomialNB","Multiclasse")
classificator2(vector,dataset_dictonary['compiler'],0.30,10,"DecisionTreeClassifier","Multiclasse")
classificator2(vector,dataset_dictonary['compiler'],0.30,10,"GaussianNB","Multiclasse")

print("Secondo test basato su test Multiclasse")
classificator2(vector,dataset_dictonary['compiler'],0.80,50,"MultinomialNB","Multiclasse")
classificator2(vector,dataset_dictonary['compiler'],0.80,50,"DecisionTreeClassifier","Multiclasse")
classificator2(vector,dataset_dictonary['compiler'],0.80,50,"GaussianNB","Multiclasse")

print("Terzo test basato su test Multiclasse")
classificator2(vector,dataset_dictonary['compiler'],0.55,0,"MultinomialNB","Multiclasse")
classificator2(vector,dataset_dictonary['compiler'],0.55,0,"DecisionTreeClassifier","Multiclasse")
classificator2(vector,dataset_dictonary['compiler'],0.55,0,"GaussianNB","Multiclasse")


print("Primo test basato su test Multiclasse one-vs-rest")
classificator3(vector,dataset_dictonary['compiler'],0.30,10,"MultinomialNB","Multiclasse")
classificator3(vector,dataset_dictonary['compiler'],0.30,10,"DecisionTreeClassifier","Multiclasse")
classificator3(vector,dataset_dictonary['compiler'],0.30,10,"GaussianNB","Multiclasse")

print("Secondo test basato su test Multiclasse one-vs-rest")
classificator3(vector,dataset_dictonary['compiler'],0.80,50,"MultinomialNB","Multiclasse")
classificator3(vector,dataset_dictonary['compiler'],0.80,50,"DecisionTreeClassifier","Multiclasse")
classificator3(vector,dataset_dictonary['compiler'],0.80,50,"GaussianNB","Multiclasse")

print("Terzo test basato su test Multiclasse one-vs-rest")
classificator3(vector,dataset_dictonary['compiler'],0.55,0,"MultinomialNB","Multiclasse")
classificator3(vector,dataset_dictonary['compiler'],0.55,0,"DecisionTreeClassifier","Multiclasse")
classificator3(vector,dataset_dictonary['compiler'],0.55,0,"GaussianNB","Multiclasse")

dataset_dictonary['opt']=targetBinario
dataset_dictonary['compiler']=targetMulticlasse
#I decide to not use previous variables for example targetMulticlasse or targetBinario because in this way I 've got without other approach the functions
y_predBinarioBinary=ManageContentForBlindSetToVectBinary(dataset_dictonary,dataset_dictonary2['instructions'])
y_predBinarioMulticlasse=ManageContentForBlindSetToVectMulticlass(dataset_dictonary,dataset_dictonary2['instructions'])
obtainInstructions('test_dataset_blind.jsonl','train-blind-dataset.csv')
instructionDict=pd.read_csv('train-blind-dataset.csv')
print(instructionDict)
instruction=list(itemgetter('instructions')(instructionDict))
buildthecsv('test_dataset_blind.jsonl','train-blind-dataset.csv',y_predBinarioBinary,y_predBinarioMulticlasse,instruction)
