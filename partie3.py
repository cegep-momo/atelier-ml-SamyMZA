#Chargement du dataset avec pour séparateur des tabulations
import pandas as pnd

dataset = pnd.read_csv("apprentissage/datas/dataset.csv",delimiter='\t')

#Suppression des lignes ayant des valeurs manquantes
dataset = dataset.dropna(axis=0, how='any')

#X = on prend toutes les données, mais uniquement les features 5 à 12 
#(ne pas hésiter à ouvrir le fichier afin de bien visualiser les features concernées)
#POINTS_ATTAQUE;POINTS_DEFFENCE;POINTS_ATTAQUE_SPECIALE;POINT_DEFENSE_SPECIALE;POINTS_VITESSE;NOMBRE_GENERATIONS
X = dataset.iloc[:, 5:12].values


#y = on prend uniquement la feature POURCENTAGE_DE_VICTOIRE (17ème feature)
y = dataset.iloc[:, 17].values

#X = on prend toutes les données, mais uniquement les features 4 à 11
# POINTS_ATTAQUE;POINTS_DEFFENCE;POINTS_ATTAQUE_SPECIALE;POINT_DEFENSE_SPECIALE;POINTS_VITESSE;NOMBRE_GENERATIONS
X = dataset.iloc[:, 5:12].values

#y = on prend uniquement la colonne POURCENTAGE_DE_VICTOIRE (16ème feature) 
#les : signifiant "Pour toutes les observations"
y = dataset.iloc[:, 16].values

#Construction du jeu d'entrainement et du jeu de tests
from sklearn.model_selection import train_test_split
X_APPRENTISSAGE, X_VALIDATION, Y_APPRENTISSAGE, Y_VALIDATION = train_test_split(X, y, test_size = 0.2, random_state = 0)


#---- ALGORITHME 1: REGRESSION LINEAIRE -----
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

#Choix de l'algorithme
algorithme = LinearRegression()

#Apprentissage à l'aide de la fonction fit
#Apprentisage effectué sur le jeu de données d'apprentissage
algorithme.fit(X_APPRENTISSAGE, Y_APPRENTISSAGE)

#Realisation de prédictions sur le jeu de tests (validation)
predictions = algorithme.predict(X_VALIDATION)

#Calcul de la précision de l'apprentissage à l'aide de la
#fonction r2_score en comparant les valeurs prédites
#(predictions) et les valeurs attendues (Y_VALIDATION)
precision = r2_score(Y_VALIDATION, predictions)

print(">> ----------- REGRESSION LINEAIRE -----------")
print(">> Precision = "+str(precision))
print("------------------------------------------")



#Choix de l'algorithme
from sklearn.ensemble import RandomForestRegressor
algorithme = RandomForestRegressor()
algorithme.fit(X_APPRENTISSAGE, Y_APPRENTISSAGE)
predictions = algorithme.predict(X_VALIDATION)
precision = r2_score(Y_VALIDATION, predictions)

print(">> ----------- FORETS ALEATOIRES -----------")
print(">> Precision = "+str(precision))
print("------------------------------------------")


import joblib
fichier = 'modele/modele_pokemon.mod'
joblib.dump(algorithme, fichier)

predictions = algorithme.predict(X_VALIDATION)
precision_apprentissage = algorithme.score(X_APPRENTISSAGE,Y_APPRENTISSAGE)
precision = r2_score(Y_VALIDATION, predictions)