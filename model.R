#*******************************************************#
# Formation :   UCAD | FST | Master2 MSI                #
# Etudiante :   Kadiatou Seydi Barry                    #
# Profil :      Ingénieur statisticienne|Data scientist # 
# Début :       20/09/2025                              #
# Fin :         27/09/2025                              #
#*******************************************************#


# Business Problème
# Prédire le score de risque de désabonnement des clients .


# 0.Installation des packages 
install.packages("visdat",dependencies = T) 
install.packages("GGally",dependencies = T)   
install.packages("cowplot",dependencies = T)   
install.packages("corrplot",dependencies = T)  
install.packages("tidyverse",dependencies = T)
install.packages("janitor",dependencies = T)
install.packages("plotly",dependencies = T)
install.packages("MLeval",dependencies = T)
install.packages("yardstick",dependencies = T)
install.packages("parallel",dependencies = T)
install.packages("doSNOW",dependencies = T)
install.packages("tictoc",dependencies = T)
install.packages("lattice",dependencies = T)
install.packages("MASS",dependencies = T)
install.packages("klaR",dependencies = T)
install.packages("e1071",dependencies = T)
install.packages("rpart",dependencies = T)
install.packages("kernlab",dependencies = T)
install.packages("plyr",dependencies = T)     
install.packages("xgboost",dependencies = T)
install.packages("Matrix",dependencies = T)
install.packages("dplyr",dependencies = T)



# 1. IMPORTATIONS DES LIBRAIRIES 
library(visdat) # Visualiser les données manquantes 
library(tidyverse) # Pour manipuler les données 
library(janitor)   # Pour toiletter les variables 
library(plotly)    # Visualiser les donnéées( Rendre intéractif les graphes) 
library(MLeval)    # Tracer la courbe ROC (permet d'estimer l'aire sous la courbe(la metric AUC))
library(yardstick) # Visualiser la matrice de confusion (representation en mosaique)
library(parallel)  # Amélorer le temps d'éxécution des algorithmes (Calculs parallèle)
library(doSNOW)    # Amélorer le temps d'éxécution des algorithmes (Calculs parallèle)
library(tictoc)    # Mésurer le Temps d'éxécution des algorithmes 
library(questionr) # Tableaux croisés
library(GGally)    # Correlogrammes
library(cowplot)   # Matrice de corrélation
library(corrplot)  # Matrice de corrélation
library(dplyr)


library(caret)     # Machine learning
library(lattice)   # librairies dont dependent caret
library(MASS)
library(klaR)
library(e1071)
library(rpart)
library(kernlab)
library(plyr)
library(xgboost)
library(Matrix)

# Pour Corriger les conflits des packages 
select = dplyr::select
filter = dplyr::filter
lag = dplyr::lag


# 2. IMPORTATION DES DONNEES-----------------------------------------------------------------------------

data_1 = read.csv("C:/Users/33616/Desktop/Master/MSI/churn-bigml-80.csv",
                  header = TRUE,
                  sep = ",",
                  dec = ".") 



data_2 = read.csv("C:/Users/33616/Desktop/Master/MSI/churn-bigml-20.csv",
                  header = TRUE,
                  sep = ",",
                  dec = ".") 

# Fusion des deux bases de données 
data_telecom = rbind(data_1,data_2)


# Toilletage des noms des colonnes 
data_telecom = clean_names(data_telecom) # Librairie janitor


# Apperçu de la base de données 
View(data_telecom)


# Nom des variables
names(data_telecom)


# Dimension du jeu de données 
dim(data_telecom)

# Statistiques de bases 
summary(data_telecom)

# Detection des données manquantes
value_missing = vis_dat(data_telecom) # Pas de données manquantes dans ce jeu de données
# Affichage
value_missing




# Recodage des classes ******************************************************************************** 
data_telecom = data_telecom %>%
  mutate_if(is.character, as.factor) 
# Structure
str(data_telecom)
# Regrouper les modalités de la variable 'state' en 3 groupes selon leur PIB ---------------------------
# Tous les Etats 
grp = levels(data_telecom$state)
length(grp)
# Groupe1 : Les 10 Etats les plus riches 
grp1 = c("CA","TX","NY","FL","IL","PA","OH","NJ","GA","WA")
length(grp1)
# Groupe1 : Les 10 Etats Moyennent riches 
grp2 = c("MA","MC","VA","MI","MO","CO","MN","TN","IN","AR")
length(grp2)
# Utiliser une boucle 
for (i in grp) {
  if (i %in% grp1) {
    data_telecom$state = fct_recode(data_telecom$state,"Parmis les 10 Etats les plus riches" = i)
  }  
  else if (i %in% grp2) {
    data_telecom$state = fct_recode(data_telecom$state,"Parmis les 10 Etats Moyennent riches" = i)
  }
  else {
    data_telecom$state = fct_recode(data_telecom$state,"Parmis les  Etats assez riches" = i)
  }
}


# Recodage de la variable "state "(fct_recode) : condition que les variables soient de type factor
data_telecom = data_telecom %>%
  mutate(state = fct_recode(state,
                            "assez_riches" = "Parmis les 10 Etats les plus riches",
                            "moyennent_riches" = "Parmis les 10 Etats Moyennent riches",
                            " plus_riches" = "Parmis les  Etats assez riches"))



### Recodage des variables(condition qu'ils soient de type factor) Remplacer "oui" par "Yes"
data_telecom$international_plan = fct_recode(data_telecom$international_plan,
                                             "Oui" = "Yes",
                                             "Non" = "No")




data_telecom$voice_mail_plan = fct_recode(data_telecom$voice_mail_plan,
                                          "Oui" = "Yes",
                                          "Non" = "No")




data_telecom$churn = fct_recode(data_telecom$churn,
                                "Oui" = "True",
                                "Non" = "False")


data_telecom$area_code = as.factor(data_telecom$area_code)
# Structure
str(data_telecom)







# ******************Analyse Exploratoire des données ***************************************************************-

# Analyse Univariée des variables -
summary(data_telecom)



#Analyse bivariée des variables -----------------------------------------------------
"Objet : L'analyse bivariée consiste à rechercher les liens entre les variables.
 L'idée consiste à rechercher les variables dependantes de la variable cible <churn>.
 En effet, l'une des meilleurs manières d'étudier la dépendance entre deux variables 
 C'est de les visualiser.
 --------------------------------> PLAN : --------------------------------------------->
 Ainsi,nous tracerons des histogrammes (fonction densité) pour comparer la distribution 
 des variables continues par rapport à la variable cible et par ailleurs des diagrrammes
 en barres poour expliquer la répartition des classes selon que le client se desabonne ou pas.
 "
# Table de l'Analyse Bivariée des variables avec 'gtsummary '----------------------------------------
# 1. Modification de la statistique en 'Moyenne(ecart-type)
# Mofification du label de l'entête .
# 1.4 Analyse bivariée ; Visualisation des données 

" Tout d'abord séparons la base de données en deux parties 

- data_cat : Base des variables categorielles 

- data_cont : et celles des variables continues 
"
# Base des variables continues 
data_cont = data_telecom %>%
  select_if(is.numeric)

# Base des variables categorielles 
data_cat = data_telecom %>%
  select_if(is.factor)

# Histogramme : Distribution des variables continues -------------------------------------------------
"
A présent , pour chacune des variables ci_dessous, nous allons tracer l'histogramme
(fonction densité) des distributions .
IL s'agit des variables : 
-Total.day.minutes : Total des minutes d'appels de la journée
-Total.night.minutes : Total des minutes d'appels de la nuit
-Total.eve.minutes : Total des minutes d'appels du soir
-Total.intl.minutes : Total des minutes d'appels externes
"# Distribution des variables continues  'loan_amount'
# Histogramme + Densité
HD1 = data_cont %>%
  ggplot(aes(x = total_day_minutes )) + 
  geom_histogram(aes(y = ..density..),col = "white",fill = "chocolate1")+
  geom_density(size = 0.8, col = "cornflowerblue")+
  labs(x = "Total des minutes d'appels de la journée ")+
  theme_minimal()


HD2 = data_cont %>%
  ggplot(aes(x = total_night_minutes)) + 
  geom_histogram(aes(y = ..density..),col = "white",fill = "chocolate2")+
  geom_density(size = 0.8, col = "cornflowerblue")+
  labs(x = "Total des minutes d'appels de la nuit ")+
  theme_minimal()


HD3 = data_cont %>%
  ggplot(aes(x = total_eve_minutes  )) + 
  geom_histogram(aes(y = ..density..),col = "white",fill = "chocolate3")+
  geom_density(size = 0.8, col = "cornflowerblue")+
  labs(x = " Total des minutes d'appels du soir")+
  theme_minimal()


HD4 = data_cont %>%
  ggplot(aes(x = total_intl_minutes)) + 
  geom_histogram(aes(y = ..density..),col = "white",fill = "chocolate4")+
  geom_density(size = 0.8, col = "cornflowerblue")+
  labs(x = " Total des minutes d'appels externes")+
  theme_minimal()
# Affichage des histogrammes 
# Combinaison de plusieurs graphes
cowplot::plot_grid(HD1,HD2,HD3,HD4)






### Comparaison de la distribution -----------------------------------------------------------------
"Après que nous ayons fini de voir la distribution des variables continues , 
 nous allons comparer ces distributions selon que le client soit fidèle ou 
 desabonné à Orange Télécom.
"# Comparaison des densités selon le churn 
D1 = data_telecom %>%
  ggplot(aes(x = total_day_minutes , col = churn))+
  geom_density(size = 0.8)+
  scale_color_brewer(palette = "Set1")+
  labs(x = "Total des minutes d'appels de la journée ")+
  theme_minimal()

D2 = data_telecom %>%
  ggplot(aes(x = total_night_minutes, col = churn))+
  geom_density(size = 0.8)+
  scale_color_brewer(palette = "Set2")+
  labs(x = "Total des minutes d'appels de la nuit ")+
  theme_minimal()

D3 = data_telecom %>%
  ggplot(aes(x = total_eve_minutes, col = churn))+
  geom_density(size = 0.8)+
  scale_color_brewer(palette = "Paired")+
  labs(x = " Total des minutes d'appels du soir")+
  theme_minimal()

D4 = data_telecom %>%
  ggplot(aes(x = total_intl_minutes, col = churn))+
  geom_density(size = 0.8)+
  scale_color_brewer(palette = "Accent")+
  labs(x = " Total des minutes d'appels externes")+
  theme_minimal()
# Affichage des densités 
# Combinaison de plusieurs graphes
cowplot::plot_grid(D1,D2,D3,D4)




# Histogramme : Distribution des variables continues "frais d'appels" -------------------------------------------------
"
A présent , pour chacune des variables ci_dessous, nous allons tracer l'histogramme
(fonction densité) des distributions .

IL s'agit des variables : 
-Total.day.charge : Total des frais d'appels de la journée
-Total.night.charge : Total des frais d'appels de la nuit
-Total.eve.charge : Total des frais d'appels du soir
-Total.intl.charge : Total des frais d'appels externes
"
# Distribution des variables continues'Total.day.charge'
# Histogramme + Densité
HD5 = data_cont %>%
  ggplot(aes(x = total_day_charge)) + 
  geom_histogram(aes(y = ..density..),col = "white",fill = "chocolate1")+
  geom_density(size = 0.8, col = "cornflowerblue")+
  labs(x = "Total des frais d'appels de la journée ")+
  theme_minimal()


HD6 = data_cont %>%
  ggplot(aes(x = total_night_charge)) + 
  geom_histogram(aes(y = ..density..),col = "white",fill = "chocolate2")+
  geom_density(size = 0.8, col = "cornflowerblue")+
  labs(x = "Total des frais d'appels de la nuit ")+
  theme_minimal()

HD7 = data_cont %>%
  ggplot(aes(x = total_eve_charge)) + 
  geom_histogram(aes(y = ..density..),col = "white",fill = "chocolate3")+
  geom_density(size = 0.8, col = "cornflowerblue")+
  labs(x = " Total des frais d'appels du soir")+
  theme_minimal()

HD8 = data_cont %>%
  ggplot(aes(x = total_intl_charge)) + 
  geom_histogram(aes(y = ..density..),col = "white",fill = "chocolate4")+
  geom_density(size = 0.8, col = "cornflowerblue")+
  labs(x = " Total des frais d'appels externes")+
  theme_minimal()
# Affichage des histogrammes 
# Combinaison de plusieurs graphes
cowplot::plot_grid(HD5,HD6,HD7,HD8)






### Comparaison de la distribution -----------------------------------------------------------------
"Après que nous ayons fini de voir la distribution des variables continues , 
 nous allons comparer ces distributions selon que le client soit fidèle ou 
 desabonné à Orange Télécom.
"# Comparaison des densités selon le churn 
D5 = data_telecom %>%
  ggplot(aes(x =total_day_charge , col = churn))+
  geom_density(size = 0.8)+
  scale_color_brewer(palette = "Set1")+
  labs(x = "Total des frais d'appels de la journée ")+
  theme_minimal()

D6 = data_telecom %>%
  ggplot(aes(x = total_night_charge, col = churn))+
  geom_density(size = 0.8)+
  scale_color_brewer(palette = "Set2")+
  labs(x = "Total des frais d'appels de la nuit ")+
  theme_minimal()

D7 = data_telecom %>%
  ggplot(aes(x = total_eve_charge, col = churn))+
  geom_density(size = 0.8)+
  scale_color_brewer(palette = "Paired")+
  labs(x = " Total des frais d'appels du soir")+
  theme_minimal()

D8 = data_telecom %>%
  ggplot(aes(x = total_intl_charge, col = churn))+
  geom_density(size = 0.8)+
  scale_color_brewer(palette = "Accent")+
  labs(x = " Total des frais d'appels externes")+
  theme_minimal()
# Affichage des densités 
# Combinaison de plusieurs graphes
cowplot::plot_grid(D5,D6,D7,D8)




### Comparaison par boxplot -------------------------------------------------------------------------
"Une autre manière plus intéressante de comparer la distribution d'une 
 variable continue c'est d'utiliser un 'boxplot' ou diagramme en boîtes.
"
# Comparaison par boxplot 
B11 = data_telecom %>%
  ggplot(aes(x = churn, y = total_day_charge, fill = churn))+
  geom_boxplot()+
  scale_fill_brewer(palette = "Blues")+
  labs(y = "Total des frais d'appels de la journée", x = "")+
  theme_minimal()
# Comparaison par boxplot 
B12 = data_telecom %>%
  ggplot(aes(x = churn, y = total_night_charge, fill = churn))+
  geom_boxplot()+
  scale_fill_brewer(palette = "OrRd")+
  labs(y = "Total des frais d'appels de la nuit", x = "")+
  theme_minimal()
# Comparaison par boxplot 
B13 = data_telecom %>%
  ggplot(aes(x = churn, y = total_eve_charge, fill = churn))+
  geom_boxplot()+
  scale_fill_brewer(palette = "PuRd")+
  labs(y = "Total des frais d'appels du soir", x = "")+
  theme_minimal()
# Comparaison par boxplot 
B14 = data_telecom %>%
  ggplot(aes(x = churn, y = total_intl_charge, fill = churn))+
  geom_boxplot()+
  scale_fill_brewer(palette = "GnBu")+
  labs(y = "Total des frais d'appels externes", x = "")+
  theme_minimal()
#Affichage des Boxplot 
# Combinaison de plusieurs graphes
cowplot::plot_grid(B11,B12,B13,B14)




# Corrélogramme ou graphique de corrélation des variables continues -----------------------------------
"
Pour étudier la corrélation des variables continues , nous pouvons les visualiser 
en nous basant  des nuages de points .L'idée sera donc de rechercher des relations 
linéaires entre les variables  pris deux à deux .
"
# Graphe des corrélations 
# Avec la fonction ggpairs() de la librairie 'GGally'
G = ggpairs(data_cont[, 1:8])+
  theme_minimal()
# Affichage 
print(G)
# Corrélogramme avec la fonction corrplot() de la librairie corrplot
mcor = data_cont %>%
  cor() %>%
  corrplot::corrplot(type = "upper",order = "hclust", tl.col = "black", tl.srt = 45)
#Affichage 
mcor


"Les graphes suivants permettent de voir de plus près l'association linéaire 
 notée sur certaines variables"
# Etude de corrélation avec les nuages de points 
# Nuages de points 
NP1 = data_cont %>%
  ggplot(aes(x =total_day_minutes ,y = total_day_charge))+
  geom_point()+
  labs(x= "Total des minutes d'appels de la journée ", y = " Total des frais d'appels de la journée")+
  theme_minimal()

NP2 = data_cont %>%
  ggplot(aes(x = total_night_minutes,y = total_night_charge))+
  geom_point()+
  labs(x= "Total des minutes d'appels de la nuit", y = " Total des frais d'appels de la nuit")+
  theme_minimal()

NP3 = data_cont %>%
  ggplot(aes(x = total_eve_minutes ,y = total_eve_charge))+
  geom_point()+
  labs(x= "Total des minutes d'appels du soir", y = "Total des frais d'appels du soir ")+
  theme_minimal()

NP4 = data_cont %>%
  ggplot(aes(x = total_intl_minutes,y = total_intl_charge))+
  geom_point()+
  labs(x= "Total des minutes d'appels externes", y = "Total des frais d'appels externes ")+
  theme_minimal()
#Affichage des Nuages de points 
# Combinaison de plusieurs graphes
cowplot::plot_grid(NP1,NP2,NP3,NP4)


# Comparaison Graphique des variables catégorielles avec ggplot2()
# Repartition des Etats d'USA selon le type de client
Diag_1 = data_cat %>%
  group_by(state, churn) %>%
  summarise(n = n()) %>%
  mutate(prop = n / sum(n), label = scales::percent(prop)) %>%
  ggplot(aes(x= state, y = prop, fill = churn))+
  geom_bar(stat = "identity", position = position_stack(), width = 0.5)+
  scale_fill_brewer(palette = "Blues","Clients Desabonnés")+
  geom_text(aes(label = label), position = position_stack(vjust = 0.5),size = 3)+
  coord_flip()+
  labs(y = "Repartition du milieu de résidence ",title = "", x = "")+
  theme_minimal()+
  theme(
    axis.text.x = element_blank(), # Spécifier l'axe des X
    panel.grid = element_blank(), # Supprimmer les grilles de fond
    plot.title = element_text(size = 3, face = "bold")) # Centrer le titre


# Repartition  selon le type de client
Diag_2 = data_cat %>%
  group_by(area_code, churn) %>%
  summarise(n = n()) %>%
  mutate(prop = n / sum(n), label = scales::percent(prop)) %>%
  ggplot(aes(x= area_code, y = prop, fill = churn))+
  geom_bar(stat = "identity", position = position_stack(), width = 0.5)+
  scale_fill_brewer(palette = "Blues","Clients Desabonnés")+
  geom_text(aes(label = label), position = position_stack(vjust = 0.5),size = 3)+
  coord_flip()+
  labs(y = "Repartition de l'indicatif régional selon le type de client",title = "", x = "")+
  theme_minimal()+
  theme(
    axis.text.x = element_blank(), # Spécifier l'axe des X
    panel.grid = element_blank(), # Supprimmer les grilles de fond
    plot.title = element_text(size = 3, face = "bold")) # Centrer le titre


Diag_3 = data_cat %>%
  group_by(international_plan, churn)%>%
  summarise(n = n()) %>%
  mutate(prop = n / sum(n), label = scales::percent(prop)) %>%
  ggplot(aes(x= international_plan, y = prop, fill = churn))+
  geom_bar(stat = "identity", position = position_stack(), width = 0.5)+
  scale_fill_brewer(palette = "Blues","Clients Desabonnés")+
  geom_text(aes(label = label), position = position_stack(vjust = 0.5),size = 3)+
  coord_flip()+
  labs(y = "Repartition du plan international",title = "", x = "")+
  theme_minimal()+
  theme(
    axis.text.x = element_blank(), # Spécifier l'axe des X
    panel.grid = element_blank(), # Supprimmer les grilles de fond
    plot.title = element_text(size = 3, face = "bold")) # Centrer le titre


Diag_4 = data_cat %>%
  group_by(voice_mail_plan, churn)%>%
  summarise(n = n()) %>%
  mutate(prop = n / sum(n), label = scales::percent(prop)) %>%
  ggplot(aes(x= voice_mail_plan, y = prop, fill = churn))+
  geom_bar(stat = "identity", position = position_stack(), width = 0.5)+
  scale_fill_brewer(palette = "Blues","Clients Desabonnés")+
  geom_text(aes(label = label), position = position_stack(vjust = 0.5),size = 3)+
  coord_flip()+
  labs(y = "Repartition des plans de messagérie vocaux selon le type de client",title = "", x = "")+
  theme_minimal()+
  theme(
    axis.text.x = element_blank(), # Spécifier l'axe des X
    panel.grid = element_blank(), # Supprimmer les grilles de fond
    plot.title = element_text(size = 3, face = "bold")) # Centrer le titre
#Affichage
cowplot::plot_grid(Diag_1,Diag_2,Diag_3,Diag_4)



# FEATURE ENGENERING------------------------------------------------------------------------------
# DUMMYVARS
# Mettre les variables categorielles en variables factices binaires (0,1)
"
En effet il est toute fois difficile de modeliser avec des variables categorielles .
Donc on transforme toutes les variables categorielles en variable numérique et binaires.
Pour cela, on fait appel appel à la fonction dummyvars de la librairie caret dediée au
machine learning.

Passage variable categorielle ----------> variable numerique binaire"
# Dummyvars permet de faire la transformation
dummy_vars = dummyVars("~.",data = data_telecom |> select(-churn))
data_telecom_dummy = predict(dummy_vars, newdata = data_telecom |> select(-churn))

# Visualiser data_telecom_dummy
class(data_telecom_dummy) # matrix

# Joindre la variable cible 'churn' dans data_telecom_dummy
data_telecom_dummy = data_telecom_dummy |> as.data.frame() |>
  mutate(churn = data_telecom |> pull(churn))

data_telecom_dummy |> glimpse()




# PRETRAITEMENT DES DONNEES--------------------------------------------------------------------------
# * Decoupage de la base en jeu d'entrainement et de test  
# Création d'index de partition avec caret
#set.seed(99)
index = createDataPartition(data_telecom_dummy$churn, p = 0.8, list = FALSE)

# Set training 
train = data_telecom_dummy[index,]
dim(train)
prop.table(table(train$churn))# Même répartition (churn = 73 et 27%)

# Set test
test = data_telecom_dummy[-index,]
dim(test) 
prop.table(table(test$churn))# Même répartition (churn = 73 et 27%)




# * Standardisation ( Mise à l'echelle des données )---------------------------------------------------
# ------> Standardiser seulement le jeu d'entrainement
# Pour cela on fait appel à la fonction preprocess()
preprocess_value = preProcess(train, method = c("center", "scale"))
# Mise à l'echelle (standardisation) du 'jeu_entrainement'
train_scaled = predict(preprocess_value, train)

# Mise à l'echelle (standardisation) du 'jeu_test'
test_scaled = predict(preprocess_value, test)




# DONNEES deséquilibrés ------------------------------------------------------------------------------
# Réechantillonnage des classes : downsample et upsample 
# On note qu'il y'a un desequilibre entre ces 2 classes 
# Cela nous permet pas l'utilisation de certains algorithmes 
# Pour cela, nous procederons à l'equilibre des classes par 2 méthodes 
# Caret Downsample et Upsample
# En effet avec 'downsample ' on aura '387 oui et 387 non' et 
# Avec 'upsample' on aura '2280 oui et 2280 non'
# Downsample 


train_scaled_downSample = downSample(x = train_scaled |> select(-churn),
                                     y = as.factor(train_scaled |> pull(churn)),
                                     yname = "churn")
# Repartition des Classes après le downsample
table(train_scaled$churn, deparse.level = 2) # Au depart
table(train_scaled_downSample$churn,deparse.level = 2)

# -----> UpSample
train_scaled_upSample = upSample(x = train_scaled |> select(-churn),
                                 y = as.factor(train_scaled |> pull(churn)),
                                 yname = "churn")
# Repartition des classes après le upsample
table(train_scaled$churn, deparse.level = 2) # Au depart
table(train_scaled_upSample$churn,deparse.level = 2)



# 5.2  Modèle d'Artillérie lourdes en Machine Learning------------------------------------------------
"
XGBOOST : << Extreme Gradient Boosting Three >>
svm : << Support Vector Machine >>
"

# Méthode d'entrainement des modèles " TrainControl pour la courbe ROC"
Control_roc = trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 3,
                           summaryFunction  = twoClassSummary,
                           classProbs = TRUE,
                           savePredictions = TRUE,
                           search = "grid") # Recherche des Meilleurs hyperparamètres


# 1 **** XGB_Tree ----------------------------------------------------------------------------------
# Calcul parallèle
# Nombre de coeur de mon pc = 8
# detectCores()
cl = makeCluster(8, type = "SOCK") 
# Register cluster so that caret will know to train in parallèle
registerDoSNOW(cl)



# HyperParamètre Tuning : Search = 'grid'
xgbTree_grid = expand.grid(
  nrounds = 100 ,
  eta = c(0.01,0.1,0.3), 
  max_depth = c(2,3,5,10),
  gamma = c(0),
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1)
# Modèle X G B Tree 
tic()
#set.seed(99)
xgbTree = train(churn~. ,data = train_scaled,
                method = "xgbTree",
                trControl = Control_roc ,
                metric = "ROC",
                tuneGrid = xgbTree_grid,
                preprocess = NULL)

# Libérer les processeurs 
stopCluster(cl)
toc()# 301.36 sec 

# Affichage du modèle 
print(xgbTree)
xgbTree$bestTune
ggplot(xgbTree) 



# 2 SVM ------------------------------------------------------------------------------------------------------
tic()
#set.seed(99)
svm = train(churn~. ,data = train_scaled, 
            method = 'svmRadial',
            preProcess = NULL,
            trControl = Control_roc  , 
            tuneLength = 8,
            metric = 'ROC')
toc() # 332.73 sec 
print(svm)
ggplot(svm)


ROC = evalm(list(xgbTree,svm),
            gnames = c("Extreme Gradient Boosting","SVM"),
            plot = "r",
            title = "Courbe ROC",
            rlinethick = 0.8)



# Prédiction du churn avec le modèle "xgbTree"  
# Prédiction à partir du jeu de test
#set.seed(99)
test_x <- test_scaled %>% select(-churn)
test_y <- as.factor(test_scaled$churn)   # s'assurer que c'est bien un facteur
predicted <- predict(object = xgbTree, newdata = test_x)
table(predicted)






# Matrice de confusion -------------------------------------------------------------------------

conf_matrix <- confusionMatrix(
  data = predicted,        # prédictions
  reference = test_y,      # vraies valeurs
  positive = "Oui",        # classe positive
  mode = "everything"      # toutes les métriques
)
# Affichage
print(conf_matrix)




# Visualisation de la matrice de confusion ------------------------------------------------------------
# Fusion des  données observées et pred
truth_predicted <- data.frame(
  truth = test_y,
  predicted = predicted
)

# Créer la matrice de confusion (yardstick)
cm <- conf_mat(truth_predicted, truth, predicted)

# Heatmap de la matrice de confusion
autoplot(cm, type = "heatmap") +
  ggtitle("Matrice de confusion du modèle xgbTree")




# Les variables important de notre model
important_var = varImp(xgbTree, scale = F)
# Visualiser le variograamme avec ggplot
g = ggplot(important_var)+
  ggtitle("Les variables les plus importants du model xgbTree ")+
  theme_minimal()
# Affichage 
ggplotly(g)







 