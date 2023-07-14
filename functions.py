import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import collections
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr, kendalltau, kruskal, chi2_contingency, shapiro, anderson, kstest, normaltest
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import axes3d
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV,learning_curve
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_log_error,confusion_matrix
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor


''' Notebook nettoyage '''

def data_duplicated(df):
    '''Retourne le nombres de lignes identiques.'''
    return df.duplicated(keep=False).sum()

def row_duplicated(df,col):
    '''Retourne le nombre de doublons de la variables col.'''
    return df.duplicated(subset = col, keep='first').sum()

def drop_lignes(df,index):
    '''Supprime les lignes des index donnés en argument'''
    df.drop(index, axis=0, inplace = True, errors = 'ignore')
    print('Suppression effectuée')
    
def missing_cells(df):
    '''Calcule le nombre de cellules manquantes sur le data set total'''
    return df.isna().sum().sum()

def missing_cells_perc(df):
    '''Calcule le pourcentage de cellules manquantes sur le data set total'''
    return df.isna().sum().sum()/(df.size)
    
def missing_general(df):
    '''Donne un aperçu général du nombre de données manquantes dans le data frame'''
    print('Nombre total de cellules manquantes :',missing_cells(df))
    print('Nombre de cellules manquantes en % : {:.2%}'.format(missing_cells_perc(df)))
    
def valeurs_manquantes(df):
    '''Prend un data frame en entrée et créer en sortie un dataframe contenant le nombre de valeurs manquantes
    et leur pourcentage pour chaque variables. '''
    tab_missing = pd.DataFrame(columns = ['Variable', 'Missing values', 'Missing (%)'])
    tab_missing['Variable'] = df.columns
    missing_val = list()
    missing_perc = list()
    
    for var in df.columns:
        nb_miss = missing_cells(df[var])
        missing_val.append(nb_miss)
        perc_miss = missing_cells_perc(df[var])
        missing_perc.append(perc_miss)
        
    tab_missing['Missing values'] = list(missing_val)
    tab_missing['Missing (%)'] = list(missing_perc)
    return tab_missing

def bar_missing(df):
    '''Affiche le barplot présentant le nombre de données présentes par variable.'''
    msno.bar(df)
    plt.title('Nombre de données présentes par variable', size=15)
    plt.show()
    
def barplot_missing(df):
    '''Affiche le barplot présentant le pourcentage de données manquantes par variable.'''
    proportion_nan = df.isna().sum().divide(df.shape[0]/100).sort_values(ascending=False)
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 30))
    ax = sns.barplot(y = proportion_nan.index, x=proportion_nan.values)
    plt.title('Pourcentage de données manquantes par variable', size=15)
    plt.show()
    
def drop_columns_empty(df,lim):
    '''Prend en entrée un data frame et un seuil de remplissage de données.
    Supprime chaque variable ayant un pourcentage de données manquantes supérieur à celui renseigné. 
    Donne en sortie le data frame filtré avec les colonnes à garder.'''
    
    tab = valeurs_manquantes(df)
    columns_keep = list()
    for row in tab.iterrows():
        if float(row[1]['Missing (%)'])>float(lim):
            print('Suppression de la variable {} avec % de valeurs manquantes {}'.format(row[1]['Variable'],
                                                                                         round(float(row[1]['Missing (%)']),2)))
            
        else :
            columns_keep.append(row[1]['Variable'])
    
    return df[columns_keep]   


def boxplot(df,ylim):
    ''' Affiche une fenêtre contenant tous les boxplots des variables sélectionnées'''
    fig = plt.figure(figsize=(20, 15))
    ax = plt.axes()
    plt.xticks(rotation=90)
    ax.set_ylim(ylim)
    sns.boxplot(data=df)
    plt.title('Boxplot des variables', size=15)
    
def multi_boxplot(df):
    ''' Affiche indépendamment tous les boxplots des variables sélectionnées'''
    fig, axs = plt.subplots(4,3,figsize=(20,20))
    axs = axs.ravel()
    
    for i,col in enumerate(df.columns):
        sns.boxplot(x=df[col], ax=axs[i])
    fig.suptitle('Boxplot pour chaque variable quantitative')
    plt.show()
        
def distribution(df,colonnes):
    ''' Affiche les histogrammes pour chaque variable renseignée.'''
    fig, axs = plt.subplots(4,3,figsize=(20,20))
    axs = axs.ravel()

    for i, col in enumerate(colonnes):
        sns.histplot(data=df, x=col, bins=30, kde=True, ax=axs[i])
        
    fig.suptitle('Distribution pour chaque variable quantitative')
    plt.show()
    
def qq_plot(df,colonnes):
    ''' Affiche le diagramme quantile-quantile entre chacune des variables renseignées et une loi normale '''
    fig = plt.figure(figsize=(15,15))
    
    for i,col in enumerate(colonnes,1):
        ax = fig.add_subplot(5,2,i)
        sm.qqplot(df[col], fit=True, line="45", ax=ax)
        ax.set_title("qq-plot entre la variable {} et une loi normale".format(col))
        
    plt.tight_layout(pad = 4)
    fig.suptitle("Diagramme quantile-quantile")
    plt.show()
        
def test_normalite(df, colonnes,level):
    ''' Calcul les différents tests de normalité pour chacune des variables passées en paramètres'''
    for col in colonnes:
        print("Tests de normalité pour la variable {}.".format(col))
        tests = [shapiro, anderson, normaltest, kstest]
        index = ['Shapiro Wilk','Anderson-Darling',"K2 de D'Agostino",'Kolmogorov-Smirnov']
        tab_result = pd.DataFrame(columns=['Stat','p-value','Resultat'], index = index)
    
        for i, fc in enumerate(tests):
            if fc==anderson:
                result = fc(df[col])
                tab_result.loc[index[i],'Stat'] = result.statistic
                if result.statistic < result.critical_values[2]:
                    tab_result.loc[index[i],'Resultat'] = 'H0'
                if result.statistic > result.critical_values[2]:
                    tab_result.loc[index[i],'Resultat'] = 'H1'
                    
            elif fc==kstest:
                stat, p = fc(df[col],cdf='norm')
                tab_result.loc[index[i],'Stat'] = stat
                tab_result.loc[index[i],'p-value'] = p
                if p < level:
                    tab_result.loc[index[i],'Resultat'] = 'H1'
                if p > level:
                    tab_result.loc[index[i],'Resultat'] = 'H0'
            
            else :
                stat, p = fc(df[col])
                tab_result.loc[index[i],'Stat'] = stat
                tab_result.loc[index[i],'p-value'] = p
                if p < level:
                    tab_result.loc[index[i],'Resultat'] = 'H1'
                if p > level:
                    tab_result.loc[index[i],'Resultat'] = 'H0'
    
        print(tab_result)
        print("-"*70)

def box_hist(df,col):
    fig, axs = plt.subplots(1,2,figsize=(20,20))
    sns.boxplot(data=df, y=col,ax=axs[0])
    sns.histplot(data=df, x=col, bins=30, kde=True, ax=axs[1]) 
    fig.suptitle('Boxplot et distribution')
    plt.show()

def bar_plot(df,colonnes):
    ''' Affiche les bar plots pour chaque variable renseignée.'''
    fig = plt.figure(figsize=(40,80))
    for i, col in enumerate(colonnes,1):
        ax = fig.add_subplot(7,2,i)
        count = df[col].value_counts()
        count.plot(kind="bar", ax=ax)
        plt.xticks(rotation=90, ha='right', fontsize = 20)
        ax.set_title(col, fontsize = 20)
    plt.tight_layout(pad = 2)
    plt.show()
    
def pie_plot(df,colonnes):
    '''Affiche un pie plot présentant la répartition de la variable renseignée.'''
    for col in colonnes :
        labels = list(df[col].value_counts().sort_index().index.astype(str))
        count = df[col].value_counts().sort_index()
        
        plt.figure(figsize=(10, 10))
        plt.pie(count,autopct='%1.2f%%')
        plt.title('Répartition de {}'.format(col), size = 20)
        plt.legend(labels)
        plt.show()

def scatter_plot(df,colonnes,var_comparaison, largeur, longueur):
    ''' Affiche le scatter plot des variables quantitatives.'''
    fig = plt.figure(figsize=(15,15))
    for i,col in enumerate(colonnes,1):
        X = df[[var_comparaison]]
        Y = df[col]
        X = X.copy()
        X['intercept'] = 1.
        result = sm.OLS(Y, X).fit()
        a,b = result.params[var_comparaison],result.params['intercept']
        equa = "y = " + str(round(a,2)) + " x + " + str(round(b,0))

        ax = fig.add_subplot(longueur,largeur,i)
        plt.scatter(x=df[var_comparaison], y=df[col])        
        plt.plot(range(-15,41),[a*x+b for x in range(-15,41)],label=equa,color='red')
        ax.set_xlabel(xlabel=var_comparaison)
        ax.set_ylabel(ylabel=col)
        plt.legend()
    plt.tight_layout(pad = 4)
    fig.suptitle("Scatter plot des variables quantitatives")
    plt.show()
    
def heat_map(df_corr):
    '''Affiche la heatmap '''
    plt.figure(figsize=(15,10))
    sns.heatmap(df_corr, annot=True, linewidth=.5)
    plt.title("Heatmap")

def tests_corr(df,colonnes,var_comparaison):
    ''' Calcul les différents tests de corrélation pour chacun des couples de variables passés en paramètres'''
    for col in colonnes:
        print("Tests de corrélation pour la variable {} par rapport à la variable {}.".format(col,var_comparaison))
        tests = [pearsonr, spearmanr, kendalltau]
        index = ['Pearson', 'Spearman', 'Kendall']
        tab_result = pd.DataFrame(columns=['Stat','p-value'], index = index)
    
        for i, fc in enumerate(tests):
            stat, p = fc(df[col],df[var_comparaison])
            tab_result.loc[index[i],'Stat'] = stat
            tab_result.loc[index[i],'p-value'] = p
        display(tab_result)
        print("-"*100)
        
def boxplot_relation(df,colonnes,var_comparaison,longueur,largeur, ordre=None,outliers=True,option=False):
    '''Affiche les boxplot des colonnes en fonctions de var_comparaison.'''
    fig = plt.figure(figsize=(20,30))
    for i,col in enumerate(colonnes,1):
        ax = fig.add_subplot(longueur,largeur,i)
        sns.boxplot(x=df[var_comparaison],y=df[col], ax=ax, order=ordre, showfliers = outliers)
        if option:
            plt.xticks(rotation=90, ha='right')
    fig.suptitle('Boxplot de chaque target en fonction de {}'.format(var_comparaison))
    plt.tight_layout(pad = 4)
    plt.show()   
        
def contingence_tab(df,var1,var2):
    data_crosstab = pd.crosstab(df[var1],df[var2], margins = False)
    return data_crosstab

def test_chi2(df, var1, var2, alpha=0.05):
    '''Test de Chi-2 pour 2 variables qualitatives.'''
    print("Test d'indépendance Chi-2 entre {} et {}".format(var1,var2))
    tab_cont = contingence_tab(df, var1, var2)
    results = chi2_contingency(tab_cont)
    print("stat = {}\np-value = {}\ndegrees of freedom = {}".format(results[0], results[1], results[2]))
    if results[1] <= alpha:
        print('Variables non indépendantes (H0 rejetée) car p = {} <= alpha = {}'.format(results[1], alpha))
    else:
        print('H0 non rejetée car p = {} > alpha = {}'.format(results[1], alpha))
    print("-"*70)
    
    
'''Notebook de modélisation'''

def cross_valid(X,y,preprocessor):
    '''cross validation de plusieurs modèles'''
    
    liste = [DummyRegressor(strategy='mean'),
             LinearRegression(),
             Ridge(),
             Lasso(),
             ElasticNet(),
             RandomForestRegressor(random_state=0),
             KNeighborsRegressor(),
             SVR(),
             XGBRegressor()
        
            ]

    #RMSLE = []
    RMSE = []
    R2 = []
    MAE = []
    MedAE = []
    FIT_TIME = []
    SCORE_TIME = []

    for model in liste:
    
        pipe = Pipeline(steps=[('preprocessor', preprocessor),('model', model)])

        score = cross_validate(pipe,X,y,cv=5,
                               scoring=(
                                   #'neg_mean_squared_log_error',
                                   'neg_root_mean_squared_error','r2','neg_mean_absolute_error', 
                                   'neg_median_absolute_error'))
    
        #rmsle = score['test_neg_mean_squared_log_error'].mean()
        rmse = score['test_neg_root_mean_squared_error'].mean()
        r2 = score['test_r2'].mean()
        mae = score['test_neg_mean_absolute_error'].mean()
        medae = score['test_neg_median_absolute_error'].mean()
        fit_time = score['fit_time'].mean()
        score_time = score['score_time'].mean()

        
        #RMSLE.append(-rmsle)
        RMSE.append(-rmse)
        R2.append(r2)
        MAE.append(-mae)
        MedAE.append(-medae)
        FIT_TIME.append(fit_time)
        SCORE_TIME.append(score_time)
    
    resultats = pd.concat([
        #pd.DataFrame(RMSLE),
        pd.DataFrame(RMSE),pd.DataFrame(R2),pd.DataFrame(MAE),pd.DataFrame(MedAE),
                               pd.DataFrame(FIT_TIME),pd.DataFrame(SCORE_TIME)],axis=1) 
    resultats.columns=[
        #'RMSLE',
        'RMSE','R2','MAE', 'MedAE','FIT_TIME','SCORE_TIME']
    resultats = resultats.rename(index = {0:'dum',1:'lr',2:'ridge',3:'lasso',4:'ElasticNet',5:'RandomForest',
                                           6:'KNR', 7:'SVR',8:'XGBR'})

    return resultats


def metrics(grid):
    res = pd.DataFrame(grid['grid_search'].cv_results_).sort_values('rank_test_r2')
    print('Meilleurs paramètres',grid['grid_search'].best_params_)
    print('Meilleur score RMSLE :',np.mean(res[res['rank_test_neg_mean_squared_log_error']==1]['mean_test_neg_mean_squared_log_error']))
    print('Meilleur score RMSE :',np.mean(res[res['rank_test_neg_root_mean_squared_error']==1]['mean_test_neg_root_mean_squared_error']))
    print('Meilleur score R2 :',grid['grid_search'].best_score_)
    print("Résultats des meilleurs paramètres :\n")
    display(res.loc[res['params']==grid['grid_search'].best_params_])
    
    return res

def bar_plot_compare(df,colonnes,neg):
    ''' Affiche les bar plots pour chaque variable renseignée.'''
    fig = plt.figure(figsize=(20,20))
    for i, col in enumerate(colonnes,1):
        ax = fig.add_subplot(4,2,i)
        if neg == True :
            sns.barplot(x = df.index,
                        y = -1*df[col])
        else :
            sns.barplot(x = df.index,
                        y = df[col])
        ax.set_yscale('log')
        ax.set_title(col)
    plt.tight_layout(pad = 4)
    fig.suptitle('Comparaison des performances des modèles')
    plt.show()

def metrics_pred(y_true, y_pred):
    '''Calcule les métriques'''
    rmsle = mean_squared_log_error(y_true= y_true, y_pred=y_pred)
    r2 =  r2_score(y_true= y_true, y_pred=y_pred)
    res = pd.DataFrame({'RMSLE': rmsle,'R2':r2},index=[0])
    return res

def plot_predict(y_true, y_pred):
    '''Fonction d'affichage des valeurs prédites en fonction des valeurs réelles.'''
    plt.figure(figsize=(10,7))
    plt.scatter(y_true, y_pred)
    plt.plot(y_true, y_true, color='red')
    plt.xlabel('Valeurs réelles')
    plt.ylabel('Valeurs prédites')
    plt.title('Valeurs prédites en fonction des valeurs réélles')
    plt.show()

def plot_pred_class(y_true, y_pred, cat):
    '''Fonction d'affichage des valeurs prédites en fonction des valeurs réelles et visualisation de la catégorie.'''
    
    plt.figure(figsize=(10,7))
    sns.scatterplot(y_true, y_pred, hue= X_test[cat])
    plt.plot(y_true, y_true, color='black')
    plt.legend(bbox_to_anchor=(1,1))
    plt.xlabel('Valeurs réelles')
    plt.ylabel('Valeurs prédites')
    plt.title('Valeurs prédites en fonction des valeurs réélles et du type de bâtiment')
    plt.show()

def plot_learning_curve(estimator, X, y):
    '''Affiche la learning curve du modèle renseigné en paramètres.'''
    train_sizes, train_scores, test_scores = learning_curve(estimator=estimator, X=X, y=y, cv=5, 
                                                            train_sizes=np.linspace(0.1, 1.0, 10), n_jobs= -1)
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training Score')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean, color='green', marker='+', markersize=5, linestyle='--', label='Validation Score')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
    plt.title('Learning Curve')
    plt.xlabel('Training Data Size')
    plt.ylabel('Performance R2 Score')
    plt.grid()
    plt.legend(loc='lower right')
    plt.show()

def plot_error(y_true, y_pred):
    '''Affiche un histogramme des erreurs du modèle.'''
    diff = (y_true - y_pred)
 
    sns.histplot(x=diff, bins=70, kde=True)
        
    plt.xlabel('Erreur')
    plt.title('Distribution des erreurs')
    plt.show()
    
    
def knn_imputer(df):
    scaler = StandardScaler().fit(df)
    scaled = scaler.transform(df)

    imputer = KNNImputer(n_neighbors=5)

    knn = imputer.fit_transform(scaled)
    filled_data = scaler.inverse_transform(knn)

    df_filled = pd.DataFrame(filled_data, columns=df.columns, index=df.index)
    
    return df_filled