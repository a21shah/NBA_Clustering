import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# Visualization
from sklearn.manifold import TSNE
import seaborn as sns # tSNE visualization
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
# Clustering
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_samples, silhouette_score

############
# CLEANING #
############

def filter_years(df):
    # Filter data so only seasons from 2010 onwards are included
    df = df[df['Year'].notna()]
    df = df[df['Year'] >= 2010]
    df['Year'] = df['Year'].astype(int)
    return df

def cleaned_df(df):
    # Clean dataframe so only 1 season stat is included for every player per season
    years = df['Year'].unique()
    for year in years:
        df_year = df[df['Year'] == year]
        df_year = df_year[df_year.duplicated(subset='Player', keep=False)]
        common_rows = df['Unnamed: 0'].isin(df_year['Unnamed: 0'])
        df.drop(df[common_rows].index, inplace = True)
        df = df.append(df_year[df_year['Tm'] == 'TOT'])
    return df

def fill_empty_values(df):
    # Remove unneccesary columns and fill nulls with 0
    df.drop(columns=['Unnamed: 0', 'blanl', 'blank2'], inplace=True)
    df.drop(df[df['USG%'].isnull()].index, inplace=True)
    df.fillna(0, inplace=True)
    return df

def season_data(df, year):
    season_data = df[df['Year'] == year]
    season_data = season_data.drop('Year', axis=1)
    return season_data

df = pd.read_csv('data/Seasons_Stats.csv')
df = filter_years(df)
df = cleaned_df(df)
df = fill_empty_values(df)

# Taking only first position listed when multiple are listed
df['Pos'] = df['Pos'].str.split('-').str[0]

#######
# PCA #
#######

features = [x for x in df.columns if (x != 'Player') &  (x != 'Pos')]
print(features)

def plot_PCA(df, year):
    df_vis = df[df['Year'] == year]
    df_vis = df_vis.drop('Year', axis=1)
    features = [x for x in df_vis.columns if (x != 'Player') &  (x != 'Pos') & (x != 'Tm')]
    x = df_vis.loc[:, features].values
    y = df_vis.loc[:,['Pos']].values
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2, random_state=0)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['pc1', 'pc2'])
    final = pd.concat([principalDf.reset_index(drop=True), df_vis['Pos'].reset_index(drop=True)], axis=1)
    print("Explained variance: ", pca.explained_variance_ratio_)
    fig = plt.figure(1, figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)

    targets = final.Pos.unique() # list of positions (PCA groups)
    colors = ['r', 'g', 'b', 'm', 'c']
    for target, color in zip(targets,colors):
        indicesToKeep = final['Pos'] == target
        ax.scatter(final.loc[indicesToKeep, 'pc1']
                , final.loc[indicesToKeep, 'pc2']
                , c = color
                , s = 50)
    ax.legend(targets)
    ax.grid()
    return final,pca,features

def plot_featImport_PCA(pcaObj, features):
     # feature importance for first dimension
    fimp1 = pd.DataFrame(data={'feat':features,'imp':pcaObj.components_[0]})
    fimp1 = fimp1.reindex(fimp1.imp.abs().sort_values(ascending = False).index)
    top_fimp1 = fimp1.nlargest(10,'imp')
    top_fimp1 = top_fimp1.iloc[::-1]
    top_fimp1 = top_fimp1.set_index("feat")
    # plt.figure(2)
    ax = top_fimp1.plot.barh()
    ax.set_xlabel('Importance', fontsize = 15)
    ax.set_ylabel('Feature', fontsize = 15)
    ax.set_title('PCA Component 1 Feature Importance', fontsize = 20)

    # repeat for second dimension
    fimp2 = pd.DataFrame(data={'feat':features,'imp':pcaObj.components_[1]})
    fimp2 = fimp2.reindex(fimp2.imp.abs().sort_values(ascending = False).index)
    top_fimp2 = fimp2.nlargest(10,'imp')
    top_fimp2 = top_fimp2.iloc[::-1]
    top_fimp2 = top_fimp2.set_index("feat")
    ax = top_fimp2.plot.barh()
    ax.set_xlabel('Importance', fontsize = 15)
    ax.set_ylabel('Feature', fontsize = 15)
    ax.set_title('PCA Component 2 Feature Importance', fontsize = 20)

def feature_variance (df, year, percent):
    df_var = df[df['Year'] == year]
    df_var = df_var.drop('Year', axis=1)
    features = [x for x in df_var.columns if (x != 'Player') &  (x != 'Pos') & (x != 'Tm')]
    x = df_var.loc[:, features].values
    y = df_var.loc[:,['Pos']].values
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=percent)
    principalComponents = pca.fit_transform(x)
    perExplained = round(100*pca.explained_variance_ratio_.sum(), 2)
    print("{}% of features explained by {} features".format(perExplained, principalComponents.shape[1]))
    return principalComponents.shape[1] # num of features

df_PCA, pca, features = plot_PCA(df, 2010)
plot_featImport_PCA(pca, features)
feature_variance(df, 2010, 0.99)


######################
# K-Means Clustering #
######################

def plot_pairwise_scatter(dataframe, columns):
    sns.pairplot(dataframe[[columns[0], columns[1], columns[2]]])
    return
    
def plot_correlation_heatmap(dataframe, columns):
    plt.figure(5)
    correlation = dataframe[[columns[0], columns[1], columns[2]]].corr()
    sns.heatmap(correlation, annot=True)
    return
    
def get_numeric_data_only(dataframe):
    return dataframe._get_numeric_data().dropna(axis=1)

def get_PCA_with_clusters(dataframe):
    columns_with_numeric_data = get_numeric_data_only(dataframe)
    nba_pca = PCA(2) #put data into 2 dimensions
    return nba_pca.fit_transform(columns_with_numeric_data)
    
def plot_PCA_with_kmeans_clusters(dataframe, data_labels, model):
    plot_columns = get_PCA_with_clusters(dataframe)   
    pred_y = model.fit_predict(plot_columns)
    plt.figure(6)
    plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=model.labels_)
    plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1],s=30, c='red')
    plt.title('K-Means Clusters', fontsize=16)
    
def generate_kmeans_model(dataframe, k):
    model = KMeans(n_clusters=k, init='k-means++', max_iter=1000, n_init=10, random_state=0)
    columns_with_numeric_data = get_numeric_data_only(dataframe)
    model.fit(columns_with_numeric_data)
    return model
    
def get_kmeans_cluster_labels(kmeans_model):
    return kmeans_model.labels_

def get_player_label(dataframe, player_name, model):
    columns_with_numeric_data = get_numeric_data_only(dataframe)
    player = columns_with_numeric_data.loc[dataframe['Player'] == player_name,:]
    player_list = player.values.tolist()
    player_label = model.predict(player_list)
    return player_label

def plot_kmeans_Silh_Score(season_numeric, min_cluster, max_cluster):
    silhouette_scores = []
    for i in range(min_cluster, max_cluster):
       # k_means = KMeans(n_clusters=i, init='k-means++', max_iter=300,
       #             n_init=10, random_state=10)
        k_means = generate_kmeans_model(season_2010, i)
        #k_means.fit(get_numeric_data_only(season_2010))
        labels = get_kmeans_cluster_labels(k_means)
        silhouette_avg = silhouette_score(season_numeric, labels)
        silhouette_scores.append(silhouette_avg)
        #print("For n_clusters =", i, "The average silhouette_score is:", silhouette_avg)
    n_clusters = list(range(min_cluster, max_cluster))
    plt.figure(7)
    plt.plot(n_clusters, silhouette_scores, '-o')
    plt.grid()
    plt.xlabel('Number of Clusters', fontsize=15)
    plt.xticks(fontsize=12)
    plt.ylabel('Score', fontsize=15)
    plt.yticks(fontsize=12)
    plt.title('K-Means Silhouette Scores ', fontsize=16)


season_2010 = season_data(df, 2010)
season_2010.mean()

columns = ["AST", "FG", "TRB"]
plot_pairwise_scatter(season_2010, columns)
plot_correlation_heatmap(season_2010, columns)

kmeans_model = generate_kmeans_model(season_2010, 4)
labels = get_kmeans_cluster_labels(kmeans_model)
plot_PCA_with_kmeans_clusters(season_2010, labels, kmeans_model)
wcss = []

for i in range(1, 11):
    k_means = KMeans(n_clusters=i, init='k-means++', max_iter=300,
                    n_init=10, random_state=0)
    #plot_columns = get_PCA_with_clusters(season_data)   
    #pred_y = model.fit_predict(plot_columns)
    #k_means.fit_predict(plot_columns)
    k_means.fit(get_numeric_data_only(season_2010))
    wcss.append(k_means.inertia_)

plt.figure(8)
plt.plot(range(1,11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')

plot_kmeans_Silh_Score(get_numeric_data_only(season_2010), 2, 11)


# for i in range(1,11):
#     k_means = KMeans(n_clusters=i, init='k-means++', max_iter=300,
#                     n_init=10, random_state=10)
#     cluster_labels = k_means.fit_predict(get_numeric_data_only(season_2010))
#     silhouette_avg = silhouette_score(get_numeric_data_only(season_2010), cluster_labels)
#     sample_silhouette_values = silhouette_samples(get_numeric_data_only(season_2010), cluster_labels)

#     for j in range(i):
#         ith_cluster_silhouette_values = \
#             sample_silhouette_values[cluster_labels == i]
        
#         ith_cluster_silhouette_values.sort()

#         size_cluster_i = ith_cluster_silhouette_values.shape[0]
#         y_upper = y_lower + size_cluster_i


#######################
# Spectral Clustering #
#######################

def season_numeric_data(season_df):
    return season_df._get_numeric_data().dropna(axis=1)

def generate_spec_model(k):
    spec = SpectralClustering(n_clusters=k, affinity='nearest_neighbors',assign_labels='kmeans')
    return spec

def __get_PCA_with_clusters(season_numeric):
    nba_pca = PCA(2) #put data into 2 dimensions
    return nba_pca.fit_transform(season_numeric)

def get_labels(model, season_numeric):
    labels = model.fit_predict(season_numeric)
    return labels

def plot_Silh_Score(season_numeric, min_cluster, max_cluster):
    silhouette_scores = []
    for i in range(min_cluster, max_cluster):
        model = generate_spec_model(i)
        labels = get_labels(model, season_numeric)
        silhouette_avg = silhouette_score(season_numeric, labels)
        silhouette_scores.append(silhouette_avg)
        # print("For n_clusters =", i, "The average silhouette_score is:", silhouette_avg)
    n_clusters = list(range(min_cluster, max_cluster))
    plt.figure(9)
    plt.plot(n_clusters, silhouette_scores, '-o')
    plt.grid()
    plt.xlabel('Number of Clusters', fontsize=15)
    plt.xticks(fontsize=12)
    plt.ylabel('Score', fontsize=15)
    plt.yticks(fontsize=12)
    plt.title('Spectral Clustering Silhouette Scores ', fontsize=16)

def plot_PCA_with_clusters(labels, season_numeric):
    plot_columns = __get_PCA_with_clusters(season_numeric)
    plt.figure(10)
    plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=labels)
    plt.xlabel('PC1', fontsize=15)
    plt.xticks(fontsize=12)
    plt.ylabel('PC2', fontsize=15)
    plt.yticks(fontsize=12)
    plt.title('Data plotted into Clusters', fontsize=16)

# TODO Needs to be changed for Spectral Clustering (if we plan on using this part of it at least)
# def get_player_label(model, season_df, season_numeric, player_name):
#     player = season_numeric.loc[season_df['Player'] == player_name,:]
#     player_list = player.values.tolist()
#     player_label = model.predict(player_list)
#     return player_label

season_2010 = season_data(df, 2010)
season_2010_numeric = season_numeric_data(season_2010)
plot_Silh_Score(season_2010_numeric, 2, 11)

# 2 clusters doesn't really make sense, but 5 performs similarly so we'll use that
spec = generate_spec_model(5)
labels = get_labels(spec, season_2010_numeric)
plot_PCA_with_clusters(labels, season_2010_numeric)

plt.show()