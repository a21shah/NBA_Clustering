from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv('data/Seasons_Stats.csv')
print(df.shape)

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

df = filter_years(df)
df = cleaned_df(df)
df = fill_empty_values(df)
df.shape

# Taking only first position listed when multiple are listed
df['Pos'] = df['Pos'].str.split('-').str[0]

# Visualize
features = [x for x in df.columns if (x != 'Player') &  (x != 'Pos')]
print(df.shape)
print(features)

def df_PCA_yr(df, year):
    df_vis = df[df['Year'] == year]
    df_vis = df_vis.drop('Year', axis=1)
    features = [x for x in df_vis.columns if (x != 'Player') &  (x != 'Pos') & (x != 'Tm')]
    x = df_vis.loc[:, features].values
    y = df_vis.loc[:,['Pos']].values
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['pc1', 'pc2'])
    final = pd.concat([principalDf.reset_index(drop=True), df_vis['Pos'].reset_index(drop=True)], axis=1)
    print("Explained variance: ", pca.explained_variance_ratio_)
    return final,pca,features

def plot_PCA(df_PCA):
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)

    targets = df_PCA.Pos.unique() # list of positions (PCA groups)
    colors = ['r', 'g', 'b', 'm', 'c']
    for target, color in zip(targets,colors):
        indicesToKeep = df_PCA['Pos'] == target
        ax.scatter(df_PCA.loc[indicesToKeep, 'pc1']
                , df_PCA.loc[indicesToKeep, 'pc2']
                , c = color
                , s = 50)
    ax.legend(targets)
    ax.grid()
    return None

def plot_featImport_PCA(pcaObj, features):
     # feature importance for first dimension
    fimp1 = pd.DataFrame(data={'feat':features,'imp':pcaObj.components_[0]})
    fimp1 = fimp1.reindex(fimp1.imp.abs().sort_values(ascending = False).index)
    # print(fimp1)
    top_fimp1 = fimp1.nlargest(10,'imp')
    top_fimp1 = top_fimp1.iloc[::-1]
    top_fimp1 = top_fimp1.set_index("feat")
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


df_PCA, pca, features = df_PCA_yr(df, 2010)
plot_PCA(df_PCA)
plot_featImport_PCA(pca, features)

# K-Means Clustering
# Note: You may need to re-run the second block of code (reading the .csv) AND the data cleaning block before running everything after this point.

year = 2010
print(df.columns)

season_data = df[df['Year'] == year]
season_data = season_data.drop('Year', axis=1)
season_data.mean()
season_data.loc[:,"FG"].mean()

# Create a pairplot and a heatmap in order to see correlations between different columns
correlation = season_data[["AST", "FG", "TRB"]].corr()
sns.pairplot(season_data[["AST", "FG", "TRB"]])
sns.heatmap(correlation, annot=True)

# Use the K-Means Clustering Method to see which players are the most similar
model = KMeans(n_clusters=6, random_state=1)
columns_with_data = season_data._get_numeric_data().dropna(axis=1)
model.fit(columns_with_data)
data_labels = model.labels_

nba_pca = PCA(2)
plot_columns = nba_pca.fit_transform(columns_with_data)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=data_labels)

# How to see which cluster a specific player belongs to
player_one = columns_with_data.loc[season_data['Player'] == 'Stephen Curry',:]
player_two = columns_with_data.loc[season_data['Player'] == 'Kyle Lowry',:]

player_one_list = player_one.values.tolist()
player_two_list = player_two.values.tolist()

player_one_label = model.predict(player_one_list)
player_two_label = model.predict(player_two_list)

print(player_one_label)
print(player_two_label)

# Split data into 80% training and 20% testing for the columns you want to use. 
# The first column is the data you will be using to predict the value of the second column
x_train, x_test, y_train, y_test = train_test_split(season_data[["FG"]], season_data[["AST"]], test_size=0.2, random_state=42)

linear_regression = LinearRegression()
linear_regression.fit(x_train, y_train)
predictions = linear_regression.predict(x_test)

lin_reg_confidence = linear_regression.score(x_test, y_test)
print("Linear Regression confidence (R^2): ", lin_reg_confidence)
print("Mean Squared Error: ", mean_squared_error(y_test, predictions))

plt.show()