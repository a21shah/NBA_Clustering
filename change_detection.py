import pandas as pd
import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt

#Changepoint detection with the Binary Segmentation search method
def plot_changePoint(array):
    model = ""  
    algo = rpt.Binseg(model=model).fit(array)
    my_bkps = algo.predict(n_bkps=1)
    # show results
    rpt.show.display(array, my_bkps, figsize=(10, 6))
    plt.title('Change Point Detection: Binary Segmentation Search Method')
    plt.show()
    return None

def plotAll(plotList):
    for item in plotList:
        plot_changePoint(item)
    return None

kMeans_performance = {'clustNum': [2,2,2,2,2,2,2], 
                    'silhouette': [0.5678, 0.5588,0.5676,0.5777,0.5661,0.5690,0.5526]}  
kMeans_per_df = pd.DataFrame(kMeans_performance)

spectral_performance = {'clustNum': [3,4,4,2,3,2,2], 
                    'silhouette': [0.4057, 0.3787, 0.3747, 0.3901, 0.3963, 20,40]}  
spectral_per_df = pd.DataFrame(spectral_performance)

kMeans_clust = np.array(kMeans_per_df['clustNum'])
kMeans_sil = np.array(kMeans_per_df['silhouette'])
spectral_clust = np.array(spectral_per_df['clustNum'])
spectral_sil = np.array(spectral_per_df['silhouette'])

plotList = [kMeans_clust, kMeans_sil, spectral_clust, spectral_sil]
plotAll(plotList)