import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def fitData(data, degrees):
    for col, deg in zip(data, degrees):
        x = np.array([1,2,3,4,5,6,7])
        y = col
        p = np.poly1d(np.polyfit(x, y, deg))
        t = np.linspace(1, 7, 200)
        plt.plot(x, y, 'o', t, p(t), '-')
        plt.show()
    return None


kMeans_performance = {'clustNum': [2,2,2,2,2,2,2], 
                    'silhouette': [0.5678, 0.5588,0.5676,0.5777,0.5661,0.5690,0.5526]}  
kMeans_per_df = pd.DataFrame(kMeans_performance)

spectral_performance = {'clustNum': [3,4,4,2,3,2,2], 
                    'silhouette': [0.4057, 0.3787, 0.3747, 0.3901, 0.3963, 0.4202,0.3498]}  
spectral_per_df = pd.DataFrame(spectral_performance)

kMeans_clust = np.array(kMeans_per_df['clustNum'])
kMeans_sil = np.array(kMeans_per_df['silhouette'])
spectral_clust = np.array(spectral_per_df['clustNum'])
spectral_sil = np.array(spectral_per_df['silhouette'])

plotList = [kMeans_clust, kMeans_sil, spectral_clust, spectral_sil]
fitData(plotList, [2, 3, 2, 3])





