from sklearn.datasets import load_wine
from sklearn.datasets import load_iris
import numpy as np

wine_data=load_wine()
x_wine=wine_data.data
means=np.mean(x_wine,axis=0)
stds=np.std(x_wine,axis=0)

if np.allclose(means,0 ) and np.allclose(stds,1):
    print('already satisfied')
else:
    stnd=(x_wine-means)/stds
    print("Standarised")

iris_data=load_iris()
x_iris=iris_data.data
meaniris=np.mean(x_iris,axis=0)
stdsiris=np.std(x_iris,axis=0)


if np.allclose(meaniris,0 ) and np.allclose(stdsiris,1):
    print('already satisfied')
else:
    stnd=(x_iris-meaniris)/stdsiris
    print("Standarised")


