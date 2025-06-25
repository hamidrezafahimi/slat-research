import numpy as np

asd = np.array([[1,2,3], [1,2,3]])

np.savetxt('/home/hamid/sdf.csv', asd.astype(np.float64), delimiter=',')