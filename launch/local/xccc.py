import numpy as np

df = np.loadtxt('/home/hamid/w/DATA/274/pattern.csv', delimiter=',')

df[220:340, 280:500] = df[220:340, 500:720]

np.savetxt('/home/hamid/w/DATA/275/pattern.csv', df, delimiter=',')

