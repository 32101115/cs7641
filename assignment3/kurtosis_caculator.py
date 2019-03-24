import sys
from numpy import median
from scipy.io import arff
from scipy.stats import kurtosis, kurtosistest

####################
#
#Optional: install seaborn through 'pip install seaborn' and uncomment related
#code to display the distribution figure
#
####################

#import seaborn as sns
#import matplotlib.pyplot as plt
#import pandas as pd

filename = sys.argv[1]
f = open(filename, 'r')
data, meta = arff.loadarff(f)
num_attr = len(data[0])
kurtosis_value = []
for i in range(0, num_attr - 1):
    attr_col = [row[i] for row in data]
    kurtosis_value.append(int(kurtosis(attr_col, fisher = False)))

row_sums = [sum(tuple(row)[0:-1]) for row in data]

#sns.set(color_codes=True)
#sns.distplot(row_sums);
#sns.plt.show()


print "Kurtosis values array: " + str(kurtosis_value) + "\n"
print "Kurtosis value of sum of all features: " + str(int(kurtosis(row_sums, fisher = False)))
