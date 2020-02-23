# -*- coding: utf-8 -*-

import pandas as  pd
import numpy  as np
import matplotlib.pyplot  as plt


uci_data=pd.read_csv('heart.csv')
train_v=pd.read_csv('train_values.csv')
train_l=pd.read_csv('train_labels.csv')
test_v=pd.read_csv('test_values.csv')

uci_arr=np.array(uci_data)

print(uci_data.loc[uci_data['age']==59 | uci_data['target']==1])