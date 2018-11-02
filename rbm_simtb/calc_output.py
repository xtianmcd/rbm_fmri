from deepnet import util
import numpy as np
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import sys

corrval = int(sys.argv[1])

X = np.load('simtb_full.npy')
model = util.ReadModel('./output/rbm_models/simtb_rbm_layer1_LAST')
params = {}
for l in model.layer:
    for p in l.param:
        params['%s_%s' % (l.name, p.name)] = util.ParameterAsNumpy(p)
for e in model.edge:
    for p in e.param:
        params['%s_%s_%s' % (e.node1, e.node2, p.name)] = util.ParameterAsNumpy(p)

W = params['input_layer_hidden1_weight']

demixer = np.transpose(W)

timecourses = np.matmul(demixer, np.transpose(X))

#print(timecourses.shape)

corrs = {}
sm=0
d=0
component_maps=[]

for spatial_map_tc in timecourses:
    cm=[]
    for simtb_tc in np.transpose(X):
        corr = pearsonr(spatial_map_tc, simtb_tc)
        cm.append(corr[0])
        if corr[0]>=corrval/100:
            corrs['rbm {} - voxel {}'.format(sm,d)]=[sm,d,corr[0],corr[1]]
        d+=1
    sm+=1
    cm = np.asarray(cm)
    cm=np.reshape(cm,(128,128))
    #sns.heatmap(cm)
    #plt.show()
    component_maps.append(cm)

with open('rbm_component_maps.csv', 'w') as mapscsv:
    mapswriter = csv.writer(mapscsv)
    for cmpnt in component_maps:
        for row in cmpnt:
            mapswriter.writerow(row)
        mapswriter.writerow('\n')
with open('rbm_component_dict_corr{}.csv'.format(corrval), 'w') as dictcsv:
    dictwriter = csv.writer(dictcsv)
    for pair in corrs:
        dictwriter.writerow(pair)
    
#print(corrs)
print(len(corrs))

