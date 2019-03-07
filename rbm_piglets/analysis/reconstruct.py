import sys
import nibabel as nib
import numpy.ma as ma
import numpy as np
from deepnet import util

## Loads 2D output from DL algorithm
# You may or may not need to use the second line, as it is just a transpose. I'm not sure of the format of your result.
#alpha_lasso=np.load('../../Output/3/DL_1000/3_100_0.15_alpha.npy')
#squished_embeddings=alpha_lasso.T

hp_str = sys.argv[1]
date_str = sys.argv[2]

model = util.ReadModel('./output_181220/piglets_rbm_layer1_LAST')
params = {}
for l in model.layer:
    for p in l.param:
        params['%s_%s' % (l.name, p.name)] = util.ParameterAsNumpy(p)
for e in model.edge:
    for p in e.param:
        params['%s_%s_%s' % (e.node1, e.node2, p.name)] = util.ParameterAsNumpy(p)

W = params['input_layer_hidden1_weight']
print("weights have shape {}".format(W.shape))
demixer = np.transpose(W)

mask = nib.load('./12_pigsMask.nii').get_data()
print("mask has shape {}".format(mask.shape))

x,y,z = mask.nonzero()
print("nonzero pixels: {}".format([x,y,z]))
print("x.shape[0]= {}".format(x.shape[0]))

#spatial_map = np.zeros((np.shape(mask)[0], np.shape(mask)[1], np.shape(mask)[2], squished_embeddings.shape[1]))
spatial_map = np.zeros((np.shape(mask)[0], np.shape(mask)[1], np.shape(mask)[2], W.shape[1]))
print("empty spatial_map shape {}".format(spatial_map.shape))

for i in range(x.shape[0]):
#    spatial_map[x[i], y[i], z[i], :] = squished_embeddings[i,:]
    spatial_map[x[i], y[i], z[i], :] = W[i,:]

img = nib.Nifti1Image(spatial_map, np.eye(4))
print("filled spatial_map shape {}".format(img.shape))

print("\n...")
nib.save(img, './rpw/rbm_piglets_weights_reconstructed_{}_{}.nii'.format(hp_str,date_str))
nib.save(img, './rpw/rbm_piglets_weights_reconstructed_last.nii')
print("image saved. exiting script.")
