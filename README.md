# rbm_fmri

This is an active repo. This repo resurrects the benchmark rbm **DeepNet** (2014) and reimplements it on novel simulated fMRI data (simulated using **simTB**) and piglet resting-state fMRI data. The rbm weights are used to render weighted volumes representing the learned Intrinsic Network Spatial Maps. 

Actual data and analysis outputs are not included, since they are quite large files. 

The rbm learns the spatial components of the simulated data with high accuracy (see jnb's in rbm_simtb)

For the piglet data, the rbm outperforms ICA and currently performs competitively with Dictionary Learning.
- It differs from DL in that DL learns temporal correlation while the RBM learns spatial information, reading in a single time step with each training step (as opposed to the entire time series for one voxel)

Currently running TPE Bayesian optimization over the deepnet hyperparameters via `hyperopt` to improve RBM performance on piglet data. 
- hp ranges determined via Hinton's text: A Practical Guide to Training Restricted Boltzmann Machines

Future aims: utilize these experiments on human fMRI data in an unsupervised fashion; define a graph over the learned networks and incorporate with T1w mri and diffusion mri data; feed multimodal data into graph conv. network
