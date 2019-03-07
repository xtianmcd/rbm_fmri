#!/bin/bash                                                                               
# Trains an rbm on simtb data                                                     
train_deepnet='python ../trainer.py'
${train_deepnet} rbm_model_l1_p.pbtxt train_p.pbtxt eval_p.pbtxt
