from deepnet import util

model = util.ReadModel('./output/rbm_models/simtb_rbm_layer1_LAST')
params = {}
for l in model.layer:
    for p in l.param:
        params['%s_%s' % (l.name, p.name)] = util.ParameterAsNumpy(p)
for e in model.edge:
    for p in e.param:
        params['%s_%s_%s' % (e.node1, e.node2, p.name)] = util.ParameterAsNumpy(p)
weights = params['input_layer_hidden1_weight']
print weights
