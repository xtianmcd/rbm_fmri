import subprocess
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

def objective(params):
    model = open("rbm_model_l1_hp.pbtxt", "w")
    model.write(
    "name: 'simtb_rbm_layer1',\n\
model_type: DBM,\n\
hyperparams {{\n\
    base_epsilon: {}\n\
    epsilon_decay : {}\n\
    epsilon_decay_half_life : {}\n\
    initial_momentum : {}\n\
    final_momentum : {}\n\
    momentum_change_steps : {}\n\
    sparsity : false\n\
    sparsity_target : 0.2\n\
    sparsity_cost : 0.00999999977648\n\
    sparsity_damping : 0.9\n\
    dropout : {}\n\
    dropout_prob : {}\n\
    apply_weight_norm : {}\n\
    weight_norm : {}\n\
    apply_l2_decay: {}\n\
    l2_decay: {}\n\
    apply_l1_decay: {}\n\
    l1_decay: {}\n\
    activation: TANH\n\
    gibbs_steps: {}\n\
    start_step_up_cd_after: {}\n\
    step_up_cd_after:{}\n\
}}\n\
\n\
layer {{\n\
    name: 'input_layer'\n\
    dimensions: 16384\n\
    is_input: true\n\
    param {{\n\
        name: 'bias'\n\
        initialization: CONSTANT\n\
    }}\n\
    data_field {{\n\
        train: 'simtbData'\n\
    }}\n\
    loss_function: SQUARED_LOSS\n\
    hyperparams {{\n\
        sparsity : false\n\
        activation: LINEAR\n\n\
        apply_l2_decay: {}\n\
        apply_l1_decay: {}\n\
    }}\n\
    performance_stats {{\n\
        compute_error: true\n\
    }}\n\
    shape: {}\n\
    shape: {}\n\
}}\n\
\n\
layer {{\n\
    name: 'hidden1'\n\
    dimensions: {}\n\
    param {{\n\
        name: 'bias'\n\
        initialization: CONSTANT\n\
    }}\n\
    performance_stats {{\n\
        compute_sparsity: true\n\
    }}\n\
    hyperparams {{\n\
        activation: TANH\n\
        apply_l2_decay: {}\n\
        sparsity: false\n\
        apply_l1_decay: {}\n\
    }}\n\
}}\n\
\n\
edge {{\n\
    node1: 'input_layer'\n\
    node2: 'hidden1'\n\
    directed: false\n\
    param {{\n\
        name: 'weight'\n\
        initialization: DENSE_GAUSSIAN_SQRT_FAN_IN \n\
        sigma : 1.0\n\
    }}\n\
}}".format(params['base_epsilon'],params['epsilon_decay'],params['epsilon_decay_half_life'],params['initial_momentum'],params['final_momentum'],params['momentum_change_steps'],params['dropout'],params['dropout_prob'],params['apply_weight_norm'],params['weight_norm'],params['apply_l2_decay'],params['l2_decay'],params['apply_l1_decay'],params['l1_decay'],params['gibbs_steps'],params['start_step_up_cd_after'],params['step_up_cd_after'],params['vis_apply_l2_decay'],params['vis_apply_l1_decay'],params['shape'],params['shape'],params['hidden_dims'],params['hidd_apply_l2_decay'],params['hidd_apply_l1_decay'],params['wt_sigma'])
    )
    model.close()
    
    train_deepnet='python ../trainer.py'
    cmd = 'python ../trainer.py rbm_model_l1_hp.pbtxt train.pbtxt eval.pbtxt'
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    model_loss = float(output.split(':')[-1])
    print("model_loss = {}".format(model_loss))
    return model_loss

if __name__ == '__main__':

    space = {'base_epsilon': 0.008,
        'epsilon_decay' : 'NONE',
        'epsilon_decay_half_life' : 5000,
        'initial_momentum' : 0.5,
        'final_momentum' : 0.9,
        'momentum_change_steps' : 3000,
        'dropout' : 'true',
        'dropout_prob' : 0.5,
        'apply_weight_norm' : 'false',
        'weight_norm' : 3,
        'apply_l2_decay': 'false',
        'l2_decay': 0.001,
        'apply_l1_decay': 'true',
        'l1_decay': 0.1,
        'gibbs_steps': 2,
        'start_step_up_cd_after': 50000,
        'step_up_cd_after':10000,
        'vis_apply_l2_decay': 'false',
        'vis_apply_l1_decay': 'true',
        'shape': 128,
        'hidden_dims' : 64,
        'hidd_apply_l2_decay': 'false',
        'hidd_apply_l1_decay': 'true',
        'wt_sigma' : 1.0
    }

    """
    trials = Trials()
    best = fmin(objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)
"""

objective(space)
