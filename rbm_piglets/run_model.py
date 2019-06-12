import subprocess
import datetime
import pandas as pd
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from tabulate import tabulate

def bash_cmd(cmd):
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    return output,error

def objective(params):
    now=datetime.datetime.now()
    date = now.year+now.month+now.day

    if params['epsilon_decay']==False:
        params

    model = open("rbm_model_l1_hp.pbtxt", "w")
    model.write(f\
    "name: 'simtb_rbm_layer1',\n\
    model_type: DBM,\n\
    hyperparams \{\n\
        \tbase_epsilon: {params['base_epsilon']}\n\
        \tepsilon_decay : {params['epsilon_choice']['epsilon_decay']}\n\
        \tepsilon_decay_half_life : {params['epsilon_choice']['epsilon_decay_half_life']}\n\
        \tinitial_momentum : {params['initial_momentum']}\n\
        \tfinal_momentum : {params['final_momentum']}\n\
        \tmomentum_change_steps : {params['momentum_change_steps']}\n\
        \tsparsity : false\n\
        \tdropout : {params['dropout_choice']['dropout']}\n\
        \tdropout_prob : {params['dropout_choice']['dropout_prob']}\n\
        \tapply_weight_norm : {params['wt_norm_choice']['apply_weight_norm']}\n\
        \tweight_norm : {params['wt_norm_choice']['weight_norm']}\n\
        \tapply_l2_decay: {params['l2_decay_choice']['apply_l2_decay']}\n\
        \tl2_decay: {params['l2_decay_choice']['l2_decay']}\n\
        \tapply_l1_decay: {params['l1_decay_choice']['apply_l1_decay']}\n\
        \tl1_decay: {params['l2_decay_choice']['l1_decay']}\n\
        \tactivation: TANH\n\
        \tgibbs_steps: {params['gibbs_steps']}\n\
        \tstart_step_up_cd_after: {params['start_step_up_cd_after']}\n\
        \tstep_up_cd_after:{params['step_up_cd_after']}\n\
    \}\n\
    \n\
    layer \{\n\
        \tname: 'input_layer'\n\
        \tdimensions: 16384\n\
        \tis_input: true\n\
        \tparam {\n\
            \t\tname: 'bias'\n\
            \t\tinitialization: {params['vis_bias_init_choice']['vis_bias_init']}\n\
            \t\tconstant: {params['vis_bias_init_choice']['vis_bias_cnst']}\n\
            \t\tsigma: {params['vis_bias_init_choice']['vis_bias_sigma']}\n\
        \t\}\n\
        \tdata_field \{\n\
            \t\ttrain: {params['train_data']}\n\
        \t\}\n\
        \tloss_function: SQUARED_LOSS\n\
        \thyperparams \{\n\
            \t\tsparsity : false\n\
            \t\tactivation: TANH\n\n\
            \t\tapply_l2_decay: {params['vis_apply_l2_decay']}\n\
            \t\tapply_l1_decay: {params['vis_apply_l1_decay']}\n\
        \t\}\n\
        \tperformance_stats \{\n\
            \t\tcompute_error: true\n\
        \t\}\n\
    \}\n\
    \n\
    layer \{\n\
        \tname: 'hidden1'\n\
        \tdimensions: {params['hidden_dims']}\n\
        \tparam \{\n\
            \t\tname: 'bias'\n\
            \t\tinitialization: {params['hidden_bias_init_choice']['hidden_bias_init']}\n\
            \t\tconstant: {params['hidden_bias_init_choice']['hidden_bias_cnst']}\n\
            \t\tsigma: {params['hidden_bias_init_choice']['hidd_bias_sigma']}\n\
        \t\}\n\
        \tperformance_stats \{\n\
            \t\tcompute_sparsity: true\n\
        \t\}\n\
        \thyperparams \{\n\
            \t\tactivation: TANH\n\
            \t\tapply_l2_decay: {params['hidd_apply_l2_decay']}\n\
            \t\tsparsity: false\n\
            \t\tapply_l1_decay: {params['hidd_apply_l1_decay']}\n\
        \t\}\n\
    \}\n\
    \n\
    edge \{\n\
        \tnode1: 'input_layer'\n\
        \tnode2: 'hidden1'\n\
        \tdirected: false\n\
        \tparam \{\n\
            \t\tname: 'weight'\n\
            \t\tinitialization: {params['wt_init_choice']['wt_init']} \n\
            \t\tconstant: {params['wt_init_choice'][wt_cnst]}\n\
            \t\tsigma : {params['wt_init_choice']['wt_sigma']}\n\
        \t\}\n\
    \}".format(,,,,,,,,,,,,,,,,,,,,,,,)\
    )
    model.close()

    train_deepnet='python ../trainer.py'
    train_cmd = 'python ../trainer.py rbm_model_l1_hp.pbtxt train.pbtxt eval.pbtxt'
    #process_train = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    #output_train, error_train = process.communicate()

    bash_cmd(train_cmd)
    bash_cmd('python reconstruct.py hp {}'.format(date))
    bash_cmd('matlab -nodisplay -nosplash -nodesktop -r\"run(\'/home/local/AIUGA/clm121/Documents/deepnet/deepnet/rbm_piglets/init_cmds.m\');run(\'/home/local/AIUGA/clm121/Documents/deepnet/deepnet/rbm_piglets/voxel_size_rbm.m\');run(\'/home/local/AIUGA/clm121/Documents/deepnet/deepnet/rbm_piglets/spm_normalize.m\');run(\'/home/local/AIUGA/clm121/Documents/deepnet/deepnet/rbm_piglets/stats.m\'); exit;\"')
    #with open('statsMaxs.txt','r') as stats:
    #    maxs[0]=stats.read()
    maxs= pd.read_csv('statsMaxs.txt',sep='\t',header=None)
    corrs = maxs.iloc[1]
    avg_corr = np.mean(corrs)

    # tab_results = tabulate(results, headers="keys", tablefmt="fancy_grid", floatfmt=".8f")
    # weights = model.get_weights()
    # # print(weights)
    # with open('../../output/hp_opt/weights.txt', 'a+') as model_summ:
    #     model_summ.write("model: {}\n\tweights:\n{}\n\tmodel_details:\n{}\n\tscore:\t{}".format(model, list(weights), tab_results, acc))
    #

    model_loss = 1-avg_corr

    with open('params_loss.txt','a') as params_loss:
        params_loss.write(f'Parameters: {params}\nLoss: {model_loss}\n\n')


    print("model_loss = {}".format(model_loss))

    return params,model_loss

if __name__ == '__main__':

    train_data = 'piglets_train'

    # in parameter.py of deepnet, looks like you can technically use l2, l1, and wt. norm (also tested experimentally), so I'm not making them mutually exclusive here

    space = {'base_epsilon'       : hp.choice('base_epsilon', [10**1,10**0,10**-1,10**-2,10**-3,10**-4]),
        'epsilon_choice'          : hp.choice('epsilon_choice', [{'epsilong_decay' :  'true', 'epsilon_decay_half_life' : hp.choice('epsilon_decay_half_life',[50,500,1000,5000,10000,50000]},{'epsilon_decay' : 'false', 'epsilon_decay_half_life' : 1000}],
        'initial_momentum'        : hp.quniform('initial_momentum', 0.0,0.9,0.1),
        'final_momentum'          : hp.quniform('final_momentum',   0.0,0.9,0.1),
        'momentum_change_steps'   : hp.choice('momentum_change_steps',[50,500,1000,5000,10000,50000]},
        'dropout_choice'          : hp.choice('dropout_choice', [{'dropout' : 'true', 'dropout_prob' : hp.quniform('dropout_prob', 0.0,0.9,0.1)}, {'dropout' : 'false', 'dropout_prob' : 0}]),
        'wt_norm_choice'          : hp.choice('wt_norm_choice', [{'apply_weight_norm' : 'true', 'weight_norm' : hp.choice('weight_norm' : [1,3,5,10,20,50,100])},{'apply_weight_norm' : 'false', 'weight_norm' : 1}]),
        'l2_decay_choice'         : hp.choice('l2_decay_choice', [{'apply_l2_decay' : 'true', 'l2_decay' : hp.choice('l2_decay', 0.00001,0.0001,0.001,0.01,0.1)},{'apply_l2_decay' : 'false', 'l2_decay' : 0.01}])
        'l1_decay_choice'         : hp.choice('l1_decay_choice', [{'apply_l1_decay' : 'true', 'l1_decay' : hp.choice('l1_decay', 0.00001,0.0001,0.001,0.01,0.1)},{'apply_l1_decay' : 'false', 'l1_decay' : 0.01}])
        'gibbs_steps'             : hp.choice('gibbs_steps', [1,3,5,10,20,50]),
        'start_step_up_cd_after'  : hp.choice('start_step_up_cd_after', [50,500,1000,5000,10000,50000]),
        'step_up_cd_after'        : hp.choice('step_up_cd_after', [50,500,1000,5000,10000,50000]),
        'train_data'              : train_data,
        'vis_bias_init_choice'    : hp.choice('vis_bias_init_choice', [{'vis_bias_init' : 'DENSE_GAUSSIAN','vis_bias_cnst' : 0.01, 'vis_bias_sigma' : hp.choice('vis_bias_sigma', [0.001,0.01,0.1])},{'vis_bias_init' : 'SPARSE_GAUSSIAN','vis_bias_cnst' : 0.01, 'vis_bias_sigma' : hp.choice('vis_bias_sigma', [0.001,0.01,0.1])},\
                                                                        {'vis_bias_init' : 'CONSTANT','vis_bias_cnst' : hp.choice('hidden_bias_cnst',[-4,-0.1,0,0.001,0.01,0.1]), 'vis_bias_sigma' : 0.001)},{'vis_bias_init' : 'DENSE_GAUSSIAN_SQRT_FAN_IN','vis_bias_cnst' : 0.01, 'vis_bias_sigma' : hp.choice('vis_bias_sigma', [0.001,0.01,0.1])},\
                                                                        {'vis_bias_init' : 'DENSE_UNIFORM','vis_bias_cnst' : 0.01, 'vis_bias_sigma' : hp.choice('vis_bias_sigma', [0.001,0.01,0.1])},{'vis_bias_init' : 'DENSE_UNIFORM_SQRT_FAN_IN','vis_bias_cnst' : 0.01, 'vis_bias_sigma' : hp.choice('vis_bias_sigma', [0.001,0.01,0.1])}]),
        'vis_apply_l2_decay'      : hp.choice('vis_apply_l2_decay', ['true','false']),
        'vis_apply_l1_decay'      : hp.choice('vis_apply_l2_decay', ['true','false']),
        'hidden_init_cnst'        : hp.quniform('initial_momentum', 0.0,0.9,0.1),
        'hidden_bias_init_choice' : hp.choice('hidden_bias_init_choice', [{'hidden_bias_init' : 'DENSE_GAUSSIAN','hidden_bias_cnst' : 0, 'hidd_bias_sigma' : hp.choice('hidd_bias_sigma', [0.001,0.01,0.1])},{'hidden_bias_init' : 'SPARSE_GAUSSIAN','hidden_bias_cnst' : 0, 'hidd_bias_sigma' : hp.choice('hidd_bias_sigma', [0.001,0.01,0.1])},\
                                                                            {'hidden_bias_init' : 'CONSTANT','hidden_bias_cnst' : hp.choice('hidden_bias_cnst',[-4,-0.1,0,0.001,0.01,0.1]), 'hidd_bias_sigma' : 0.001},{'hidden_bias_init' : 'DENSE_GAUSSIAN_SQRT_FAN_IN','hidden_bias_cnst' : 0, 'hidd_bias_sigma' : hp.choice('hidd_bias_sigma', [0.001,0.01,0.1])},\
                                                                            {'hidden_bias_init' : 'DENSE_UNIFORM','hidden_bias_cnst' : 0, 'hidd_bias_sigma' : hp.choice('hidd_bias_sigma', [0.001,0.01,0.1])},{'hidden_bias_init' 'DENSE_UNIFORM_SQRT_FAN_IN','hidden_bias_cnst' : 0, 'hidd_bias_sigma' : hp.choice('hidd_bias_sigma', [0.001,0.01,0.1])}])
        'hidden_dims'             : hp.choice('hidden_dims',[6,40,64,128]),
        'hidd_apply_l2_decay'     : hp.choice('hidd_apply_l2_decay', ['true','false']),
        'hidd_apply_l1_decay'     : hp.choice('hidd_apply_l1_decay', ['true','false']),
        'wt_init_choice'          : hp.choice('wt_init_choice', [{'wt_init' : 'DENSE_GAUSSIAN','wt_cnst' : 0.01, 'wt_sigma' : hp.choice('wt_sigma', [0.001,0.01,0.1])},{'wt_init' : 'SPARSE_GAUSSIAN','wt_cnst' : 0.01, 'wt_sigma' : hp.choice('wt_sigma', [0.001,0.01,0.1])},\
                                                                    {'wt_init' : 'CONSTANT','wt_cnst' : hp.choice('wt_cnst',[-4,-0.1,0,0.001,0.01,0.1]))}, 'wt_sigma' : 0.001,{'wt_init' : 'DENSE_GAUSSIAN_SQRT_FAN_IN','wt_cnst' : 0.01, 'wt_sigma' : hp.choice('wt_sigma', [0.001,0.01,0.1])},\
                                                                    {'wt_init' : 'DENSE_UNIFORM','wt_cnst' : 0.01, 'wt_sigma' : hp.choice('wt_sigma', [0.001,0.01,0.1])},{'wt_init' : 'DENSE_UNIFORM_SQRT_FAN_IN','wt_cnst' : 0.01, 'wt_sigma' : hp.choice('wt_sigma', [0.001,0.01,0.1])}]),
    }

    # SCALE THE DATA!

    trials = Trials()
    best = fmin(objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)

    # objective(space)    
