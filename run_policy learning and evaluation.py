import OPL
import OPE
import DGP
import utils
import all_obj_fun
from scipy.optimize import minimize, minimize_scalar
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
import argparse
import math
import os
import joblib
parser = argparse.ArgumentParser(description="Credit Allocation")
parser.add_argument("-num_train", type=int, default=500)
parser.add_argument("-num_test", type=int, default=1000)
parser.add_argument("-num_cov", type=int, default=10)
parser.add_argument("-num_bins", type=int, default=5)
parser.add_argument("-max_iter", type=int, default=1000)
parser.add_argument("-SAMP_N", type=int, default=1)  # sample splitting
parser.add_argument("-amb_radius", type=float, default=0.2)
parser.add_argument("-norm_met", type=str, default="normalized", choices=["normalized", "unnormalized"])
parser.add_argument("-treat_met", type=str, default="continuous", choices=["continuous", "discrete"])
parser.add_argument("-robust_met", type=str, default="robust", choices=["robust", "non-robust"])
parser.add_argument("-kernel_met", type=str, default="epanechnikov", choices=["epanechnikov", "gaussian"])
parser.add_argument("-propensity_met", type=str, default="unknown", choices=["unknown", "known"])
parser.add_argument("-p_norm_bds", type=float, default=2)
parser.add_argument("-p_norm", default=np.inf)
parser.add_argument("-threshold", type=float, default=0.001)
parser.add_argument("-alpha_res", type=float, default=1e-4)
parser.add_argument("-init_alpha", type=float, default=5)
parser.add_argument("-alpha_update", type=str, default="minimize", choices=["update_one", "minimize"])

args = parser.parse_args()
if __name__ == '__main__':
    M = 100
    n_list = [(1000,2000)]
    policy_list = [('conti-unknown', 'robust'), ('conti-unknown', 'non-robust')]
    if not os.path.isdir('./result'):
        os.mkdir('./result')
    for (num_train, num_test) in n_list:
        load_data = joblib.load('./data/'+str(num_train)+'_'+str(num_test)+'_0-'+str(M))


        '''policy learning'''
        if not os.path.isdir('./result/'+str(num_train)+'_'+str(num_test)):
            os.mkdir('./result/'+str(num_train)+'_'+str(num_test))
        for (treat_met, robust_met) in policy_list:
            if not os.path.isdir('./result/'+str(num_train)+'_'+str(num_test)+ '/train_'+ treat_met+'_'+robust_met):
                os.mkdir('./result/'+str(num_train)+'_'+str(num_test)+ '/train_'+ treat_met+'_'+robust_met)
            for i_exp in range(0, M):
                '''policy learning'''
                train_dict, test_dict = load_data[i_exp]
                if treat_met in ['conti-known', 'conti-unknown']:
                    args.treat_met = 'continuous'
                    if treat_met == 'conti-known':
                        args.propensity_met = 'known'
                    else:
                        args.propensity_met = 'unknown'
                else:
                    args.treat_met = 'discrete'
                args.robust_met = robust_met
                dict_train_forOPL = utils.initialize_setting(args=args, data_dict=train_dict)
                pol_dict_train = OPL.policy_learning(args=args, pol_dict=dict_train_forOPL)
                joblib.dump(pol_dict_train, './result/'+str(num_train)+'_'+str(num_test)+ '/train_'+ treat_met+'_'+robust_met+'/exp_'+str(i_exp))


        '''policy evaluation'''

        '''load learned policy'''
        # for (treat_met, robust_met) in [('discrete', 'robust')]:
        for (treat_met, robust_met) in policy_list:
            if not os.path.isdir('./result/'+str(num_train)+'_'+str(num_test)+ '/test_'+ treat_met+'_'+robust_met):
                os.mkdir('./result/'+str(num_train)+'_'+str(num_test)+ '/test_'+ treat_met+'_'+robust_met)
            for i_exp in range(0, M):
                train_dict, test_dict = load_data[i_exp]
                pol_dict_learned = joblib.load('./result/'+str(num_train)+'_'+str(num_test)+ '/train_'+ treat_met+'_'+robust_met+'/exp_'+str(i_exp))
                evaluated_policy = pol_dict_learned['pi_param_list'][-1]
                if treat_met in ['conti-known', 'conti-unknown']:
                    args.treat_met = 'continuous'
                    if treat_met == 'conti-known':
                        args.propensity_met = 'known'
                    else:
                        args.propensity_met = 'unknown'
                else:
                    args.treat_met = 'discrete'

                '''start evaluation'''
                eval_dict = {}
                for evaluator_type in ['robust', 'min']:
                    for delta in [0.05, 0.1, 0.2, 0.3, 0.4]:
                        dict_evaluation = utils.initialize_setting(args=args, data_dict=test_dict)
                        dict_evaluation['amb_radius'] = delta
                        pol_dict_test = OPE.policy_evaluation(args=args, pol_dict=dict_evaluation, evaluated_policy=evaluated_policy, evaluator_type=evaluator_type)
                        eval_dict[evaluator_type + '_' + str(delta)] = pol_dict_test['phi_list'][-1]

                        joblib.dump(eval_dict, './result/'+str(num_train)+'_'+str(num_test)+ '/test_'+treat_met+'_'+robust_met+'/exp_'+str(i_exp))
