import DGP
import utils
import all_obj_fun
from scipy.optimize import minimize, minimize_scalar
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
import argparse
import math

def weight_generation(delta, num_samp):
    Y = np.random.normal(size=num_samp)
    def objective(alpha):
        res = (-alpha*np.log(np.mean(np.exp(-Y/alpha)))-alpha*delta)
        return -res
    opt_alpha = minimize_scalar(objective, bounds=[1e-4,1e4], method='bounded')
    if not opt_alpha.success:
        raise ValueError("optimization fail")
    alpha = opt_alpha.x
    weight = np.exp(-Y/alpha) / np.sum(np.exp(-Y/alpha))
    return weight.reshape(-1, 1)

def policy_evaluation(args, pol_dict, evaluated_policy, evaluator_type):

    if args.treat_met == "continuous":
        pol_dict['pi_param_list'].append(evaluated_policy)
        pol_dict["h"] = pow((4 / 3 * (np.std(pol_dict["T"]) ** 5) *
                                  1 / pol_dict["T"].shape[0]), 1 / 5)  # Gaussian rule of thumb
    elif args.treat_met == "discrete":
        pol_dict["T"] = pol_dict["discrete_T"]
        pol_dict['pi_param_list'].append(evaluated_policy) # e.g., if bins are [1, 5, 10], then we have 2 catogary 0:1-5, 1: 5-10.
    '''Initialize alpha'''
    pol_dict['alpha_list'].append(args.init_alpha)

    if evaluator_type == "robust":
        pol_dict['robust_met'] = 'robust'
        '''Update alpha'''
        problem_alpha = all_obj_fun.OptimizationProblem()
        new_alpha = minimize(problem_alpha.objective_fun_alpha,
                                x0=pol_dict["alpha_list"][-1],
                                bounds=((1e-1, None), ),
                                options={"disp": True,},
                                jac=problem_alpha.der1_phi,
                                args=pol_dict.items())
        pol_dict['alpha_list'].append(new_alpha.x[0])
        pol_dict['phi_list'].append(-new_alpha.fun)

    elif evaluator_type == "min":
        random_dist_num = 1000
        X = pol_dict['X']
        X_C = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        if args.treat_met == "continuous":
            evaluated_pi_X = np.matmul(X_C, evaluated_policy)
            Y_of_piX = DGP.true_Y_fun(X=X, T=evaluated_pi_X.squeeze(), DGP_coef=pol_dict['DGP_coef'])
            res_list = np.zeros(random_dist_num)
            for i in range(0, random_dist_num):
                weights = weight_generation(delta=pol_dict['amb_radius'], num_samp=pol_dict['Y'].shape[0])
                res_list[i] = weights.T @ Y_of_piX
            Q_min = np.min(res_list)
        elif args.treat_met == "discrete":
            pi_X_all = np.matmul(X_C, evaluated_policy)
            action = np.argmax(pi_X_all, axis=1)
            pi_X_select = pi_X_all[np.arange(X.shape[0]), action].reshape(-1,1)
            Y_of_piX = DGP.true_Y_fun(X=X, T=pi_X_select.squeeze(), DGP_coef=pol_dict['DGP_coef'])
            res_list = np.zeros(random_dist_num)
            for i in range(0, random_dist_num):
                weights = weight_generation(delta=pol_dict['amb_radius'], num_samp=pol_dict['Y'].shape[0])
                res_list[i] = weights.T @ Y_of_piX
            Q_min = np.min(res_list)
        pol_dict['phi_list'].append(Q_min)
    return pol_dict