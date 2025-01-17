import DGP
import utils
import all_obj_fun
from scipy.optimize import minimize, minimize_scalar
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
import argparse
import math

def policy_learning(args, pol_dict):

    if args.treat_met == "continuous":
        pol_dict['pi_param_list'].append(np.random.normal(loc=0.0, scale=1.0, size=(args.num_cov + 1, 1)))
        pol_dict["h"] = pow((4 / 3 * (np.std(pol_dict["T"]) ** 5) *
                                  1 / pol_dict["T"].shape[0]), 1 / 5)  # Gaussian rule of thumb
    elif args.treat_met == "discrete":
        pol_dict["T"] = pol_dict["discrete_T"]
        # pol_dict['pi_param_list'].append(0.1*np.ones((args.num_cov + 1, args.num_bins - 1)))
        pol_dict['pi_param_list'].append(np.random.normal(loc=0.0, scale=1.0, size=(args.num_cov + 1, args.num_bins - 1))) # e.g., if bins are [1, 5, 10], then we have 2 catogary 0:1-5, 1: 5-10.
    '''Initialize alpha'''
    pol_dict['alpha_list'].append(args.init_alpha)

    alpha_diff = np.inf
    if args.robust_met == "robust":
        for k_idx in range(0, args.max_iter):
            if np.abs(alpha_diff) > args.alpha_res:
                '''Update pi'''
                problem_pi = all_obj_fun.OptimizationProblem()
                constraint = all_obj_fun.lin_policy_constraint(args.p_norm_bds, pol_dict["p_norm"])
                new_policy = minimize(problem_pi.objective_fun_pi, constraints=constraint,
                                      x0=pol_dict["pi_param_list"][-1],
                                      options={"disp": True},
                                      args=pol_dict.items())
                if args.treat_met == "continuous":
                    pol_dict["pi_param_list"].append(new_policy.x.reshape(-1, 1))
                elif args.treat_met == "discrete":
                    pol_dict["pi_param_list"].append(new_policy.x.reshape(args.num_cov + 1, np.max(pol_dict["discrete_T"]) + 1))
                pol_dict['W_list'].append(new_policy.fun)
                '''Update alpha'''
                problem_alpha = all_obj_fun.OptimizationProblem()
                new_alpha = minimize(problem_alpha.objective_fun_alpha,
                                        x0=pol_dict["alpha_list"][-1],
                                        bounds=((1e-1, None), ),
                                        options={"disp": True, "gtol": 1e-1},
                                        jac=problem_alpha.der1_phi,
                                        args=pol_dict.items())
                pol_dict['alpha_list'].append(new_alpha.x[0])
                pol_dict['phi_list'].append(-new_alpha.fun)
                alpha_diff = pol_dict['alpha_list'][-1] - pol_dict['alpha_list'][-2]

    elif args.robust_met == "no robust":
        problem_pi = all_obj_fun.OptimizationProblem()
        constraint = all_obj_fun.lin_policy_constraint(args.p_norm_bds, pol_dict["p_norm"])
        new_policy = minimize(problem_pi.objective_fun_pi,
                              x0=pol_dict["pi_param_list"][-1], constraints=constraint,
                              options={"disp": True},
                              args=pol_dict.items())
        if args.treat_met == "continuous":
            pol_dict["pi_param_list"].append(new_policy.x.reshape(-1, 1))
        elif args.treat_met == "discrete":
            pol_dict["pi_param_list"].append(
                new_policy.x.reshape(args.num_cov + 1, np.max(data_dict["discrete_T"]) + 1))
    return pol_dict