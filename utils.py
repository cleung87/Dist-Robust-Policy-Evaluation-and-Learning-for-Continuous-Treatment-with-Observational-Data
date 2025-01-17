import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
import DGP
import all_obj_fun
from scipy.optimize import minimize, minimize_scalar
from sklearn.linear_model import LinearRegression, LogisticRegression
import argparse
import math

def initialize_setting(args, data_dict):
    pol_dict = {"prop_score": None, "X": data_dict["X"], "T": data_dict["T"], "Y": data_dict["Y"], 'DGP_coef':data_dict['DGP_coef'],
                     "kernel_met": args.kernel_met, 'propensity_met':args.propensity_met, "amb_radius": args.amb_radius, "lr": None, "treat_met": args.treat_met,
                     "norm_met": args.norm_met, "p_norm": args.p_norm, "threshold": args.threshold,
                "robust_met": args.robust_met, 'alpha_list': [], 'pi_param_list': [], 'W_list': [], 'phi_list':[]}

    if args.propensity_met == 'known':
        pol_dict['prop_score'] = data_dict['prop_score']
    else:
        pol_dict['prop_score'] = None
    if args.treat_met == "continuous":
        lr = LinearRegression()
        lr.fit(data_dict["X"], data_dict["T"])
    elif args.treat_met == "discrete":
        data_dict["discrete_T"] = treat_discretization(data_dict["T"], args.num_bins)
        lr = LogisticRegression(multi_class="multinomial", solver="lbfgs")
        lr.fit(data_dict["X"], data_dict["discrete_T"])
        pol_dict['discrete_T'] = data_dict["discrete_T"]
    pol_dict['lr'] = lr
    return pol_dict
def treat_discretization(cont_T, n_bins=10):
    '''Discretize the treatment vector 'tau' according to uniform binning.'''
    t_lo = np.min(cont_T)
    t_hi = np.max(cont_T)
    bins = np.linspace(t_lo, t_hi, n_bins)
    # check = cont_T.squeeze()
    T_binned = np.digitize(cont_T, bins)
    # for i in range(0, n_bins):
    #     if bins[i]
    T_binned = np.where(T_binned >= n_bins, n_bins - 1, T_binned) - 1
    return T_binned

def data_split(sim_data_dict, train_size, test_size):
    '''Total number of samples'''
    trainind = np.array(range(0, train_size))
    testind = np.array(range(train_size, train_size+test_size))

    data_train_dict = {}
    data_test_dict = {}

    for key in sim_data_dict:
        if key=='DGP_coef':
            data_train_dict[key] = sim_data_dict[key]
            data_test_dict[key] = sim_data_dict[key]
        else:
            data_train_dict[key] = sim_data_dict[key][trainind]
            data_test_dict[key] = sim_data_dict[key][testind]

    return data_train_dict, data_test_dict

def kernel_cal(kernel_met):
    if kernel_met == "gaussian":
        return lambda u: np.exp(-0.5 * u ** 2) / (np.sqrt(2 * np.pi))

    elif kernel_met == "epanechnikov":
        return lambda u: np.abs(0.75 * (1 - u ** 2) * (np.where(np.abs(u) <= 1, 1, 0)))

def norm_T_Q_est(x, t, lr, treat_met):
    if treat_met == "continuous":
        T_hat = lr.predict(x)
        resid = t - T_hat

        mu_resid = np.mean(resid)
        sigma_resid = np.std(resid)
        ret = norm.pdf(t - np.matmul(x, lr.coef_.reshape(-1, 1)), loc=mu_resid, scale=sigma_resid)
    elif treat_met == "discrete":
        est_prob = lr.predict_proba(x)
        ret = np.array([est_prob[i, t[i]] for i in range(est_prob.shape[0])])
    return ret.reshape(-1, 1)

'''Learn regressor from data'''
# This part can be generalized to an implicit regressor
def regressor_fun_est(X, T, Y, alpha, policy, robust_met, treat_met):
    comb_feat = np.hstack((X, T.reshape(-1, 1)))
    if robust_met == "robust":
        ult_response = np.exp(-Y / alpha)
    elif robust_met == "no robust":
        ult_response = Y

    regressor_lr = LinearRegression()
    regressor_lr.fit(comb_feat, ult_response)

    if treat_met == "continuous":
        policy_X = np.matmul(X, policy.reshape(-1, 1))
    elif treat_met == "discrete":
        policy_X_temp = np.matmul(X, policy.reshape(X.shape[1], -1))
        policy_X = np.array([np.argmax(policy_X_temp[i, :]) for i in range(policy_X_temp.shape[0])])

    est_comb_feat = np.hstack((X, policy_X.reshape(-1, 1)))
    reg_est = regressor_lr.predict(est_comb_feat)
    return reg_est

def der1_alpha_regressor_fun_est(X, T, Y, alpha, policy, treat_met):
    comb_feat = np.hstack((X, T.reshape(-1, 1)))
    ult_response = Y * np.exp(-Y / alpha)

    regressor_lr = LinearRegression()
    regressor_lr.fit(comb_feat, ult_response)

    if treat_met == "continuous":
        policy_X = np.matmul(X, policy.reshape(-1, 1))
    elif treat_met == "discrete":
        policy_X_temp = np.matmul(X, policy.reshape(X.shape[1], -1))
        policy_X = np.array([np.argmax(policy_X_temp[i, :]) for i in range(policy_X_temp.shape[0])])

    est_comb_feat = np.hstack((X, policy_X.reshape(-1, 1)))
    reg_est = 1 / (alpha ** 2) * regressor_lr.predict(est_comb_feat)
    return reg_est


def der2_alpha_regressor_fun_est(X, T, Y, alpha, policy, treat_met):
    comb_feat = np.hstack((X, T.reshape(-1, 1)))
    ult_response1 = Y * np.exp(-Y / alpha)
    ult_response2 = (Y ** 2) * np.exp(-Y / alpha)

    regressor_lr1 = LinearRegression()
    regressor_lr2 = LinearRegression()
    regressor_lr1.fit(comb_feat, ult_response1)
    regressor_lr2.fit(comb_feat, ult_response2)

    if treat_met == "continuous":
        policy_X = np.matmul(X, policy.reshape(-1, 1))
    elif treat_met == "discrete":
        policy_X_temp = np.matmul(X, policy.reshape(X.shape[1], -1))
        policy_X = np.array([np.argmax(policy_X_temp[i, :]) for i in range(policy_X_temp.shape[0])])

    est_comb_feat = np.hstack((X, policy_X.reshape(-1, 1)))
    reg_est = -2 / (alpha ** 3) * regressor_lr1.predict(est_comb_feat) + \
              1 / (alpha ** 4) * regressor_lr2.predict(est_comb_feat)
    return reg_est