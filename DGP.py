import numpy as np
import utils
import os
import joblib

from scipy.stats import multivariate_normal, norm
from scipy.optimize import minimize_scalar


def true_Y_fun(X, T, DGP_coef):
    # X.shape = [n,p] # T.sahpe = [n,] Y.sahpe = [n,]
    beta_T = DGP_coef['beta_T']
    beta_x = DGP_coef['beta_x']
    beta_x_T = DGP_coef['beta_x_T']
    seed = DGP_coef['seed']
    noise_sigma = DGP_coef['noise_sigma']
    np.random.seed(seed)

    n = X.shape[0]
    lin_T = (beta_T * T)
    lin_X = (np.matmul(X, beta_x.reshape(-1, 1))).squeeze()
    lin_XT = T * (np.matmul(X, beta_x_T.reshape(-1, 1))).squeeze()
    noise_Y = np.random.normal(loc=0, scale=noise_sigma, size=(n,))

    Y = 5 + lin_X + lin_T + lin_XT + noise_Y

    return Y.reshape(-1, 1)

def simulate(num_cov=10, train_size=2000, test_size=2000, seed=0):
    np.random.seed(seed)
    num_samp = train_size + test_size
    '''Preliminary results'''
    beta_x = np.ones((num_cov))
    beta_T = 1  # by default
    beta_x_T = np.ones((num_cov))
    beta_x_quad_T = np.ones((num_cov))


    '''sparse interaction terms'''
    sparse_entries_1 = np.random.choice(range(num_cov), size=int(round(0.3 * num_cov)), replace=False)
    beta_x_T[sparse_entries_1] = 0
    sparse_entries_2 = np.random.choice(range(num_cov), size=int(round(0.3 * num_cov)), replace=False)
    beta_x[sparse_entries_2] = 0
    sparse_entries_3 = np.random.choice(range(num_cov), size=int(round(0.2 * num_cov)), replace=False)
    beta_x_quad_T[sparse_entries_3] = 0

    '''simulate the covariates'''
    X = np.random.uniform(low=-0.2, high=0.2, size=(num_samp, num_cov))
    sim_norm_T = norm.rvs(0, np.sqrt(0.1), num_samp)
    theta_T = beta_x_quad_T
    T = sim_norm_T + np.matmul(X, theta_T) + 1 * X[:, 0] + 2 * X[:, 1] - 3 * X[:, 2]

    '''simulate treatment for each sample'''
    true_Q = norm.pdf(T - np.matmul(X, theta_T) - 1 * X[:, 0] - 2 * X[:, 1] + 3 * X[:, 2], loc=0, scale=np.sqrt(0.1))
    '''simulate outcome for each sample'''
    noise_sigma=0.1
    DGP_coef = {'beta_T': beta_T, 'beta_x': beta_x, 'beta_x_T': beta_x_T, 'beta_x_quad_T': beta_x_quad_T, 'seed': seed, 'noise_sigma': noise_sigma}
    Y = true_Y_fun(X, T, DGP_coef)

    data_dict = {'X': X, 'T': T.reshape(-1, 1), 'Y': Y, 'prop_score':true_Q.reshape(-1, 1), 'DGP_coef':DGP_coef}
    train_dict, test_dict = utils.data_split(data_dict, train_size, test_size)
    return train_dict, test_dict


if __name__ == "__main__":
    M=100
    if not os.path.isdir('./data'):
        os.mkdir('./data')
    for (num_train, num_test) in [(1500, 2000),(2500, 2000)]:
        data_list = []
        for i_exp in range(0, M):
            train_dict, test_dict = simulate(num_cov=10, train_size=num_train,test_size=num_test, seed=i_exp)
            data_list.append((train_dict, test_dict))
        joblib.dump(data_list, './data/'+str(num_train)+'_'+str(num_test)+'_0-'+str(M))

