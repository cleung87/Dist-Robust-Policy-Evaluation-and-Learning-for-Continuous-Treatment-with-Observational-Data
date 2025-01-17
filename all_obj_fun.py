import pandas as pd
import numpy as np
import utils
class OptimizationProblem:
    def __init__(self):
        self.iteration = 0

    def IPW_calculation(self, pi_X, alpha):
        X = self.args["X"]
        T = self.args["T"]
        lr = self.args["lr"]
        treat_met = self.args["treat_met"]


        if treat_met == "continuous":
            h = self.args["h"] # bandwidth
            kernel_fun = utils.kernel_cal(self.args["kernel_met"])
            if self.args['propensity_met'] == 'known':
                self.GPS = self.args['prop_score']
            else:
                self.GPS = np.clip(utils.norm_T_Q_est(X, pi_X, lr, treat_met), self.args["threshold"], np.inf)
            N = self.GPS.shape[0]
            self.S_cal = 1 / (N * h) * np.sum(kernel_fun(1 / h * (pi_X - T)) / self.GPS)
            self.IPW = 1 / h * 1 / self.GPS * kernel_fun((pi_X - T) / h)


        elif treat_met == "discrete":
            self.GPS = np.clip(utils.norm_T_Q_est(X, T, lr, treat_met), self.args["threshold"], np.inf)
            ind_approx_matrix = np.zeros((pi_X.shape[0], self.args['num_T_category']))
            ind_approx = np.zeros((pi_X.shape[0], 1))
            for k in range(0, self.args['num_T_category']):
                ind_approx_k = np.exp(pi_X[:, k]).reshape(-1,1)/np.sum(np.exp(pi_X), axis=1).reshape(-1,1)
                ind_approx_matrix[:, k] = ind_approx_k.squeeze()
            for i in range(0, ind_approx_matrix.shape[0]):
                category = T[i, 0]
                ind_approx[i, 0] = ind_approx_matrix[i, category]
            self.ind_approx = ind_approx
            self.S_cal = np.mean(self.ind_approx/self.GPS)
            self.IPW = self.ind_approx / self.GPS


    def objective_fun_pi(self, pi_param, args):
        # pi_param will be automatically be processed as (num_params, )
        self.args = dict(args)
        X = self.args['X']
        T = self.args['T']
        Y = self.args['Y']
        X_C = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        alpha = self.args["alpha_list"][-1]

        if self.args["treat_met"] == 'continuous':
            pi_X = np.matmul(X_C, pi_param.reshape(-1, 1)) # linear policy class pi_para = (p+1, 1)
        elif self.args["treat_met"] == 'discrete':
            self.args['num_T_category'] = np.max(T)+1
            pi_X = np.matmul(X_C, pi_param.reshape(X_C.shape[1], self.args['num_T_category']))  # linear policy class pi_para = (p+1, the num of T_category)

        self.IPW_calculation(pi_X, alpha)
        W = np.mean(self.IPW * np.exp(-Y/alpha))
        objective = W

        '''xxx non robust to modify'''
        return objective

    def objective_fun_alpha(self, alpha, args):
        # pi_param will be automatically be processed as (num_params, )
        self.args = dict(args)
        X = self.args['X']
        T = self.args['T']
        Y = self.args['Y']
        X_C = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        pi_param = self.args['pi_param_list'][-1]

        if self.args["treat_met"] == 'continuous':
            pi_X = np.matmul(X_C, pi_param.reshape(-1, 1)) # linear policy class pi_para = (p+1, 1)
        elif self.args["treat_met"] == 'discrete':
            self.args['num_T_category'] = np.max(T)+1
            pi_X = np.matmul(X_C, pi_param.reshape(X_C.shape[1], self.args['num_T_category']))  # linear policy class pi_para = (p+1, the num of T_category)

        self.pi_X = pi_X
        self.IPW_calculation(pi_X, alpha)
        Y_max = np.max(Y)
        eta = self.args["amb_radius"]
        if self.args["norm_met"] == "unnormalized":
            phi = - alpha * eta - alpha * np.log(np.mean(self.IPW * np.exp(-(Y-Y_max)/alpha))) + Y_max
        elif self.args["norm_met"] == "normalized":
            phi = - alpha * eta - alpha * np.log(np.mean(self.IPW * np.exp(-(Y-Y_max)/alpha))) + Y_max + alpha * np.log(self.S_cal)
        objective = -1 * phi

        return objective

    def der1_phi(self, alpha, args):
        T = self.args['T']
        Y = self.args["Y"]
        pi_X = self.pi_X
        eta = self.args["amb_radius"]
        GPS = self.GPS

        if self.args["treat_met"] == "continuous":
            h = self.args['h']
            kernel_fun = utils.kernel_cal(self.args["kernel_met"])
            if self.args["robust_met"] == "robust":
                if self.args["norm_met"] == "normalized":
                    term_1_denomin = np.mean(1 / h * kernel_fun(1 / h * (pi_X - T)) / GPS)
                elif self.args["norm_met"] == "unnormalized":
                    term_1_denomin = 1
                term_1_num = np.mean(1 / h * kernel_fun(1 / h * (pi_X - T)) / GPS * np.exp(-Y / alpha))
                term_1 = term_1_num / term_1_denomin

                term_2_denomin = term_1_denomin
                term_2_num = 1 / (alpha ** 2) * np.mean(1 / h * kernel_fun(1 / h * (pi_X - T)) / GPS * Y * np.exp(-Y / alpha))
                term_2 = term_2_num / term_2_denomin

                first_der = eta + np.log(term_1) + alpha * term_2 / term_1

        elif self.args["treat_met"] == "discrete":
            if self.args["robust_met"] == "robust": # only DROPE DROPL needs to compute the derivative of alpha
                ind_approx = self.ind_approx
                GPS = self.GPS

                if self.args["norm_met"] == "normalized":
                    term_1_denomin = np.mean(ind_approx / GPS)
                elif self.args["norm_met"] == "unnormalized":
                    term_1_denomin = 1
                term_1_num = np.mean(ind_approx / GPS * np.exp(-Y / alpha))
                term_1 = term_1_num / term_1_denomin

                term_2_denomin = term_1_denomin
                term_2_num = 1 / (alpha ** 2) * np.mean(ind_approx / GPS * Y * np.exp(-Y / alpha))
                term_2 = term_2_num / term_2_denomin
                first_der = eta + np.log(term_1) + alpha * term_2 / term_1
        return first_der

    def callback(self, xk):
        self.iteration += 1

        check_res = {"ind_approx": self.ind_approx, "est_GPS_W": self.GPS,
                     "ult_Y": self.ult_Y, "W_h": self.W_h}
        check_res_df = pd.DataFrame(check_res)
        check_res_df.to_csv("check_"+str(self.iteration)+".csv", index=True)
        print("This is the " + str(self.iteration) + " iterations.")


########################################################################################################################

def lin_policy_constraint(p_norm_bds, p_norm):
    if np.isinf(p_norm):
        ine_con = lambda policy: np.max(policy)
    elif ((p_norm >= 1) and (p_norm < np.inf)):
        ine_con = lambda policy: (np.mean((np.abs(policy) ** p_norm)) ** (1 / p_norm))

    cons = [{'type': 'ineq', 'fun': lambda policy: p_norm_bds - ine_con(policy)},
            {'type': 'ineq', 'fun': lambda policy: ine_con(policy)}]

    return cons
