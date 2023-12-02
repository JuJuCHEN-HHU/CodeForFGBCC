import math
import numpy as np
import scipy.sparse as ssp
from scipy.special import digamma
from scipy.stats import entropy, dirichlet
from scipy.optimize import minimize
import pandas as pd
import os
import time
from utility import get_acc
from utility import get_fscore



class FGBCC:
    def __init__(self, df_label, tuples):
        num_items, num_workers, num_classes = tuples.max(axis=0) + 1
        self.num_items, self.num_workers, self.num_classes = num_items, num_workers, num_classes

        self.sigma_jk_vec = np.zeros(shape=(num_workers, num_classes, num_classes))
        self.lambda_jk_vec = np.zeros(shape=(num_workers, num_classes, num_classes))
        self.eta2_jk_vec = np.zeros(shape=self.lambda_jk_vec.shape)
        self.gamma_vec = np.zeros(num_classes)

        self.y_is_one_lij = []
        self.y_is_one_lji = []
        for k in range(num_classes):
            selected = (tuples[:, 2] == k)
            coo_ij = ssp.coo_matrix((np.ones(selected.sum()), tuples[selected, :2].T), shape=(num_items, num_workers),
                                    dtype=np.bool_)
            self.y_is_one_lij.append(coo_ij.tocsr())
            self.y_is_one_lji.append(coo_ij.T.tocsr())
        self.phi_ik = np.zeros((num_items, num_classes))
        for l in range(num_classes):
            self.phi_ik[:, [l]] += self.y_is_one_lij[l].sum(axis=-1)
        self.phi_ik /= self.phi_ik.sum(axis=-1, keepdims=True)
        self.alpha_vec = self.phi_ik.sum(axis=0)
        self.N_jkl = np.zeros(shape=self.lambda_jk_vec.shape)
        for l in range(self.num_classes):
            for k in range(self.num_classes):
                self.N_jkl[:, k, l] += self.y_is_one_lji[l].dot(self.phi_ik[:, k])
        q = 0
        for j in range(num_workers):
            if len(df_label[df_label['worker'] == j]) > 30:
                q = q + 1
        mark1 = q / num_workers
        if num_workers < 100 and num_items >= 1000:
            self.eta2_jk_vec.fill(30)
            self.lambda_jk_vec.fill(2)
            for k in range(self.num_classes):
                self.lambda_jk_vec[:, k, k] = 6
        elif (500 <= num_items < 1000 and num_workers >= 150) or (num_items >= 1000 and num_workers >= 100 and mark1 > 0.5):
            self.eta2_jk_vec.fill(15)
        else:
            self.eta2_jk_vec.fill(4)
            for k in range(self.num_classes):
                self.lambda_jk_vec[:, k, k] = 2

        self.mu_jk_vec = self.lambda_jk_vec
        self.sigma_jk_vec = self.eta2_jk_vec

        self.N_exp = np.exp(self.lambda_jk_vec + self.eta2_jk_vec / 2)
        self.zeta_jk_vec = self.N_exp.sum(axis=-1)


    def f_lambda(self, x):
        dim = self.eta2_jk_vec.shape
        self.lambda_jk_vec = x.reshape(dim)
        f = 0
        f += (self.N_jkl * self.lambda_jk_vec).sum()
        N_exp = np.exp(self.lambda_jk_vec + self.eta2_jk_vec / 2)
        N_exp = N_exp.sum(axis=-1)
        f -= ((self.N_jkl.sum(axis=-1))[:, :, None] * 1.0/self.zeta_jk_vec[:, :, None] * N_exp[:, :, None]).sum()
        for j in range(self.num_workers):
            for k in range(self.num_classes):
                f -= 0.5 * (np.power(self.lambda_jk_vec[j, k, :] - self.mu_jk_vec[j, k, :], 2) * 1.0 / self.sigma_jk_vec[j, k, :]).sum()
        return -f


    def gradientQ(self):
        self.dQlambda = np.zeros(shape=np.prod(self.eta2_jk_vec.shape))
        i = 0
        N_exp = np.exp(self.lambda_jk_vec + self.eta2_jk_vec / 2)
        for j in range(self.num_workers):
          for k in range(self.num_classes):
              for a in range(self.num_classes):
                  self.dQlambda[i] = self.N_jkl[j][k][a] - (self.lambda_jk_vec[j][k][a] - self.mu_jk_vec[j][k][a]) *\
                  1.0/ self.sigma_jk_vec[j][k][a] - self.N_jkl[j][k].sum() * 1.0/self.zeta_jk_vec[j][k] * N_exp[j][k][a]
                  i = i + 1


    def df_lambda(self, x):
        dim = self.eta2_jk_vec.shape
        self.lambda_jk_vec = x.reshape(dim)
        self.gradientQ()
        return -self.dQlambda


    def opt_lambda(self):
        x0 = self.lambda_jk_vec.reshape(1, -1)
        res = minimize(self.f_lambda, x0, method='L-BFGS-B', jac=self.df_lambda, tol=0.00001,
                       options={'disp': False, 'maxiter': 35})
        return res.x


    def opt_eta2_jk(self):
        for j in range(self.num_workers):
            for k in range(self.num_classes):
                    for l in range(self.num_classes):
                      init = 5
                      temp_eta = self.eta2_jk_vec[j][k][l]
                      temp_log_eta = np.log(temp_eta)
                      df_eta = 1
                      while abs(df_eta) > 0.00001:
                          temp_eta = np.exp(temp_log_eta)
                          if math.isnan(np.exp(self.lambda_jk_vec[j][k][l] + temp_eta / 2)):
                              print("A NAN appears!!!!")
                              print("A NAN appears!!!!")
                              init = init*2
                              temp_log_eta = math.log(init)
                              temp_eta = init
                          N_exp = np.exp(self.lambda_jk_vec[j][k][l] + temp_eta / 2)
                          df_eta = - self.N_jkl[j][k].sum(axis=-1) * 0.5 / self.zeta_jk_vec[j, k] * N_exp
                          df_eta -= (0.5 * 1.0 / self.sigma_jk_vec[j][k][l])
                          df_eta += 0.5 * 1.0 / temp_eta
                          if abs(df_eta) > 0.00001:
                              d2f_eta = - self.N_jkl[j][k].sum(axis=-1) * 0.25 / self.zeta_jk_vec[j, k] * N_exp
                              d2f_eta -= 0.5 * 1.0 / (temp_eta * temp_eta)
                              temp_log_eta = temp_log_eta - (df_eta * temp_eta)/(df_eta * temp_eta + d2f_eta * temp_eta * temp_eta)
                      self.eta2_jk_vec[j][k][l] = np.exp(temp_log_eta)


    def computeElbo(self):
        ELBO = ((self.gamma_vec - 1) * self.Eq_log_tau_k).sum() + \
               (self.N_jkl * self.Eq_log_v_jkl).sum()
        for j in range(self.num_workers):
            for k in range(self.num_classes):
                for l in range(self.num_classes):
                    ELBO -= 0.5 * np.log(self.sigma_jk_vec[j][k][l])
                    ELBO -= 0.5 * (np.power(self.lambda_jk_vec[j][k][l] - self.mu_jk_vec[j][k][l], 2) + self.eta2_jk_vec[j][k][l])\
                            * 1.0 / self.sigma_jk_vec[j][k][l]

        ELBO += dirichlet.entropy(self.gamma_vec)
        ELBO += entropy(self.phi_ik.reshape(self.num_items, -1).T).sum()
        for j in range(num_workers):
            for k in range(self.num_classes):
                    for l in range(self.num_classes):
                        ELBO += 0.5 * (np.log(self.eta2_jk_vec[j][k][l]))
        return ELBO


    def run(self):
        ELBO = 1.0
        maxIter = 1000
        i = 0
        while i < maxIter:
            self.N_jkl = np.zeros(shape=self.lambda_jk_vec.shape)
            for l in range(self.num_classes):
                for k in range(self.num_classes):
                    self.N_jkl[:, k, l] += self.y_is_one_lji[l].dot(self.phi_ik[:, k])

            self.gamma_vec = self.alpha_vec + self.phi_ik.sum(axis=0)

            self.Eq_log_tau_k = digamma(self.gamma_vec) - digamma(self.gamma_vec.sum())

            self.N_exp = np.exp(self.lambda_jk_vec + self.eta2_jk_vec / 2)
            self.zeta_jk_vec = self.N_exp.sum(axis=-1)

            self.opt_lambda()

            self.N_exp = np.exp(self.lambda_jk_vec + self.eta2_jk_vec / 2)
            self.zeta_jk_vec = self.N_exp.sum(axis=-1)
            self.opt_eta2_jk()

            self.N_exp = np.exp(self.lambda_jk_vec + self.eta2_jk_vec / 2)
            self.zeta_jk_vec = self.N_exp.sum(axis=-1)

            N_exp = self.N_exp.sum(axis=-1)
            self.Eq_log_v_jkl = self.lambda_jk_vec - 1.0 / self.zeta_jk_vec[:, :,None] * \
                                 N_exp[:, :, None] - np.log(self.zeta_jk_vec[:, :, None]) + 1
            self.phi_ik[:] = self.Eq_log_tau_k[None, :] - 1
            for l in range(self.num_classes):
                for k in range(self.num_classes):
                    self.phi_ik[:, k] += self.y_is_one_lij[l].dot(self.Eq_log_v_jkl[:, k, l])
            self.phi_ik = np.exp(self.phi_ik)
            self.phi_ik /= self.phi_ik.reshape(self.num_items, -1).sum(axis=-1)[:, None]
            oldELBO = ELBO
            ELBO = self.computeElbo()
            self.mu_jk_vec = self.lambda_jk_vec
            self.sigma_jk_vec = self.eta2_jk_vec
            i += 1
            if abs((ELBO - oldELBO) / oldELBO) <= 1e-6:
                break
        return self.phi_ik, ELBO


datasets = ['aircrowd6', 'fej2013', 'valence5', 'valence7', 'WS', 'bluebird', 'rte', 'ZenCrowd_all', 'ZenCrowd_in',
                'ZenCrowd_us', 'CF', 'CF_amt', 'fact_eval', 'MS', 's4_Dog_data', 's4_Face_Sentiment_Identification',
                's5_AdultContent', 'web', 'd_jn-product', 'd_sentiment', 'SP', 'SP_amt', 'trec', 'sentiment']
datasets_b = ['bluebird', 'd_jn-product', 'd_sentiment', 'rte', 'SP', 'SP_amt', 'trec', 'ZenCrowd_all',
                  'ZenCrowd_in', 'ZenCrowd_us']
datasets_m = ['aircrowd6', 'fej2013', 'valence5', 'valence7', 'WS', 'CF', 'CF_amt', 'fact_eval', 'MS',
                  's4_Dog_data', 's4_Face_Sentiment_Identification',
                  's5_AdultContent', 'sentiment', 'web']

if __name__ == "__main__":
    iteration = 1
    total_accu = 0
    total_time = 0
    total_fscore = 0

    for dataset in datasets:
        print(dataset)
        sum_acc = 0
        sum_fscore = 0
        sum_time = 0
        accuracies = []
        fscores = []
        times = []
        for i in range(iteration):
            tempaccuracies = []
            tempfscores = []
            temptime = []
            truth_file = '../../datasets/' + dataset + '/truth.csv'
            datafile = "../../datasets/" + dataset + "/label.csv"
            starttime = time.time()
            df_label = pd.read_csv(datafile)
            df_truth = pd.read_csv(truth_file)
            num_items, num_workers, num_classes = df_label.values.max(axis=0) + 1
            phi_ik, ELBO = FGBCC(df_label, df_label.values).run()
            duration = time.time() - starttime
            accuracy = get_acc(phi_ik, df_truth)
            print(dataset + ":      " + str(accuracy) + "    :    " + str(duration))
            temptime.append(str(duration))
            tempaccuracies.append(str(accuracy))
            times.append(temptime)
            accuracies.append(tempaccuracies)
            sum_acc += accuracy
            sum_time += duration
            if dataset in datasets_b:
                fscore = get_fscore(phi_ik, df_truth)
                tempfscores.append(str(fscore))
                fscores.append(tempfscores)
                sum_fscore += fscore
        if dataset in datasets_b:
            folder = r'../../output/binary'
            if not os.path.isdir(folder):
                os.mkdir(folder)
            folder = folder + '/' + dataset
            if not os.path.isdir(folder):
                os.mkdir(folder)
            # time
            f = open(folder + '/' + 'time_FGBCC', 'w')
            for tempresults in times:
                f.write('\t'.join(tempresults) + '\n')
            f.close()
            # accuracy
            f = open(folder + '/' + 'accuracy_FGBCC', 'w')
            for tempresults in accuracies:
                f.write('\t'.join(tempresults) + '\n')
            f.close()
            # fscore
            f = open(folder + '/' + 'fscore_FGBCC', 'w')
            for tempresults in fscores:
                f.write('\t'.join(tempresults) + '\n')
            f.close()
        else:
            folder = r'../../output/multiclass'
            if not os.path.isdir(folder):
                os.mkdir(folder)
            folder = folder + '/' + dataset
            if not os.path.isdir(folder):
                os.mkdir(folder)
            # time
            f = open(folder + '/' + 'time_FGBCC', 'w')
            for tempresults in times:
                f.write('\t'.join(tempresults) + '\n')
            f.close()
            # accuracy
            f = open(folder + '/' + 'accuracy_FGBCC', 'w')
            for tempresults in accuracies:
                f.write('\t'.join(tempresults) + '\n')
            f.close()
        total_accu += (sum_acc / iteration)
        total_time += (sum_time / iteration)
        total_fscore += (sum_fscore / iteration)
    print("total_accu: " + str(total_accu / len(datasets)))
    print("total_time: " + str(total_time / len(datasets)))
    print("total_fscore: " + str(total_fscore / len(datasets_b)))
