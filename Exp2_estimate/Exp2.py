# coding:utf-8

import sys, os
sys.path.append(os.pardir)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pdb

# from sklearn.mixture import GMM
from sklearn.mixture import GaussianMixture as GMM
from tqdm import tqdm
from multiprocessing import Pool, Process

from models import uGMModel, uEpaMixModel
from misc.utils import *

def Exp2(frames, LearningRate, iterate, path):

    frame_num = frames.shape[0]

    core_num = np.int(input('input core number : '))
    pool = Pool(core_num)

    mixture_list = np.array([5, 10, 20])
    # mixture_list = np.array([5,6,7,8,9,10,11,12,13,14,15,20, 30, 40, 50])

    for mixture in mixture_list:

        feedattrs = [[frames[i:i+3], LearningRate, iterate, mixture] \
                     for i in range(len(frames)-2)]

        # pdb.set_trace()

        ress = list(pool.map(analyze, feedattrs))
        
        print('{}-th mixture finished'.format(mixture))

        # get objective of final iteration
        result = np.array([[res.hist_all[-1], res.hist_move[-1], res.obj_min]\
                           for res in ress])

        # get objective transition by iteration
        hist_all = np.array([np.array(res.hist_all) for res in ress])
        hist_move = np.array([np.array(res.hist_move) for res in ress])

        # save objective value as .pkl file 
        with open('{}/obj_{}mix.pkl'.format(path, mixture), 'wb') as f:
            pickle.dump(result, f)
        with open('{}/hist_all_{}mix.pkl'.format(path, mixture), 'wb') as f:
            pickle.dump(hist_all, f)
        with open('{}/hist_move_{}mix.pkl'.format(path, mixture), 'wb') as f:
            pickle.dump(hist_move, f)

        for i, res in enumerate(ress):

            with open('{}/params_{}mix_{}batch.pkl'.format(path, mixture, i),\
                      'wb') as f:
                pickle.dump(res.__dict__, f)
            
        
    return result, hist_all, hist_move

                    
def analyze(feedattr):
    # feedattr contains [frame, LearningRate, iterate, mixture]
    frames, LearningRate, iterate, mixture = feedattr
                    
    analyzer = Analyzer(frames = frames,
                        LearningRate = LearningRate,
                        iterate = iterate,
                        mixture = mixture)
    analyzer.fit()
    
    return analyzer


class Analyzer:

    def __init__(self,
                 frames,
                 OuterDrop = 5,
                 LearningRate = 0.1,
                 iterate = 200,
                 mixture = 10):

        self.f_origin = frames

        self.OuterDrop = OuterDrop

        self.f_len = frames.shape[0]

        self.y_len = frames.shape[1] - 2*self.OuterDrop
        self.ygrid = np.arange(self.y_len)

        self.x_len = frames.shape[2] - 2*self.OuterDrop
        self.xgrid = np.arange(self.x_len)

        self.f = frames[:,
                        self.OuterDrop:self.OuterDrop + self.y_len,
                        self.OuterDrop:self.OuterDrop + self.x_len]

        self.lr = LearningRate

        self.mix = np.int(mixture)

        self.params = None

        self.modelinit = None
        self.model_all = None
        self.model_move = None

        self.a = None
        self.b = None

        self.it = iterate

        self.hist_all = []
        self.hist_move = []

        self.obj_min = None

    def fit(self):

        self.modelinit = initializer(frame = self.f_origin[0])
        
        std_params = self.modelinit.NormApprox(n_comp = self.mix)
        self.a = self.modelinit.a
        self.b = self.modelinit.b

        print('initial GMM fitting finished')

        p = np.log(self.f/(1-self.f))/self.a - self.b/self.a
        z = self.a * p + self.b
        U_p = 1/self.a * np.log(1 + np.exp(z))
        self.obj_min = U_p - self.f * p
        self.obj_min = np.sum(self.obj_min)

        # uGMModel construction
        self.model_all = uGMModel(xy_lim = np.array([self.x_len, self.y_len]),
                                  mixture_size = self.mix,
                                  frame_num = self.f_len,
                                  logistic_coefficient = np.array([self.a, self.b]))
        self.model_all.params['mus'] = std_params['mus'] - self.OuterDrop
        self.model_all.params['covs'] = std_params['covs']
        self.model_all.params['pi'] = std_params['pi']
        
        self.model_all.loss(f = self.f)
        self.hist_all.append(np.sum(self.model_all.lossvalue))

        # uEpaMixModel costruction
        self.model_move = uGMModel(xy_lim = np.array([self.x_len, self.y_len]),
                                   mixture_size = self.mix,
                                   frame_num = self.f_len,
                                   logistic_coefficient = np.array([self.a, self.b]))
        self.model_move.params['mus'] = std_params['mus'] - self.OuterDrop
        self.model_move.params['covs'] = std_params['covs']
        self.model_move.params['pi'] = std_params['pi']

        for i in tqdm(range(self.it)):

            # all gradient computing
            grad = self.model_all.gradient(f = self.f)
            for key in grad.keys():
                grad[key] = np.mean(np.sum(grad[key], axis = (1,2)), axis = 0)

            grad['covs_inv'] += np.identity(2)
                
            self.model_all.params['pi'] -= self.lr * grad['pi']
            self.model_all.params['mus'] -= self.lr * grad['mus']
            self.model_all.params['covs'] -= self.lr * np.linalg.inv(grad['covs_inv'])
            self.model_all.params['move'] -= self.lr * grad['move']

            # weight normalize
            self.model_all.params['pi'][self.model_all.params['pi']<0] = 1e-3/self.mix
            self.model_all.params['pi'] = self.model_all.params['pi']\
                                          /np.sum(self.model_all.params['pi'])

            # loss record
            self.hist_all.append(np.sum(self.model_all.lossvalue))

            # only move gradient computing
            grad_move = self.model_move.gradient_move(f = self.f)
            grad_move = np.mean(np.sum(grad_move, axis = (1,2)), axis = 0)
            self.model_move.params['move'] -= self.lr * grad_move

            # loss record
            self.hist_move.append(np.sum(self.model_move.lossvalue))

        
class initializer:

    def __init__(self, frame):
        
        self.f = frame
        self.xgrid = np.arange(frame.shape[1])
        self.ygrid = np.arange(frame.shape[0])

        self.a = None
        self.b = None

        self.params = None

        self.p = None

    def ComputeLogisticCoef(self):

        fmin = np.min(self.f)

        self.b = np.log(fmin/(1 - fmin))

        f = (1. - 1e-4) * self.f + 1e-4 * 1/2
        self.a = np.sum(np.log(f/(1-f)) - self.b)

        return self.a, self.b

    def Samplize(self, sample_num = 1e+5):

        _, _  = self.ComputeLogisticCoef()

        f = (1. - 1e-4) * self.f + 1e-4 * 1/2

        p = np.log(f/(1-f))/self.a - self.b/self.a

        p_vec = p.reshape(p.size)
        p_vec[p_vec<0] = 0
        p_vec = p_vec/np.sum(p_vec)

        idx_sample = np.random.choice(range(p.size),
                                      np.int(sample_num),
                                      p = p_vec)
        xy_sample = np.array([np.array([idx%self.f.shape[1],
                                        idx//self.f.shape[0]])
                              for idx in idx_sample])

        return xy_sample

    def NormApprox(self, sample_num = 1e+5, n_comp = 10):

        sample = self.Samplize(sample_num = sample_num)

        gmm = GMM(n_components = n_comp, covariance_type = 'full')
        gmm.fit(sample)

        self.params = {}
        self.params['mus'] = gmm.means_
        # self.params['covs'] = gmm.covars_
        self.params['covs'] = gmm.covariances_
        self.params['pi'] = gmm.weights_

        return self.params
        
