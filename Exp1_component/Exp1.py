# coding:utf-8

import sys, os
sys.path.append(os.pardir)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pdb

from sklearn.mixture improt GMM
from tqdm import tqdm
from multiprocessing import Pool

from models import uGMModel, uEpaMixModel
from misc.utils import *

def Exp1(frames, LearningRate, iterate):

    frame_num = frame.shape[0]

    core_num = np.int(input('input core number : '))
    pool = Pool(core_num)

    mixture_list = np.array([5,6,7,8,9,10,11,12,13,14,15,
                             20, 30, 40, 50])

    result = {}
    
    for mixture in mixture_list:

        result[str(mixture) + 'mixture'] = list(pool.map(analyze,
                                                         frames,
                                                         np.ones(frame_num) * LearningRate,
                                                         np.ones(frame_num) * iterate,
                                                         np.ones(frame_num) * mixture))
        print('{}-th mixture finished'.format(mixture))

    return result
    

def analyze(frame, LearningRate, iterate, mixture):

    analyzer = Analyzer(frame = frame,
                        LearningRate = LearningRate,
                        iterate = iterate,
                        mixture = mixture)
    analyzer.fit()

    return analyzer


class Analyzer:

    def __init__(self,
                 frame,
                 OuterDrop = 5
                 LearningRate = 1.
                 iterate = 200,
                 mixture = 10):

        self.f = frame

        self.OuterDrop = OuterDrop

        self.y_len = frame.shape[0] - 2*self.OuterDrop
        self.ygrid = np.arange(self.y_len)

        self.x_len = frame.shape[1] - 2*self.OuterDrop
        self.xgrid = np.arange(self.x_len)

        self.f_center = frame[self.OuterDrop:self.OuterDrop + self.y_len,
                              self.OuterDrop:self.OuterDrop + self.x_len]

        self.lr = LearningRate

        self.mix = np.int(mixture)

        self.params = None

        self.modelinit = None
        self.model_G = None
        self.model_E = None

        self.a = None
        self.b = None

        self.it = iterate

        self.hist_G = []
        self.hist_E = []

    def fit(self):

        self.modelinit = initializer(frame = self.f)
        
        std_params = self.modelinit.NormApprox(n_comp = self.mix)
        self.a = self.modelinit.a
        self.b = self.modelinit.b

        print('initial GMM fitting finished')

        # uGMModel construction
        self.model_G = uGMModel(xy_lim = np.array([self.x_len, self.y_len]),
                                mixture_size = self.mix,
                                frame_num = 1,
                                logistic_coefficient = np.array([self.a, self.b]))
        self.model_G.params['mus'] = std_params['mus'] - self.OuterDrop
        self.model_G.params['covs'] = std_params['covs']
        self.model_G.params['pi'] = std_params['pi']
        
        self.model_G.loss(f = self.f)
        self.hist_G.append(np.mean(self.model_G.lossvalue))

        # uEpaMixModel costruction
        self.model_E = uEpaMixModel(xy_lim = np.array([self.x_len, self.y_len]),
                                    mixture_size = self.mix,
                                    frame_num = 1,
                                    logistic_coefficient = np.array([self.a, self.b]))
        self.model_E.params['mus'] = std_params['mus'] - self.OuterDrop
        self.model_E.params['covs'] = std_params['covs']
        self.model_E.params['pi'] = std_params['pi']

        for i in tqdm(range(self.it)):
             grad = self.model_E.gradient(f = self.f)
             [grad[key] = np.mean(np.sum(grad, axis = (1,2)), axis = 0)
              for key in grad.keys()]

             self.model_E.params['pi'] -= self.lr * grad['pi']
             self.model_E.params['mus'] -= self.lr * grad['mus']
             self.model_E.params['covs'] -= self.lr * np.linalg.inv(grad['covs'])

             self.hist_E.append(np.mean(self.model_E.lossvalue))



        
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

        f = self.f + 1e-6
        self.a = np.sum(np.log(f/(1-f)) - self.b)

        return self.a, self.b

    def Samplize(self, sample_num = 1e+5):

        _, _ self.ComputeLogisticCoef

        p = np.log(f/(1-f))/self.a - self.b/self.a
        p = p/np.sum(p)

        p_vec = p.reshape(p.size)

        idx_sample = np.random.choice(range(p.size),
                                      np.int(sample_num),
                                      p = p_vec)
        xy_sample = np.array([np.array([idx%self.f.shape[1],
                                        idx//self.f.shape[0]])
                              for idx in idx_sample])

        return xy_sample

    def NormApprox(self, sample_num = 1e+5, n_comp = 10):

        sample = self.Samplize(sample_num = sample_num)

        gmm = GMM(n_components = n_cmop, covariance_type = 'full')
        gmm.fit(sample)

        self.params = {}
        self.params['mus'] = gmm.means_
        self.params['covs'] = gmm.covars_
        self.params['pi'] = gmm.weights_

        return self.params
        
