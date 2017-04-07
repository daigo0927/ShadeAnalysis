# coding:utf-8

import sys, os
sys.path.append(os.pardir)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pdb

from sklearn.mixture import GMM
# from sklearn.mixture import GaussianMixture as GMM
from tqdm import tqdm
from multiprocessing import Pool, Process

from models import uGMModel, uEpaMixModel
from misc.utils import *

def Exp1(frames, LearningRate, iterate, path):

    frame_num = frames.shape[0]

    core_num = np.int(input('input core number : '))
    pool = Pool(core_num)

    # mixture_list = np.array([5, 10])
    mixture_list = np.array([5,6,7,8,9,10,11,12,13,14,15,20, 30, 40, 50, 60, 80, 100])

    result = {}
    hist_E = {}

    for mixture in mixture_list:

        feedattrs = [[frame, LearningRate, iterate, mixture] for frame in frames]

        ress = list(pool.map(analyze, feedattrs))
        
        print('{}-th mixture finished'.format(mixture))

        result['{}mixture'.format(mixture)] = \
                            np.array([[res.hist_E[-1], res.hist_G[-1], res.obj_min] \
                                      for res in ress])
        hist_E['{}mixture'.format(mixture)] = \
                            np.array([np.array(res.hist_E) for res in ress])
                    
    with open('{}/obj_result.pkl'.format(path), 'wb') as f:
        pickle.dump(result, f)
    with open('{}/hist_E.pkl'.format(path), 'wb') as f:
        pickle.dump(hist_E, f)
        
    return result, hist_E

                    
def analyze(feedattr):
    # feedattr contains [frame, LearningRate, iterate, mixture]
    frame, LearningRate, iterate, mixture = feedattr
                    
    analyzer = Analyzer(frame = frame,
                        LearningRate = LearningRate,
                        iterate = iterate,
                        mixture = mixture)
    analyzer.fit()
    
    return analyzer


class Analyzer:

    def __init__(self,
                 frame,
                 OuterDrop = 5,
                 LearningRate = 0.1,
                 iterate = 200,
                 mixture = 10):

        self.f_origin = frame

        self.OuterDrop = OuterDrop

        self.y_len = frame.shape[0] - 2*self.OuterDrop
        self.ygrid = np.arange(self.y_len)

        self.x_len = frame.shape[1] - 2*self.OuterDrop
        self.xgrid = np.arange(self.x_len)

        self.f = frame[self.OuterDrop:self.OuterDrop + self.y_len,
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

        self.obj_min = None

    def fit(self):

        self.modelinit = initializer(frame = self.f_origin)
        
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
        self.model_G = uGMModel(xy_lim = np.array([self.x_len, self.y_len]),
                                mixture_size = self.mix,
                                frame_num = 1,
                                logistic_coefficient = np.array([self.a, self.b]))
        self.model_G.params['mus'] = std_params['mus'] - self.OuterDrop
        self.model_G.params['covs'] = std_params['covs']
        self.model_G.params['pi'] = std_params['pi']
        
        self.model_G.loss(f = self.f)
        self.hist_G.append(np.sum(self.model_G.lossvalue))

        # uEpaMixModel costruction
        self.model_E = uEpaMixModel(xy_lim = np.array([self.x_len, self.y_len]),
                                    mixture_size = self.mix,
                                    frame_num = 1,
                                    logistic_coefficient = np.array([self.a, self.b]))
        self.model_E.params['mus'] = std_params['mus'] - self.OuterDrop
        self.model_E.params['covs'] = std_params['covs'] * 100
        self.model_E.params['pi'] = std_params['pi']

        for i in tqdm(range(self.it)):
            grad = self.model_E.gradient(f = self.f)
            for key in grad.keys():
                grad[key] = np.mean(np.sum(grad[key], axis = (1,2)), axis = 0)

            grad['covs_inv'] += np.identity(2)

                
            self.model_E.params['pi'] -= self.lr * grad['pi']
            self.model_E.params['mus'] -= self.lr * grad['mus']
            self.model_E.params['covs'] -= self.lr * np.linalg.inv(grad['covs_inv'])

            self.model_E.params['pi'][self.model_E.params['pi']<0] = 1e-3/self.mix
            self.model_E.params['pi'] = self.model_E.params['pi']\
                                        /np.sum(self.model_E.params['pi'])
            
            self.hist_E.append(np.sum(self.model_E.lossvalue))

        
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
        self.params['covs'] = gmm.covars_
        self.params['pi'] = gmm.weights_

        return self.params
        
