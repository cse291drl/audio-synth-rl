from typing import Iterable, Sequence, Optional
import numpy as np

import torch
from torch._C import Size
import torch.nn as nn
import torch.nn.functional as F

from synth.dexed import Dexed

class SynthParamsLoss:
    """ A 'dynamic' loss which handles different representations of learnable synth parameters
    (numerical and categorical). The appropriate loss can be computed by passing a PresetIndexesHelper instance
    to this class constructor.

    The categorical loss is categorical cross-entropy. """
    def __init__(self, normalize_losses: bool, categorical_loss_factor=0.2,
                 prevent_useless_params_loss=True,
                 cat_bce=True, cat_softmax=False, cat_softmax_t=0.1):
        """

        :param idx_helper: PresetIndexesHelper instance, created by a PresetDatabase, to convert vst<->learnable params
        :param normalize_losses: If True, losses will be divided by batch size and number of parameters
            in a batch element. If False, losses will only be divided by batch size.
        :param categorical_loss_factor: Factor to be applied to the categorical cross-entropy loss, which is
            much greater than the 'corresponding' MSE loss (if the parameter was learned as numerical)
        :param prevent_useless_params_loss: If True, the class will search for useless params (e.g. params which
            correspond to a disabled oscillator and have no influence on the output sound). This introduces a
            TODO describe overhead here
        :param cat_softmax: Should be set to True if the regression network does not apply softmax at its output.
            This implies that a Categorical Cross-Entropy Loss will be computed on categorical sub-vectors.
        :param cat_softmax_t: Temperature of the softmax activation applied to cat parameters
        :param cat_bce: Binary Cross-Entropy applied to independent outputs (see InverSynth 2019). Very bad
            perfs but option remains available.
        """
        self.normalize_losses = normalize_losses
        if cat_bce and cat_softmax:
            raise ValueError("'cat_bce' (Binary Cross-Entropy) and 'cat_softmax' (implies Categorical Cross-Entropy)"
                             "cannot be both set to True")
        self.cat_bce = cat_bce
        self.cat_softmax = cat_softmax
        self.cat_softmax_t = cat_softmax_t
        self.cat_loss_factor = categorical_loss_factor
        # self.prevent_useless_params_loss = prevent_useless_params_loss
        # Numerical loss criterion
        self.numerical_criterion = nn.MSELoss(reduction='mean')
        # Pre-compute indexes lists (to use less CPU). 'num' stands for 'numerical' (not number)
        self.num_indexes = Dexed.get_numerical_params_indexes()
        self.cat_indexes = Dexed.get_categorical_params_indexes()

    def __call__(self, u_out: torch.Tensor, u_in: torch.Tensor):
        """ Categorical parameters must be one-hot encoded. """
        # At first: we search for useless parameters (whose loss should not be back-propagated)
        useless_num_learn_param_indexes, useless_cat_learn_param_indexes = list(), list()
        batch_size = u_in.shape[0]
        # if self.prevent_useless_params_loss:
        #     for row in range(batch_size):
        #         num_indexes, cat_indexes = self.idx_helper.get_useless_learned_params_indexes(u_in[row, :])
        #         useless_num_learn_param_indexes.append(num_indexes)
        #         useless_cat_learn_param_indexes.append(cat_indexes)
        num_loss = 0.0  # - - - numerical loss - - -
        if len(self.num_indexes) > 0:
            # if self.prevent_useless_params_loss:
            #     # apply a 0.0 factor for disabled parameters (e.g. Dexed operator w/ output level 0.0)
            #     for row in range(u_in.shape[0]):
            #         for num_idx in self.num_indexes:
            #             if num_idx in useless_num_learn_param_indexes[row]:
            #                 u_in[row, num_idx] = 0.0
            #                 u_out[row, num_idx] = 0.0
            num_loss = self.numerical_criterion(u_out[:, self.num_indexes], u_in[:, self.num_indexes])
        cat_loss = 0.0  # - - - categorical loss - - -
        if len(self.cat_indexes) > 0:
            # For each categorical output (separate loss computations...)
            for cat_learn_indexes in self.cat_indexes:  # type: list
                # don't compute cat loss for disabled parameters (e.g. Dexed operator w/ output level 0.0)
                rows_to_remove = list()
                # if self.prevent_useless_params_loss:
                #     for row in range(batch_size):  # Need to check cat index 0 only
                #         if cat_learn_indexes[0] in useless_cat_learn_param_indexes[row]:
                #             rows_to_remove.append(row)
                useful_rows = None  # None means that the full batch is useful
                if len(rows_to_remove) > 0:  # If this batch contains useless inferred params
                    useful_rows = list(range(0, batch_size))
                    for row in rows_to_remove:
                        useful_rows.remove(row)
                # Direct cross-entropy computation. The one-hot target is used to select only q output probabilities
                # corresponding to target classes with p=1. We only need a limited number of output probabilities
                # (they actually all depend on each other thanks to the softmax output layer).
                target_one_hot = u_in[:, cat_learn_indexes].bool()  # Will be used for tensor-element selection
                if useful_rows is not None:  # Some rows can be discarded from loss computation
                    target_one_hot = target_one_hot[useful_rows, :]
                q_odds = u_out[:, cat_learn_indexes]  # contains all q odds required for BCE or CCE
                # The same rows must be discarded from loss computation (if the preset didn't use this cat param)
                if useful_rows is not None:
                    q_odds = q_odds[useful_rows, :]
                # - - - - - Categorical CE - - - - -
                # softmax T° if required: q_odds might not sum to 1.0 already if no softmax was applied before
                if self.cat_softmax:
                    q_odds = torch.softmax(q_odds / self.cat_softmax_t, dim=1)
                # Then the cross-entropy can be computed (simplified formula thanks to p=1.0 one-hot odds)
                q_odds = q_odds[target_one_hot]  # CE uses only 1 odd per output vector (thanks to softmax)
                # batch-sum and normalization vs. batch size
                param_cat_loss = - torch.sum(torch.log(q_odds)) / (batch_size - len(rows_to_remove))
                # CCE and BCE: add the temp per-param loss
                cat_loss += param_cat_loss
                # TODO instead of final factor: maybe divide the each cat loss by the one-hot vector length?
                #    maybe not: cross-entropy always uses only 1 of the odds... (softmax does the job before)
            if self.normalize_losses:  # Normalization vs. number of categorical-learned params
                cat_loss = cat_loss / len(self.cat_indexes)
        # losses weighting - Cross-Entropy is usually be much bigger than MSE. num_loss
        return num_loss + cat_loss * self.cat_loss_factor

class paramDistribution():
    
    def __init__(self,action_logits, cov_mat):
        pass


def generateLearnableIndices(param_length, numerical_set, categorical_set):
    learnable_num = []
    learnable_cat = []
    learn_idx = 0
    for vst_idx in range(param_length):
        if vst_idx in numerical_set:
            learnable_num.append(learn_idx)
            learn_idx += 1
    num_len = learn_idx + 1
    learn_idx = 0
    for vst_idx in range(param_length):
        if vst_idx in categorical_set:
            n_classes = Dexed.get_param_cardinality(vst_idx)
            cat_list = []
            for idx in range(n_classes):
                cat_list.append(learn_idx)
                learn_idx += 1
            learnable_cat.append(cat_list)
    cat_len = learn_idx+1
    return learnable_num, learnable_cat, num_len, cat_len
            
    

class presetParam():
    '''
    This class is a wrapper for parameters, allowing one to switch between the learnable tensor which is onehot encoded,
    and the numpy array which is required for plugging into the synthesizer
    '''
    def __init__(self, params, param_length = 155, learnable = False, device = 'cpu'):
        '''
        Params: either tuple of tensors if learnable
        '''
        self.params = params
        if isinstance(params, tuple):
            self._batch_size = self.params[0].shape[0]
            if param_length is None:
                self.preset_length = self.params[0].shape[1]
            else:
                self.preset_length = param_length
            self.device = device
        else:
            self._batch_size = self.params.shape[0]
            if param_length is None:
                self.preset_length = self.params.shape[1]
            else:
                self.preset_length = param_length
            self.device = device
        # print(preset_params.shape)
        
        self.numerical_set = set(Dexed.get_numerical_params_indexes_learnable())
        self.categorical_set = set(Dexed.get_categorical_params_indexes_learnable())
        self.learnable_num, self.learnable_cat,\
            self.learnable_num_size, self.learnable_cat_size = generateLearnableIndices(self.preset_length,\
                                                                          self.numerical_set,self.categorical_set)
        self.learnable = learnable
        if learnable:
            # Undo things
            pass
        
    def to_learnable(self):
        '''
        Returns a tuple (numerical_tensor, categorical_tensor)
        '''
        # Pre-allocation of learnable tensor
        if self.learnable:
            return self.params
        else:
            num_tensor = torch.empty((self._batch_size, self.learnable_num_size),
                                            device=self.device, requires_grad=False)
            # print(len(self.params))
            # Numerical/categorical in VST preset are *always* stored as numerical (whatever their true
            # meaning is). So we turn only numerical to numerical/categorical
            learn_indexes = 0 # For debug
            cat_counter = 0
            num_counter = 0
            for vst_idx in range(self.preset_length):
                if vst_idx in self.numerical_set:  # learned as numerical: OK, simple copy
                    idx = self.learnable_num[num_counter]  # type: int
                    num_tensor[:, idx] = self.params[:,vst_idx]
                    num_counter += 1
                    learn_indexes += 1
            cat_tensor = torch.empty((self._batch_size, self.learnable_cat_size),
                                            device=self.device, requires_grad=False)
            for vst_idx in range(self.preset_length):
                if vst_idx in self.categorical_set:  # Learnable params only:  # learned as categorical: one-hot encoding
                    assert isinstance(self.learnable_cat[cat_counter],Iterable) # Sanity check
                    n_classes = Dexed.get_param_cardinality(vst_idx)
                    classes = torch.round(self.params[:,vst_idx] * (n_classes - 1))
                    classes = classes.type(torch.int64)  # index type required
                    classes_one_hot = torch.nn.functional.one_hot(classes, num_classes=n_classes)
                    cat_index = self.learnable_cat[cat_counter]
                    # print(f"cat_index {cat_index}")
                    # print(f"n_classes {n_classes}")
                    assert len(cat_index) == n_classes
                    cat_tensor[:, cat_index] = classes_one_hot.type(torch.float)
                    cat_counter += 1
                    learn_indexes += n_classes
            return (num_tensor, cat_tensor)
        
    def to_params(self, sample = False): # returns numpy
        '''
        Returns synthsizer params (one numpy array)
        '''
        if self.learnable:
            param_tensor = torch.empty((self._batch_size,self.preset_length),device=self.device, requires_grad=False)
            learn_indexes = 0 # For debug
            cat_counter = 0
            num_counter = 0
            for vst_idx in range(self.preset_length):
                if vst_idx in self.numerical_set:
                    param_tensor[:,vst_idx] = self.params[0][:,self.learnable_num[num_counter]]
                    num_counter+= 1
                    learn_indexes+= 1
            for vst_idx in range(self.preset_length):
                if vst_idx in self.categorical_set:
                    assert isinstance(self.params,Iterable) # Ensure one-hot encoding varible
                    n_classes = Dexed.get_param_cardinality(vst_idx)
                    cat_index = self.learnable_cat[cat_counter]
                    param_tensor[:,vst_idx] = torch.argmax(self.params[1][:,cat_index]).type(torch.float)/(n_classes - 1)
                    cat_counter += 1
                    learn_indexes += n_classes
            # set all oscillators on
            param_tensor[:,[44, 66, 88, 110, 132, 154]] = 1.0
            # set default filter
            param_tensor[:,0] = 1.0
            param_tensor[:,1] = 0.0
            param_tensor[:,2] = 1.0
            param_tensor[:,3] = 0.5
            param_tensor[:,13] = 0.5
            param_arr = param_tensor.detach().numpy()
            return param_arr
        else:
            return self.params
  
# DEPRECATED VERSION      
# def get_mapping_dict():
#     numerical_set = set(Dexed.get_numerical_params_indexes_learnable())
#     categorical_set = set(Dexed.get_categorical_params_indexes_learnable())
#     preset_length = 155
#     mapping_dict = {'Numerical':[],'Categorical':[]}
#     learn_idx = 0
#     for vst_idx in range(preset_length):
#         if vst_idx in categorical_set:
#             n_classes = Dexed.get_param_cardinality(vst_idx)
#             cat_list = []
#             for idx in range(n_classes):
#                 cat_list.append(learn_idx)
#                 learn_idx += 1
#             mapping_dict['Categorical'].append({vst_idx:cat_list})
#         elif vst_idx in numerical_set:
#             mapping_dict['Numerical'].append({vst_idx:learn_idx})
#             learn_idx += 1
#     return mapping_dict
        