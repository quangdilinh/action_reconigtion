#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import pickle
# import torch
import tqdm

import os 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
pd.set_option('mode.chained_assignment', None)
import numpy as np
import matplotlib.pyplot as plt
from tools import util_eval
import re
from matplotlib.pyplot import MultipleLocator
from tools import util_loc
from tools import util_vis
from tools import util_eval
from pprint import pprint

# key: value = {file_name: video_id}
_FILENAME_TO_ID = {
    "Rear_view_user_id_26223_NoAudio_3":1,
    "Rear_view_user_id_26223_NoAudio_5":2,
    "Rear_view_user_id_42897_NoAudio_5":3,
    "Rear_view_user_id_42897_NoAudio_7":4,
    "Rear_view_user_id_49989_NoAudio_5":5,
    "Rear_view_user_id_49989_NoAudio_7":6,
    "Rear_view_user_id_51953_NoAudio_3":7,
    "Rear_view_user_id_51953_NoAudio_7":8,
    "Rear_view_user_id_96715_NoAudio_5":9,
    "Rear_view_user_id_96715_NoAudio_7":10,
    "Dashboard_user_id_26223_NoAudio_3":1,
    "Dashboard_user_id_26223_NoAudio_5":2,
    "Dashboard_user_id_42897_NoAudio_5":3,
    "Dashboard_user_id_42897_NoAudio_7":4,
    "Dashboard_user_id_49989_NoAudio_5":5,
    "Dashboard_user_id_49989_NoAudio_7":6,
    "Dashboard_user_id_51953_NoAudio_3":7,
    "Dashboard_user_id_51953_NoAudio_7":8,
    "Dashboard_user_id_96715_NoAudio_5":9,
    "Dashboard_user_id_96715_NoAudio_7":10,
    "Right_side_window_user_id_26223_NoAudio_3":1,
    "Right_side_window_user_id_26223_NoAudio_5":2,
    "Right_side_window_user_id_42897_NoAudio_5":3,
    "Right_side_window_user_id_42897_NoAudio_7":4,
    "Right_side_window_user_id_49989_NoAudio_5":5,
    "Right_side_window_user_id_49989_NoAudio_7":6,
    "Right_side_window_user_id_51953_NoAudio_3":7,
    "Right_side_window_user_id_51953_NoAudio_7":8,
    "Right_side_window_user_id_96715_NoAudio_5":9,
    "Right_side_window_user_id_96715_NoAudio_7":10,
}
    

    
def get_classification(sequence_class_prob):
    '''
        given list of class and prob return the max_label and prob
    '''
    labels_index = np.argmax(sequence_class_prob, axis=1)
    probs= np.max(sequence_class_prob, axis=1)
    return labels_index, probs

def activity_localization(prob_sq, vid, action_threshold):
    """
    利用阈值对时序得分曲线二值化，确定最有可能含有动作的时间区间
    Use thresholds to binarize the timing score curve to determine the time interval most likely to contain actions.
    
    prob_sq: sequence of probs for each class of action:
    [action_id, prob]
    action_id are from 0 - 15
    """
    
    action_idx, action_probs = get_classification(prob_sq)
    action_tag = np.zeros(action_idx.shape)
    action_tag[action_probs >= action_threshold] = 1
    activities_idx = []
    startings = []
    endings = []
    clip_classfication = []
    for i in range(len(action_tag)):
        if action_tag[i] ==1:
            activities_idx.append(action_idx[i])
            start = i
            end = i+1
            startings.append(start)
            endings.append(end)
            clip_classfication.append([int(vid), action_idx[i], start, end])
    ''' 
    returned clip_classification:
        [
            video_id,
            action_idx at activity aboved thread hold,
            start: index start,
            end: index end = start +1
        ]
    '''
    return clip_classfication

def smoothing(x, k=3):
    ''' Applies a mean filter to an input sequence. The k value specifies the window
    size. window size = 2*k
    '''
    l = len(x)
    # array from -k to l-1 with step size 1
    s = np.arange(-k, l - k)
    # array from k to l+k with step size 1
    e = np.arange(k, l + k)
    s[s < 0] = 0
    e[e >= l] = l - 1
    y = np.zeros(x.shape)
    for i in range(l):
        if int(np.argmax(x[i], axis=-1) ) in [9, 10,14]:
            y[i] = x[i]
        else:
            y[i] = np.mean(x[s[i]:e[i]], axis=0)
    return y


def load_k_fold_probs(pickle_dir, view, k=5):


    probs = []
    for i in range(k):
        with open(os.path.join(pickle_dir, "A1_{}_vmae_16x4_crop_fold_{}.pkl".format(view, i)), "rb") as fp:
            vmae_16x4_probs = pickle.load(fp)
        probs.append(vmae_16x4_probs)
    return probs
        
    
def multi_view_ensemble(avg_dash_seq, avg_right_seq, avg_rear_seq):
    '''
        add buffer to result
        prioritize result overrall by view manually by grant higher weight
    '''

    # alpha, beta, sigma = 0.3, 0.4, 0.3
    '''
        alpha: weight for dashboard
        beta: weight for right
        sigma: weight for rear
    selectively get the probability by view
        rear: 3, 14
        right: 5, 6, 7, 8, 15 -> rest
        dash: 13
        avg by view weight: 0, 1, 2, 4,9,10,11,12

    '''
    alpha, beta, sigma = 0.3, 0.4, 0.3
    prob_ensemble = avg_dash_seq * alpha + avg_right_seq * beta + avg_rear_seq * sigma
    prob_ensemble[:, 3:4] = np.array(avg_rear_seq)[:, 3:4]
    prob_ensemble[:, 4:5] = prob_ensemble[:, 4:5]
    prob_ensemble[:, 5:6] = np.array(avg_right_seq)[:, 5:6]
    prob_ensemble[:, 6:7] = np.array(avg_right_seq)[:, 6:7]
    prob_ensemble[:, 7:9] = np.array(avg_right_seq)[:, 7:9]
    prob_ensemble[:, 13:14] = (np.array(avg_dash_seq)[:, 13:14])*2
    prob_ensemble[:, 14:15] = np.array(avg_rear_seq)[:, 14:15]*1.5
    prob_ensemble[:, 15:] = np.array(avg_right_seq)[:, 15:]*1.5
    return prob_ensemble


def main():
    '''
        1. load the view
        2. with every right_view prob
            2.1 find the dash_vd and rear_vd name
            2.2 with every element in pickcle result, create dictionary of video name with every view

    '''
    localization = []
    clip_classification = []
    pickle_dir = "pickles/A2"
    # each view & fold will have pickel format: ['filename', 'clip_idx', "prob"]
    k_flod_dash_probs = load_k_fold_probs(pickle_dir, "dash")
    k_flod_right_probs = load_k_fold_probs(pickle_dir, "right")
    k_flod_rear_probs = load_k_fold_probs(pickle_dir, "rear")


    all_model_results = dict()
    print(k_flod_right_probs[0].keys())
    for right_vid in k_flod_right_probs[0].keys():
        dash_vid = "Dashboard_"+re.search("user_id_[0-9]{5}_NoAudio_[0-9]", right_vid)[0]
        rear_vid = "Rear_view_"+re.search("user_id_[0-9]{5}_NoAudio_[0-9]", right_vid)[0]

        '''
            dash_prob: {
                dash_vid: probs np.array of probs that vid
            }
        '''

        all_dash_probs = np.stack([np.array(list(map(np.array, dash_prob[dash_vid]))) for dash_prob in k_flod_dash_probs])
        all_right_probs = np.stack([np.array(list(map(np.array, right_prob[right_vid]))) for right_prob in k_flod_right_probs])
        all_rear_probs = np.stack([np.array(list(map(np.array, rear_prob[rear_vid]))) for rear_prob in k_flod_rear_probs])

        avg_dash_seq = np.mean(all_dash_probs, axis=0)
        avg_right_seq = np.mean(all_right_probs, axis=0)
        avg_rear_seq = np.mean(all_rear_probs, axis=0)
        
        avg_dash_seq = smoothing(np.array(avg_dash_seq), k=1)
        avg_right_seq = smoothing(np.array(avg_right_seq), k=1)
        avg_rear_seq = smoothing(np.array(avg_rear_seq), k=1)
    
        length = min(avg_dash_seq.shape[0], avg_right_seq.shape[0], avg_rear_seq.shape[0])
        avg_dash_seq, avg_right_seq, avg_rear_seq = avg_dash_seq[:length, :],avg_right_seq[:length, :],avg_rear_seq[:length, :]

        prob_ensemble = multi_view_ensemble(avg_dash_seq, avg_right_seq, avg_rear_seq)
        vid = _FILENAME_TO_ID[right_vid]
        prob_seq = np.array(prob_ensemble)
        prob_seq = np.squeeze(prob_seq)
        clip_classification += activity_localization(prob_seq, vid, 0.1)

        all_model_results[vid] = {
            "dash":all_dash_probs,
            "rear":all_rear_probs,
            "right":all_right_probs
        }
        # print(clip_classification)

    clip_classification = pd.DataFrame(clip_classification, columns=["video_id", "label", "start", "end"])
    loc_segments = util_loc.clip_to_segment(clip_classification)
    loc_segments = util_loc.reclassify_segment(loc_segments, all_model_results)
    loc_segments =  util_loc.correct_with_prior_constraints(loc_segments)
    with open("A2_submission.txt", "w") as fp:
        for (vid, label, start, end) in loc_segments:
            fp.writelines("{} {} {} {}\n".format(int(vid), label, start, end))
    
main()