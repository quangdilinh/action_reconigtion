
import os 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
pd.set_option('mode.chained_assignment', None)

import numpy as np

_DRINKING = 1
_PHONE_CALL_RIGHT = 2
_PHONE_CALL_LEFT = 3
_EATING = 4
_TEXT_RIGHT = 5
_TEXT_LEFT = 6
_REACHING_BEHIND = 7
_ADJUST_CONTROL_PANE = 8
_PICK_UP_FROM_FLOOR_DRIVER = 9
_PICK_UP_FROM_FLOOR_PASSENGER = 10
_TALK_TO_PASSENGER_RIGHT = 11
_TALK_TO_PASSENGER_BACKSEAT = 12
_YAWNING = 13
_HAND_ON_HEAD = 14
_SINGING_OR_DANCE = 15

_INVALID_ACTION_LENGTH = 32




def process_overlap(action_segments):
    """
    """
    # 处理相交、较远的 segments
    action_segments = action_segments[action_segments["end"]!=0]
    action_segments = action_segments.sort_values(by=["start"]).reset_index(drop=True)
    for row_index in range(len(action_segments)-1):
        former_action_label = int(action_segments.loc[row_index, "label"])
        latter_action_label = int(action_segments.loc[row_index+1, "label"])
        former_action_start = int(action_segments.loc[row_index, "start"])
        former_action_end = int(action_segments.loc[row_index, "end"])

        latter_action_start = int(action_segments.loc[row_index+1, "start"])
        latter_action_end = int(action_segments.loc[row_index+1, "end"])

        former_latter_length_ratio = abs(former_action_end-former_action_start) / abs(latter_action_end-latter_action_start)
        latter_former_length_ratio = abs(latter_action_end-latter_action_start) / abs(former_action_end-former_action_start)
        # 两个框重合, 对发短信打电话
        if (latter_action_start - former_action_end <= 4) and (former_action_label in [_PHONE_CALL_RIGHT,_PHONE_CALL_LEFT,_TEXT_RIGHT,_TEXT_LEFT]) and (latter_action_label in [_PHONE_CALL_RIGHT,_PHONE_CALL_LEFT,_TEXT_RIGHT,_TEXT_LEFT]):
            if former_action_label in [_PHONE_CALL_RIGHT,_PHONE_CALL_LEFT] and latter_action_label in [_TEXT_RIGHT,_TEXT_LEFT]:
                action_segments.loc[row_index, "start"] = 0
                action_segments.loc[row_index, "end"] = 0

                action_segments.loc[row_index+1, "end"] = former_action_end
                action_segments.loc[row_index+1, "start"] = former_action_start
                action_segments.loc[row_index+1, "label"] = former_action_label

            if former_action_label in [_TEXT_RIGHT,_TEXT_LEFT] and latter_action_label in [_PHONE_CALL_RIGHT,_PHONE_CALL_LEFT]:
                action_segments.loc[row_index, "start"] = 0
                action_segments.loc[row_index, "end"] = 0

        elif latter_action_start - former_action_end <= 2:
            if (latter_action_end - latter_action_start) > (former_action_end - former_action_start):
                if former_action_label in [_YAWNING, _REACHING_BEHIND] and (former_action_end - former_action_start)<=3 and (former_latter_length_ratio>0.3):
                    continue
                if former_action_label in [_EATING, _HAND_ON_HEAD] and (former_action_end-former_action_start)<=3:
                    # 短的为吃东西
                    continue
                if former_action_label == _SINGING_OR_DANCE and latter_action_label not in [_PHONE_CALL_RIGHT,_PHONE_CALL_LEFT, _TALK_TO_PASSENGER_RIGHT, _TALK_TO_PASSENGER_BACKSEAT, _YAWNING]:
                    continue
                if latter_action_label == _SINGING_OR_DANCE and former_action_label not in [_PHONE_CALL_RIGHT,_PHONE_CALL_LEFT, _TALK_TO_PASSENGER_RIGHT, _TALK_TO_PASSENGER_BACKSEAT, _YAWNING]:
                    continue
                if max(former_action_end, latter_action_end) - min(former_action_start, latter_action_start) <= 24:
                    action_segments.loc[row_index+1, "start"] =  min(former_action_start, latter_action_start)
                    action_segments.loc[row_index+1, "end"] = max(former_action_end, latter_action_end)
                    action_segments.loc[row_index+1, "label"] = latter_action_label
                    action_segments.loc[row_index, "end"] = 0
                    action_segments.loc[row_index, "start"] = 0
                elif (former_action_label in [_TALK_TO_PASSENGER_RIGHT, _TALK_TO_PASSENGER_BACKSEAT]) and (latter_action_label in [_TALK_TO_PASSENGER_RIGHT, _TALK_TO_PASSENGER_BACKSEAT]):
                    action_segments.loc[row_index+1, "start"] =  min(former_action_start, latter_action_start)
                    action_segments.loc[row_index+1, "end"] = max(former_action_end, latter_action_end)
                    action_segments.loc[row_index+1, "label"] = latter_action_label
                    action_segments.loc[row_index, "end"] = 0
                    action_segments.loc[row_index, "start"] = 0
            elif (latter_action_end - latter_action_start) < (former_action_end - former_action_start):
                if latter_action_label in [_YAWNING, _REACHING_BEHIND] and (latter_action_end - latter_action_start)<=3:
                    # 短的为打哈欠
                    continue
                if former_action_label in [_EATING, _HAND_ON_HEAD] and (former_action_end-former_action_start) <= 3:
                    # 短的为吃东西
                    continue
                if former_action_label == _SINGING_OR_DANCE and latter_action_label not in [_PHONE_CALL_RIGHT,_PHONE_CALL_LEFT, _TALK_TO_PASSENGER_RIGHT, _TALK_TO_PASSENGER_BACKSEAT, _YAWNING]:
                    continue
                if latter_action_label == _SINGING_OR_DANCE and former_action_label not in [_PHONE_CALL_RIGHT,_PHONE_CALL_LEFT, _TALK_TO_PASSENGER_RIGHT, _TALK_TO_PASSENGER_BACKSEAT, _YAWNING]:
                    continue

                if max(former_action_end, latter_action_end) - min(former_action_start, latter_action_start) <= 24:
                    action_segments.loc[row_index+1, "start"] = min(former_action_start, latter_action_start)
                    action_segments.loc[row_index+1, "end"] = max(former_action_end, latter_action_end)               
                    action_segments.loc[row_index+1, "label"] = former_action_label
                    action_segments.loc[row_index, "end"] = 0
                    action_segments.loc[row_index, "start"] = 0  
                elif (former_action_label in [_TALK_TO_PASSENGER_RIGHT, _TALK_TO_PASSENGER_BACKSEAT]) and (latter_action_label in [_TALK_TO_PASSENGER_RIGHT, _TALK_TO_PASSENGER_BACKSEAT]):
                    if (latter_former_length_ratio< 0.3):
                        action_segments.loc[row_index+1, "start"] = min(former_action_start, latter_action_start)
                        action_segments.loc[row_index+1, "end"] = max(former_action_end, latter_action_end)               
                        action_segments.loc[row_index+1, "label"] = former_action_label
                        action_segments.loc[row_index, "end"] = 0
                        action_segments.loc[row_index, "start"] = 0                         

    action_segments = action_segments[action_segments["end"]!=0]            
    return action_segments

def merge_same_action(vid_clip_classification, action_label, merge_threshold):
    """
        get all the vid classification in the clip with same action_label
        sort by start index

    """
    data_video_label = vid_clip_classification[vid_clip_classification["label"]== action_label]
    data_video_label = data_video_label.reset_index()
    data_video_label = data_video_label.sort_values(by=["start"])
    for j in range(len(data_video_label)-1):
        # check if next action id from end to start in the merge threashold
        # if it also valid action lenght, start of j+1 = start
        if data_video_label.loc[j+1, "start"] - data_video_label.loc[j, "end"] <= merge_threshold:
            if abs(data_video_label.loc[j, "start"] - data_video_label.loc[j+1, "end"]) > _INVALID_ACTION_LENGTH:
                continue
            # extend the start and end of j+1
            data_video_label.loc[j+1, "start"] = data_video_label.loc[j, "start"]
            # remove current start and end 
            data_video_label.loc[j, "end"] = 0
            data_video_label.loc[j, "start"] = 0
    # filter out end = 0
    data_video_label = data_video_label[data_video_label["end"]!=0]
    return data_video_label

def remove_noisy_action(noisy_actions, noise_length_threshold=2):
    '''
        remove long action (1,7,8,9,10) below threashold
    '''
    short_term_actions =  noisy_actions[noisy_actions["label"].isin([1, 7, 8, 9, 10])]
    long_term_actions = noisy_actions[~noisy_actions["label"].isin([1, 7, 8, 9, 10])]
    long_term_actions  = long_term_actions[(long_term_actions["end"] - long_term_actions["start"] > noise_length_threshold)]
    clean_actions = pd.concat([short_term_actions, long_term_actions], join="inner")
    clean_actions = clean_actions.drop(columns=['index'])
    return clean_actions


def merge_and_remove(clip_classification, merge_threshold=16):
    """
    1-cluster clip level classification to action segments
    2-remove the noise action segments
    3-process the action overlap
    """
    output = pd.DataFrame([[0, 0, 0, 0]], columns=["video_id", "label", "start", "end"])
    clip_classification = clip_classification.reset_index(drop=True)
    clip_classification = clip_classification.sort_values(by=["video_id", "label"])
    for vid in clip_classification["video_id"].unique():
        # cleaning & selecting stuff
        vid_clip_classification = clip_classification[clip_classification["video_id"]==vid]
        vid_action_labels = vid_clip_classification["label"].unique()
        vid_ation_segments = pd.DataFrame([[0, 0, 0, 0]], columns=["video_id", "label", "start", "end"])
        vid_all = pd.DataFrame([[0, 0, 0, 0]], columns=["video_id", "label", "start", "end"])
        # try merge same action to get 1 action only
        for action_label in vid_action_labels:
            data_video_label = merge_same_action(vid_clip_classification, action_label, merge_threshold)
            vid_all = vid_all._append(data_video_label)
        
        vid_all = process_overlap(vid_all)
        output = output._append(vid_all)
    output = output[output["end"]!=0]
    output = remove_noisy_action(output, noise_length_threshold=2)
    output = output.sort_values(by=["video_id", "start"])
    return output


def clip_to_segment(clip_level_classification):
    ''' 
        get classification with label != 0
        merge and remove
    '''
    data_filtered = clip_level_classification[clip_level_classification["label"]!=0]
    data_filtered["start"] = data_filtered["start"].map(lambda x: int(float(x)))
    data_filtered["end"] = data_filtered["end"].map(lambda x: int(float(x)))
    data_filtered = data_filtered.sort_values(by=["video_id", "label"])
    '''
        output of data filtered: activity with label != 0
            data_filtered = [
                video_id,
                label = action_id,
                start: start_index,
                end: end_index
            ]
    '''
    results = merge_and_remove(data_filtered, merge_threshold=10)
    return results  


def reclassify_segment(loc_segments, all_model_results):
    '''
        reclassify _TALK_TO_PASSENGER_RIGHT and _TALK_TO_PASSENGER_BACKSEAT by right and rear view
        filter out action with end = 0
    '''
    loc_segments = loc_segments.reset_index(drop=True)
    for idx, row_data in loc_segments.iterrows():
        # hardcode to choose between action _TALK_TO_PASSENGER_RIGHT and _TALK_TO_PASSENGER_BACKSEAT
        if int(row_data["label"]) in [_TALK_TO_PASSENGER_RIGHT, _TALK_TO_PASSENGER_BACKSEAT]:
            vid = row_data["video_id"]
            start = row_data["start"]
            end = row_data["end"]
            pred = 0
            # another hardcode for right and rear probs result to improve talk to passenger
            # improve prediction by right and rear view to improve talk to passenger
            for segments_prob_seq in all_model_results[vid]["right"]:
                pred += np.array(list(map(np.array, segments_prob_seq[max(0, start):end])))
            for segments_prob_seq in all_model_results[vid]["rear"]:
                pred += np.array(list(map(np.array, segments_prob_seq[max(0, start):end])))
            
            # hard code for activity 12 and 11
            # choose the better avg probs
            prob_12 = np.mean(pred, axis=0)[_TALK_TO_PASSENGER_BACKSEAT]
            prob_11 = np.mean(pred, axis=0)[_TALK_TO_PASSENGER_RIGHT]
            label = _TALK_TO_PASSENGER_BACKSEAT if prob_12 > prob_11 else _TALK_TO_PASSENGER_RIGHT
            loc_segments.loc[idx, "label"] = label
            # filter out duration > 30s
            if abs(end-start) > 30:
                loc_segments.loc[idx, "end"] = 0
                loc_segments.loc[idx, "start"] = 0
    # filter out end = 0
    loc_segments = loc_segments[loc_segments["end"]!=0]    
    return loc_segments


def correct_with_prior_constraints(loc_segments):
    '''
        loop through all video
            check for missing action_id(label)
                try readd action with duration >8s
                hard code fix talk to right and backseat

    '''
    prediction = loc_segments.groupby("video_id")
    submission = []
    
    for vid in range(1, 11):
        prediction_by_vid = prediction.get_group(vid).reset_index(drop=True)
        unique_labels = np.unique(prediction_by_vid.label.values)
        miss_labels = list(set([l for l in range(1, 16)]).difference(set(unique_labels)))
        prediction_by_label = prediction_by_vid.groupby("label")
        for c in range(1, 16):
            # for each class, add duration(end - start) try correct miss_action
            # mainly correct talk to passenger right and backseat
            try:
                sub_set = prediction_by_label.get_group(c).reset_index(drop=True)
                if len(sub_set) <= 1:
                    for idx, row_data in sub_set.iterrows():
                        submission.append([int(row_data["video_id"]), int(row_data["label"]), int(row_data["start"]), int(row_data["end"])])
                else:
                    sub_set["diff"] = sub_set["end"] - sub_set["start"]
                    sub_set = sub_set.sort_values(by=["diff"], ascending=False)
                    for idx, row_data in enumerate(sub_set.values):
                        # row_data = [video_id, action_id, start, end]
                        if idx == 0:

                            submission.append([int(row_data[0]), int(row_data[1]), int(row_data[2]), int(row_data[3])]) 
                        else:
                            '''
                                :) wth? duplicate confused action?
                                if have talk to right and cannot detect talk to back seat
                                    add the prediction talk to back seat the same as talk to right
                            '''
                            if row_data[1] == _TALK_TO_PASSENGER_RIGHT and _TALK_TO_PASSENGER_BACKSEAT in miss_labels:
                                submission.append([int(row_data[0]), _TALK_TO_PASSENGER_BACKSEAT, int(row_data[2]), int(row_data[3]) ]) 
                                miss_labels.remove(_TALK_TO_PASSENGER_BACKSEAT)
                            if row_data[1] == _TALK_TO_PASSENGER_BACKSEAT and _TALK_TO_PASSENGER_RIGHT in miss_labels:
                                submission.append([int(row_data[0]), _TALK_TO_PASSENGER_RIGHT, int(row_data[2]), int(row_data[3])]) 
                
                                miss_labels.remove(_TALK_TO_PASSENGER_RIGHT)
                            else:
                                # re add missed action with duration > 8s
                                if abs(row_data[3]-row_data[2]) > 8:
                                    recall_label = miss_labels[0]
                                    submission.append([int(row_data[0]), recall_label, int(row_data[2]), int(row_data[3]) ]) 
                                    miss_labels.remove(recall_label) 
            except:
                print("{} does not have {}".format(vid, c))                  
    return submission