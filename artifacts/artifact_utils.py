import sys
sys.path.append("../")
from constants import *
import pandas as pd
import ast
from collections import Counter
import csv
import warnings
import h5py

def check_count_statisfied(target_artifacts, cur_out_artifacts):
    for artifact_name, count in target_artifacts.items():
        if count != len(cur_out_artifacts[artifact_name]):
            # count hasn't yet been reached
            return False
    return True
        

def get_filenames_by_artifacts(target_artifacts):
    '''
     target_artifacts is {artifact name: count, ....}

    returns {artifact name : [(filename, pos), ...], ...}
    '''
    # load summary file
    summary = pd.read_csv(DATASET_SUMMARY)
#     summary = summary.sample(frac=1) # to shuffle
    
    target_artifact_names = list(target_artifacts.keys())
    out_artifacts = {}

    
    # Initial artifacts dict with empty lists
    for artifact_name in target_artifact_names:
        out_artifacts[artifact_name] = []
    
    for _, sample in summary.iterrows():
        
        # Ensure that the counts have not yet been met
        if check_count_statisfied(target_artifacts, out_artifacts):
            return out_artifacts
        
        annotations = sample["annotations"]
        filename = DATASET_PATH + sample["file_name"]
        frequency = sample["sample_frequency"]
        assert(int(frequency) == float(frequency))
        frequency = int(frequency)
        
        annoations_list = ast.literal_eval(annotations)
        
        for annoation in annoations_list:
        
            annoation_name, annoation_time = annoation
            if annoation_name in target_artifact_names:
                
                # Check to ensure the count hasn't yet been satisfied
                if len(out_artifacts[annoation_name]) < target_artifacts[annoation_name]:
                    out_artifacts[annoation_name].append((filename, int(annoation_time * frequency)))
                    
    if not (check_count_statisfied(target_artifacts, out_artifacts)):
        warnings.warn("COUNT NOT MET")
    return out_artifacts