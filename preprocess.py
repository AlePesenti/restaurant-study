### Import and manage data 
def load_and_preprocess_data(file_path):
    
    data_raw = open(file_path)
    
    dialog_acts = []
    utterances = []
    
    for line in data_raw:
        line = line.strip().lower()    # /n weghalen en lowercase

        dialog_act, utterance = line.split(" ", 1)

        dialog_acts.append(dialog_act)
        utterances.append(utterance)    

    return(dialog_acts, utterances)


### Grouped Split
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

def grouped_split(utterances, dialog_acts): 
    X = np.array(utterances)
    y = np.array(dialog_acts)
    groups = np.array(utterances)
    sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True , random_state=42)
    sgkf.get_n_splits()

    train_index, test_index = next(sgkf.split(X, y, groups))

    trainX = X[train_index]
    trainY = y[train_index]
    testX  = X[test_index]
    testY  = y[test_index]

    return(trainX, testX, trainY, testY)