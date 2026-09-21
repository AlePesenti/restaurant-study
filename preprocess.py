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