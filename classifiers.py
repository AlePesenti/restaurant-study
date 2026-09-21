### 1. Rule-based baseline

keyword = { 
    "thankyou": ["thank"],
    "bye"     : ["see you", "bye"],
    "ack"     : ["okay"],
    "affirm"  : ["yes", "right"],
    "reqalts" : ["how", "else"],
    "request" : ["what", "number", "address", "post","type"],
    "confirm" : ["is it"],
    "deny"    : ["dont want", "change", "wrong"],
    "hello"   : ["hi", "hello"],
    "inform"  : ["looking for", "any", "food", "restaurant"],
    "negate"  : ["no", "not"],
    "null"    : ["cough", "sil"],
    "repeat"  : ["repeat"],
    "reqmore" : ["more"],
    "restart" : ["restart", "start over"],
}

def rule_based_baseline(utterance):
    
    for dialog_act, keywords in keyword.items():
        for word in keywords:
            if word in utterance:
                return dialog_act
            
    return "inform"  # Most are inform so thats why if it is not matched we return inform
            
            
            
def rule_based_classifier(X_test):
    
    predicted_y = []
    
    for utterance in X_test:
        prediction = rule_based_baseline(utterance)
        predicted_y.append(prediction)
    
    return(predicted_y)
