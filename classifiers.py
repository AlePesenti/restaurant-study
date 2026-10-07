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


# 1. Logistic regression
from sklearn.linear_model import LogisticRegression

def log_reg(X_train, y_train, X_test):

    log_reg_classifier = LogisticRegression(max_iter=1000)
    log_reg_classifier.fit(X_train, y_train)

    log_reg_predictions = log_reg_classifier.predict(X_test)

    return(log_reg_predictions, log_reg_classifier.predict(X_test))


# 2. SVM
from sklearn.svm import LinearSVC

def svm(X_train, y_train, X_test):

    svm_classifier = LinearSVC(max_iter=5000, random_state=42)
    svm_classifier.fit(X_train, y_train)

    svm_predictions = svm_classifier.predict(X_test)

    return(svm_predictions, svm_classifier.predict(X_test))