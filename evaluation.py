### Quantitive ecaluation 
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score

def evaluate(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    
    print("Accuracy:", accuracy)
    print("Balanced accuracy", balanced_accuracy)
    print("")