Part 1a  
 
Requirements 
To run this implementation the following packages are required: 
- numpy	 
- scikit-learn 
- sentence-transformers 
 
How to run: 
 
1. Make sure the dataset is at: Data/dailog_acts.dat 
 
2. If the embeddings are not at: Data/embeddings.npz, run embeddings.py. 
 
3. Run main.py to run the complete classification pipeline. The results are then printed. 
 
 
 
main.py 
This file contains the classification pipeline from start to end. In the DATA_PATH variable the path to the dataset is stored. 
 
To make sure this works properly, store the data at: Data/dailog_acts.dat 
 
In main.py the functions from the other files are imported and used. Running main.py will load the dataset and create the splits (normal and grouped), creates the bag-of-words, load the embeddings, train the classifiers, and then evaluate these classifiers of the test data. The evaluation results are printed in the terminal.  
 
 
preprocess.py 
This file contains the function to load and pre-process the data. All the input is converted to lowercase. The function to create the group split is also in this file. For the grouped split, StratifiedGroupKFold is used with the utterances as the grouping variable. 7 splits are used and the first one is chosen. This makes the split around 15/85. (100/7 ~ 14.3)   
 
 
classifiers.py 
This file contains the different classifiers used (rule-based, logistic regression and the support vector machine). It also contains the keywords that are used by the rule-based classifier 
 
 
embeddings.py 
Here the utterances are encoded into frozen DistilBERT sentence embeddings. Since the encoder doesn't learn from data we can safely encode the dataset before any spilt is made. We encode solely the unique utterances (4,823 out of 23,998 rows) since it's cheaper and it would give identical vectors for repeated sentences. The embeddings script only needs to be run once to produce embeddings.npz. Which is stored in: Data/embeddings.npz. 
 
 
evaluation.py 
This file contains the evaluation method. Currently the reported measures are accuracy and balanced accuracy. 

In addition, macro F1 is reported to give equal importance to the different dialog-act classes and to account for the class imbalance in the dataset.
 
 
error_analysis.py
This file contains the functions used for the error analysis. It can generate a classification report and confusion matrix, find hard examples that are misclassified by all machine-learning models, and find the most common classification errors.
 
 
run_error_analysis.py
This file is used to run the error analysis separately from the main classification pipeline. It runs the BoW + SVM classifier on the normal split and prints the classification report, confusion matrix, and the most common classification errors. It also checks specific utterances from the test set.
 
 
*NOT IN THIS DRAFT* 
- Evaluation file to run on the held-out set 
- Prompt-based interface