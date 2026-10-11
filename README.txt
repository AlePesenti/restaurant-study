INFOMAIR - Project Part 1
 
REQUIREMENTS 
python 3.11 

Install the dependencies with:
    
    pip install -r requirements.txt 

This installs scikit-learn, numpy, sentence-transformers, pandas, Levenshtein, pyttsx3 and notebook. 

The repository is self-contained, the DistilBERT embedding cache and the trained classifiers are all included. 
Nothing has to be regennerated first.


HOW TO RUN: 
 
    pip install -r requirements.txt

    python main.py                                      Part 1a: train and evaluate
    python evaluate_heldout.py    path to test.dat      Part 1a: held-out evaluation
    python run_interactive_prompt.py                    Part 1a: classify utterances
    python run_dialog_system.py                         Part 1b: the dialog system
 
 The two variants compareeddd in the Part 2 user study are:
    
    python run_dialog_system.py                     without speech output
    python run_dialog_system.py --tts               with speech output


 

PART 1A - DIALOG ACT CLASSIFIERS

 
main.py 
This file contains the classification pipeline from start to end. In the DATA_PATH variable the path to the dataset is stored. 
To make sure this works properly, store the data at: Data/dailog_acts.dat 
In main.py the functions from the other files are imported and used.\

    python main.py

Running main.py will load the dataset and create the splits (normal and grouped), creates the bag-of-words, load the embeddings, train the classifiers, and then evaluate these classifiers of the test data. 
The evaluation results are printed in the terminal.  
 
 
preprocess.py 
This file contains the function to load and pre-process the data. All the input is converted to lowercase. 
The function to create the group split is also in this file. For the grouped split, StratifiedGroupKFold is used with the utterances as the grouping variable. 
7 splits are used and the fold whose size is closest to 15% of the data is selected.   
 
 
classifiers.py 
This file contains the different classifiers used (rule-based, logistic regression and the support vector machine). 
It also contains the keywords that are used by the rule-based classifier 
 
 
embeddings.py 
Here the utterances are encoded into frozen DistilBERT sentence embeddings. 
Since the encoder doesn't learn from data we can safely encode the dataset before any spilt is made. 
We encode solely the unique utterances (4,823 out of 23,998 rows) since it's cheaper and it would give identical vectors for repeated sentences. 
    
    python embeddings.py

Only needs to be run to regenerate Data/embeddings.npz, which is already included in the repository.
 
 
evaluation.py 
This file contains the evaluation method. 
Currently the reported measures are accuracy and balanced accuracy. 
In addition, macro F1 is reported to give equal importance to the different dialog-act classes and to account for the class imbalance in the dataset.
 

evaluate_heldout.py
Loads a held-out .dat file with the same structure as the provided data, runs inference with the saved classifiers, and reports accuracy, balanced accuracy
and macro F1 for both of them.

    python evaluate_heldout.py Data/dialog_acts_test.dat

With no argument it looks for Data/dialog_acts_test.dat. The bag-of-words models are used by default, because they need only scikit-learn. 
Add --emb to evaluate the embedding models instead; this additionally requires sentence-transformers and re-encodes the held-out utterances with DistilBERT.
Results are both printed and also written to results/heldout.txt


run_interactive_prompt.py
The interactive prompt requested in part1a: type an utterance, see the predicted dialog act. 
Input is converted to lowercase.

    python run_interactive_prompt.py

Type baseline, log_reg or svm to switch classifier while running, and exit to quit. 
Empty lines and Ctrl+C are handled without crashing.


error_analysis.py
This file contains the functions used for the error analysis. 
It can generate a classification report and confusion matrix, find hard examples that are misclassified by all machine-learning models, and find the most common classification errors.
 
 
run_error_analysis.py
This file is used to run the error analysis separately from the main classification pipeline. 
It runs the BoW + SVM classifier on the normal split and prints the classification report, confusion matrix, and the most common classification errors. 
It also checks specific utterances from the test set.

    python run_error_analysis.py
 
 



 PART 1B - DIALOG SYSTEM
 
 
run_dialog_system.py
The terminal conversation that collects the user's preferences and recommends a restaurant. 
It loads the dialog act classifier saved by main.py rather than retraining one, so the running system is exactly the model whose scores are reported in the paper.

    python run_dialog_system.py
    python run_dialog_system.py --tts
    python run_dialog_system.py --fallback semantic
    python run_dialog_system.py --debug

    --tts               speak the system utterances out loud (the configurable feature compared in the Part 2 user study)
    --fallback          which slot extraction fallback to use: levenshtein (default) or semantic
    --debug             print the predicted dialog act for each turn

    python run_dialog_system.py --help    shows the same list (generated by argparse)

The conversation ends when the user says goodbye.


state_machine.py
The state transition function, it takes the current dialog state and a classified utterance, and returns the next state together with the system utterance. 
Also contains create_dialog_state, which builds the initial state.


slot_extraction.py
Maps an utterance onto ontology values. Keyword matching is always tried first
When nothing is found, one of two fallbacks runs, selected with --fallback:

    levenshtein     maps the closest value by edit distance, which recovers typos and misspellings

    semantic        maps the closest value by cosine similarity between DistilBERT embeddings



responses.py
String templates for every system utterance,as the assignment requires. 
It includes the natural language explanations used when a recommendation is justified by the inference rules.



reasoning.py
The six inference rules over the extended restaurant properties and the restaurant lookup. 
antecedent_rules gives the antecedents and consequents of each rule.
reasoning_component applies them to the candidate restaurants and returns the chosen one, the remaining alternatives, and if the choice was contradictory. 
restaurant_lookup filters the restaurants by the collected preferences.


ontology.py
The closed set of values the system can talk about: the areas, food types and price ranges read from Data/restaurant_info.csv, and the four consequents of the inference rules. 
Read once here and imported by the other modules, so the file is parsed a single time and every component agrees on the allowed values.




INTERACTION LOGGING


Every dialog turn is appended to logs/dialogs.jsonl as one JSON object per line. 
The session-level fields are repeated on every line, so the file loads directly into a dataframe without any further processing:

    import pandas as pd
    df = pd.read_json("logs/dialogs.jsonl", lines=True)

Grouping by session_id gives the turns of each dialog; filtering on tts compares the two variants of the configurable feature.
Fields:

    session_id          identifier of the dialog
    timestamp           time of the turn
    tts                 speech output enabled or not
    fallback            which one was in use
    turn                turn number in the dialog
    state_before        previous dialog state
    user_utterance      what the user typed(in lowercase)
    dialog_act          dialog act predictedby classifier
    state_after         dialog state after
    system_response     sytem reply
    area                area preference collected in runtime
    food                food type preference collected in runtime
    pricerange          price range preference collected in runtime
    recommendation      the recommended restaurant, or null if none is found

Stalled conversation are the ones with state_before = state_after




DATA AND EXTRA FILES

Data/dailog_acts.dat                dialog act dataset
Data/all_dialogs.txt                the dialog_acts_test

Data/restaurant_info.csv            restaurant properties
Data/restaurant_info_extended.csv   extended properties, used by the reasoning
Data/embeddings.npz                 cached DistilBERT vectors, regenerated by embeddings.py

models/bow_models.joblib            vectorizer and the two bag-of-words classifiers, written by main.py
models/embedding_models.joblib      the two classifiers trained on embeddings

results/heldout.txt                 written by evaluate_heldout.py
logs/dialogs.jsonl                  written by the dialog system