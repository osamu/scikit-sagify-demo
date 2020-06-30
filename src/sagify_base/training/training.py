import os
import json
import joblib
import pandas as pd
from sklearn import tree

def train(input_data_path, model_save_path, hyperparams_path=None):
    """
    The function to execute the training.

    :param input_data_path: [str], input directory path where all the training file(s) reside in
    :param model_save_path: [str], directory path to save your model(s)
    :param hyperparams_path: [optional[str], default=None], input path to hyperparams json file.
    Example:
        {
            "max_leaf_nodes": 10,
            "n_estimators": 200
        }
    """
    print('Start Training')

    # TODO: If exists, read in hyperparams file JSON content
    with open(hyperparams_path) as tc:
        trainingParams = json.load(tc)

    # TODO: Write your modeling logic
    input_files = [ os.path.join(input_data_path, file) for file in os.listdir(input_data_path) ]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(input_data_path, "training"))
    raw_data = [ pd.read_csv(file, header=None) for file in input_files ]
    train_data = pd.concat(raw_data)

    # labels are in the first column
    train_y = train_data.loc[:,0]
    train_X = train_data.loc[:,1:]

    # Here we only support a single hyperparameter. Note that hyperparameters are always passed in as
    # strings, so we need to do any necessary conversions.
    max_leaf_nodes = trainingParams.get('max_leaf_nodes', None)
    if max_leaf_nodes is not None:
        max_leaf_nodes = int(max_leaf_nodes)

    # Now use scikit-learn's decision tree classifier to train the model.
    clf = tree.DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes)
    clf = clf.fit(train_X, train_y)

    # TODO: save the model(s) under 'model_save_path'
    with open(os.path.join(model_save_path, 'decision-tree-model.pkl'), 'wb') as out:
            joblib.dump(clf, out)
    print('Training complete.')
