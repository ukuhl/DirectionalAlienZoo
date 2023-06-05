# -*- coding: utf-8 -*-
import numpy as np
import sklearn
import random

# Fit model
def build_model():
    import pandas as pd
    import random
    random.seed(42)
    from joblib import dump, load
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score

    # MODEL / DATA DISTRIBUTION TAKEN FROM EXPERIMENT 2 in
    # Kuhl, U., Artelt, A., & Hammer, B. (2023). Let's go to the Alien Zoo: Introducing an experimental framework 
    # to study usability of counterfactual explanations for machine learning. Frontiers in Computer Science, 5, 20.
    # Data available at: https://github.com/ukuhl/IntroAlienZoo/tree/689adf4b10fe8d670059d7cd1afd4ab5bc72c181/BackEnd/modelData
    source_data_file_path="modelData/AlienZooDataSet_EXP2.csv" # source data
    depthTree=5
    fin_data_file_path="modelData/dataset_IAZ_EXP2.npz" # where to load train / test data
    model_file_path="modelData/model_IAZ_EXP2.joblib" # where to load model

    df = pd.read_csv(source_data_file_path)

    df = df[["Var1", "Var2", "Var3", "Var4", "Var5", "GR"]]
    X = df[["Var1", "Var2", "Var3", "Var4", "Var5"]].to_numpy()
    y = df["GR"].to_numpy()

    print(X.shape)
    print(y.shape)

    # FOLLOWING COMMENTS CONTAIN CODE FOR MODEL COMPUTATION
    # FOR CONVENIENCE, we'll load the precomputed model below
    # # settings for compute balanced data + max tree depth
    # bins=5
    #
    # # Binning
    # _, bin_values = np.histogram(y, bins=bins)
    # y_binning = [list(bin_values).index(bin_values[np.argmin(np.abs(bin_values - y_))]) for y_ in y]
    #
    # # Split into training and test set
    # X_train, X_test, y_train, y_test, y_train_bins, y_test_bins = train_test_split(X, y, y_binning, test_size=0.33, random_state=42)
    #
    # # Resample data set to get a balanced data set
    # # Apply method from imbalanced learn to get a balanced data set
    # X_train_ = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)    # Add true targets to the input as an additional dimension
    # X_test_ = np.concatenate((X_test, y_test.reshape(-1, 1)), axis=1)
    #
    # # Apply imbalanced learn
    # X_train_, _ = SMOTE().fit_resample(X_train_, y_train_bins)
    # X_test_, _ = SMOTE().fit_resample(X_test_, y_test_bins)
    #
    # #  Split into input and output
    # X_train_final, y_train_final = X_train_[:,:X.shape[1]], X_train_[:, -1]
    # X_test_final, y_test_final = X_test_[:,:X.shape[1]], X_test_[:, -1]
    #
    # # Fit model
    # model = DecisionTreeRegressor(max_depth=depthTree, random_state=42)
    # model.fit(X_train_final, y_train_final)
    #
    # # Evaluate
    # y_pred = model.predict(X_test_final)
    # print(f"R^2: {r2_score(y_test_final, y_pred)}")
    # print(f"MSE: {mean_squared_error(y_test_final, y_pred)}")
    #
    # # Save dataset and model
    # np.savez(fin_data_file_path, X_train=X_train_final, X_test=X_test_final, y_train=y_train_final, y_test=y_test_final)
    # dump(model, model_file_path)

    # Load pre-computed dataset and model
    dat=np.load(fin_data_file_path)
    model = load(model_file_path)

    #print(dat.files)
    X_train_final=dat['X_train']
    X_test_final=dat['X_test']
    y_train_final=dat['y_train']
    y_test_final=dat['y_test']

    # Evaluate
    y_pred = model.predict(X_test_final)
    print(f"R^2: {r2_score(y_test_final, y_pred)}")
    print(f"MSE: {mean_squared_error(y_test_final, y_pred)}")

    return {"model": model, "X_train": X_train_final, "y_train": y_train_final}

# Compute counterfactual
def get_leafs_from_tree(tree_, classifier=False):
    leafs = []
    def traversal(node, path):
        if tree_.feature[node] != sklearn.tree._tree.TREE_UNDEFINED:
            feature_id = tree_.feature[node]
            threshold = tree_.threshold[node]

            traversal(tree_.children_left[node], path + [(node, feature_id, threshold, "<")])
            traversal(tree_.children_right[node], path + [(node, feature_id, threshold, ">")])
        else:
            if classifier is False:
                path += [(node, -1, tree_.value[node].flatten()[0])]
            else:
                path += [(node, -1, np.argmax(tree_.value[node]))]
            leafs.append(path)

    traversal(0, [])

    return leafs

def leaf_to_path(path_to_leaf):
    return [p[0] for p in path_to_leaf]


def compute_overlap_of_path(p1, p2):
    if p1 == [] or p2 == []:
        return 0
    if p1[0] != p2[0]:
        return 0
    else:
        return 1 + compute_overlap_of_path(p1[1:], p2[1:])


def compute_change(x, threshold, direction):
    eps = 1.e-5

    if direction == "<":
        if x >= threshold:
            return -1. * np.abs(threshold - x) - eps
        else:
            return 0
    elif direction == ">":
        if x <= threshold:
            return np.abs(threshold - x) + eps
        else:
            return 0


def compute_path_adjustment(x_orig, x_orig_path, target_path):
    r = {}
    overlap = 0
    y = target_path[-1][2]

    for i in range(len(target_path) - 1):
        if x_orig_path[i+1] != target_path[i+1][0]: # Starting point found?
            for j in range(i, len(target_path) - 1):
                feature_id = target_path[j][1]
                threshold = target_path[j][2]
                direction = target_path[j][3]
                x = x_orig[feature_id]

                delta = compute_change(x, threshold, direction)
                if delta != 0:
                    if feature_id in r:
                        if abs(delta) > abs(r[feature_id]):
                            r[feature_id] = delta
                    else:
                        r[feature_id] = delta

            break
        else:
            overlap += 1
    return overlap, y, r


def apply_adjustment(x_orig, adjustment):
    x_new = np.array(x_orig)

    for i, d in adjustment.items():
        x_new[i] += d

    return x_new


def score_adjustments(x_orig, x_orig_path, leafs_path, dist):
    r = []

    for leaf_path in leafs_path:
        _, y, adjustment = compute_path_adjustment(x_orig, x_orig_path, leaf_path)
        x_new = apply_adjustment(x_orig, adjustment)
        cost = dist(x_new)

        r.append((cost, y, adjustment))

    r.sort(key=lambda item: item[0])

    return r

def compute_counterfactual_of_model(model, x, y_pred, trialNo, expGroup, X_train=None, y_train=None, features_whitelist = [0, 1, 2, 3, 4]):
    x = x.flatten()

    # Compute counterfactual
    # Computation varies according to group:
    # 0 - control, no explanation
    # group 1, only upwards explanations
    # group 2, only downwards explanations
    # group 3, mixed explanations

    # Enumerate all leafs
    leafs = get_leafs_from_tree(model.tree_, classifier=False)
    # make new variable to store info on directionality:
    dirMarker = expGroup
    if expGroup == 3:
        # for odd trials, compute upwards CFs
        if trialNo % 2:
            dirMarker = 1
        # for even ones, compute downwards CFs
        else:
            dirMarker = 2

    ## Next, we filter leafs depending on group:
    # group 1: retain only those better than the current prediction
    # group 2: retain only those worse than the current prediction
    # group 3: retain only those better or worse than the current prediction (random draw!)
    if dirMarker == 1:
        # for better predictions
        leafs = list(filter(lambda z: z[-1][2] > y_pred, leafs))
    elif dirMarker == 2:
        # for worse predictions
        leafs = list(filter(lambda z: z[-1][2] < y_pred, leafs))
    else:
        print("Invalid marker for directionality, sth. went wrong.")

    # Compute path of sample
    path_of_x = list(model.decision_path([x]).indices)

    # Score and sort all counterfactuals of the sample
    regularization = lambda z: np.linalg.norm(x - z, ord=1)
    counterfactuals = score_adjustments(x, path_of_x, leafs, regularization)

    # Rounding
    counterfactuals = [np.round(apply_adjustment(x, cf[2])) for cf in counterfactuals]

    # Filter our all invalid counterfactuals - rounding might result in invalid counterfactuals!
    ## again, we filter leafs depending on group:
    # group 1: retain only those better than the current prediction
    # group 2: retain only those worse than the current prediction
    # group 3: retain only those better or worse than the current prediction
    if dirMarker == 1:
        # for better predictions
        counterfactuals = list(filter(lambda cf: model.predict([cf]) > y_pred, counterfactuals))
    elif dirMarker == 2:
        # for worse predictions
        counterfactuals = list(filter(lambda cf: model.predict([cf]) < y_pred, counterfactuals))
    else:
        self.send_custom_error(400, "Invalid marker for directionality.")

    # Choose a counterfactual -> simply take the first one (closest)
    x_cf = [-1000 for _ in range(x.shape[0])] if len(counterfactuals) == 0 else counterfactuals[0]

    return x_cf, dirMarker
