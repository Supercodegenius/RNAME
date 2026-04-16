import pandas as pd

import os

from scripts.CreateNoisedData import create_noised_data
from scripts.EntityMatching import EntityMatchingWithHP, EntityMatchingWithHP_GPU

import datetime

from optuna.samplers import TPESampler

GroundTruth_path = r"data\GroundTruthOrbisALL(Unclean).csv"

Optuna_path = r"optuna_studies"
Model_path = r"saved_models"
Validation_path = r"validation_data"

Test_name = "ALL(Uncleaned)"


def printT(text):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") , f" : {text}")

if __name__ == "__main__":
    printT("Creating Noise")
    ground_truth, noised_data, _, _, validation_df = create_noised_data(
        data_path=GroundTruth_path,
        name_col='Name',  # specify your text column
        index_col='Index',          # optional: specify your ID column
        noise_level=0.6,
        noise_count=1,            # creates 2 noised versions per original entry
        split_pos_neg=False,
        return_validation=True
    )

    indexers = [
            {
                'type': 'cosine_similarity',
                'tokenizer': 'words',           # word-based cosine similarity
                'ngram': 1,
                'num_candidates': 5,            # max 5 candidates per name-to-match
                'cos_sim_lower_bound': 0.2,     # lower bound on cosine similarity
            },
            {
                'type': 'cosine_similarity',
                'tokenizer': 'characters',      # 2character-based cosine similarity
                'ngram': 2,
                'num_candidates': 5,
                'cos_sim_lower_bound': 0.2,
            },
            {'type': 'sni', 
            'window_length': 5}
    ]

    em_params = {
    'name_only': True,         # only consider name information for matching
    'entity_id_col': 'Index',  # important to set both index and name columns to pick up
    'name_col': 'Name',
    'indexers': indexers,
    'supervised_on': False,    # no supervided model (yet) to select best candidates
    'without_rank_features': False,
    'with_legal_entity_forms_match': True,   # add feature that indicates match of legal entity forms (e.g. ltd != co)
    'return_sm_features' : True,
    }

    printT("Initializing")
    NameMatcher = EntityMatchingWithHP_GPU(em_params,    
        optimise_parameters={
            "learning_rate": ("float",[0.1,0.3]),
            "max_depth": ("int",[3,10]),
            "min_child_weight": ("int",[1,5]),
            "n_estimators": ("int",[50,400]),
            "subsample": ("float",[0.5,1.0]),
            "eval_metric": ("categorical", ["logloss","aucpr","auc"])
            },
        feature_selection=False,
        timeout = 50*60,
        sampler=TPESampler(seed=42),
        random_seed=42)

    printT("Fitting Indexers")
    NameMatcher.fit(ground_truth)
    printT("Optimizing Model")
    NameMatcher.fit_classifier(noised_data, create_negative_sample_fraction=0.5)

    printT("Creating Output Folders")
    os.makedirs(Optuna_path,exist_ok=True)
    os.makedirs(Model_path,exist_ok=True)
    os.makedirs(Validation_path,exist_ok=True)

    printT("Saving Optuna Study")
    NameMatcher.save_study(os.path.join(Optuna_path,fr"{Test_name}.csv"))
    printT("Saving Optimized Model")
    NameMatcher.save(os.path.join(Model_path,fr"{Test_name}.pkl"))
    printT("Saving validation Dataset")
    validation_df.to_csv(os.path.join(Validation_path,fr"{Test_name}.csv"))