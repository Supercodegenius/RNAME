"""
Some of this code will look obviously inefficient but that's to make it easier to read and minimising the amount of data open at any one time
    The structures will predict on the portfolio, real_world_test set and validation sets sequentially rather than at the same time for clarity
"""

import pandas as pd

import os

import datetime

from emm import PandasEntityMatching

from xgboost import DMatrix

Output_path = r"TestingOutputs"

Model_path = r"saved_models"
Validation_path = r"validation_data"
Data_path = r"data"

def make_dmatrix(data):
    return DMatrix(data)

def printT(text,**kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") , f": {text}", **kwargs)

def distill_outputs(df,
                   threshold = 0.5):
    df = df[df.best_match]

    df.loc[df.nm_score <= threshold , "gt_entity_id"] = pd.NA
    df.loc[df.nm_score <= threshold , "gt_name"] = pd.NA
    df.loc[df.nm_score <= threshold , "score_0"] = pd.NA
    df.loc[df.nm_score <= threshold , "score_1"] = pd.NA
    df.loc[df.nm_score <= threshold , "nm_score"] = pd.NA

    return df[["uid","gt_entity_id","name","gt_name","score_0","score_1","nm_score"]]

def reassign_best_match(df):
    df = df.sort_values(by=["uid","nm_score", "gt_entity_id"], ascending=False, na_position="last")
    gb = df.groupby(["uid"])

    df["best_rank"] = gb["nm_score"].transform(lambda x: range(1, len(x) + 1))
    df["best_match"] = (df["best_rank"] == 1) & (df["nm_score"].notnull()) & (df["nm_score"] > 0)

    return df

# Model per country and partition data by country
def structure_1(output_path, cleanliness, modelFra, modelDeu, modelUsa):
    def run_transformation(filepath, input_filename, output_file_desc,cleaned=""):
        transformed_dfs = []
        for country, model in zip(["France", "Germany", "USA"], [modelFra, modelDeu, modelUsa]):
            df = pd.read_csv(os.path.join(filepath,f"{input_filename}{country}{cleaned}.csv"))
            transformed_dfs.append(model.transform(make_dmatrix(df)))
        transformed_dfs = pd.concat(transformed_dfs)
        transformed_dfs.to_csv(os.path.join(output_path,f"Structure_1_{output_file_desc}_full.csv"))
        distill_outputs(transformed_dfs).to_csv(os.path.join(output_path,f"Structure_1_{output_file_desc}_short.csv"))
    
    # Transform all the portfolio data
    print("portfolio", end=" | ")
    run_transformation(Data_path,"Portfolio","Portfolio")

    # Transform all the Real World Test Set Data
    print("Real World Test Set", end=" | ")
    run_transformation(Data_path,"Validation","RWTS")

    # Transform all the Real World Test Set Data
    print("Validation Set", end=" | ")
    run_transformation(Validation_path,"","Validation", f"({cleanliness})")

    print()


# One model for all data
def structure_2(output_path, cleanliness, modelAll):
    def run_transformation(filepath, input_filename, output_file_desc,cleaned=""):
        df = pd.read_csv(os.path.join(filepath,f"{input_filename}All{cleaned}.csv"))
        transformed_df = modelAll.transform(make_dmatrix(df))
        transformed_df.to_csv(os.path.join(output_path,f"Structure_2_{output_file_desc}_full.csv"))
        distill_outputs(transformed_df).to_csv(os.path.join(output_path,f"Structure_2_{output_file_desc}_short.csv"))
    
    # Transform all the portfolio data
    print("portfolio", end=" | ")
    run_transformation(Data_path,"Portfolio","Portfolio")

    # Transform all the Real World Test Set Data
    print("Real World Test Set", end=" | ")
    run_transformation(Data_path,"Validation","RWTS")

    # Transform all the Real World Test Set Data
    print("Validation Set", end=" | ")
    run_transformation(Validation_path,"","Validation", f"({cleanliness})")

    print()

# Model per country + single model for unknown location
#   We don't exactly have any "unknown location data" for these tests so pass
def structure_3(output_path, cleanliness, modelFra, modelDeu, modelUsa, modelAll):
    pass

# Model per country + single model for all data
def structure_4(output_path, cleanliness, modelFra, modelDeu, modelUsa, modelAll):
    def run_transformation(filepath, input_filename, output_file_desc,cleaned=""):
        transformed_dfs = []
        for country, model in zip(["France", "Germany", "USA"], [modelFra, modelDeu, modelUsa]):
            df = pd.read_csv(os.path.join(filepath,f"{input_filename}{country}{cleaned}.csv"))
            transformed_dfs.append(model.transform(make_dmatrix(df)))

        df = pd.read_csv(os.path.join(filepath,f"{input_filename}All{cleaned}.csv"))
        transformed_dfs.append(model.transform(make_dmatrix(df)))
        transformed_dfs = pd.concat(transformed_dfs)

        transformed_dfs = reassign_best_match(transformed_dfs)
        transformed_dfs.to_csv(os.path.join(output_path,f"Structure_4_{output_file_desc}_full.csv"))
        distill_outputs(transformed_dfs).to_csv(os.path.join(output_path,f"Structure_4_{output_file_desc}_short.csv"))

    # Transform all the portfolio data
    print("portfolio", end=" | ")
    run_transformation(Data_path,"Portfolio","Portfolio")

    # Transform all the Real World Test Set Data
    print("Real World Test Set", end=" | ")
    run_transformation(Data_path,"Validation","RWTS")

    # Transform all the Real World Test Set Data
    print("Validation Set", end=" | ")
    run_transformation(Validation_path,"","Validation", f"({cleanliness})")

    print()

# Model per country with all data entered
def structure_5(output_path, cleanliness, modelFra, modelDeu, modelUsa):
    def run_transformation(filepath, input_filename, output_file_desc,cleaned=""):
        transformed_dfs = []
        for model in zip([modelFra, modelDeu, modelUsa]):
            df = pd.read_csv(os.path.join(filepath,f"{input_filename}ALL{cleaned}.csv"))
            transformed_dfs.append(model.transform(make_dmatrix(df)))

        transformed_dfs = pd.concat(transformed_dfs)
        transformed_dfs = reassign_best_match(transformed_dfs)

        transformed_dfs.to_csv(os.path.join(output_path,f"Structure_5_{output_file_desc}_full.csv"))
        distill_outputs(transformed_dfs).to_csv(os.path.join(output_path,f"Structure_5_{output_file_desc}_short.csv"))
    
    # Transform all the portfolio data
    print("portfolio", end=" | ")
    run_transformation(Data_path,"Portfolio","Portfolio")

    # Transform all the Real World Test Set Data
    print("Real World Test Set", end=" | ")
    run_transformation(Data_path,"Validation","RWTS")

    # Transform all the Real World Test Set Data
    print("Validation Set", end=" | ")
    run_transformation(Validation_path,"","Validation", f"({cleanliness})")

    print()



if __name__ == "__main__":
    os.makedirs(Output_path,exist_ok=True)

    printT("Initializing")
    for cleanliness in ["Uncleaned","Cleaned"]:
        cleanliness_output = os.path.join(Output_path,cleanliness)
        os.makedirs(cleanliness_output,exist_ok=True)

        model_France = PandasEntityMatching.load(os.path.join(Model_path,f"France({cleanliness}).pkl"))
        model_Germany = PandasEntityMatching.load(os.path.join(Model_path,f"Germany({cleanliness}).pkl"))
        model_USA = PandasEntityMatching.load(os.path.join(Model_path,f"USA({cleanliness}).pkl"))
        model_ALL = PandasEntityMatching.load(os.path.join(Model_path,f"ALL({cleanliness}).pkl"))

        printT(f"Structure 1: Model Per Country - {cleanliness}",end=" - ")
        structure_1(cleanliness_output, cleanliness, model_France, model_Germany, model_USA)

        printT(f"Structure 2: Single Global Model - {cleanliness}",end=" - ")
        structure_2(cleanliness_output, cleanliness, model_ALL)

        # printT(f"Structure 3: Model Per Country + Global Model for Ambiguous Locations - {cleanliness}")
        # structure_3(cleanliness_output, cleanliness, model_France, model_Germany, model_USA, model_ALL)

        printT(f"Structure 4: Model Per Country + Global Model for All Locations - {cleanliness}",end=" - ")
        structure_4(cleanliness_output, cleanliness, model_France, model_Germany, model_USA, model_ALL)

        printT(f"Structure 5: Model Per Country for All Data - {cleanliness}",end=" - ")
        structure_5(cleanliness_output, cleanliness, model_France, model_Germany, model_USA)




