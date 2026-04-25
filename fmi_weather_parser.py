#
# This is for parsing fml's wind csv files 
# Returns cvs file with avarage daily windspeeds and gust
# Also creates csv file with daily avarage.
#

import pandas as pd

def read_fml_file(filename, resultname):
    
    df = pd.read_csv(filename)

    df = pd.DataFrame(df)

    df["datetime"] = pd.to_datetime(
        df["Year"].astype(str) + "-" +
        df["Month"].astype(str) + "-" +
        df["Day"].astype(str) + " " +
        df["Time [Local time]"]
    )
    df.drop(["Month", "Day", "Year", "Time [Local time]"], axis=1, inplace=True)

    #daily avg with grouby first need to floats
    df["Maximum wind speed [m/s]"] = df["Maximum wind speed [m/s]"].apply(pd.to_numeric,errors="coerce")
    

    df["datetime"] = pd.to_datetime(df["datetime"])


    df["date"] = df["datetime"].dt.date
    
    daily_avg = df.groupby("date").mean(numeric_only=True)

    #print(daily_avg)
    daily_avg.to_csv("Parsed_wind datasets/avgs_"+resultname ,index=True)
    df.to_csv("Parsed_wind datasets/"+resultname, index=False)
    return df



def summary_Statics(dataframe):
    #Creates summary of statics mean max for different wind info
    #Half vibe coded

    df = pd.read_csv("Parsed_wind datasets/"+dataframe)
    #print("for_sum")
    #print(df.dtypes)

    features = df.groupby("Observation station").agg({
        "Wind speed [m/s]": ["mean", "max", "std"],
        "Maximum wind speed [m/s]": ["mean", "max"],
        "Maximum gust speed [m/s]": ["mean", "max"]
    })
    
    features.columns = ["_".join(col) for col in features.columns]
    features = features.reset_index()  
    return features
    #print(features) 

def derived_features(key):
    df = pd.read_csv("Parsed_wind datasets/"+key)
    #print("for_features")
    #print(df.dtypes)

    features = df.groupby("Observation station").agg({
        "Wind speed [m/s]": ["mean", "std"],
        "Maximum gust speed [m/s]": "max",
        "Maximum wind speed [m/s]": "max"
    })
    features.columns = ["_".join(col) for col in features.columns]
    features = features.reset_index()  
    return features

