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

    #daily avg with grouby
    df["Maximum wind speed [m/s]"] = df["Maximum wind speed [m/s]"].apply(pd.to_numeric,errors="coerce")
    print(df.dtypes)

    df["datetime"] = pd.to_datetime(df["datetime"])


    df["date"] = df["datetime"].dt.date

    daily_avg = df.groupby("date").mean(numeric_only=True)

    print(daily_avg)
    daily_avg.to_csv(resultname + "_avgs",index=True)
    df.to_csv(resultname, index=False)