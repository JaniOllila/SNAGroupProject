import pandas as pd
import datetime as dt
from geopy.distance import geodesic
import csv
import fmi_weather_parser as fmi
from sklearn.preprocessing import StandardScaler
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression

stations = {
    "Hailuoto":(65, 24.7),
    "Kokkola":(63.8, 23.1),
    "Kokemaki":(61.3, 22.3),
    "Tornio":(65.9,24.2),
    "Oulu_lento":(64.9,25.3),
    "Olkiluoto":(61.1,21.3),
    "Hamina":(60.4,27),
    "Pori":(61.6,21.4),
    "Jyvaskyla":(62.4,25.7),
}

stations_dataset = {
    "Hailuoto":"Hailuoto_Marjaniemi.csv",
    "Kokkola":"Kokkola_Santahaka.csv",
    "Kokemaki":"Kokemäki_Tulkkila.csv",
    "Tornio":"Tornio_Kaakkuri.csv",
    "Oulu_lento":"Oulu_airport.csv",
    "Olkiluoto":"Rauma_Kylmäpihlaja.csv",
    "Hamina":"Kotka_Rankki.csv",
    "Pori":"Pori_Tahkoluoto_harbour.csv",
    "Jyvaskyla":"Jyvaskyla_airport.csv",
}
id_to_stat = {
    "Hailuoto":"Hailuoto Marjaniemi",
    "Kokkola":"Kokkola Santahaka",
    "Kokemaki":"Kokemäki Tulkkila",
    "Tornio":"Tornio Kaakkuri",
    "Oulu_lento":"Oulu airport",
    "Olkiluoto":"Rauma Kylmäpihlaja",
    "Hamina":"Kotka Rankki",
    "Pori":"Pori Tahkoluoto harbour",
    "Jyvaskyla":"Jyväskylä airport AWOS",
}

def distlatlon(coord1, coord2):
    return geodesic(coord1, coord2).km

def csv_write(datapair, fields, filename):
    # writing to csv file
    with open(filename, 'w', newline="", encoding="utf-8") as csvfile:

        # creating a csv dict writer object
        writer = csv.writer(csvfile)

        # writing headers (field names)
        writer.writerow(fields)

        writer.writerows(datapair)

def read_wind_dataset(key):
    return fmi.read_fml_file("Wind datasets/" + stations_dataset[key],stations_dataset[key])

#-----------------Task 1-2-----------------------------------
#read and filter data 
data_turbine = pd.read_csv("global_power_plant_database.csv")
df = pd.DataFrame(data_turbine)
filtered_df = df[df['country_long'].str.contains('Finland')]
filtered_df = filtered_df[filtered_df['primary_fuel'].str.contains('Wind')]

#print(filtered_df.info())

#-----------------Task 3-----------------------------------
#basically assaing weather station to turrbine/windfarm
vals = filtered_df.filter(items=["name","latitude","longitude"])
#print(vals.info())
list_lat_long = vals.values.tolist()
list_loc_lat_long_id = []
for loc, lat, long in list_lat_long:
    id=""
    save = 10000
    for key in stations:
        distance = distlatlon(stations[key], (lat,long))
        if distance < save: 
            id=key
            save = distance
        data_append = loc,lat,long,id

    list_loc_lat_long_id.append(data_append)



filename = "turbine_dataset"
fields = ['location', 'latitude','longitude','station_id']

csv_write(list_loc_lat_long_id, fields, filename)
#print(list_loc_lat_long_id)
df_dict = {}
for key in stations_dataset:
    df = read_wind_dataset(key)
    df_dict.update({key:df})


#-----------------Task 4-----------------------------------
#kinda useless because task 5

turbines_df = pd.read_csv(filename)

iter = turbines_df
sums = []
for index, row in iter.iterrows():
    #print(row["station_id"])
    sum = fmi.summary_Statics(stations_dataset[row["station_id"]])
    sums.append(sum)
    
sums_df = pd.concat(sums)
#print(sums_df)

turbines_df = turbines_df.reset_index(drop=True)
sums_df = sums_df.reset_index(drop=True)

single_vector = pd.concat([turbines_df, sums_df], axis=1)

#single_vector = pd.concat([turbines_df, Summary], axis=1)
single_vector.drop("Observation station", axis=1, inplace=True)
single_vector.to_csv("results", index=False)
#print(single_vector)

#-----------------Task 5-----------------------------------

def rolling_mean(path):

    df = df = pd.read_csv("Parsed_wind datasets/"+ path)

    df = df.sort_values(["Observation station", "datetime"])

    df["rolling_mean_3h"] = (
        df.groupby("Observation station")["Wind speed [m/s]"]
        .rolling(window=3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["rolling_std_3h"] = (
        df.groupby("Observation station")["Wind speed [m/s]"]
        .rolling(window=3)
        .std()
        .reset_index(level=0, drop=True)
    )

    rolling_features = df.groupby("Observation station").agg({
        "rolling_mean_3h": "mean",
        "rolling_std_3h": "mean"
    }).reset_index()

    df = df.dropna()

    return rolling_features

derived_list = []
rolling_features_list = []
#print(turbines_df)
iter = turbines_df.drop_duplicates(subset=["station_id"])

for index, row in iter.iterrows():
    #print(row["station_id"])
    der = fmi.derived_features(stations_dataset[row["station_id"]])
    rolls = rolling_mean(stations_dataset[row["station_id"]])
    derived_list.append(der)
    rolling_features_list.append(rolls)

data = pd.concat(derived_list)
data2 = pd.concat(rolling_features_list)

single_vector_rolling_derived = pd.concat([data, data2], axis=1)

#print(single_vector_rolling_derived)

# drop non-numeric column and save it for later
ids = single_vector_rolling_derived["Observation station"]
ids = ids.loc[:, ~ids.columns.duplicated()]

#print(type(ids["Observation station"]))
#print(ids["Observation station"].shape)

X = single_vector_rolling_derived.drop(columns=["Observation station"])

def scale_func(X_scale_this):
    scaler = StandardScaler()

    X_scaling = scaler.fit_transform(X_scale_this)

    # back to DataFrame
    return pd.DataFrame(X_scaling, columns=X_scale_this.columns)
      
X_scaled = scale_func(X)

X_scaled_unmodf = X_scaled

#print(X_scaled_unmodf)


#-----------------Task 6-----------------------------------
#every data frame in one dataframe

ys = []
for key in df_dict:
    y = df_dict[key]
    ys.append(y)

df_all_wind = pd.concat(ys)

def risk_score_calc(scaled_data, y_list):

    #print(y)
    model = LogisticRegression()
    model.fit(scaled_data.drop(columns=["risk_score"], errors="ignore"), y_list)

    risk_prob = model.predict_proba(scaled_data.drop(columns=["risk_score"], errors="ignore"))[:, 1]

    scaled_data["risk_score_model"] = risk_prob

    #print(X_scaled["risk_score_model"].describe())
    return scaled_data

y = (df_all_wind.groupby("Observation station")["Maximum gust speed [m/s]"].max() > 28).astype(int)

X_scaled = risk_score_calc(X_scaled, y)

#-----------------Task 7-----------------------------------
#Bringing X_scaled and turbine locations together first

X_scaled["Observation station"] = ids["Observation station"].values

iter = turbines_df

list_ids = []
for index, row in iter.iterrows():
    st = id_to_stat[row["station_id"]]
    turbines_df
    list_ids.append(st)

#df_temp = pd.concat(list_ids)
#print(df_temp)
turbines_df["Observation station"] = list_ids

#print(turbines_df)

df_combined = pd.merge(turbines_df, X_scaled, on="Observation station", how="inner")

#print(df_combined)
df_combined.to_csv("combined.csv", index=False)

G = nx.Graph()

for _, row in df_combined.iterrows():
    G.add_node(row["location"],
    lat=row["latitude"],
    long=row["longitude"])
    #risk=row["risk_score_model"]

#print(G.nodes["Huikku Hailuoto"])

#calculate distance between nodes and assig edge between them if below treshold
threshold = 300

for i, row1 in df_combined.iterrows():
    for j, row2 in df_combined.iterrows():
        if i < j:
            dist = distlatlon((row1["latitude"], row1["longitude"]),(row2["latitude"], row2["longitude"]))
            if dist <= threshold:
                G.add_edge(
                    row1["location"],
                    row2["location"],
                    weight=1
                )
        
print(G.number_of_nodes())
print(G.number_of_edges())
print(G.nodes(data=True))

import matplotlib.pyplot as plt

plt.figure(figsize=(10,7))

#pos1 = nx.spring_layout(G, iterations=30)
pos = {row["location"]: (row["longitude"], row["latitude"]) for _, row in df_combined.iterrows()}

nx.draw(G,pos, with_labels=True, node_size=200)

plt.show()

#-----------------Task 8-----------------------------------

df_test = df_combined.drop(df_combined.columns.difference(["Wind speed [m/s]_mean",  "Wind speed [m/s]_std",  "Maximum gust speed [m/s]_max",  "rolling_std_3h"]), axis=1)

similarity_matrix = cosine_similarity(df_test)

#print(similarity_matrix)

alpha = 0.3

df = df_combined

for i in range(len(df)):
    for j in range(i+1, len(df)):
        dist = distlatlon((df.iloc[i]["latitude"], df.iloc[j]["longitude"]),(df.iloc[i]["latitude"], df.iloc[j]["longitude"]))

        sim = similarity_matrix[i,j]

        dist_score = 1 / (1 + dist)  # normalize distance to calc weight accurately

        weight = alpha * sim + (1 - alpha) * dist_score  #weight calc how close physically and how similar based on similarity matrix

        if(G.has_edge(df.iloc[i]["location"],df.iloc[j]["location"])):
            G[df.iloc[i]["location"]][df.iloc[j]["location"]]["weight"] = weight

#For edge weights
#for u, v, d in G.edges(data=True):
#    print(u, v, d["weight"])


#-----------------Task 9-----------------------------------

degrees = dict(G.degree())

#biggest_node = max(degrees, key=degrees.get)
#print("Highest degree node:", biggest_node, "Degree:", degrees[biggest_node])
df_degree = pd.DataFrame({
    "location": list(degrees.keys()),
    "degree_value": list(degrees.values())
})

#plt.hist(degree_values, bins=20)
#plt.xlabel("Degree")
#plt.ylabel("Frequency")
#plt.show()

avg_clust = nx.average_clustering(G, weight="weight")
connect_comp = nx.connected_components(G)

#print(avg_clust)
#print(connect_comp)


d = {"avg_clust":avg_clust,"number_of_con_comp":connect_comp}
#the data is not in dataframes yet might do it but seem unnessesary

#-----------------Task 10-----------------------------------

deg_cent = nx.degree_centrality(G)

bet_cent = nx.betweenness_centrality(G, weight="weight")

clo_cent = nx.closeness_centrality(G)

df_centrality = pd.DataFrame({
    "location": list(deg_cent.keys()),
    "degree_centrality": list(deg_cent.values()),
    "betweenness_centrality": list(bet_cent.values()),
    "closeness_centrality": list(clo_cent.values())
})

#print(df_centrality)

df_ranked = df_centrality.sort_values(by="betweenness_centrality", ascending=False)

#print(df_ranked)

df_analysis = df_centrality.merge(df_combined[["location","risk_score_model"]],on="location")

df_analysis.sort_values(by=["risk_score_model", "betweenness_centrality"],ascending=False)

#These nodes have high risk score and also important node in network this might change if edge weights are recalc or changed 
critical = df_analysis[
    (df_analysis["risk_score_model"] > df_analysis["risk_score_model"].mean()) & (df_analysis["betweenness_centrality"] > df_analysis["betweenness_centrality"].mean())
]

print("critical locations in network: ")
print(critical)

#-----------------Task 11-----------------------------------
#df_all_wind has all of the wind data in one frame this can be groubed() with datetime
#then roll thourg dataset and calc mean for week/month and if its above certain treshold
#can identify month or weeks with high gust, wind and maxminm wind speeds.(here as reminder)
#risk_score_calc to calc risk scores this function is kinda ok but might not work well because uses .max gust speed.
#Many other functins need to be made from one time use code to make work (here as reminder)

#update can calc how turbine has different risk score depending month not perfect ---------->but need to sleep
#need to automate and probaply should calc std for each month more reliable results.

def month_avg(key):
    df = pd.read_csv("Parsed_wind datasets/"+key)

    df["datetime"] = pd.to_datetime(df["datetime"])

    df["month"] = df["datetime"].dt.month
    
    month_avg = df.groupby("month").mean(numeric_only=True)
    
    #month_avg = month_avg.groupby("datetime or any other feature").agg({
    #    "Wind speed [m/s]": ["mean", "std"],
    #    "Maximum gust speed [m/s]": "max"
    #})

    df["Observation station"] = key

    return month_avg

#monthly_avg = []

#for key in stations_dataset:
#    one_df = month_avg(stations_dataset[key])
#    one_df = one_df.agg({
#        "Wind speed [m/s]": ["mean", "std"],
#        "Maximum gust speed [m/s]": "max"
#    }).reset_index()
#    print(one_df)
#    monthly_avg.append(one_df)

#monthly_avg_df = pd.concat(monthly_avg)
for key in stations_dataset:
    month = month_avg(stations_dataset[key])
    y = (month["Maximum gust speed [m/s]"] > (month["Maximum gust speed [m/s]"].mean()) ).astype(int)
    print(y)
    df_scaled = scale_func(month)
    risk_score_df_monthly = risk_score_calc(df_scaled,y)
    print(risk_score_df_monthly)




#df_all_wind["datetime"] = pd.to_datetime(df_all_wind["datetime"])

#df_all_wind["month"] = df_all_wind["datetime"].dt.month

#df_monthly = df_all_wind.groupby(
#    ["Observation station", pd.Grouper(key="datetime", freq="MS")]).agg({
#        "Wind speed [m/s]": ["mean", "std"],
#        "Maximum gust speed [m/s]": "max"
#    }).reset_index()

#print(df_monthly)

#df_scaled = scale_func(df_monthly)
#y = (df_monthly["Maximum gust speed [m/s]"] > 20).astype(int)
#risk_score_df_monthly = risk_score_calc(df_scaled,y)

#rint(risk_score_df_monthly)