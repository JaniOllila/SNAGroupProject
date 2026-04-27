import pandas as pd
import datetime as dt
from geopy.distance import geodesic
import csv
import fmi_weather_parser as fmi
from sklearn.preprocessing import StandardScaler
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

stations = {
    "Hailuoto":(65, 24.7),
    "Kokkola":(63.8, 23.1),
    "Kokemaki":(61.3, 22.3),
    "Tornio":(65.9,24.2),
    "Oulu_lento":(64.9,25.3),
    "Olkiluoto":(61.1,21.3),
    "Hamina":(60.4,27),
    "Pori":(61.6,21.4),
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
print(single_vector)

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

print(single_vector_rolling_derived)

# drop non-numeric column and save it for later
ids = single_vector_rolling_derived["Observation station"]
ids = ids.loc[:, ~ids.columns.duplicated()]

print(type(ids["Observation station"]))
print(ids["Observation station"].shape)

X = single_vector_rolling_derived.drop(columns=["Observation station"])

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# back to DataFrame (optional but nice)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

X_scaled_unmodf = X_scaled
print("--------------scaled-----------")
print(X_scaled_unmodf)


#-----------------Task 6-----------------------------------
#every data frame in one dataframe

ys = []
for key in df_dict:
    y = df_dict[key]
    ys.append(y)

df_all = pd.concat(ys)

y = (df_all.groupby("Observation station")["Maximum gust speed [m/s]"].max() > 28).astype(int)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_scaled.drop(columns=["risk_score"], errors="ignore"), y)

risk_prob = model.predict_proba(X_scaled.drop(columns=["risk_score"], errors="ignore"))[:, 1]

X_scaled["risk_score_model"] = risk_prob

print(X_scaled["risk_score_model"].describe())



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

print(df_combined)
df_combined.to_csv("combined.csv", index=False)

G = nx.Graph()

for _, row in df_combined.iterrows():
    G.add_node(row["location"],
    lat=row["latitude"],
    long=row["longitude"])
    #risk=row["risk_score_model"]

#print(G.nodes["Huikku Hailuoto"])

#calculate distance between nodes and assig edge between them if below treshold
threshold = 200

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

print(similarity_matrix)

alpha = 0.3

df = df_combined

for i in range(len(df)):
    for j in range(i+1, len(df)):
        dist = distlatlon((df.iloc[i]["latitude"], df.iloc[j]["longitude"]),(df.iloc[i]["latitude"], df.iloc[j]["longitude"]))

        sim = similarity_matrix[i,j]

        dist_score = 1 / (1 + dist)  # normalize distance to calc weight accurately
        print(dist_score)
        weight = alpha * sim + (1 - alpha) * dist_score  #weight calc how close physically and how similar based on similarity matrix
        print(weight)
        if(G.has_edge(df.iloc[i]["location"],df.iloc[j]["location"])):
            G[df.iloc[i]["location"]][df.iloc[j]["location"]]["weight"] = weight

for u, v, d in G.edges(data=True):
    print(u, v, d["weight"])


#-----------------Task 9-----------------------------------

degrees = dict(G.degree())

biggest_node = max(degrees, key=degrees.get)
print("Highest degree node:", biggest_node, "Degree:", degrees[biggest_node])

degree_values = list(degrees.values())

plt.hist(degree_values, bins=20)
plt.xlabel("Degree")
plt.ylabel("Frequency")
#plt.show()

avg_clust = nx.average_clustering(G, weight="weight")
connect_comp = nx.number_connected_components(G)

print(avg_clust)
print(connect_comp)

#X_scaled_unmodf.drop(columns=["Observation station","risk_score_model","rolling_mean_3h","Maximum wind speed [m/s]_max"])
#--------------was replaced by fmi weather parser---------->>>>>>>>>>>.------------------
#filtered_df.to_csv("results_turbine", index=False)

data_wind = pd.read_csv("GlobalWeatherRepository.csv")
df = pd.DataFrame(data_wind)
filtered_df_wind = df[df['country'].str.contains('Finland')]

filtered_df_wind_parsed2 = filtered_df_wind.loc[:, filtered_df_wind.columns.intersection(["country","location_name","latitude","longitude","wind_kph","wind_degree","wind_direction","gust_kph","last_updated_epoch","last_updated"])]

filtered_df_wind_parsed2["datetime"] = pd.to_datetime(filtered_df_wind_parsed2["last_updated_epoch"], unit="s", utc = True).dt.tz_convert('Europe/Helsinki')

#print(filtered_df_wind_parsed2)

#filtered_df_wind_parsed2.to_csv("results", index=False)

#print(filtered_df_wind_parsed)
#filtered_df_wind_parsed = filtered_df_wind.drop(filtered_df_wind.columns.difference(["country","location_name","latitude","longitude","wind_kph","wind_degree","wind_direction","gust_kph"]), axis=1, inplace=True)
#filtered_df_wind = filtered_df_wind[filtered_df_wind['primary_fuel'].str.contains('Wind')]
#"country","location_name","latitude","longitude","wind_kph","wind_degree","wind_direction","gust_kph"