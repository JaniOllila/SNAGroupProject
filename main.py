import pandas as pd
import datetime as dt
from geopy.distance import geodesic
import csv
import fmi_weather_parser as fmi
from sklearn.preprocessing import StandardScaler
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

stations = {
    "Hailuoto":(65, 24.7),
    "Kokkola":(63.8, 23.1),
    "Kokemaki":(61.3, 22.3),
    "Tornio":(65.9,24.2),
    "Oulu_lento":(64.9,25.3),
    "Olkiluoto":(61.1,21.3),
    "Hamina":(60.4,27),
    "Pori":(61.6,21.4),
    #"Jyvaskyla":(62.4,25.7),
    "Kemi_ajos":(65.7,24.5)
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
    "Kemi_ajos":"Kemi_Ajos.csv"
    #"Jyvaskyla":"Jyvaskyla_airport.csv",
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
    #Jyvaskyla":"Jyväskylä airport AWOS",
    "Kemi_ajos":"Kemi Ajos"
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

#csv_write(list_loc_lat_long_id, fields, filename)
#df_dict = {}
#for key in stations_dataset:
#    df = read_wind_dataset(key)
#    df_dict.update({key:df})

#create one pandas dataframe containing all wind data and write csv file
#ys = []
#for key in df_dict:
#    y = df_dict[key]
#    ys.append(y)

#df_all_wind_tocsv = pd.concat(ys)

#df_all_wind_tocsv.to_csv("csv_all_wind.csv", index=False)

df_all_wind = pd.read_csv("csv_all_wind.csv")

#-----------------Task 4-----------------------------------

turbines_df = pd.read_csv(filename)

iter = turbines_df
sums = []
for index, row in iter.iterrows():
    sum = fmi.summary_Statics(stations_dataset[row["station_id"]])
    sums.append(sum)
    
sums_df = pd.concat(sums)

turbines_df = turbines_df.reset_index(drop=True)
sums_df = sums_df.reset_index(drop=True)

single_vector = pd.concat([turbines_df, sums_df], axis=1)


single_vector.drop("Observation station", axis=1, inplace=True)
#single_vector.to_csv("results", index=False)

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
iter = turbines_df.drop_duplicates(subset=["station_id"])

for index, row in iter.iterrows():
    der = fmi.derived_features(stations_dataset[row["station_id"]])
    rolls = rolling_mean(stations_dataset[row["station_id"]])
    derived_list.append(der)
    rolling_features_list.append(rolls)

data = pd.concat(derived_list)
data2 = pd.concat(rolling_features_list)

single_vector_rolling_derived = pd.concat([data, data2], axis=1)
single_vector_rolling_derived_cp = single_vector_rolling_derived.loc[:, ~single_vector_rolling_derived.columns.duplicated()]

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

#-----------------Task 6-----------------------------------

def risk_score_calc(scaled_data, y_list):

    model = LogisticRegression()
    model.fit(scaled_data.drop(columns=["risk_score"], errors="ignore"), y_list)

    risk_prob = model.predict_proba(scaled_data.drop(columns=["risk_score"], errors="ignore"))[:, 1]

    scaled_data["risk_score_model"] = risk_prob

    return scaled_data

y = (df_all_wind.groupby("Observation station")["Maximum gust speed [m/s]"].max() > 30).astype(int)
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

turbines_df["Observation station"] = list_ids

df_combined = pd.merge(turbines_df, X_scaled, on="Observation station", how="inner")

single_vector_rolling_derived_cp = pd.merge(turbines_df, single_vector_rolling_derived_cp, on="Observation station", how="inner")
single_vector_rolling_derived_cp["risk_score_model"] = df_combined["risk_score_model"]

df_combined.to_csv("combined.csv", index=False)

single_vector_rolling_derived_cp.to_csv("Unscaled_combined.csv", index=False)

G = nx.Graph()

for _, row in df_combined.iterrows():
    G.add_node(row["location"],
    lat=row["latitude"],
    long=row["longitude"],
    risk_sc=row["risk_score_model"])
    #risk=row["risk_score_model"]

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
        



#score_dict = dict(G.nodes(data="risk_sc"))
#for node in score_dict:
#    print(score_dict[node])

#-----------------Task 8-----------------------------------

df_test = df_combined.drop(df_combined.columns.difference(["Wind speed [m/s]_mean",  "Wind speed [m/s]_std",  "Maximum gust speed [m/s]_max",  "rolling_std_3h","rolling_mean_3h"]), axis=1)

similarity_matrix = cosine_similarity(df_test)

alpha = 0.7

df = df_combined

for i in range(len(df)):
    for j in range(i+1, len(df)):

        dist = distlatlon((df.iloc[i]["latitude"], df.iloc[i]["longitude"]),(df.iloc[j]["latitude"], df.iloc[j]["longitude"]))
        
        sim = similarity_matrix[i,j]


        dist = (dist/20)**1.2 #IF this smaller then dist score up
        dist_score = (1 / (1 + dist))  # normalize distance to calc weight accurately


        if(dist_score == 1): # avoid cases where same coordinates
            weight1 = 1
        else:
            weight1 = (alpha * sim) + ((1) * (dist_score))  #weight calc how close physically and how similar based on similarity matrix

        if(G.has_edge(df.iloc[i]["location"],df.iloc[j]["location"])):
            G[df.iloc[i]["location"]][df.iloc[j]["location"]]["weight"] = weight1
        #elif(weight1>0.6):
        #else:
        #    G.add_edge(df.iloc[i]["location"],df.iloc[j]["location"],weight=weight1)

#For edge weights
#for u, v, d in G.edges(data=True):
#    print(u, v, d["weight"])



pos = {row["location"]: (row["longitude"], row["latitude"]) for _, row in df_combined.iterrows()}

pos["Tahkoluoto Offshore tv"] = (21.7833,61.7)


#nx.draw(G,pos,node_size=200,with_labels=True)
labels = nx.get_edge_attributes(G, "weight")
#, with_labels=True
labels_round = {}
for edge, weight in labels.items():
    rounded = {edge:f"{weight:.3f}"}
    labels_round.update(rounded)

plt.figure(figsize=(10,7))
nx.draw_networkx_nodes(G, pos, node_size=200)
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G, pos, font_size=10)

nx.draw_networkx_edge_labels(G, pos, edge_labels=labels_round)
plt.tight_layout()
#nx.draw_networkx_edge_labels(G,pos,edge_labels=labels,connectionstyle="arc3")
plt.savefig("plots_and_fiqures/network.png")
#plt.show()

nx.write_gexf(G, "graph.gexf")
#-----------------Task 9-----------------------------------

degrees = dict(G.degree())

#biggest_node = max(degrees, key=degrees.get)
#print("Highest degree node:", biggest_node, "Degree:", degrees[biggest_node])
df_degree = pd.DataFrame({
    "location": list(degrees.keys()),
    "degree_value": list(degrees.values())
})

degree_values = list(degrees.values())
#plt.hist(degree_values, bins=20)
#plt.xlabel("Degree")
#plt.ylabel("Frequency")
#plt.savefig("plots_and_fiqures/degree_frequency.png")
#plt.show()
#
avg_clust = nx.average_clustering(G, weight="weight")
connect_comp = nx.number_connected_components(G)
diameter = nx.diameter(G)
print(connect_comp)

d = {"avg_clust":avg_clust,"number_of_con_comp":connect_comp,"diameter":diameter}

df_network_metrics = pd.DataFrame(data=d, index=[0])
df_network_metrics.to_csv("network_metrics.csv")
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

df_ranked = df_centrality.sort_values(by="betweenness_centrality", ascending=False)

df_analysis = df_centrality.merge(df_combined[["location","risk_score_model"]],on="location")

df_analysis.sort_values(by=["risk_score_model", "betweenness_centrality"],ascending=False)

#These nodes have high risk score and also important node in network this might change if edge weights are recalc or changed 
critical = df_analysis[
    (df_analysis["risk_score_model"] > df_analysis["risk_score_model"].mean()) & (df_analysis["betweenness_centrality"] > df_analysis["betweenness_centrality"].mean())
]

#for _, row in df_analysis.iterrows():
#    G[row["location"]]["betweenes"] = row["betweenness_centrality"]

nx.set_node_attributes(G, bet_cent, "betweenness")
nx.set_node_attributes(G, deg_cent, "degree_centrality")
nx.set_node_attributes(G, clo_cent, "closeness_centrality")

print(G.number_of_nodes())
print(G.number_of_edges())
print(G.nodes(data=True))


print("critical locations in network: ")
print(critical)
color_map = []
for node in G:
    if node in critical["location"].to_list():
        color_map.append("red")
    else:
        color_map.append("steelblue")
        
plt.figure(figsize=(10,7))  


nx.draw_networkx_nodes(G, pos, node_size=200,node_color=color_map)
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G, pos, font_size=9)   
nx.draw_networkx_edge_labels(G,pos,labels_round, font_size=6)


plt.tight_layout()
plt.savefig("plots_and_fiqures/network2.png")
plt.show()

network_node_metrics = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')
network_node_metrics = network_node_metrics.drop(columns=["lat","long"]).to_csv("network_node_metrics.csv")

#-----------------Task 11-----------------------------------

def month_avg(key, all_wind=0):
    if(all_wind == 0):
        df = pd.read_csv("Parsed_wind datasets/"+key)
    else:
        df = df_all_wind

    df["datetime"] = pd.to_datetime(df["datetime"])

    df["month"] = df["datetime"].dt.month
    
    month_avg = df.groupby("month").mean(numeric_only=True)
    month_std= df.groupby("month").std(numeric_only=True)

    month_std = month_std.drop(month_std.columns.difference(["Wind speed [m/s]"]),axis=1)
    month_avg = month_avg.join(month_std, rsuffix="_std")
    
    return month_avg

def day_avg(key, all_wind=0):
    if(all_wind == 0):
        df = pd.read_csv("Parsed_wind datasets/"+key)
    else:
        df = df_all_wind

    df["datetime"] = pd.to_datetime(df["datetime"])

    df["day"] = df["datetime"].dt.hour
    
    day_avg = df.groupby("day").mean(numeric_only=True)
    day_std= df.groupby("day").std(numeric_only=True)

    day_std = day_std.drop(day_std.columns.difference(["Wind speed [m/s]"]),axis=1)
    day_avg = day_avg.join(day_std, rsuffix="_std")
    
    return day_avg

def day_avg_year(key, all_wind=0):
    if(all_wind == 0):
        df = pd.read_csv("Parsed_wind datasets/"+key)
    else:
        df = df_all_wind

    df["datetime"] = pd.to_datetime(df["datetime"])

    df["day"] = df["datetime"].dt.day_of_year
    
    day_avg = df.groupby("day").mean(numeric_only=True)
    day_std= df.groupby("day").std(numeric_only=True)

    day_std = day_std.drop(day_std.columns.difference(["Wind speed [m/s]"]),axis=1)
    day_avg = day_avg.join(day_std, rsuffix="_std")
    
    return day_avg

list_dfs = []
list_dfs2 = []
for key in stations_dataset:
    month = month_avg(stations_dataset[key])
    y = (month["Maximum gust speed [m/s]"] > (month["Maximum gust speed [m/s]"].mean() + 0.2) ).astype(int)
    month_cp = month.drop(["Maximum wind speed [m/s]"],axis=1)
    df_scaled = scale_func(month_cp)
    risk_score_df_monthly = risk_score_calc(df_scaled,y)
    month["station_id"] = key
    list_dfs.append(month)
    risk_score_df_monthly["station_id"] = key
    list_dfs2.append(risk_score_df_monthly)
    
df_month_risk_score_each_station = pd.concat(list_dfs)
#print(df_month_risk_score_each_station)

plt.figure(figsize=(12,7))
plt.subplot(1,2,1)
for i in list_dfs:
    plt.plot(i.index,i["Wind speed [m/s]"], label=i["station_id"][1])

plt.legend()
plt.title("Wind Trends through year per station")
plt.xlabel("Month")
plt.ylabel("Wind (m/s) mean")
plt.xticks(rotation=45)

plt.subplot(1,2,2)
for i in list_dfs2:
    plt.plot(i.index,i["risk_score_model"], label=i["station_id"][1])

plt.legend()
plt.title("Risk score Trends through year per station")
plt.xlabel("Month")
plt.ylabel("Risk score")
plt.xticks(rotation=45)
plt.savefig("plots_and_fiqures/Month_based_risk_sc.png")
#plt.show()

all_stations_month_avg = month_avg("place",1)

y = (all_stations_month_avg["Maximum gust speed [m/s]"] > (all_stations_month_avg["Maximum gust speed [m/s]"].mean())).astype(int)
month_cp = all_stations_month_avg.drop(["Maximum wind speed [m/s]"],axis=1)
df_scaled = scale_func(month_cp)
risk_score_df_monthly = risk_score_calc(df_scaled,y)
#print(risk_score_df_monthly)

values_month_avg = all_stations_month_avg["Maximum gust speed [m/s]"].to_list()

df_day_avgs = day_avg("location",1)
#print(df_day_avgs)
#print(df_day_avgs.info())

plt.figure(figsize=(12,7))
plt.subplot(1,2,1)
plt.bar(all_stations_month_avg.index,values_month_avg)
plt.xlabel("month")
plt.ylabel("Maximum gust speed [m/s]")
plt.title("Monthly distrupution of Maximum gust speed")

plt.subplot(1,2,2)
values_month_avg = risk_score_df_monthly["risk_score_model"].to_list()
plt.bar(all_stations_month_avg.index,values_month_avg)
plt.xlabel("month")
plt.ylabel("Risk_score")
plt.title("Monthly risk_score")
plt.savefig("plots_and_fiqures/Month_based_risk_colloctive_sc.png")
#plt.show()

plt.figure(figsize=(10,7))

risk_score_all = df_combined["risk_score_model"].to_list()
stations_lista = df_combined["location"].to_list()
plt.bar(stations_lista,risk_score_all, width=0.5)
plt.xlabel("Station")
plt.ylabel("Risk_score")
plt.title("risk_score for each station")
plt.xticks(rotation = 90)
plt.tight_layout()
plt.savefig("plots_and_fiqures/Risk_score_of_each_stat.png")
#plt.show()


y = (df_day_avgs["Wind speed [m/s]"] > (df_day_avgs["Wind speed [m/s]"].mean())).astype(int)
day_cp = df_day_avgs.drop(["Maximum wind speed [m/s]"],axis=1)
df_scaled = scale_func(day_cp)
risk_score_df_day = risk_score_calc(df_scaled,y)


plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(df_day_avgs.index, df_day_avgs["Wind speed [m/s]"], label="Mean Wind")
#print(df_day_avgs['Wind speed [m/s]_std'])
#lower = df_day_avgs["Wind speed [m/s]"] - df_day_avgs['Wind speed [m/s]_std']
#upper = df_day_avgs["Wind speed [m/s]"] + df_day_avgs['Wind speed [m/s]_std']

#plt.fill_between(df_day_avgs.index, lower, upper, color='gray', alpha=0.3)
plt.legend()
plt.title("Wind Trends during one day")
plt.xlabel("Hours")
plt.ylabel("Wind (m/s)")
plt.xticks(rotation=45)

plt.subplot(1,2,2)
plt.plot(risk_score_df_day.index, risk_score_df_day["risk_score_model"], label="score")

plt.legend()
plt.title("Risk score Trends during day")
plt.xlabel("Hours")
plt.ylabel("score")
plt.xticks(rotation=45)
plt.savefig("plots_and_fiqures/day_wind_vs_score.png")

df_day_avgs_year = day_avg_year("location",1)

y = (df_day_avgs_year.groupby("day")["Maximum gust speed [m/s]"].max() > 9).astype(int)

#y = (df_day_avgs_year["Wind speed [m/s]"] > (df_day_avgs_year["Wind speed [m/s]"].mean())).astype(int)
day_cp = df_day_avgs_year.drop(["Maximum wind speed [m/s]"],axis=1)
df_scaled = scale_func(day_cp)
risk_score_df_day_year = risk_score_calc(df_scaled,y)

def rolling_mean_func(df,column,window):
    df["rolling_mean_5"] = (
    df[column]
    .rolling(window=window)
    .mean()
    .reset_index(level=0, drop=True))

    return df

risk_score_df_day_year = rolling_mean_func(risk_score_df_day_year,"risk_score_model",5)
df_day_avgs_year = rolling_mean_func(df_day_avgs_year,"Wind speed [m/s]",20)

plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.plot(df_day_avgs_year.index, df_day_avgs_year["rolling_mean_5"], label="Mean Wind")

plt.legend()
plt.title("Wind Trends during year")
plt.xlabel("days")
plt.ylabel("Wind (m/s)")
plt.xticks(rotation=45)

plt.subplot(1,2,2)
plt.plot(risk_score_df_day_year.index, risk_score_df_day_year["rolling_mean_5"], label="score")
plt.legend()
plt.title("Risk score Trends during day")
plt.xlabel("days")
plt.ylabel("score")
plt.xticks(rotation=45)
plt.tight_layout()
#plt.show()
plt.savefig("plots_and_fiqures/day_wind_vs_score_year.png")


#tulos = df_all_wind[df_all_wind["Maximum gust speed [m/s]"] > 25]
#print(tulos)


#-----------------Task 12-----------------------------------
#in file probality.py