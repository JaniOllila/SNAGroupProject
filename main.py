import pandas as pd
import datetime as dt
from geopy.distance import geodesic
import csv
import fmi_weather_parser as fmi

stations = {
    "Hailuoto":(65, 24.7),
    "Kokkola":(68, 23.1),
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
    fmi.read_fml_file("Wind datasets/" + stations_dataset[key],stations_dataset[key])


data_turbine = pd.read_csv("global_power_plant_database.csv")
df = pd.DataFrame(data_turbine)
filtered_df = df[df['country_long'].str.contains('Finland')]
filtered_df = filtered_df[filtered_df['primary_fuel'].str.contains('Wind')]

print(filtered_df.info())

vals = filtered_df.filter(items=["name","latitude","longitude"])
print(vals.info())
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

daily_avg = read_wind_dataset("Hailuoto")

print(list_loc_lat_long_id)

#returns dataframe of summary statics
Summary = fmi.summary_Statics(stations_dataset["Hailuoto"])
print(Summary)

turbines_df = pd.read_csv(filename)

single_vector = pd.concat([turbines_df, Summary], axis=1)
single_vector.drop("Observation station", axis=1, inplace=True)
single_vector.to_csv("results", index=False)
print(single_vector)
#--------------maybe replaced by fmi weather parser.----------------------
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