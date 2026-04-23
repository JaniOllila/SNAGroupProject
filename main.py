import pandas as pd
import datetime as dt
from geopy.distance import geodesic

data_turbine = pd.read_csv("global_power_plant_database.csv")
df = pd.DataFrame(data_turbine)
filtered_df = df[df['country_long'].str.contains('Finland')]
filtered_df = filtered_df[filtered_df['primary_fuel'].str.contains('Wind')]

print(filtered_df.info())

helsinki = (60.18,24.93)

vals = filtered_df.filter(items=["name","latitude","longitude"])
print(vals.info())
list_lat_long = vals.values.tolist()
list_loc_lat_long_dist = []
for loc, lat, long in list_lat_long:
    distance = geodesic(helsinki, (lat,long)).km
    data_append = loc,lat,long,distance
    list_loc_lat_long_dist.append(data_append)

#print(list_lat_long)
print(list_loc_lat_long_dist)

#print("Filtered DataFrame:")
#print(filtered_df)
filtered_df.to_csv("results_turbine", index=False)

data_wind = pd.read_csv("GlobalWeatherRepository.csv")
df = pd.DataFrame(data_wind)
filtered_df_wind = df[df['country'].str.contains('Finland')]

filtered_df_wind_parsed2 = filtered_df_wind.loc[:, filtered_df_wind.columns.intersection(["country","location_name","latitude","longitude","wind_kph","wind_degree","wind_direction","gust_kph","last_updated_epoch","last_updated"])]

filtered_df_wind_parsed2["datetime"] = pd.to_datetime(filtered_df_wind_parsed2["last_updated_epoch"], unit="s", utc = True).dt.tz_convert('Europe/Helsinki')

#filtered_df_wind_parsed2["datetime"] = df.apply(lambda x: x['dt'].tz_localize('UTC').tz_convert('Europe/Finland'), axis=1)

filtered_df_wind_parsed2["datetime"] = pd.to_datetime(filtered_df_wind_parsed2["last_updated_epoch"], unit="s", utc = True).dt.tz_convert('Europe/Helsinki')

print(filtered_df_wind_parsed2)

filtered_df_wind_parsed2.to_csv("resultss", index=False)

#print(filtered_df_wind_parsed)
#filtered_df_wind_parsed = filtered_df_wind.drop(filtered_df_wind.columns.difference(["country","location_name","latitude","longitude","wind_kph","wind_degree","wind_direction","gust_kph"]), axis=1, inplace=True)
#filtered_df_wind = filtered_df_wind[filtered_df_wind['primary_fuel'].str.contains('Wind')]
#"country","location_name","latitude","longitude","wind_kph","wind_degree","wind_direction","gust_kph"