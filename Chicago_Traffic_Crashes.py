#!/usr/bin/env python
# coding: utf-8

# # Investigation on Traffic Crashes in Chicago

# 
# # Table of Contents
# 1. [Abstract](#abstract)
# 2. [Exploratory Data Analysis](#exploratory-data-analysis)
# 3. [Modeling](#modeling)
# 4. [Conclusion](#conclusion)

# <a id='abstract'></a>
# # I. Abstract
# 
# Traffic crash is not only an major topic in Geographic AI, but also directly ralated to people's daily life. Nearly 1.3 million people die internationally every year from car accidents and in addition up to 50 million people are injured. Machine learning can be a helpful tool to analyze and reduce the risk of crashes. In this work, an data analysis is demonstrated on Chicago traffic crashes (20016-2018), followed by a prediction of risk and severity of crashes based on random forrest model. This work is a promising application for safe route planning, emergency vehicle allocation, roadway design and where to place additional traffic control devices.

# <a id='exploratory-data-analysis'></a>
# # II. Exploratory Data Analysis

# ## Data Source
# Traffic crash data can be obtained at [Chicago Data Portal](https://data.cityofchicago.org/), using SODA API or exporting directly from the website. The links and descriptions of datasets are listed below:
# 1. [Traffic Crashes - Crashes](https://data.cityofchicago.org/Transportation/Traffic-Crashes-Crashes/85ca-t3if): Major dataset for this project.
# 2. [Traffic Crashes - Vehicles](https://data.cityofchicago.org/Transportation/Traffic-Crashes-Vehicles/68nd-jvt3): Information of related vehicles.
# 3. [Traffic Crashes - People](https://data.cityofchicago.org/Transportation/Traffic-Crashes-People/u6pd-qa9d):Information of related people.
# 4. [Chicago Traffic Tracker - Congestion Estimates by Segments](https://data.cityofchicago.org/Transportation/Chicago-Traffic-Tracker-Congestion-Estimates-by-Se/n4j6-wkkf): Geo information of traffic segments of Chicago arterial streets (nonfreeway streets).
# 5. [Chicago Traffic Tracker - Congestion Estimates by Regions](https://data.cityofchicago.org/Transportation/Chicago-Traffic-Tracker-Congestion-Estimates-by-Re/t2qc-9pjd): Geo information of Chicago Regions.
# 
# 

# ## Data Processing

# In[297]:


import pandas as pd
import numpy as np
import geopandas as gpd
from shapely import wkt
import matplotlib.pyplot as plt
import folium
from folium import plugins
import seaborn as sns


# Load the dataset into dataframe and have a glance.

# In[298]:


crash_raw = pd.read_csv("data/Traffic_Crashes_-_Crashes.csv", parse_dates=['CRASH_DATE','DATE_POLICE_NOTIFIED'])
crash_raw.head()


# In[299]:


crash_raw.shape


# Select crashes in 2016-2018

# In[300]:


crash = crash_raw[(crash_raw['CRASH_DATE'] < pd.datetime(2019,1,1)) & (crash_raw['CRASH_DATE'] >= pd.datetime(2016,1,1))].copy()





# Deal with outliers and missing values in geo coordinates(LATITUDE, LONGITUDE), INJURIES_FATAL and POSTED_SPEED_LIMIT, then convert dataframe to geo dataframe.

# In[301]:


crash = crash[crash['LOCATION'].notnull()]
#crash.boxplot(column=['LONGITUDE', 'LATITUDE'])
# drop rows with outliers in LATITUDE and LONGITUDE (outlier defiend as being out of 10 std range)
crash = crash[np.abs(crash['LATITUDE'] - crash['LATITUDE'].mean()) <= (10 * crash['LATITUDE'].std())]
crash = crash[np.abs(crash['LONGITUDE'] - crash['LONGITUDE'].mean()) <= (10 * crash['LONGITUDE'].std())]

crash = crash[crash['INJURIES_FATAL'].notnull()]
crash = crash[crash['MOST_SEVERE_INJURY'].notnull()]

crash = crash[(crash['POSTED_SPEED_LIMIT'] > 0) & (crash['POSTED_SPEED_LIMIT'] % 5 == 0) ]
# convert datagframe to geo-dataframe
crash['LOCATION'] = crash['LOCATION'].apply(wkt.loads)
#crs = {'init': 'epsg:4326'}
crash = gpd.GeoDataFrame(crash, geometry='LOCATION')

crash.shape


# ## Data Analysis

# ### An overview of geographic distribution
# What does the 245k traffic crashes look like?

# In[302]:


crash.plot(markersize=0.01, edgecolor='red',figsize=(12,12));
plt.axis('off');
plt.title('Crash in Chicago from 2016 to 2018')


# In[303]:


"""crash.info()
selected_clomuns = ['RD_NO','CRASH_DATE','POSTED_SPEED_LIMIT',
                    'TRAFFIC_CONTROL_DEVICE', 'DEVICE_CONDITION','WEATHER_CONDITION','LIGHTING_CONDITION',
                    'ROAD_DEFECT','INTERSECTION_RELATED_I',
                    'STREET_NO','STREET_DIRECTION','STREET_NAME ',
                    'LATITUDE','LONGITUDE','LOCATION ',
                    'CRASH_HOUR','CRASH_DAY_OF_WEEK','CRASH_MONTH ',
                    'MOST_SEVERE_INJURY '
                   ]"""


# ### Number of crashes by a single feature

# In[304]:


crash['CRASH_TYPE'].value_counts().plot(kind='bar', title ="CRASH_TYPE", figsize=(10, 5), legend=True, fontsize=12)


# In[305]:


crash['DAMAGE'].value_counts().plot(kind='bar', title ="DAMAGE", figsize=(10, 5), legend=True, fontsize=12)


# In[306]:


crash['MOST_SEVERE_INJURY'].value_counts().plot(kind='bar', title ="MOST_SEVERE_INJURY", figsize=(10, 5), legend=True, fontsize=12)


# In[307]:


crash['WEATHER_CONDITION'].value_counts().plot(kind='bar', title ="WEATHER_CONDITION", figsize=(10, 5), legend=True, fontsize=12)


# In[308]:


crash['ROADWAY_SURFACE_COND'].value_counts().plot(kind='bar', title ="road surface condition", figsize=(10, 5), legend=True, fontsize=12)


# ### Number of crashes by time-related varables

# In[309]:


crash.groupby(['CRASH_MONTH']).count()['RD_NO'].plot(kind='bar', title ="Crash by Month", figsize=(10, 5), legend=True, fontsize=12)


# More crashes happen in Sep to Oct, namely late autumn and early winter.

# *Crashes by day of week (Sunday == 1)*

# In[310]:


crash.groupby(['CRASH_DAY_OF_WEEK']).count()['RD_NO'].plot(kind='bar', title ="Crash by Day", figsize=(10, 5), legend=True, fontsize=12)


# More crashes happen in Friday 

# *Crashes by hours*

# In[311]:


crash.groupby(['CRASH_HOUR']).count()['RD_NO'].plot(kind='bar', title ="Crash by Hour", figsize=(10, 5), legend=True, fontsize=12)


# More crashes happen in rush hours 

# ### Number of crashes by location-realated variables

# *Dynamic heapmap showing geographic distribution of crash by month in 2018*

# In[312]:


map_chicago = folium.Map(location=[41.830994, -87.647345],
                         tiles = "Stamen Terrain",
                         zoom_start = 10) 

crash2018 = crash[(crash['CRASH_DATE'] < pd.datetime(2019,1,1)) & (crash['CRASH_DATE'] >= pd.datetime(2018,1,1))].copy()

heatmap = []
for i in range(1, 13):
    df = crash2018[crash2018['CRASH_MONTH'] == i]
    df1 = df.sample(int(len(df)*0.3))
    cood = [[row["LATITUDE"], row["LONGITUDE"]] for idx, row in df1.iterrows()]
    heatmap.append(cood)
    
plugins.HeatMapWithTime(heatmap, radius=3, auto_play=True,max_opacity=0.8).add_to(map_chicago)
map_chicago


# *Dynamic heapmap showing geographic distribution of crash by hour in 2018*

# In[313]:


map_chicago = folium.Map(location=[41.830994, -87.647345],
                         tiles = "Stamen Terrain",
                         zoom_start = 10) 

crash2018 = crash[(crash['CRASH_DATE'] < pd.datetime(2019,1,1)) & (crash['CRASH_DATE'] >= pd.datetime(2018,1,1))].copy()

heatmap = []
for i in range(0,24):
    df = crash2018[crash2018['CRASH_HOUR'] == i]
    df1 = df.sample(int(len(df)*0.1))
    cood = [[row["LATITUDE"], row["LONGITUDE"]] for idx, row in df1.iterrows()]
    heatmap.append(cood)
    
plugins.HeatMapWithTime(heatmap, radius=5, auto_play=True,max_opacity=0.8).add_to(map_chicago)
map_chicago


# In[314]:


#map_chicago.save('crash heatmap.html')


# *number of crashes by street*

# In[315]:


crash['STREET_NAME'].value_counts()[:min(20, len(crash))].plot(kind='bar', title ="Crash by Street", figsize=(10, 5), legend=True, fontsize=12)


# <a id='modeling'></a>
# # Modeling

# In[316]:



features = ['POSTED_SPEED_LIMIT',
            'TRAFFIC_CONTROL_DEVICE', 'DEVICE_CONDITION','WEATHER_CONDITION','LIGHTING_CONDITION',
            'ROAD_DEFECT',
            'STREET_NO','STREET_DIRECTION','STREET_NAME',
            'LATITUDE','LONGITUDE',
            'CRASH_HOUR','CRASH_DAY_OF_WEEK','CRASH_MONTH',
            'MOST_SEVERE_INJURY','DAMAGE','FIRST_CRASH_TYPE','TRAFFICWAY_TYPE',
            'INJURIES_FATAL', 'INJURIES_INCAPACITATING'
            ]


# In[317]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler


# In[318]:


# Convert geo-dataframe into a regular dataframe.
df = pd.DataFrame(crash[features])
#df.head()
#ss_lat = StandardScaler()
#df['LATITUDE'] = ss_lat.fit_transform(df['LATITUDE'].values.reshape(-1,1)).flatten()
#ss_lon = StandardScaler()
#df['LONGITUDE'] = ss_lon.fit_transform(df['LONGITUDE'].values.reshape(-1,1)).flatten()

#df.shape
df.describe()


# In[319]:


df.info()


# In[320]:


"""
cols = ['POSTED_SPEED_LIMIT', 'TRAFFIC_CONTROL_DEVICE', 'DEVICE_CONDITION',
       'WEATHER_CONDITION', 'LIGHTING_CONDITION', 'ROAD_DEFECT', 
       'STREET_DIRECTION', 'STREET_NAME', 
       'CRASH_DAY_OF_WEEK', 'CRASH_MONTH'
       ]

for i in range(len(cols)):
    print(cols[i])
    df[cols[i]].value_counts()
"""


# In[321]:


corr_mat = df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr_mat, vmax=.8, square=True)


# In[322]:


# Encoding catagory variables
from sklearn.preprocessing import LabelEncoder
lblE = LabelEncoder()
for i in df:
    if df[i].dtype == 'object':
        lblE.fit(df[i])
        df[i] = lblE.transform(df[i])
df = pd.get_dummies(df)
df.head()


# In[323]:


X_train, X_test, y_train, y_test = train_test_split(df.drop(['INJURIES_FATAL', 'INJURIES_INCAPACITATING'], axis=1), 
                                                    df['INJURIES_FATAL'], test_size=0.33, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[324]:


def rmse(x,y): return np.sqrt(((x-y)**2).mean())
def print_score(m):
    res = [rmse(m.predict(X_train), y_train), 
           rmse(m.predict(X_test), y_test),
           m.score(X_train, y_train), 
           m.score(X_test, y_test)]
    
    if hasattr(m, 'oob_score_'):res.append(m.oob_score_)
    print ("\nRMSE for train set: ", res[0],
           "\nRMSE for test set: ", res[1],
           "\nScore for train set: ", res[2],
           "\nScore for test set: ", res[3]
          )


# In[325]:



rfr = RandomForestRegressor(n_estimators=50)
rfr.fit(X_train, y_train)
print_score(rfr)


# In[326]:


f_imp = pd.DataFrame(data={'importance':rfr.feature_importances_,'features':X_train.columns}).set_index('features')
f_imp = f_imp.sort_values('importance', ascending=False)
f_imp


# Besides the non-surprising result that 'MOST_SEVERE_INJURY' is most related, we can see in the feature importance table that:
# 1. The street where the crash happened are more related to the number of death.
# 2. The hour and day of week are more related to the number of death, comparede to the month.
# 3. Weather and traffic control device seems not related to the number of death, which is not a straightforward result.

# <a id='conclusion'></a>
# # Conclusion
# 
# By analyzing the modeling the traffic crash data, we have successfully predict the severity of crashes based on Random Forrest Regression with a score of 0.89. Future works will be predicting the probobility of crashes per raod segment and per hour. 

# In[ ]:




