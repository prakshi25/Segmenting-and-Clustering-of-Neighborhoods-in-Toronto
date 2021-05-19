#!/usr/bin/env python
# coding: utf-8

# <h2>Segmenting and Clustering Neighborhoods in Toronto | Part-1</h2>

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_html('https://en.wikipedia.org/w/index.php?title=List_of_postal_codes_of_Canada:_M&oldid=945633050')[0]
df.head()


# In[3]:


df.drop(df.index[df['Borough'] == 'Not assigned'], inplace = True)
df.head()


# In[4]:


df['Neighbourhood'] = df.groupby(['Postcode', 'Borough'])['Neighbourhood'].transform(lambda x: ', '.join(x))
df = df.drop_duplicates()
df


# In[5]:


df = df.reset_index()
del df['index']
df.head()


# In[6]:


df['Neighbourhood'].replace('Not assigned', df['Borough'])
df.head()


# In[7]:


df.shape


# In[8]:


df.to_csv('toronto_part1.csv',index=False)


# <h2> Segmenting and Clustering Neighborhoods in Toronto | Part-2 </h2>

# In[9]:


get_ipython().system('pip install geocoder')


# In[10]:


import pandas as pd
import numpy as np
import geocoder


# In[11]:


def get_latilong(post_code):
    lati_long_coords = None
    while(lati_long_coords is None):
        g = geocoder.arcgis('{}, Toronto, Ontario'.format(post_code))
        lati_long_coords = g.latlng
    return lati_long_coords
    
get_latilong('M3A')


# In[12]:


postal_codes = df['Postcode']
coords = [ get_latilong(post_code) for post_code in postal_codes.tolist() ]


# In[13]:


df_coords = pd.DataFrame(coords, columns=['Latitude', 'Longitude'])
df['Latitude'] = df_coords['Latitude']
df['Longitude'] = df_coords['Longitude']


# In[14]:


df[df.Postcode == 'M3A']


# In[15]:


df.head(15)


# In[16]:


df.to_csv('toronto_part2.csv',index=False)


# <h2> Segmenting and Clustering Neighborhoods in Toronto | Part-3 </h2>

# In[17]:


get_ipython().system('pip install geopy')
get_ipython().system('pip install folium')


# In[18]:


import folium
import requests 
import json 
import matplotlib.cm as cm
import matplotlib.colors as colors
import pandas as pd

from pandas.io.json import json_normalize 
from sklearn.cluster import KMeans
from geopy.geocoders import Nominatim 

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[19]:


df = pd.read_csv('toronto_part2.csv')
print(df.shape)
df.head()


# In[20]:


address = 'Toronto, Ontario Canada'
geolocator = Nominatim(user_agent="minnieraval@gmail.com")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Toronto Canada are {}, {}.'.format(latitude, longitude))


# In[21]:


map_toronto = folium.Map(location=[latitude, longitude], zoom_start=11)
for lat, lng, borough, neighborhood in zip(df['Latitude'], df['Longitude'], df['Borough'], df['Neighbourhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=4,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#87cefa',
        fill_opacity=0.5,
        parse_html=False).add_to(map_toronto)
map_toronto


# In[22]:


toronto_data = df[df['Borough'].str.contains("Toronto")].reset_index(drop=True)
print(toronto_data.shape)
toronto_data.head()


# In[23]:


map_toronto = folium.Map(location=[latitude, longitude], zoom_start=11)

for lat, lng, label in zip(toronto_data['Latitude'], toronto_data['Longitude'], toronto_data['Neighbourhood']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker([lat, lng], radius=5, popup=label, color='blue', fill=True, fill_color='#3186cc', fill_opacity=0.7,parse_html=False).add_to(map_toronto)  
map_toronto


# In[24]:


# Foursquare API
CLIENT_ID = 'A0AYCTB2MQTTB1IZIYMMPROSCCPOOBLM4DCP1LRDUKUXATJK' # Put Your Client Id
CLIENT_SECRET = 'MHKU2OHEAD5HEHJRXMPUYSI3LIITYJEK2MP2LBPY5KIVEXGX' # Put You Client Secret 
VERSION = '20180604'
LIMIT = 30
print('Your credentails:')
print('CLIENT_ID: Hidden')
print('CLIENT_SECRET: Hidden')


# In[25]:


def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
        
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, CLIENT_SECRET, VERSION, lat, lng, radius, LIMIT)
        
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        venues_list.append([( name, lat, lng, v['venue']['name'], v['venue']['location']['lat'], v['venue']['location']['lng'], v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighbourhood', 'Neighbourhood Latitude', 'Neighbourhood Longitude', 'Venue', 'Venue Latitude', 'Venue Longitude', 'Venue Category']
    
    return(nearby_venues)


# In[26]:


df = toronto_data
toronto_venues = getNearbyVenues(names=df['Neighbourhood'], latitudes=df['Latitude'],longitudes=df['Longitude'])


# In[27]:


print(toronto_venues.shape)
toronto_venues.head()


# In[28]:


toronto_venues.groupby('Neighbourhood').count()


# In[29]:


print('There are {} uniques categories.'.format(len(toronto_venues['Venue Category'].unique())))


# In[30]:


toronto_onehot = pd.get_dummies(toronto_venues[['Venue Category']], prefix="", prefix_sep="")

toronto_onehot['Neighbourhood'] = toronto_venues['Neighbourhood'] 

fixed_columns = [toronto_onehot.columns[-1]] + list(toronto_onehot.columns[:-1])
toronto_onehot = toronto_onehot[fixed_columns]

toronto_onehot.head()


# In[31]:


toronto_onehot.shape


# In[32]:


toronto_grouped = toronto_onehot.groupby('Neighbourhood').mean().reset_index()
toronto_grouped


# In[33]:


toronto_grouped.shape


# In[34]:


num_top_venues = 5
for neigh in toronto_grouped['Neighbourhood']:
    print("----"+neigh+"----")
    temp = toronto_grouped[toronto_grouped['Neighbourhood'] == neigh].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# In[35]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    return row_categories_sorted.index.values[0:num_top_venues]


# In[36]:


import numpy as np
num_top_venues = 10
indicators = ['st', 'nd', 'rd']

columns = ['Neighbourhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighbourhood'] = toronto_grouped['Neighbourhood']

for ind in np.arange(toronto_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(toronto_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.shape


# In[37]:


from sklearn.cluster import KMeans
import sklearn.cluster.k_means_
km = KMeans(n_clusters=3, init='k-means++', max_iter=100, n_init=1, 
  verbose=True)


# In[38]:


kclusters = 10
toronto_grouped_clustering = toronto_grouped.drop('Neighbourhood', 1)
kmeans = KMeans(n_clusters=kclusters, random_state=1).fit(toronto_grouped_clustering)
print(kmeans.labels_[0:10])
print(len(kmeans.labels_))


# In[39]:


df.head()


# In[40]:


neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)
toronto_merged = df
toronto_merged = toronto_merged.join(neighborhoods_venues_sorted.set_index('Neighbourhood'), on='Neighbourhood')
toronto_merged.head()


# In[41]:


map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

x = np.arange(kclusters)
ys = [i+x+(i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

markers_colors = []
for lat, lon, poi, cluster in zip(toronto_merged['Latitude'], toronto_merged['Longitude'], toronto_merged['Neighbourhood'],kmeans.labels_):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker([lat, lon], radius=5, popup=label, color=rainbow[cluster-1], fill=True, fill_color=rainbow[cluster-1], fill_opacity=0.7).add_to(map_clusters)
map_clusters

