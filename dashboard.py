
import pandas as pd
import streamlit as st
import numpy as np
import folium
import geopandas
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
from datetime import datetime
import plotly.express as px
import os

st.set_page_config(layout='wide')


@st.cache(allow_output_mutation=True)
def get_data(path):
    data = pd.read_csv(path)
    return data


@st.cache(allow_output_mutation=True)
def get_geofile(url):
    geofile = geopandas.read_file(url)
    return geofile


def set_feature(data):
    data['date'] = pd.to_datetime(data['date'], format="%Y-%m-%d")
    data['price_m2'] = data['price'] / data['sqft_lot']
    return data


def overview_data(data):
    f_attributes = st.sidebar.multiselect('Entre colmns', data.columns)
    f_zipcode = st.sidebar.multiselect(
        'Entre zipcode', data['zipcode'].unique())

    st.title('Data Overview')

    if(f_zipcode != []) & (f_attributes != []):
        data = data.loc[data['zipcode'].isin(f_zipcode), f_attributes]

    elif(f_zipcode != []) & (f_attributes == []):
        data = data.loc[data['zipcode'].isin(f_zipcode)]

    elif(f_zipcode == []) & (f_attributes != []):
        data = data.loc[:, f_attributes]

    else:
        data = data.copy()

    st.dataframe(data.head())

    st.title('Average metrics')

    c1, c2 = st.beta_columns((1, 1))
    # Average metrics
    df1 = data[['id', 'zipcode']].groupby('zipcode').count().reset_index()
    df2 = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df3 = data[['sqft_living', 'zipcode']].groupby(
        'zipcode').mean().reset_index()
    df4 = data[['price_m2', 'zipcode']].groupby('zipcode').mean().reset_index()

    # merge
    m1 = pd.merge(df1, df2, on='zipcode', how='inner')
    m2 = pd.merge(m1, df3, on='zipcode', how='inner')
    df = pd.merge(m2, df4, on='zipcode', how='inner')

    df.columns = ['ZIPCODE', 'TOTAL HOUSES',
                  'PRICE', 'SQRT LIVING', 'PRICE/M2']

    c1.header('Avarage Values')
    c1.dataframe(df, height=300)

    # Statistics Descriptive

    num_attributes = data.select_dtypes(include=['int64', 'float64'])
    media = pd.DataFrame(num_attributes.apply(np.mean))
    mediana = pd.DataFrame(num_attributes.apply(np.median))
    std = pd.DataFrame(num_attributes.apply(np.std))

    max_ = pd.DataFrame(num_attributes.apply(np.max))
    min_ = pd.DataFrame(num_attributes.apply(np.min))

    df1 = pd.concat([max_, min_, media, mediana, std], axis=1).reset_index()

    df1.columns = ['attributes', 'max', 'min', 'mean', 'mediana', 'std']

    c2.header('Statistics Descriptive')
    c2.dataframe(df1, height=300)

    return None


def portfolio_density(data, geofile):

    st.title('Region Overview')

    c1, c2 = st.beta_columns((1, 1))
    c1.header('Portfolio Density')

    df = data.sample(1000)

    # Base Map - Folium

    density_map = folium.Map(
        location=[data['lat'].mean(),
                  data['long'].mean()],
        default_zoom_start=15)

    maker_cluster = MarkerCluster().add_to(density_map)

    for name, row in df.iterrows():
        folium.Marker([row['lat'], row['long']],
                      popup='Sold R$ {0} on: {1}. Features: {2} sqft, {3} bedrooms, \
                        {4} bathrooms, year built: {5}'.format(
                            row['price'],
                            row['date'],
                            row['sqft_living'],
                            row['bedrooms'],
                            row['bathrooms'],
                            row['yr_built'])
                      ).add_to(maker_cluster)

    with c1:
        folium_static(density_map)

    # Region Price Map

    c2.header('Price Density')

    df = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df.columns = ['ZIP', 'PRICE']

    geofile = geofile[geofile['ZIP'].isin(df['ZIP'].tolist())]

    region_price_map = folium.Map(
        location=[data['lat'].mean(),
                  data['long'].mean()],
        default_zoom_start=15)

    folium.Choropleth(data=df,
                      geo_data=geofile,
                      columns=['ZIP', 'PRICE'],
                      key_on='feature.properties.ZIP',
                      fill_color='YlOrRd',
                      fill_opacity=0.7,
                      line_opacity=0.2,
                      legend_name='AVG PRICE').add_to(region_price_map)

    with c2:
        folium_static(region_price_map)

    return None


def commercial_distribution(data):

    st.sidebar.title('Commercial Options')
    st.title('Commercial attributess')

    data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')

    # --------Average Price per Year
    st.header('Average Price per Year built')

    # filter
    min_year_built = int(data['yr_built'].min())
    max_year_built = int(data['yr_built'].max())
    median_year_built = int(data['yr_built'].median())

    st.sidebar.subheader('Select Max Year Built')
    f_year_built = st.sidebar.slider(
        'Year Built', min_year_built, max_year_built, median_year_built)

    # data selection
    df = data.loc[data['yr_built'] < f_year_built]
    df = df[['yr_built', 'price']].groupby('yr_built').mean().reset_index()

    # plot
    fig = px.line(df, x='yr_built', y='price')
    st.plotly_chart(fig, use_container_width=True)

    # -------- Average Price per Day

    st.header('Average Price per Day')

    st.sidebar.subheader('Select Max Date')

    # filters
    min_date = datetime.strptime(data['date'].min(), '%Y-%m-%d')
    max_date = datetime.strptime(data['date'].max(), '%Y-%m-%d')
    

    f_date = st.sidebar.slider('Date', min_date, max_date, max_date)

    # data Select
    data['date'] = pd.to_datetime(data['date'])
    df = data.loc[data['date'] < f_date]
    df = df[['date', 'price']].groupby('date').mean().reset_index()

    # plot
    fig = px.line(df, x='date', y='price')
    st.plotly_chart(fig, use_container_width=True)

    # -------- Histograma

    st.header('Price Distribution')
    st.sidebar.subheader('Select Max Price')

    # filter
    min_price = int(data['price'].min())
    max_price = int(data['price'].max())
    median_price = int(data['price'].median())

    # data filtering
    f_price = st.sidebar.slider('price', min_price, max_price, median_price)
    df = data.loc[data['price'] < f_price]

    # data plot
    fig = px.histogram(df, x='price', nbins=50)
    st.plotly_chart(fig, use_container_width=True)

    return df


def attributes_distribution(data):
    st.sidebar.title('Attibutes Options')
    st.title('House attributes')

    # filters
    f_bedrooms = st.sidebar.selectbox(
        'Max number of bedrooms', sorted(set(data['bedrooms'].unique())), index=len(data['bedrooms'].unique()) -1 ) 

    f_bathrooms = st.sidebar.selectbox(
        'Max number of bathrooms', sorted(set(data['bathrooms'].unique())), index=len(data['bathrooms'].unique()) -1)

    c1, c2 = st.beta_columns(2)

    # house per bedrooms
    c1.header('House per bedrooms')
    df = data[data['bedrooms'] < f_bedrooms]
    fig = px.histogram(df, x='bedrooms', nbins=19)
    c1.plotly_chart(fig, use_container_width=True)

    # house per bathrooms
    c2.header('House per bathrooms')
    df = data[data['bathrooms'] < f_bathrooms]
    fig = px.histogram(df, x='bathrooms', nbins=10)
    c2.plotly_chart(fig, use_container_width=True)

    # filters
    f_floors = st.sidebar.selectbox(
        'Max number of floor', sorted(set(data['floors'].unique())), index=len(data['floors'].unique()) -1)
    f_waterview = st.sidebar.checkbox('Ony house with Water View')

    c1, c2 = st.beta_columns(2)

    # house per floors
    c1.header('House per floors')
    df = data[data['floors'] < f_floors]
    fig = px.histogram(df, x='floors', nbins=7)
    c1.plotly_chart(fig, use_container_width=True)

    # house per water view
    c2.header('House per waterfront')
    if f_waterview:
        df = data[data['waterfront'] == 1]
    else:
        df = data.copy()

    fig = px.histogram(df, x='waterfront', nbins=2)
    c2.plotly_chart(fig, use_container_width=True)

    return df


def business_recommendations(data):
    st.title('Business Recommendations')
    st.header('Purchasing Recommendation Report')

    # Group the median of properties by region
    dfzip = data[['zipcode', 'price']].groupby('zipcode').median(
    ).reset_index().rename(columns={'price': 'median_price'})

    data = pd.merge(data, dfzip, how='inner', on='zipcode')

    # recommendation list
    data['recomendation'] = data[['price', 'median_price', 'condition']].apply(
        lambda x: 'buy' if (x['price'] < x['median_price']) &
                           (x['condition'] == 5) else 'not buy', axis='columns')

    data['condition'] = data['condition'].map({1: 'too bad',
                                               2: 'bad',
                                               3: 'good',
                                               4: 'very good',
                                               5: 'great'})

    data = data[data['recomendation'] == 'buy'].copy()
    df3 = data[['id', 'price', 'median_price', 'condition',
               'recomendation']].sort_values('price', ascending=False).reset_index(drop=True)

    st.dataframe(df3)
    st.write(f'{df3.shape[0]} properties are recommended for purchase')

    # mapa
    st.header('Location of Recommended Properties:')

    houses = data[['id', 'lat', 'long', 'price']]

    fig = px.scatter_mapbox(houses,
                            lat='lat',
                            lon='long',
                            size='price',
                            color_continuous_scale=px.colors.cyclical.IceFire,
                            size_max=15,
                            zoom=10)

    fig.update_layout(mapbox_style='open-street-map')
    fig.update_layout(height=600, margin={'r': 0, 't': 0, 'l': 0, 'b': 0})
    # fig.show()
    st.plotly_chart(fig)

    return data

def sell_repport(data):

    st.header('Sales Recommended Report')

    data['date_month'] = pd.to_datetime(data['date']).dt.month
    data['seasonality'] = np.nan

    data['seasonality'] = data['date_month'].apply(
        lambda x: 'winter' if (x == 12 or x <= 2) else 
                  'spring' if (3 <= x < 6) else
                  'summer' if (6 <= x <= 8) else 'Autumn')

    data['sale_price'] = data[['seasonality', 'price']].apply(
        lambda x: x['price']*1.30 if x['seasonality'] == 'Spring' else
                  x['price']*1.25 if x['seasonality'] == 'summer' else
                  x['price']*1.20 if x['seasonality'] == 'winter' else
                  x['price']*1.10, axis='columns')

    data['profit'] = data[['sale_price', 'price']].apply(
        lambda x: x['sale_price'] - x['price'], axis='columns')
        

    

    st.dataframe(data[['id', 'zipcode', 'seasonality', 'median_price', 'sale_price', 'profit']] )


if __name__ == '__main__':
    # ETL
    # data extration
    path = os.path.join(os.path.abspath('.'), 'datasets', 'kc_house_data.csv')
    url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'

    data = get_data(path)
    geofile = get_geofile(url)

    # transformation
    data = set_feature(data)

    overview_data(data)

    portfolio_density(data, geofile)

    data = commercial_distribution(data)

    data = attributes_distribution(data)

    # Business Recommendations
    data = business_recommendations(data)

    # best time for sale
    sell_repport(data)