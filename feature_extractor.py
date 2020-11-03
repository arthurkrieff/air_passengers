import pandas as pd
import os
import geopy.distance
from datetime import timedelta
from pandas.tseries.holiday import USFederalHolidayCalendar
pd.options.mode.chained_assignment = None
 
class FeatureExtractor(object):
    def __init__(self):
        self.connect = dict()
 
    def fit(self, X_df, y_array):
        airports = X_df['Departure'].unique()
        for i in airports:
            self.connect[i] = X_df[X_df['Departure'] == i]['Arrival'].nunique()
 
    def transform(self, X_df):
        ## Loading data
        path = os.path.dirname(__file__)
        external_data = pd.read_csv(os.path.join(path, 'external_data.csv'))
        weather_data = external_data.iloc[:,:24]
        airports_data = external_data.iloc[:20,24:]
        
        
        ## Temporal variables
        X_df["DateOfDeparture"] = pd.to_datetime(X_df["DateOfDeparture"], format='%Y-%m-%d')
        X_df["Weekday"] = pd.DatetimeIndex(X_df["DateOfDeparture"]).weekday
        X_df["Month"] = pd.DatetimeIndex(X_df["DateOfDeparture"]).month
        X_df["Week"] = pd.DatetimeIndex(X_df["DateOfDeparture"]).week
        X_df["Year"] = pd.DatetimeIndex(X_df["DateOfDeparture"]).year
        
        ## Journeys
        X_df = X_df.join(pd.get_dummies(X_df['Departure']) + pd.get_dummies(X_df['Arrival']))
        
        ## Distances
        coordinates = airports_data[["IATA", "Latitude", "Longitude"]]
        coordinates["Coordinates"] = None
        
        for i in range(len(coordinates)):
            coordinates.iat[i, 3] = [coordinates.iloc[i,1], coordinates.iloc[i,2]]
        coordinates = coordinates.drop(columns=["Latitude", "Longitude"])
        
        X_df = X_df.merge(coordinates.add_suffix("_Dep"), how='left', left_on="Departure", 
                          right_on="IATA_Dep")
        X_df = X_df.merge(coordinates.add_suffix("_Arr"), how='left', left_on="Arrival", 
                          right_on="IATA_Arr")
        X_df = X_df.drop(columns=["IATA_Arr","IATA_Dep"])
        
        X_df['Distance_km'] = 1/X_df.apply(lambda x: 
                                         geopy.distance.geodesic(x["Coordinates_Dep"],x["Coordinates_Arr"]).km, axis=1)
        
        ## Population
        pop = airports_data[['IATA','Pop_2012']]
        X_df = X_df.merge(pop.add_suffix("_Dep"), how='left', left_on="Departure", right_on="IATA_Dep")
        X_df = X_df.merge(pop.add_suffix("_Arr"), how='left', left_on="Arrival", right_on="IATA_Arr")
        X_df = X_df.drop(columns=["IATA_Dep","IATA_Arr"])
        
        ## Connectivity index
        X_df['Connectivity'] = X_df['Arrival'].map(self.connect) + X_df['Departure'].map(self.connect)
        
        ## Holidays
        holidays = USFederalHolidayCalendar().holidays(start='2011-01-01', end='2013-12-31')
        holidays = holidays.append(holidays + timedelta(days = -1))
        X_df["Holidays"] = X_df["DateOfDeparture"].isin(holidays)*1
            
        ## Preprocessing
        X_encoded = X_df.join(pd.get_dummies(X_df["Month"], prefix="m"))
        X_encoded = X_encoded.join(pd.get_dummies(X_df["Weekday"], prefix="Weekday"))
        X_encoded = X_encoded.join(pd.get_dummies(X_df["Week"], prefix="Week"))
        X_encoded = X_encoded.join(pd.get_dummies(X_df["Year"], prefix="Year"))
        
        ## Drop columns
        X_encoded = X_encoded.drop(columns=["DateOfDeparture", "Departure", "Arrival",
                                            "Weekday", "Month", "Week", "Year", 
                                            "Coordinates_Dep", "Coordinates_Arr"])
        
        ## Feature selection
        X_encoded = X_encoded.drop(columns=['std_wtd','Pop_2012_Dep','Week_11','Week_12','Week_15','Week_18',
                                            'Week_19','Week_23','Week_24','Week_25','Week_28','Week_30','Week_32',
                                            'Week_33','Week_34','Week_39','Week_40','Week_42','Week_43'])
        X_array = X_encoded.values     
        
        return X_array