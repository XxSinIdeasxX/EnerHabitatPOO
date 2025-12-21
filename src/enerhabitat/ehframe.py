import pandas as pd
import pvlib
import pytz
import warnings

from datetime import datetime

from .ehtools import *

class Location:
    """
    Location class to handle climate data from an EPW file.

    Attributes:
        epw (str): Path to the EPW file containing climate data.
        city (str): City of the location.
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        altitude (float): Altitude of the location in meters.
        timezone (pytz.timezone): Timezone of the location.

    Methods:
        meanDay(day, month, year): Calculates the ambient temperature per second for the average day
    """
    
    def __init__(self, epw_file:str):
        self.epw = epw_file
        
    @property
    def epw(self):
        return self.__epw_path

    @epw.setter
    def epw(self, file):
        """
        EPW file containing climate data. Attributes timezone, longitude, latitude, altitude are taken from this file.
        """
        datos=[]
        
        with open(file,'r') as epw:
            datos=epw.readline().split(',')
            
        self.__epw_path = file
        self.city = str(datos[1]) + ", " + str(datos[2])
        self.latitude = float(datos[6])
        self.longitude = float(datos[7])
        self.altitude = float(datos[9])
        
        tmz = int(datos[8].split('.')[0])
        self.timezone = pytz.timezone('Etc/GMT'+f'{(-tmz):+}')

    def meanDay(self,
        day = "15",
        month = "current_month",
        year = "current_year"
        ) -> pd.DataFrame:
        """
        Calculates the ambient temperature per second for the average day based on Location data.

        Args:
            day (str, optional): Day of interest. Defaults to 15.
            month (str, optional): Month of interest. Defaults to current month.
            year (str, optional): Year of interest. Defaults to current year.

        Returns:
            DataFrame: Predicted ambient temperature ( Ta ), global ( Ig ), beam ( Ib ) 
            and diffuse irradiance ( Id ) per second for the average day of the specified month and year.
        """
        
        if month == "current_month": month = datetime.now().month
        if year == "current_year": year = datetime.now().year

        f1 = f'{year}-{month}-{day} 00:00'
        f2 = f'{year}-{month}-{day} 23:59'


        epw_data = self.__epw_data(year=year)

        dia_promedio = pd.date_range(start=f1, end=f2, freq='1s',tz=self.timezone)
        location = pvlib.location.Location(latitude = self.latitude, 
                                           longitude= self.longitude, 
                                           altitude= self.altitude,
                                           tz=self.timezone)

        dia_promedio = location.get_solarposition(dia_promedio)
        del dia_promedio['apparent_zenith']
        del dia_promedio['apparent_elevation']

        sunrise,_ = get_sunrise_sunset_times(dia_promedio)
        tTmax,Tmin,Tmax = calculate_tTmaxTminTmax(month, epw_data)

        # Calculate ambient temperature y add to the DataFrame
        dia_promedio = add_temperature_model(dia_promedio, Tmin, Tmax, sunrise, tTmax)

        # Add Ig, Ib, Id y Tn a dia_promedio 
        dia_promedio = add_IgIbId_Tn(dia_promedio, epw_data, month, f1, f2, self.timezone)

        # Add DeltaTn
        DeltaTa= dia_promedio.Ta.max() - dia_promedio.Ta.min()
        dia_promedio['DeltaTn'] = calculate_DtaTn(DeltaTa)

        return dia_promedio
    
    def info(self):
        """
        Prints Location information.
        """
        print(f'City: {self.city}')
        print(f'Latitude: {self.latitude}°')
        print(f'Longitude: {self.longitude}°')
        print(f'Altitude: {self.altitude} m')
        print(f'Timezone: {self.timezone}')
        print(f'File: {self.epw}')
        
    def __epw_data(self, year = None, warns = False, alias = True):
        """
        Reads Location's EPW file and returns a formatted DataFrame.
            year : None default to leave intact the year or change if desired. It raises a warning.
            alias : True default, to change columns to To, Ig, Ib, Ws, RH, ...
            warns : False default, True to enable warnings.
        """
        
        names = ['Year',
                 'Month',
                 'Day',
                 'Hour',
                 'Minute',
                 'Data Source and Uncertainty Flags',
                 'Dry Bulb Temperature',
                 'Dew Point Temperature',
                 'Relative Humidity',
                 'Atmospheric Station Pressure',
                 'Extraterrestrial Horizontal Radiation',
                 'Extraterrestrial Direct Normal Radiation',
                 'Horizontal Infrared Radiation Intensity',
                 'Global Horizontal Radiation',
                 'Direct Normal Radiation',
                 'Diffuse Horizontal Radiation',
                 'Global Horizontal Illuminance',
                 'Direct Normal Illuminance',
                 'Diffuse Horizontal Illuminance',
                 'Zenith Luminance',
                 'Wind Direction',
                 'Wind Speed',
                 'Total Sky Cover',
                 'Opaque Sky Cover',
                 'Visibility',
                 'Ceiling Height',
                 'Present Weather Observation',
                 'Present Weather Codes',
                 'Precipitable Water',
                 'Aerosol Optical Depth',
                 'Snow Depth',
                 'Days Since Last Snowfall',
                 'Albedo',
                 'Liquid Precipitation Depth',
                 'Liquid Precipitation Quantity']

        rename = {'Dry Bulb Temperature'       :'To',
                 'Relative Humidity'           :'RH',
                 'Atmospheric Station Pressure':'P' ,
                 'Global Horizontal Radiation' :'Ig',
                 'Direct Normal Radiation'     :'Ib',
                 'Diffuse Horizontal Radiation':'Id',
                 'Wind Direction'              :'Wd',
                 'Wind Speed'                  :'Ws'}

        data = pd.read_csv(self.epw, skiprows=8, header=None, names=names, usecols=range(35))
        data.Hour = data.Hour -1
        if year != None:
            data.Year = year
            if warns == True:
                warnings.warn("Year has been changed, be carefull")
        try:
            data['tiempo'] = data.Year.astype('str') + '-' + data.Month.astype('str')  + '-' + data.Day.astype('str') + ' ' + data.Hour.astype('str') + ':' + data.Minute.astype('str') 
            data.tiempo = pd.to_datetime(data.tiempo,format='%Y-%m-%d %H:%M')
        except:
            data.Minute = 0
            data['tiempo'] = data.Year.astype('str') + '-' + data.Month.astype('str')  + '-' + data.Day.astype('str') + ' ' + data.Hour.astype('str') + ':' + data.Minute.astype('str') 
            data.tiempo = pd.to_datetime(data.tiempo,format='%Y-%m-%d %H:%M')

        data.set_index('tiempo',inplace=True)
        del data['Year']
        del data['Month']
        del data['Day']
        del data['Hour']
        del data['Minute']
        if alias:
            data.rename(columns=rename,inplace=True)
        
        return data
    
class System:
    """_summary_

    Attributes:
        _type_: _description_
        
    Methods:
        _method_(self, args): _description_
    """
    
    def __init__(self, location:Location, tilt = 90, azimuth = 0, absortance = 0.8):
        self.tilt = tilt
        self.azimuth = azimuth
        self.absortance = absortance
        self.location = location
    
    @property
    def location(self):
        return self.__inst_location
    @location.setter
    def location(self, loc:Location):
        """
        Location object containing climate data.
        """
        self.__inst_location = loc
        self._updated = True

    @property
    def tilt(self):
        return self.__tilt
    @tilt.setter
    def tilt(self, angle:float):
        """
        Tilt angle of the surface in degrees.
        """
        self.__tilt = angle
        self.__updated = True
        
    @property
    def azimuth(self):
        return self.__azimuth
    @azimuth.setter
    def azimuth(self, angle:float):
        """
        Azimuth angle of the surface in degrees.
        """
        self.__azimuth = angle
        self.__updated = True