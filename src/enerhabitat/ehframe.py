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


        epw_data = self.__epw_format_data(year=year)

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
        print(f'Timezone: {self.timezone}')
        print(f'Latitude: {self.latitude}°')
        print(f'Longitude: {self.longitude}°')
        print(f'Altitude: {self.altitude} m')
        print(f'File: {self.epw}')
        
    def __epw_format_data(self, year = None, warns = False, alias = True):
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

    
    def __init__(self, location:Location, tilt = 90, azimuth = 0, absortance = 0.8):
        self.tilt = tilt
        self.azimuth = azimuth
        self.absortance = absortance
        self.location = location
        
        self.__updated = True
        self.__tsa_dataframe = None
    
    @property
    def location(self):
        return self.__instance_location
    @location.setter
    def location(self, loc:Location):
        """
        Location object containing climate data.
        """
        self.__instance_location = loc
        self.__invalidate_cache()

    @property
    def tilt(self):
        return self.__tilt
    @tilt.setter
    def tilt(self, angle:float):
        """
        Tilt angle of the surface in degrees.
        """
        if angle != getattr(self, "__tilt", None):
            self.__tilt = angle
            self.__invalidate_cache()
        
    @property
    def azimuth(self):
        return self.__azimuth
    @azimuth.setter
    def azimuth(self, angle:float):
        """
        Azimuth angle of the surface in degrees.
        """
        if angle != getattr(self, "__azimuth", None):
            self.__azimuth = angle
            self.__invalidate_cache()

    @property
    def absortance(self):
        return self.__absortance
    @absortance.setter
    def absortance(self, value:float):
        """
        Surface absortance of the system's external material.
        """
        if value != getattr(self, "__absortance", None):
            self.__absortance = value
            self.__invalidate_cache()

    def Tsa(self,
            # solar_absortance:float=None,
            # surface_tilt:float=None,
            # surface_azimuth:float=None
            ) -> pd.DataFrame: 
        """
        Sun-air temperature per second for the average day experienced
        by a surface based on a meanDay dataframe from System's Location
        (Ta, Ig, Ib and Id).

        Returns:
            DataFrame: Predicted sun-air temperature ( Tsa ) and solar irradiance ( Is )
            per second for the average day.
        """
        """
        if solar_absortance is not None:
            self.absortance = solar_absortance
        if surface_tilt is not None:
            self.tilt = surface_tilt
        if surface_azimuth is not None:
            self.azimuth = surface_azimuth
        """
        if self.__tsa_dataframe is None or self.__updated:
            self.__tsa_dataframe = self.__calc_tsa()  # el método que calcula Tsa
            self.__updated = False
        return self.__tsa_dataframe
    
    def __calc_tsa(self) -> pd.DataFrame:
        meanDay_dataframe = self.location.meanDay()
        absortance = self.absortance
        tilt = self.tilt
        azimuth = self.azimuth
        
        global ho
        outside_convection_heat_transfer = ho

        if tilt == 0:
            LWR = 3.9
        else:
            LWR = 0.

        total_irradiance = pvlib.irradiance.get_total_irradiance(
            surface_tilt=tilt,
            surface_azimuth=azimuth,
            dni=meanDay_dataframe['Ib'],
            ghi=meanDay_dataframe['Ig'],
            dhi=meanDay_dataframe['Id'],
            solar_zenith=meanDay_dataframe['zenith'],
            solar_azimuth=meanDay_dataframe['azimuth']
        )

        # Add Is
        meanDay_dataframe['Is'] = total_irradiance.poa_global

        # Add Tsa
        meanDay_dataframe['Tsa'] = meanDay_dataframe.Ta + meanDay_dataframe.Is*absortance/outside_convection_heat_transfer - LWR

        return meanDay_dataframe
    
    def solveCS(
        constructive_system:list,
        Tsa_dataframe:pd.DataFrame,
        AC = False,
        energia=False
        )->pd.DataFrame:
        """
        Solves the constructive system's inside temperature with the Tsa simulation dataframe.

        Args:
            constructive_system (list): List of tuples from outside to inside with material and width.
            Tsa_dataframe (DataFrame): Predicted sun-air temperature ( Tsa ) per second for the average day DataFrame.

        Returns:
            Ti (DataFrame): Interior temperature for the constructive system.
            ET (float): Energy transfer if energia=True.
            Qcool, Qheat (float): Cooling energy and heating energy values if AC=True.
        """

        global La     # Length of the dummy frame
        global Nx     # Number of elements to discretize
        global ho     # Outside convection heat transfer
        global hi     # Inside convection heat transfer
        global dt     # Time step

        global AIR_DENSITY
        global AIR_HEAT_CAPACITY

        SC_dataframe = Tsa_dataframe.copy()

        propiedades = materials_dict()

        cs = set_construction(propiedades, constructive_system)
        k, rhoc, dx = set_k_rhoc(cs, Nx)
        mass_coeff, a_static, b_static, c_static = prepare_static_coefficients(k, rhoc, dx, dt, ho, hi)

        d = np.empty(Nx)
        P = np.empty(Nx)
        Q = np.empty(Nx)
        Tn_aux = np.empty(Nx)
        capacitance_factor = hi * dt / (AIR_DENSITY * AIR_HEAT_CAPACITY * La)

        T = np.full(Nx, SC_dataframe.Tn.mean())
        SC_dataframe['Ti'] = SC_dataframe.Tn.mean()

        SC_dataframe = SC_dataframe.iloc[::dt]
        Tsa_vals = SC_dataframe['Tsa'].to_numpy()
        Ti_vals = SC_dataframe['Ti'].to_numpy()
        Ti_new = np.empty_like(Ti_vals)
        n_steps = Tsa_vals.shape[0]

        C = 1
        ET = 0.0

        if AC:  # AC = True
            while C > 5e-4: 
                Told = T.copy()
                Qcool = Qheat = 0.
                for idx in range(n_steps):
                    calculate_coefficients(mass_coeff, T, Tsa_vals[idx], ho, Ti_vals[idx], hi, d)
                    # Llamado de funcion para Acc
                    T, Ti = solve_PQ_AC(a_static, b_static, c_static, d, T, Nx, Ti_vals[idx], hi, La, dt)
                    if (T[Nx-1] > Ti):
                        Qcool += hi*dt*(T[Nx-1]-Ti)
                    if (T[Nx-1] < Ti):
                        Qheat += hi*dt*(Ti-T[Nx-1])
                    Ti_vals[idx] = Ti
                Tnew = T.copy()
                C = abs(Told - Tnew).mean()

            SC_dataframe['Ti'] = Ti_vals

            return SC_dataframe['Ti'], Qcool, Qheat

        else:
            ET = 0.0
            while C > 5e-4: 
                Told = T.copy()
                ET_iter = 0.
                for idx in range(n_steps):
                    tint_prev = Ti_vals[idx]
                    calculate_coefficients(mass_coeff, T, Tsa_vals[idx], ho, tint_prev, hi, d)
                    T, tint_new = solve_PQ(a_static, b_static, c_static, d, T, Nx, tint_prev, capacitance_factor, P, Q, Tn_aux)
                    Ti_new[idx] = tint_new
                    if T[Nx-1] > tint_new:
                        ET_iter += hi * (T[Nx - 1] - tint_new) * dt
                Ti_vals[:] = Ti_new
                C = np.abs(Told - T).mean()
                ET = ET_iter

            SC_dataframe['Ti'] = Ti_vals

            if energia: return SC_dataframe['Ti'], ET
            else: return SC_dataframe['Ti']

    def __invalidate_cache(self):
        self.__updated = True