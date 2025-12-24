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
        
    def info(self):
        """
        Prints Location information.
        """
        print("<class 'enerhabitat.Location'>")
        print(f'City: {self.city}')
        print(f'Timezone: {self.timezone}')
        print(f'Latitude: {self.latitude}°')
        print(f'Longitude: {self.longitude}°')
        print(f'Altitude: {self.altitude} m')
        print(f'File: {self.epw}')
        
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
    """
    System class to model a constructive system and calculate its interior temperature
    based on the sun-air temperature experienced by the surface.
    Attributes:
        location (Location): Location object containing climate data.
        tilt (float): Tilt angle of the surface in degrees.
        azimuth (float): Azimuth angle of the surface in degrees.
        absortance (float): Surface absortance of the system's external material.
        layers (list): List of tuples from outside to inside with material and width.
    
    Methods:
        Tsa(absortance, tilt, azimuth): Calculates the sun-air temperature per second for the average day experienced by a surface.
        solve(energy): Solves the constructive system's inside temperature.
        solveAC(): Solves the constructive system's required cooling and heating energy to maintain the inside temperature.
    """
    """
        add_layer(material, width): Adds a layer to the constructive system.
        remove_layer(index): Removes a layer from the constructive system by index.
        
        set_solver(...): Class method to set solver parameters.
        solver_info(): Class method to print current solver parameters.
    """
    
    __La = 2.5    # Length of the dummy frame
    __Nx = 200    # Number of elements to discretize
    __ho = 13     # Outside convection heat transfer
    __hi = 8.6    # Inside convection heat transfer
    __dt = 600    # Time step in seconds

    # Propiedades del aire empleadas en el modelo lumped-capacitance
    __AIR_DENSITY = 1.1797660470258469
    __AIR_HEAT_CAPACITY = 1005.458757
    
    __solver_version = 0

    def __init__(self, location:Location , tilt = 90, azimuth = 0, absortance = 0.8, layers = []):
        self.tilt = tilt
        self.azimuth = azimuth
        self.absortance = absortance
        self.location = location
        self.layers= layers
        
        self.__updated = True
        self.__tsa_dataframe = None
        self.__solve_dataframe = None
    
    def info(self):
        """
        Prints System information.
        """
        print("<class 'enerhabitat.System'>")
        print(f"Location: {self.location.city}")
        print(f"Tilt: {self.tilt}°")
        print(f"Azimuth: {self.azimuth}°")
        print(f"Absortance: {self.absortance}")
        print("Layers:")
        for i, (material, width) in enumerate(self.layers):
            print(f"  Layer {i+1}: {material}, {width} m")
    
    @property
    def layers(self):
        return self.__capas
    @layers.setter
    def layers(self, capas:list):
        """
        List of tuples from outside to inside with material and width.
        Example: [('Brick',0.1), ('Insulation',0.05), ('Adobe',0.02)]
        """
        self.__capas = capas
        self.__invalidate_cache()
    
    def add_layer(self, material:str, width:float):
        """
        Adds a layer to the constructive system.

        Args:
            material (str): Material name.
            width (float): Width of the material in meters.
        """
        self.__capas.append((material, width))
        self.__invalidate_cache()
        return self.layers
    
    def remove_layer(self, index:int):
        """
        Removes a layer from the constructive system by index.

        Args:
            index (int): Index of the layer to remove.
        """
        if index < 0 or index >= len(self.__capas):
            raise IndexError("Layer index out of range.")
        del self.__capas[index]
        self.__invalidate_cache()
        return self.layers
    
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
        if self.__tsa_dataframe is None or self.__updated or self.__tsa_solver_version != self.__class__.__solver_version:
            self.__tsa_dataframe = self.__calc_tsa()  # el método que calcula Tsa
            self.__updated = False
        return self.__tsa_dataframe
    
    def solve(self, energy=False) -> pd.DataFrame:
        """
        Solves the constructive system's inside temperature with the Tsa simulation dataframe.

        Args:
            energy (bool): If True, returns also the energy transfer ET.
        
        Returns:
            Ti (DataFrame): Interior temperature for the constructive system.
        """
        constructive_system = self.layers
        if len(constructive_system) == 0:
            raise ValueError("Constructive system layers are not defined.")
        
        recalculate = (self.__updated or 
                        self.__solve_dataframe is None or 
                        self.__solve_solver_version != self.__class__.__solver_version or
                        self.__last_solve != 'temp'
                        )

        if recalculate:    
            self.__solve_dataframe, self.__solve_energy = self.__calc_solve(energia=True)
            self.__updated = False
        if energy:
            return self.__solve_dataframe, self.__solve_energy
        else:
            return self.__solve_dataframe
    
    def solveAC(self) -> pd.DataFrame:
        """
        Solves the constructive system's required cooling and heating energy to 
        maintain the interior temperature with the Tsa simulation dataframe.

        Returns:
            Ti (DataFrame): Interior temperature for the constructive system.
            Qcool, Qheat (float): Cooling energy and heating energy values.
        """
        constructive_system = self.layers
        if len(constructive_system) == 0:
            raise ValueError("Constructive system layers are not defined.")
        
        recalculate = (self.__updated or 
                        self.__solve_dataframe is None or 
                        self.__solve_solver_version != self.__class__.__solver_version or
                        self.__last_solve != 'ac'
                       )

        if recalculate:
            self.__solve_dataframe, self.__solve_qcool, self.__solve_qheat = self.__calc_solve(AC=True)
            self.__updated = False
        return self.__solve_dataframe, self.__solve_qcool, self.__solve_qheat

    def __calc_tsa(self) -> pd.DataFrame:
        tsa_dataframe = self.location.meanDay()
        absortance = self.absortance
        tilt = self.tilt
        azimuth = self.azimuth
        
        outside_convection_heat_transfer = self.__class__.__ho

        if tilt == 0:
            LWR = 3.9
        else:
            LWR = 0.

        total_irradiance = pvlib.irradiance.get_total_irradiance(
            surface_tilt=tilt,
            surface_azimuth=azimuth,
            dni=tsa_dataframe['Ib'],
            ghi=tsa_dataframe['Ig'],
            dhi=tsa_dataframe['Id'],
            solar_zenith=tsa_dataframe['zenith'],
            solar_azimuth=tsa_dataframe['azimuth']
        )

        # Add Is
        tsa_dataframe['Is'] = total_irradiance.poa_global

        # Add Tsa
        tsa_dataframe['Tsa'] = tsa_dataframe.Ta + tsa_dataframe.Is*absortance/outside_convection_heat_transfer - LWR

        self.__tsa_solver_version = self.__class__.__solver_version
        
        return tsa_dataframe

    def __calc_solve(self, AC=False) -> pd.DataFrame:
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
        cls = self.__class__
        
        La = cls.__La # Length of the dummy frame
        Nx = cls.__Nx # Number of elements to discretize
        ho = cls.__ho # Outside convection heat transfer
        hi = cls.__hi # Inside convection heat transfer
        dt = cls.__dt # Time step
        AIR_DENSITY = cls.__AIR_DENSITY
        AIR_HEAT_CAPACITY = cls.__AIR_HEAT_CAPACITY

        SC_dataframe = self.Tsa().copy()
        constructive_system = self.layers
        
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

        self.__solve_solver_version = self.__class__.__solver_version
        
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
            self.__last_solve = 'ac'
            return SC_dataframe['Ti'], Qcool, Qheat

        else:
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
            self.__last_solve = 'temp'
            return SC_dataframe['Ti'], ET

    def __invalidate_cache(self):
        self.__updated = True

    @classmethod
    def set_solver(cls, *,
                   La=None, Nx=None, ho=None, hi=None, dt=None,
                   air_density=None, air_heat_capacity=None):
        if La is not None: cls.__La = La
        if Nx is not None: cls.__Nx = Nx
        if ho is not None: cls.__ho = ho
        if hi is not None: cls.__hi = hi
        if dt is not None: cls.__dt = dt
        if air_density is not None: cls.__AIR_DENSITY = air_density
        if air_heat_capacity is not None: cls.__AIR_HEAT_CAPACITY = air_heat_capacity
        cls.__solver_version += 1

    @classmethod
    def solver_info(cls):
        print(f"La: {cls.__La}")
        print(f"Nx: {cls.__Nx}")
        print(f"ho: {cls.__ho}")
        print(f"hi: {cls.__hi}")
        print(f"dt: {cls.__dt}")
        print(f"AIR_DENSITY: {cls.__AIR_DENSITY}")
        print(f"AIR_HEAT_CAPACITY: {cls.__AIR_HEAT_CAPACITY}")
    