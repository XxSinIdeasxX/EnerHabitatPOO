import pvlib
import pytz

from datetime import datetime
from .ehtools import *

La = 2.5    # Length of the dummy frame
Nx = 200     # Number of elements to discretize
ho = 13     # Outside convection heat transfer
hi = 8.6    # Inside convection heat transfer
dt = 600     # Time step in seconds

def meanDay(
    epw_file : str,
    day = "15",
    month = "current_month",
    year = "current_year"
    ) -> pd.DataFrame:
    """
    Calculates the ambient temperature per second for the average day based on EPW file data.
    
    Args:
        epw_file (str): Path to the EPW file. 
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
    
    epw, latitud, longitud, altitud, timezone = readEPW(epw_file,year,alias=True,warns=False)
    timezone=pytz.timezone('Etc/GMT'+f'{(-timezone):+}')
    
    dia_promedio = pd.date_range(start=f1, end=f2, freq='1s',tz=timezone)
    location = pvlib.location.Location(latitude = latitud, 
                                       longitude=longitud, 
                                       altitude=altitud,
                                       tz=timezone)

    dia_promedio = location.get_solarposition(dia_promedio)
    del dia_promedio['apparent_zenith']
    del dia_promedio['apparent_elevation']
    
    sunrise,_ = get_sunrise_sunset_times(dia_promedio)
    tTmax,Tmin,Tmax = calculate_tTmaxTminTmax(month, epw)

    # Calculate ambient temperature y add to the DataFrame
    dia_promedio = add_temperature_model(dia_promedio, Tmin, Tmax, sunrise, tTmax)
    
    # Add Ig, Ib, Id y Tn a dia_promedio 
    dia_promedio = add_IgIbId_Tn(dia_promedio, epw, month, f1, f2, timezone)
    
    # Add DeltaTn
    DeltaTa= dia_promedio.Ta.max() - dia_promedio.Ta.min()
    dia_promedio['DeltaTn'] = calculate_DtaTn(DeltaTa)
    
    return dia_promedio

def Tsa(
    meanDay_dataframe:pd.DataFrame,
    solar_absortance: float,
    surface_tilt: float,
    surface_azimuth: float,
    ) -> pd.DataFrame: 
    """
    Calculates the sun-air temperature per second for the average day experienced
    by a surface based on a meanDay dataframe.
    
    Args:
        meanDay_dataframe (DataFrame): Data frame containing ambient temperature ( Ta ), global ( Ig ), direct ( Ib ) and diffuse irradiance ( Id ). 
        solar_absortance (float): Solar absortance of the system's external material.
        surface_tilt (float): Surface tilt relative to the ground, 90° == Vertical.
        surface_azimuth (float): Deviation from true north, 0° == North.
    
    Returns:
        DataFrame: Predicted sun-air temperature ( Tsa ) and solar irradiance ( Is )
        per second for the average day.
    """
    
    global ho
    outside_convection_heat_transfer = ho
    
    if surface_tilt == 0:
        LWR = 3.9
    else:
        LWR = 0.
        
    total_irradiance = pvlib.irradiance.get_total_irradiance(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        dni=meanDay_dataframe['Ib'],
        ghi=meanDay_dataframe['Ig'],
        dhi=meanDay_dataframe['Id'],
        solar_zenith=meanDay_dataframe['zenith'],
        solar_azimuth=meanDay_dataframe['azimuth']
    )
    
    # Add Is
    meanDay_dataframe['Is'] = total_irradiance.poa_global
    
    # Add Tsa
    meanDay_dataframe['Tsa'] = meanDay_dataframe.Ta + meanDay_dataframe.Is*solar_absortance/outside_convection_heat_transfer - LWR
       
    return meanDay_dataframe
  
def solveCS(
    constructive_system:list,
    Tsa_dataframe:pd.DataFrame,
    AC = False,
    Energia=False
    )->pd.DataFrame:
    """
    Solves the constructive system's inside temperature with the Tsa simulation dataframe.

    Args:
        constructive_system (list): list of tuples from outside to inside with material and width.
        Tsa_dataframe (DataFrame): Predicted sun-air temperature ( Tsa ) per second for the average day DataFrame.
        
    Returns:
        DataFrame: Interior temperature ( Ti ) for the constructive system.
    """
    
    global La     # Length of the dummy frame
    global Nx     # Number of elements to discretize
    global ho     # Outside convection heat transfer
    global hi     # Inside convection heat transfer
    global dt     # Time step

    SC_dataframe = Tsa_dataframe.copy()
       
    propiedades = read_materials()
    
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
    solver = solve_PQ_AC if AC else solve_PQ
    
    if AC:  # AC = True
        while C > 5e-4: 
            Told = T.copy()
            print('calculo')
            for idx in range(n_steps):
                tint_prev = Ti_vals[idx]
                calculate_coefficients(mass_coeff, T, Tsa_vals[idx], ho, tint_prev, hi, d)
                T, tint_new = solver(a_static, b_static, c_static, d, T, Nx, tint_prev, capacitance_factor, P, Q, Tn_aux)
                Ti_new[idx] = tint_new
            Ti_vals[:] = Ti_new
            C = np.abs(Told - T).mean()
        #    FD   = (SC_dataframe.Ti.max() - SC_dataframe.Ti.min())/(SC_dataframe.Ta.max()-SC_dataframe.Ta.min())
        #    FDsa = (SC_dataframe.Ti.max() - SC_dataframe.Ti.min())/(SC_dataframe.Tsa.max()-SC_dataframe.Tsa.min())

        SC_dataframe['Ti'] = Ti_vals
        resultados = SC_dataframe['Ti']
    
    else:
        ET = 0.0
        while C > 5e-4: 
            Told = T.copy()
            ET_iter = 0.
            for idx in range(n_steps):
                tint_prev = Ti_vals[idx]
                calculate_coefficients(mass_coeff, T, Tsa_vals[idx], ho, tint_prev, hi, d)
                T, tint_new = solver(a_static, b_static, c_static, d, T, Nx, tint_prev, capacitance_factor, P, Q, Tn_aux)
                Ti_new[idx] = tint_new
                if T[Nx-1] > tint_new:
                    ET_iter += hi * (T[Nx - 1] - tint_new) * dt
            Ti_vals[:] = Ti_new
            C = np.abs(Told - T).mean()
            ET = ET_iter
        #    FD   = (SC_dataframe.Ti.max() - SC_dataframe.Ti.min())/(SC_dataframe.Ta.max()-SC_dataframe.Ta.min())
        #    FDsa = (SC_dataframe.Ti.max() - SC_dataframe.Ti.min())/(SC_dataframe.Tsa.max()-SC_dataframe.Tsa.min())

        SC_dataframe['Ti'] = Ti_vals
        resultados = SC_dataframe['Ti']
    if Energia:
        return resultados, ET
    else:
        return resultados
        
