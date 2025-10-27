import pvlib
import pytz

from datetime import datetime
from .ehtools import materials_file, materials_list, materials_dict, readEPW
from .ehcalcs import meanDay, Tsa, solveCS

La = 2.5    # Length of the dummy frame
Nx = 200     # Number of elements to discretize
ho = 13     # Outside convection heat transfer
hi = 8.6    # Inside convection heat transfer
dt = 600     # Time step in seconds

# Propiedades del aire empleadas en el modelo lumped-capacitance
AIR_DENSITY = 1.1797660470258469
AIR_HEAT_CAPACITY = 1005.458757
        
        
        
