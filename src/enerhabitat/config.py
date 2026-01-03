import configparser
import os

class Config:
    """
    Global configuration class for EnerHabitat simulations.
    
    Attributes:
        file (str): Path to the materials configuration file.
        La (float): Length of the dummy frame (m).
        Nx (int): Number of discretization elements.
        ho (float): Outdoor convective coefficient (W/m²K).
        hi (float): Indoor convective coefficient (W/m²K).
        dt (float): Time step (seconds).
        AIR_DENSITY (float): Density of air (kg/m³).
        AIR_HEAT_CAPACITY (float): Heat capacity of air (J/kgK).
        
    Methods:
        materials_list(): Returns the list of materials in the configuration file.
        materials_dict(): Returns a dictionary of materials and their properties.
        reset(): Resets configuration parameters to default values.
        info(): Prints the current configuration parameters.
        to_dict(): Returns the current configuration parameters as a dictionary.
    """
    def __init__(self):
        self.reset()
        self.__materials_file = "materials.ini"   # Default configuration file path

    def materials_list(self):
        """
        Returns the list of materials contained in the configuration file

        Returns:
            list: List of materials in the configuration file
        """
        config = configparser.ConfigParser()
        config.read(self.file)
        materiales = config.sections()
        return materiales

    def materials_dict(self):
        """
        Returns a dictionary with the list of materials and their properties

        Returns:
            dict: _description_
        """
        data = configparser.ConfigParser()
        data.read(self.file)

        class Material:
            def __init__(self, k, rho, c):
                self.k = k
                self.rho = rho
                self.c = c

        materiales = {}
        for material_i in data.sections():
            k = float(data[material_i]['k'])
            rho = float(data[material_i]['rho'])
            c = float(data[material_i]['c'])
            materiales[material_i] = Material(k, rho, c)

        return materiales
    
    def reset(self):
        self.__La = 2.5
        self.__Nx = 200
        self.__ho = 13
        self.__hi = 8.6
        self.__dt = 600
        self.__AIR_DENSITY = 1.1797660470258469
        self.__AIR_HEAT_CAPACITY = 1005.458757

        self.version = 0
        
    def info(self):
        print("<enerhabitat.Config -- Current config Parameters>")
        print(f"Materials file: \t\t\t{self.__materials_file}")
        print(f"La (Length of dummy frame): \t\t{self.La} m")
        print(f"Nx (Number of discretization elements):\t{self.Nx}")
        print(f"ho (Outdoor convective coefficient): \t{self.ho} W/m²K")
        print(f"hi (Indoor convective coefficient): \t{self.hi} W/m²K")
        print(f"dt (Time step): \t\t\t{self.dt} seconds")
        print(f"\nAIR_DENSITY: \t\t\t\t{self.AIR_DENSITY} kg/m³")
        print(f"AIR_HEAT_CAPACITY: \t\t\t{self.AIR_HEAT_CAPACITY} J/kgK")
    
    def to_dict(self):
        return {
            "La": self.La,
            "Nx": self.Nx,
            "ho": self.ho,
            "hi": self.hi,
            "dt": self.dt,
            "AIR_DENSITY": self.AIR_DENSITY,
            "AIR_HEAT_CAPACITY": self.AIR_HEAT_CAPACITY,
        }

    @property
    def file(self):
        try:    
            # Verificar si el archivo existe
            if not os.path.isfile(self.__materials_file):
                raise FileNotFoundError()
            return self.__materials_file
        except FileNotFoundError:
            print(f"Error: {self.__materials_file} not found")
            
    @file.setter
    def file(self, new_file):
        try:    
            # Verificar si el archivo existe
            if not os.path.isfile(new_file):
                raise FileNotFoundError()

            # Actualizar la configuración global si se proporcionó una nueva ruta
            self.__materials_file = new_file

        except FileNotFoundError:
            print(f"Error: {new_file} not found")    
        
    @property
    def La(self):
        return self.__La
    @La.setter
    def La(self, value):
        self.__La = value
        self.version += 1
        
    @property
    def Nx(self):
        return self.__Nx
    @Nx.setter
    def Nx(self, value):
        self.__Nx = value
        self.version += 1
        
    @property
    def ho(self):
        return self.__ho
    @ho.setter
    def ho(self, value):
        self.__ho = value
        self.version += 1
        
    @property
    def hi(self):
        return self.__hi
    @hi.setter
    def hi(self, value):
        self.__hi = value
        self.version += 1
        
    @property
    def dt(self):
        return self.__dt
    @dt.setter
    def dt(self, value):
        self.__dt = value
        self.version += 1
        
    @property
    def AIR_DENSITY(self):
        return self.__AIR_DENSITY
    @AIR_DENSITY.setter
    def AIR_DENSITY(self, value):
        self.__AIR_DENSITY = value
        self.version += 1
        
    @property
    def AIR_HEAT_CAPACITY(self):
        return self.__AIR_HEAT_CAPACITY
    @AIR_HEAT_CAPACITY.setter
    def AIR_HEAT_CAPACITY(self, value):
        self.__AIR_HEAT_CAPACITY = value
        self.version += 1
    
# Global configuration instance
config = Config()
