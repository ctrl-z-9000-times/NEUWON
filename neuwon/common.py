import numpy as np

F = 96485.3321233100184 # Faraday's constant, Coulumbs per Mole of electrons
R = 8.31446261815324 # Universal gas constant
celsius = 37 # Human body temperature
T = celsius + 273.15 # Human body temperature in Kelvins

Real = np.dtype('f4')
epsilon = np.finfo(Real).eps
Location = np.dtype('u4')
ROOT = np.iinfo(Location).max

def docstring_wrapper(property_name, docstring):
        def get_prop(self):
            return self.__dict__[property_name]
        def set_prop(self, value):
            self.__dict__[property_name] = value
        return property(get_prop, set_prop, None, docstring)
