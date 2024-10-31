# coding: utf-8

""" Utility functions from Spectroscopy Made Hard """

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

from six import string_types

from spag.periodic_table import pt_list, pt_dict

# Functions to import when using 'from spag.utils_smh import *'
# __all__ =  ["element_to_species", "element_to_atomic_number",
#             "species_to_element", "species_to_atomic_number",
#             "atomic_number_to_species", "atomic_number_to_element",
#             "element_matches_atomic_number",
#             "elems_isotopes_ion_to_species", "species_to_elems_isotopes_ion"]

################################################################################
## Roman numeral conversion functions

def int_to_roman(n):
    """ Convert an integer to Roman numerals. """
    roman_int_dict = {'I': 1, 'IV': 4, 'V': 5, 'IX': 9, 'X': 10, 'XL': 40, \
                      'L': 50, 'XC': 90, 'C': 100, 'CD': 400, 'D': 500, \
                      'CM': 900, 'M': 1000}
    roman_numeral = ''
    for numeral, value in sorted(roman_int_dict.items(), key=lambda x: x[1], reverse=True):
        while n >= value:
            roman_numeral += numeral
            n -= value
    
    return roman_numeral

def roman_to_int(roman):
    """ Convert a Roman numeral to an integer, up to several thousand. """
    roman = roman.upper()
    roman_int_dict = {'I': 1, 'IV': 4, 'V': 5, 'IX': 9, 'X': 10, 'XL': 40, \
                      'L': 50, 'XC': 90, 'C': 100, 'CD': 400, 'D': 500, \
                      'CM': 900, 'M': 1000}
    value = 0
    for i in range(len(roman)):
        if i > 0 and roman_int_dict[roman[i]] > roman_int_dict[roman[i - 1]]:
            value += roman_int_dict[roman[i]] - 2 * roman_int_dict[roman[i - 1]]
        else:
            value += roman_int_dict[roman[i]]
    return value


################################################################################
## SMH Molecular Species Identification by Atomic Number (Z)

common_molecule_name_to_Z = {
    'Mg-H': 12,'H-Mg': 12,
    'C-C':  6,
    'C-N':  7, 'N-C':  7, #TODO
    'C-H':  6, 'H-C':  6,
    'O-H':  8, 'H-O':  8,
    'Fe-H': 26,'H-Fe': 26,
    'N-H':  7, 'H-N':  7,
    'Si-H': 14,'H-Si': 14,
    'Ti-O': 22,'O-Ti': 22,
    'V-O':  23,'O-V':  23,
    'Zr-O': 40,'O-Zr': 40
    }
common_molecule_name_to_species = {
    'Mg-H': 112,'H-Mg': 112,
    'C-C':  606,
    'C-N':  607,'N-C':  607,
    'C-H':  106,'H-C':  106,
    'O-H':  108,'H-O':  108,
    'Fe-H': 126,'H-Fe': 126,
    'N-H':  107,'H-N':  107,
    'Si-H': 114,'H-Si': 114,
    'Ti-O': 822,'O-Ti': 822,
    'V-O':  823,'O-V':  823,
    'Zr-O': 840,'O-Zr': 840
    }
common_molecule_species_to_elems = {
    112: ["Mg", "H"],
    606: ["C", "C"],
    607: ["C", "N"],
    106: ["C", "H"],
    108: ["O", "H"],
    126: ["Fe", "H"],
    107: ["N", "H"],
    114: ["Si", "H"],
    822: ["Ti", "O"],
    823: ["V", "O"],
    840: ["Zr", "O"]
    }

################################################################################
## Element, atomic number, and species conversion functions

def element_to_species(element_repr):
    """
    element_repr: str
        A string representation of the element. Typical examples include...
        'Fe I' or 'Fe' or 'fe' -> 26.0, 'C-H' -> 106.0
    
    Converts a string from the astronomical representation of an element and its
    ionization state (or SMH-known molecules) to a floating point. 
    """
    
    if not isinstance(element_repr, string_types):
        raise TypeError("element must be represented by a string-type.")
    
    # If first character is lowercase, capitalize it
    element_repr = element_repr.capitalize()
    # if element_repr[0].islower():
    #     element_repr = element_repr.title()

    # Separate the element and the ionization state
    if " " in element_repr:
        element, ionization_str = element_repr.split()[:2]
    else:
        element, ionization_str = element_repr, "I"  # Default to neutral atom if no ionization state is provided
        
    # Handle unknown elements or molecules
    if element not in pt:
        try:
            return common_molecule_name_to_species[element]
        except KeyError:
            return float(element_repr) # Don't know what this element is
    
    # Convert Roman numeral ionization to integer
    ionization = max([0, roman_to_int(ionization_str) - 1, 0]) * 0.1
    
    # Find the atomic number of the element and add the ionization
    transition = pt.index(element) + 1 + ionization
    return transition


def element_to_atomic_number(element_repr):
    """
    element_repr: str
        A string representation of the element. Typical examples include...
        'Fe I' or 'Fe' or 'fe' -> 26, 'C-H' -> 6
        
    Converts a string representation of an element (ionization state is ignored)
    to a floating point representation of the element's atomic number.
    """
    
    if not isinstance(element_repr, string_types):
        raise TypeError("element must be represented by a string-type")
    
    element = element_repr.title().strip().split()[0]
    try:
        index = pt.index(element)
    except IndexError:
        raise ValueError("unrecognized element '{}'".format(element_repr))
    except ValueError:
        try:
            return common_molecule_name_to_Z[element]
        except KeyError:
            raise ValueError("unrecognized element '{}'".format(element_repr))
    return 1 + index

def species_to_element(species):
    """
    species: float
        A floating point representation of the species. Typical examples might
        be 26.0, 26.1, 26.2, etc. (only one decimal place is allowed)
    
    Converts a floating point representation of a species to the astronomical 
    string representation of the element and its ionization state.
    """

    if not isinstance(species, (float, int)):
        raise TypeError("species must be represented by a floating point-type")
    
    if round(species, 1) != species:
        # Then you have isotopes, but we will ignore that
        species = int(species * 10) / 10.

    if species + 1 >= len(pt) or 1 > species:
        # Don't know what this element is. It's probably a molecule.
        try:
            elems = common_molecule_species_to_elems[species]
            return "-".join(elems)
        except KeyError:
            # No idea
            return str(species)
        
    atomic_number = int(species)
    element = pt[atomic_number - 1]
    ionization = int(round(10 * (species - atomic_number)) + 1)

    if element in ("C", "H", "He"):
        # Special cases where no ionization state is usually shown
        return element

    # Convert the ionization number to Roman numerals
    return "%s %s" % (element, int_to_roman(ionization))

def species_to_atomic_number(species):
    """
    species: float
        A floating point representation of the species. Typical examples might
        be 26.0, 26.1, 26.2, etc.
    
    Converts a floating point representation of a species to the atomic number
    of the element.
    """
    
    if not isinstance(species, (float, int)):
        raise TypeError("species must be represented by a floating point-type")
    
    if round(species,1) != species:
        # Then you have isotopes, but we will ignore that
        species = int(species*10)/10.

    if species + 1 >= len(pt) or 1 > species:
        # Don"t know what this element is. It"s probably a molecule.
        try:
            elems = common_molecule_species_to_elems[species]
            return "-".join(elems)
        except KeyError:
            # No idea
            return str(species)
        
    atomic_number = int(species)
    return atomic_number
    
def atomic_number_to_element(Z, species=None):
    """
    Z: int
        The atomic number of the element.
    species: float (default: None)
        The decimal for the species of the element (26.0 -> 0, 26.1 -> 1)
        
    Converts an atomic number to the astronomical string representation of the
    element and its ionization state, if 'species' is provided. Otherwise, it
    returns the element's symbol.
    """
    
    if species is not None:
        return species_to_element(int(Z) + (species*0.1))
    else:
        return pt_dict[int(Z)]


def atomic_number_to_species(Z, species=None):
    """
    Z: int
        The atomic number of the element.
    species: float (default: None)
        The decimal for the species of the element (26.0 -> 0, 26.1 -> 1)
        
    Converts an atomic number to the astronomical string representation of the
    element and its ionization state, if 'species' is provided. Otherwise, it
    assumes the element is neutral (i.e., no ionization state).
    """
    
    if species is not None:
        return int(Z) + (species * 0.1)
    else:
        return float(Z)


################################################################################
## Molecule Species Identification by Elements, Isotopes, and Ionization State

def elems_isotopes_ion_to_species(elem1,elem2,isotope1,isotope2,ion):
    """
    elem1: str
        The first element in the molecule.
    elem2: str
        The second element in the molecule.
    isotope1: int
        The isotope of the first element.
    isotope2: int
        The isotope of the second element.
    ion: int
        The ionization state of the molecule.
    
    Converts the elements, isotopes, and ionization state of a molecule to a
    floating point representation of the species, using the MOOG formatting.
    """
    
    Z1 = int(element_to_species(elem1.strip()))
    if isotope1==0: isotope1=''
    else: isotope1 = str(isotope1).zfill(2)

    # print(Z1,elem1,int(ion-1), isotope1)
    if elem2.strip()=='': # Atom
        print("Atom: ", elem1, Z1, int(ion-1), isotope1)
        mystr = "{}.{}{}".format(Z1,int(ion-1),isotope1)
    else: # Molecule
        #assert ion==1,ion
        Z2 = int(element_to_species(elem2.strip()))

        # If one isotope is specified but the other isn't, use a default mass
        # These masses are taken from MOOG for Z=1 to 95
        amu = [1.008,4.003,6.941,9.012,10.81,12.01,14.01,16.00,19.00,20.18,
               22.99,24.31,26.98,28.08,30.97,32.06,35.45,39.95,39.10,40.08,
               44.96,47.90,50.94,52.00,54.94,55.85,58.93,58.71,63.55,65.37,
               69.72,72.59,74.92,78.96,79.90,83.80,85.47,87.62,88.91,91.22,
               92.91,95.94,98.91,101.1,102.9,106.4,107.9,112.4,114.8,118.7,
               121.8,127.6,126.9,131.3,132.9,137.3,138.9,140.1,140.9,144.2,
               145.0,150.4,152.0,157.3,158.9,162.5,164.9,167.3,168.9,173.0,
               175.0,178.5,181.0,183.9,186.2,190.2,192.2,195.1,197.0,200.6,
               204.4,207.2,209.0,210.0,210.0,222.0,223.0,226.0,227.0,232.0,
               231.0,238.0,237.0,244.0,243.0]
        amu = [int(round(x,0)) for x in amu]
        if isotope1 == '':
            if isotope2 == 0:
                isotope2 = ''
            else:
                isotope1 = str(amu[Z1-1]).zfill(2)
        else:
            if isotope2 == 0:
                isotope2 = str(amu[Z2-1]).zfill(2)
            else:
                isotope2 = str(isotope2).zfill(2)
        # Swap if needed
        if Z1 < Z2:
            mystr = "{}{:02}.{}{}{}".format(Z1,Z2,int(ion-1),isotope1,isotope2)
        else:
            mystr = "{}{:02}.{}{}{}".format(Z2,Z1,int(ion-1),isotope2,isotope1)

    return float(mystr)

def species_to_elems_isotopes_ion(species):
    """
    species: float
        A floating point representation of the species, using the MOOG format.
        Typical examples might be 260.0, 260.1, 260.2, etc.
    
    Converts a floating point representation of a species to the elements,
    isotopes, and ionization state of a molecule, using the MOOG formatting.
    """
    
    element = species_to_element(species)
    if species >= 100:
        # Molecule
        Z1 = int(species/100)
        Z2 = int(species - Z1*100)
        elem1 = species_to_element(Z1).split()[0]
        elem2 = species_to_element(Z2).split()[0]
        # All molecules that we use are unionized
        ion = 1
        if species == round(species,1):
            # No isotope specified
            isotope1 = 0
            isotope2 = 0
        else: #Both isotopes need to be specified!
            isotope1 = int(species*1000) - int(species*10)*100
            isotope2 = int(species*100000) - int(species*1000)*100
            if isotope1 == 0 or isotope2 == 0: 
                raise ValueError("molecule species must have both isotopes specified: {} -> {} {}".format(species,isotope1,isotope2))
        # Swap if needed
    else:
        # Element
        try:
            elem1,_ion = element.split()
        except ValueError as e:
            if element == 'C':
                elem1,_ion = 'C','I'
            elif element == 'H':
                elem1,_ion = 'H','I'
            elif element == 'He':
                elem1,_ion = 'He','I'
            else:
                print(element)
                raise e
        ion = len(_ion)
        assert _ion == 'I'*ion, "{}; {}".format(_ion,ion)
        if species == round(species,1):
            isotope1 = 0
        elif species == round(species,4):
            isotope1 = int(species*10000) - int(species*10)*1000
        elif species == round(species,3):
            isotope1 = int(species*1000) - int(species*10)*100
        else:
            raise ValueError("problem determining isotope: {}".format(species))
        elem2 = ''
        isotope2 = 0
    return elem1,elem2,isotope1,isotope2,ion


################################################################################
## Validation/Testing Functions

def element_matches_atomic_number(elem, Z):
    """
    elem: str
        Element symbol.
    Z: int
        Atomic number.
        
    Returns True if the element symbol matches the atomic number, and False
    otherwise.    
    """
    
    if elem != pt_dict[Z]:
        return False
    else:
        return True

