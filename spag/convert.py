# coding: utf-8

""" Utility functions from Spectroscopy Made Hard """

from __future__ import (division, print_function, absolute_import, unicode_literals)


from six import string_types
import numpy as np

import warnings

## Regular expressions
import re
m_XH = re.compile(r'\[(\D+)/H\]')
m_XFe= re.compile(r'\[(\D+)/Fe\]')

## SPAG imports
import spag.periodic_table  as pt
from spag.periodic_table import pt_list, pt_dict
from spag.utils import *
from spag.solar import *


################################################################################
## SMH Molecular Species and Uncommon Isotopes Identification by Atomic Number (Z)

common_molecule_name_to_Z = {
    'Mg-H': 12,'H-Mg': 12,
    'C-C':  6,
    'C-N':  7, 'N-C':  7,
    'C-H':  6, 'H-C':  6,
    'O-H':  8, 'H-O':  8,
    'Fe-H': 26,'H-Fe': 26,
    'N-H':  7, 'H-N':  7,
    'Si-H': 14,'H-Si': 14,
    'Ti-O': 22,'O-Ti': 22,
    'V-O':  23,'O-V':  23,
    'Zr-O': 40,'O-Zr': 40
    }
common_molecule_Z_to_name = {
    v: k for k, v in common_molecule_name_to_Z.items()
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
common_molecule_species_to_name = {
    v: k for k, v in common_molecule_name_to_species.items()
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
common_molecule_name_to_colname = {
    'C-N':  'N', 'N-C':  'N',
    'C-H':  'C', 'H-C':  'C',
    'C-C':  'C-C',
    'O-H':  'O-H', 'H-O':  'O-H',
    'N-H':  'N-H', 'H-N':  'N-H',
    }
common_molecule_colname_to_name = {
    v: k for k, v in common_molecule_name_to_colname.items()
}

common_isotope_colname_to_Z = {
    'Ca2': 20,
    'Ti1': 22,
    'Ti':  22,
    'V2':  23,
    'Cr2': 24,
    'Mn2': 25,
    'Fe2': 26,
    'Sr1': 38,
}
common_isotope_name_to_species = {
    'Ca II': 20.1,
    'Ti I':  22.0,
    'Ti II': 22.1,
    'V II':  23.1,
    'Cr II': 24.1,
    'Mn II': 25.1,
    'Fe II': 26.1,
    'Sr I':  38.0,
}
common_isotope_colname_to_species = {
    'Ca2': 20.1,
    'Ti1': 22.0,
    'Ti':  22.1,
    'V2':  23.1,
    'Cr2': 24.1,
    'Mn2': 25.1,
    'Fe2': 26.1,
    'Sr1': 38.0,
}
common_isotope_name_to_colname = {
    'Ca II': 'Ca2',
    'Ti I': 'Ti1',
    'Ti II': 'Ti',
    'V II': 'V2',
    'Cr II': 'Cr2',
    'Mn II': 'Mn2',
    'Fe II': 'Fe2',
    'Sr I': 'Sr1',
}
common_isotope_colname_to_name = {
    v: k for k, v in common_isotope_name_to_colname.items()
}
common_isotope_species_to_colname = {
    v: k for k, v in common_isotope_colname_to_species.items()
}

################################################################################
## Baseline Functions for the Conversions

def getelem(elem, lower=False, keep_species=False):
    """
    Converts an element's common name to a standard formatted chemical symbol
    """
        
    if isinstance(elem, string_types):
        prefix = None
        try:
            prefix,elem_ = identify_prefix(elem)
            elem = elem_
        except ValueError:
            pass

        if pt.element_query(elem) != None: # No ionization, e.g. Ti
            # print('1')
            elem = pt.element_query(elem).symbol
                
        elif elem in common_molecule_name_to_colname.keys():
            # print('2')
            elem = common_molecule_name_to_colname[elem]

        elif elem[-1]=='I': #Check for ionization
            # print('3')
            if ' ' not in elem:
                if elem[0]=='I':
                    assert elem=='I'*len(elem)
                    elem = 'I'
                elif elem[-2:]=='II':
                    elem = elem[:-2] + ' II'
                elif elem[-1]=='I':
                    elem = elem[:-1] + ' I'
            
            species = element_to_species(elem)
            if species in common_isotope_species_to_colname.keys():
                elem = common_isotope_species_to_colname[species]
            elif species in common_molecule_species_to_name.keys():
                elem = common_molecule_name_to_colname[common_molecule_species_to_name[species]]
            else:
                elem = species_to_element(species)
                elem = elem.split()[0]

        else:
            # print('4')
            species = element_to_species(elem)
            if species in common_isotope_species_to_colname.keys():
                elem = common_isotope_species_to_colname[species]
            elif species in common_molecule_species_to_name.keys():
                elem = common_molecule_name_to_colname[common_molecule_species_to_name[species]]
            else:
                elem = species_to_element(species)
                elem = elem.split()[0]

    elif isinstance(elem, (int, np.integer)):
        # print('5')

        Z = int(elem)
        try:
            elem = pt.element_query(Z).symbol
            if keep_species: elem = elem + ' ' + 'I'*get_default_ion(elem)
        except:
            if Z in common_molecule_species_to_name.keys():
                elem = common_molecule_name_to_colname[common_molecule_species_to_name[Z]]

    elif isinstance(elem, float):
        # print('6')
        species = elem
        
        if species in common_isotope_species_to_colname.keys():
            elem = common_isotope_species_to_colname[species]
        elif species in common_molecule_species_to_name.keys():
            elem = common_molecule_name_to_colname[common_molecule_species_to_name[species]]
        
        if isinstance(elem, float):
            elem = species_to_element(species)
            if not keep_species: elem = elem.split()[0]

    if lower: elem = elem.replace(' ','').lower()
    return elem

def get_default_ion(elem):
    """
    Returns the default ionization state for an element
    """

    # default ionization states for elements in SPAG



    default_to_1 = ['O','Na','Mg','Al','Si','K','Ca','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Pb',
                    'Rb','Mo','Ru','Rh','Os','Ir','U','Li','Ga','Tc',]
    default_to_2 = ['Sc','Ti','Sr','Y','Zr','Ba','La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Hf',
                    'Nb','Th']

    elem = getelem(elem)
    if elem in default_to_1:
        return 1
    elif elem in default_to_2:
        return 2
    else:
        warnings.warn("get_default_ion: {} not in defaults, returning 0".format(elem))
        return 0

def get_star_abunds(starname,data,type):
    """
    Input: starname, DataFrame, and type of abundance to extract ('eps', 'XH', 'XFe', 'e_', 'ul')
    Returns: a pandas Series of abundances for a star by extracting the columns of the specified type
    """
    assert type in ['eps','XH','XFe','e_','ul']
    star = data.ix[starname]
    colnames = _getcolnames(data,type)
    if len(colnames)==0: raise ValueError("{} not in data".format(type))
    abunds = np.array(star[colnames])
    elems = [getelem(elem) for elem in colnames]
    return pd.Series(abunds,index=elems)

def format_elemstr(elem):
    """
    Capitalizes the first letter of an element string
    """
    assert len(elem) <= 2 and len(elem) >= 1
    return elem[0].upper() + elem[1:].lower()

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
    element = element_repr.capitalize()
    # if element[0].islower():
    #     element = element.title()

    # Separate the element and the ionization state
    if " " in element_repr:
        element, ionization_str = element.split()[:2]
    else:
        element, ionization_str = element, "I"  # Default to neutral atom if no ionization state is provided
        
    # Handle unknown elements or molecules
    if element not in pt_list:
        try:
            return common_isotope_colname_to_species[element]
        except KeyError:
            try:
                return common_molecule_name_to_species[element_repr]
            except KeyError:
                print(f"Unknown element: {element_repr}")
                return str(element_repr) # Don't know what this element is
    
    # Convert Roman numeral ionization to integer
    ionization = max([0, roman_to_int(ionization_str) - 1, 0]) * 0.1
    
    # Find the atomic number of the element and add the ionization
    species = pt_list.index(element) + 1 + ionization
    return species

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
        index = pt_list.index(element)
    except IndexError:
        raise ValueError("unrecognized element '{}'".format(element_repr))
    except ValueError:
        try:
            return common_isotope_colname_to_Z[element]
        except KeyError:
            try:
                return common_molecule_name_to_Z[element]
            except KeyError:
                raise ValueError("unrecognized element '{}'".format(element_repr))
    return 1 + index

def element_to_ion(element_repr, state=None):
    """
    element_repr: str
        A string representation of the element. Typical examples include...
        'Fe' or 'fe' -> 'Fe I', 'Fe II', etc.
    state: int (default: None)
        The ionization state of the element.
        
    Converts a string representation of an element to thier default string
    representation of the element's ionization state.
    """

    if not isinstance(element_repr, string_types):
        raise TypeError("element must be represented by a string-type")
    
    if state is None:
        try:
            elem = element_repr.title().strip().split()[0]
            if not elem.isalpha():
                elem = elem[:-1]
            ionization_state = get_default_ion(elem)
            return f"{elem} {int_to_roman(ionization_state)}"

        except ValueError:
            molecule = element_repr.strip().split()[0]
            Z = element_to_atomic_number(molecule)
            elem = atomic_number_to_element(Z)
            print(f"element_to_ion: {molecule} -> {Z} -> {elem}")

            ionization_state = get_default_ion(elem)
            if ionization_state == 0:
                return f"{elem}"
            else:
                return f"{elem} {int_to_roman(ionization_state)}"
    else:
        return f"{element_repr.title()} {int_to_roman(state)}"

def species_to_element(species):
    """
    species: float
        A floating point representation of the species. Typical examples might
        be 26.0, 26.1, 26.2, etc. (only one decimal place is allowed)
    
    Converts a floating point representation of a species to the astronomical 
    string representation of the element and its ionization state.
    """

    if not isinstance(species, (float, int)):
        raise TypeError(f"species must be represented by a floating point-type: {species}, {type(species)}")
    
    if round(species, 1) != species:
        # Then you have isotopes, but we will ignore that
        species = int(species * 10) / 10.

    if species + 1 >= len(pt_list) or 1 > species:
        # Don't know what this element is. It's probably a molecule.
        try:
            elems = common_molecule_species_to_elems[species]
            return "-".join(elems)
        except KeyError:
            # No idea
            return str(species)
        
    atomic_number = int(species)
    element = pt_list[atomic_number - 1]
    ionization = int(round(10 * (species - atomic_number) + 1))

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

    if species + 1 >= len(pt_list) or 1 > species:
        # Don"t know what this element is. It"s probably a molecule.
        try:
            elems = common_molecule_species_to_elems[species]
            molecule = "-".join(elems)
            Z = element_to_atomic_number(molecule)
            # print(f"species_to_atomic_number: {species} -> {elems} -> {molecule} -> {Z}")
            return Z
        except KeyError:
            # No idea
            return str(species)
        
    atomic_number = int(species)
    return atomic_number

def species_to_ion(species):
    """
    species: float
        A floating point representation of the species. Typical examples might
        be 26.0, 26.1, 26.2, etc.
    
    Converts a floating point representation of a species to the astronomical
    string representation of the element and its ionization state.
    """
    
    return species_to_element(species)

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
        # if molecule
        if Z not in pt_dict:
            try:
                return common_molecule_name_to_Z[Z]
            except KeyError:
                return str(Z)
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

def atomic_number_to_ion(Z, state=None):
    """
    Z: int
        The atomic number of the element.
    state: int (default: None)
        The ionization state of the element.
        
    Converts an atomic number to the astronomical string representation of the
    element and its ionization state, if 'state' is provided. Otherwise, it
    returns the default element's ionization or just the element's symbol.
    """
    
    if state is None:
        ionization_state = get_default_ion(pt_dict[Z])
        if ionization_state == 0:
            return pt_dict[Z]
        else:
            return f"{pt_dict[Z]} {int_to_roman(ionization_state)}"
    else:
        return element_to_ion(pt_dict[Z], state)

def ion_to_species(ion):
    """
    ion: str
        The astronomical string representation of the element and its ionization
        state. Typical examples include 'Fe I', 'Fe II', etc.
    
    Converts the astronomical string representation of an element and its ionization
    state to a floating point representation of the species.
    """
    
    if not isinstance(ion, string_types):
        raise TypeError("ion must be represented by a string-type")
    
    if ion[0].islower():
        ion = ion.title()
    
    try:
        element, ionization_str = ion.split()
        Z = element_to_atomic_number(element)
        ionization = roman_to_int(ionization_str) - 1
        return Z + (ionization * 0.1)
    except ValueError:
        return float(element_to_species(ion))

def ion_to_element(ion):
    """
    ion: str
        The astronomical string representation of the element and its ionization
        state. Typical examples include 'Fe I', 'Fe II', etc.
    
    Converts the astronomical string representation of an element and its ionization
    state to the element's symbol. (Note: This)
    """
    
    if not isinstance(ion, string_types):
        raise TypeError("ion must be represented by a string-type")
    
    Z = ion_to_atomic_number(ion)
    elem = atomic_number_to_element(Z)
    # print(f"ion_to_element: {ion} -> {Z} -> {elem}")
    return elem

def ion_to_atomic_number(ion):
    """
    ion: str
        The astronomical string representation of the element and its ionization
        state. Typical examples include 'Fe I', 'Fe II', etc.
    
    Converts the astronomical string representation of an element and its ionization
    state to the element's atomic number.
    """
    
    if not isinstance(ion, string_types):
        raise TypeError("ion must be represented by a string-type")
    
    species = ion_to_species(ion)
    Z = species_to_atomic_number(species)
    # print(f"ion_to_atomic_number: {ion} -> {species} -> {Z}")
    return Z

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
# Utility functions for column names

def epscol(elem):
    """
    Returns the log(eps) column name for an element
    """
    return 'eps'+getelem(elem,lower=True)

def errcol(elem):
    """
    Returns the error column name for an element
    """
    try:
        return 'e_'+getelem(elem,lower=True)
    except ValueError:
        if elem=="alpha": return "e_alpha"
        else: raise
    
def e_ABcol(elem1, elem2):
    """
    Input a tuple of elements, returns the error column name for the pair
    """
    return f"e_[{getelem(elem1)}/{getelem(elem2)}]"

def ulcol(elem):
    """
    Returns the upper limit column name for an element
    """
    try:
        return 'ul'+getelem(elem,lower=True)
    except ValueError:
        if elem=="alpha": return "ulalpha"
        else: raise
    
def XHcol(elem, keep_species=False):
    """
    Returns the [X/H] column name for an element
    """
    try:
        return '['+getelem(elem, keep_species=keep_species)+'/H]'
    except ValueError:
        if elem=="alpha": return "[alpha/H]"
        else: raise

def ulXHcol(elem, keep_species=False):
    """
    Returns the ul[X/H] column name for an element
    """
    try:
        return 'ul['+getelem(elem, keep_species=keep_species)+'/H]'
    except ValueError:
        if elem=="alpha": return "ul[alpha/H]"
        else: raise

def XFecol(elem, keep_species=False):
    """
    Returns the [X/Fe] column name for an element
    """
    try:
        return '['+getelem(elem, keep_species=keep_species)+'/Fe]'
    except ValueError:
        if elem=="alpha": return "[alpha/Fe]"
        else: raise

def ulXFecol(elem, keep_species=False):
    """
    Returns the ul[X/Fe] column name for an element
    """
    try:
        return 'ul['+getelem(elem, keep_species=keep_species)+'/Fe]'
    except ValueError:
        if elem=="alpha": return "ul[alpha/Fe]"
        else: raise

def llXFecol(elem, keep_species=False):
    """
    Returns the ll[X/Fe] column name for an element
    """
    try:
        return 'll['+getelem(elem, keep_species=keep_species)+'/Fe]'
    except ValueError:
        if elem=="alpha": return "ll[alpha/Fe]"
        else: raise
        
def ABcol(elems):
    """
    Input a tuple of elements, returns the column name for the pair
    Note: by default the data does not have [A/B]
    """
    A,B = elems
    return '['+getelem(A)+'/'+getelem(B)+']'

## ---

def _getcolnames(df,prefix):
    """
    Returns a list of all columns with a specific prefix
    """
    allnames = []
    for col in df:
        try:
            this_prefix,elem = identify_prefix(col)
        except ValueError:
            continue
        if this_prefix==prefix: allnames.append(col)
    return allnames

def epscolnames(df):
    """
    Returns a list of all log(eps) columns
    """
    return _getcolnames(df,'eps')

def errcolnames(df):
    """
    Returns a list of all error columns
    """
    return _getcolnames(df,'e_')

def ulcolnames(df):
    """
    Returns a list of all upper limit columns
    """
    allnames = []
    for col in df:
        if col.startswith('ul') and not col.startswith('ul['):
            allnames.append(col)
    return allnames

def llcolnames(df):
    """
    Returns a list of all lower limit columns
    """
    allnames = []
    for col in df:
        if col.startswith('ll') and not col.startswith('ll['):
            allnames.append(col)
    return allnames

def XHcolnames(df):
    """
    Returns a list of all [X/H] columns
    """
    allnames = []
    for col in df:
        if col.startswith('[') and col.endswith('/H]'):
            allnames.append(col)
    return allnames

def ulXHcolnames(df):
    """
    Returns a list of all ul[X/H] columns
    """
    allnames = []
    for col in df:
        if col.startswith('ul[') and col.endswith('/H]'):
            allnames.append(col)
    return allnames
    
def XFecolnames(df):
    """
    Returns a list of all [X/Fe] columns
    """
    allnames = []
    for col in df:
        if col.startswith('[') and col.endswith('/Fe]'):
            allnames.append(col)
    return allnames

def ulXFecolnames(df):
    """
    Returns a list of all ul[X/Fe] columns
    """
    allnames = []
    for col in df:
        if col.startswith('ul[') and col.endswith('/Fe]'):
            allnames.append(col)
    return allnames

## ---

def make_epscol(species):
    """
    Converts species to a formatted log(eps) column name
    """
    return epscol(species)

def make_ulcol(species):
    """
    Converts species to a formatted upper limit column name
    """
    return ulcol(species)

def make_errcol(species):
    """
    Converts species to a formatted error column name
    """
    return errcol(species)

def make_XHcol(species):
    """
    Converts species to a formatted [X/H] column name
    """
    return XHcol(species)

def make_ulXHcol(species):
    """
    Converts species to a formatted ul[X/H] column name
    """
    return ulXHcol(species)

def make_XFecol(species):
    """
    Converts species to a formatted [X/Fe] column name
    """
    return XFecol(species)

def make_ulXFecol(species):
    """
    Converts species to a formatted ul[X/Fe] column name
    """
    return ulXFecol(species)

################################################################################
# Quick abundance conversion functions

def XH_from_eps(eps, elem, precision=2, version='asplund2009'):
    """
    Converts log(eps) to [X/H]
    """
    return normal_round(eps - get_solar(elem, version)[0], precision=precision)

def eps_from_XH(XH, elem, precision=2, version='asplund2009'):
    """
    Converts [X/H] to log(eps)
    """
    return normal_round(XH + get_solar(elem, version)[0], precision=precision)

def XFe_from_eps(eps, FeH, elem, precision=2, version='asplund2009'):
    """
    Converts log(eps) to [X/Fe]
    """
    return normal_round(eps - get_solar(elem, version)[0] - FeH, precision=precision)

def eps_from_XFe(XFe, FeH, elem, precision=2, version='asplund2009'):
    """
    Converts [X/Fe] to log(eps)
    """
    return  normal_round(XFe+ get_solar(elem, version)[0] + FeH, precision=precision)

def XFe_from_XH(XH, FeH, precision=2):
    """
    Converts [X/H] to [X/Fe]
    """
    return normal_round(XH - FeH, precision=precision)

def XH_from_XFe(XFe, FeH, precision=2):
    """
    Converts [X/Fe] to [X/H]
    """
    return normal_round(XFe + FeH, precision=precision)

## ---

def getcolion(col):
    """
    Returns the ionization state of an element column
    """
    prefix,elem = identify_prefix(col)
    if '.' in elem: int(ion = elem.split('.')[1])
    else: ion = get_default_ion(elem)
    ionstr = 'I'
    for i in range(ion): ionstr += 'I'
    return ionstr

def species_from_col(col):
    """
    Returns the numerical species from the input column name, using the default species.
    Accepts the following columns formats:
        formats: 'epsx', 'e_x', 'ulx', '[x/Fe]', '[x/H]'
    """

    if col.startswith('e_'): col = col.replace('e_', '')
    if col.startswith('eps'): col = col.replace('eps', '')
    if col.startswith('ul'): col = col.replace('ul', '')
    if col.startswith('['): 
        if col.endswith('/Fe]'): col = col.replace('[', '').replace('/Fe]', '')
        elif col.endswith('/H]'): col = col.replace('[', '').replace('/H]', '')
        else: raise ValueError(f"Column {col} not recognized for species extraction")
    elem = col.lower()

    if elem.title() in common_isotope_colname_to_species.keys():
        return common_isotope_colname_to_species[elem.title()]
    if elem.title() in common_molecule_colname_to_name.keys():
        return common_molecule_name_to_species[common_molecule_colname_to_name[elem.title()]]

    if elem.isalpha():
        default_state = get_default_ion(elem.title())
        if default_state == 1:
            species = element_to_species(elem.title()) + 0.0
        elif default_state == 2:
            species = element_to_species(elem.title()) + 0.1
        else:
            # print(elem)
            if elem.title() in pt_list: #default to ground ionization state (##.0)
                species = element_to_species(elem.title()) + 0.0
            else:
                raise ValueError(f"Element {elem} not recognized from epscol {col}")
        
    elif elem[:-1].isalpha() and elem[-1].isdigit():
        species = element_to_species(elem[:-1].title()) + (0.1 * float(elem[-1]) - 0.1)

    elif '-' in elem:
        # Handle common molecules
        try:
            species = common_molecule_name_to_species[elem.upper()]
        except KeyError:
            raise ValueError(f"Element {elem} not recognized from epscol {col}")
    else:
        raise ValueError(f"Invalid epscol format: {col}")
    
    return species

def ion_from_col(col):
    """
    Returns the ionization state from the input column name. See `species_from_col` for details.
    """
    return species_to_ion(species_from_col(col))

def jinabasecol_from_col(col):
    """
    Returns the numerical species from the input column name, using the default species.
    Accepts the following columns formats:
        formats: 'epsx', 'e_x', 'ulx', '[x/Fe]', '[x/H]'
    """

    if col.startswith('e_'): col = col.replace('e_', '')
    if col.startswith('eps'): col = col.replace('eps', '')
    if col.startswith('ul'): col = col.replace('ul', '')
    if col.startswith('['): 
        if col.endswith('/Fe]'): col = col.replace('[', '').replace('/Fe]', '')
        elif col.endswith('/H]'): col = col.replace('[', '').replace('/H]', '')
        else: raise ValueError(f"Column {col} not recognized for species extraction")
    elem = col.lower()

    if elem=="ti1": return 'Ti'  # Ti I
    if elem=="ti": return 'TiII'  # Ti II
    if elem=="v2": return 'VII'  # V II
    if elem=="fe2": return 'FeII'  # Fe II
    if elem=="cr2": return 'CrII'  # Cr II
    if elem=="sr1": return 'Sr'  # Sr I
    if elem=="c": return 'C'  # C-H
    if elem=="n-h": return 'N-H'  # N-H
    if elem=="n": return 'N'  # C-N
    if elem=="o-h": return 'O'  # O-H
    if elem=="ca2": return 'CaII'  # Ca II
    if elem=="mn2": return 'MnII'  # Mn II

    if elem.isalpha():
        return elem.title()  # Return the element name capitalized
    else:
        raise ValueError(f"Column {col} not recognized for species extraction")

################################################################################
# Utility functions operating on standardized DataFrame columns

def XHcol_from_epscol(df):
    """
    Converts log(eps) columns to [X/H] columns
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        epscols = epscolnames(df)
        asplund = get_solar(epscols)
        for col in epscols:
            if XHcol(col) in df: warnings.warn("{} already in DataFrame, replacing".format(XHcol(col)))
            eps_star = pd.to_numeric(df[col], errors='coerce')  # ensures proper float conversion
            eps_solar = float(asplund[col])
            df[XHcol(col)] = (eps_star - eps_solar).apply(lambda x: normal_round(x, precision=2))

def XHcol_from_XFecol(df):
    """
    Converts [X/Fe] columns to [X/H] columns
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        XFecols = XFecolnames(df)
        assert '[Fe/H]' in df
        asplund = get_solar(XFecols)
        feh = df['[Fe/H]']
        for col in XFecols:
            if XHcol(col) in df: warnings.warn("{} already in DataFrame, replacing".format(XHcol(col)))
            df[XHcol(col)] = df[col] + feh

def XFecol_from_epscol(df):
    """
    Converts log(eps) columns to [X/Fe] columns
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        epscols = epscolnames(df)
        assert 'epsfe' in epscols
        asplund = get_solar(epscols)
        epsfe_star = pd.to_numeric(df['epsfe'], errors='coerce')
        epsfe_solar = float(asplund['epsfe'])
        FeH = df['epsfe'].astype(float) - float(asplund['epsfe'])
        for col in epscols:
            if col=='epsfe': continue
            if XFecol(col) in df: warnings.warn("{} already in DataFrame, replacing".format(XFecol(col)))
            eps_star = pd.to_numeric(df[col], errors='coerce')
            eps_solar = float(asplund[col])
            XH = (eps_star - eps_solar).apply(lambda x: normal_round(x, precision=2))
            df[XFecol(col)] = (XH - FeH).apply(lambda x: normal_round(x, precision=2))

def epscol_from_XHcol(df):
    """
    Converts [X/H] columns to log(eps) columns
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        XHcols = XHcolnames(df)
        asplund = get_solar(XHcols)
        for col in XHcols:
            if epscol(col) in df: warnings.warn("{} already in DataFrame, replacing".format(epscol(col)))
            df[epscol(col)] = df[col] + float(asplund[col])

def XFecol_from_XHcol(df):
    """
    Converts [X/H] columns to [X/Fe] columns
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        XHcols = XHcolnames(df)
        assert '[Fe/H]' in XHcols
        feh = df['[Fe/H]']
        for col in XHcols:
            if col=='[Fe/H]': continue
            if XFecol(col) in df: warnings.warn("{} already in DataFrame, replacing".format(XFecol(col)))
            df[XFecol(col)] = df[col] - feh

def epscol_from_XFecol(df):
    """
    Converts [X/Fe] columns to log(eps) columns
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        XFecols = XFecolnames(df)
        assert '[Fe/H]' in df
        asplund = get_solar(XFecols)
        feh = df['[Fe/H]']
        for col in XFecols:
            df[epscol(col)] = df[col] + feh + float(asplund[col])

def XHcol_from_XFecol(df):
    """
    Converts [X/Fe] columns to [X/H] columns
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        XFecols = XFecolnames(df)
        assert '[Fe/H]' in df
        asplund = get_solar(XFecols)
        feh = df['[Fe/H]']
        for col in XFecols:
            if XHcol(col) in df: warnings.warn("{} already in DataFrame, replacing".format(XHcol(col)))
            df[XHcol(col)] = df[col] + feh

def ulcol_from_ulXHcol(df):
    """
    Converts the ul[X/H] columns to upper limit log(eps) columns
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ulXHcols = ulXHcolnames(df)
        asplund = get_solar(ulXHcols)
        for col in ulXHcols:
            if ulcol(col) in df: warnings.warn("{} already in DataFrame, replacing".format(ulcol(col)))
            ulXH_star = pd.to_numeric(df[col], errors='coerce')  # ensures proper float conversion
            eps_solar = float(asplund[col])
            df[ulcol(col)] = (ulXH_star + eps_solar).apply(lambda x: normal_round(x, precision=2))

def ulXHcol_from_ulcol(df):
    """
    Converts the upper limit log(eps) columns to ul[X/H] columns
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ulcols = ulcolnames(df)
        asplund = get_solar(ulcols)
        for col in ulcols:
            if ulXHcol(col) in df: warnings.warn("{} already in DataFrame, replacing".format(ulXHcol(col)))
            ul_eps_star = pd.to_numeric(df[col], errors='coerce')  # ensures proper float conversion
            ul_eps_solar = float(asplund[col])
            df[ulXHcol(col)] = (ul_eps_star - ul_eps_solar).apply(lambda x: normal_round(x, precision=2))

def ulXHcol_from_ulXFecol(df):
    """
    Converts the ul[X/Fe] columns to ul[X/H] columns
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        XFecols = XFecolnames(df)
        assert '[Fe/H]' in df
        asplund = get_solar(XFecols)
        feh = df['[Fe/H]']
        for col in XFecols:
            if col=='[Fe/H]': continue
            if ulXHcol(col) in df: warnings.warn("{} already in DataFrame, replacing".format(ulXHcol(col)))
            ulXFe_star = pd.to_numeric(df[col], errors='coerce')  # ensures proper float conversion
            eps_solar = float(asplund[col])
            df[ulXHcol(col)] = ulXFe_star + feh
            
def ulXFecol_from_ulcol(df):
    """
    Converts the upper limit log(eps) columns to ul[X/Fe] columns
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        epscols = epscolnames(df)
        assert 'epsfe' in epscols
        asplund = get_solar(epscols)
        FeH = df['epsfe'].astype(float) - float(asplund['epsfe'])
        ulcols = ulcolnames(df)
        for epscol, ulcol in zip(epscols, ulcols):
            if ulcol=='ulfe': continue
            if ulXFecol(epscol) in df: warnings.warn("{} already in DataFrame, replacing".format(ulXFecol(ulcol)))
            ul_eps_star = pd.to_numeric(df[ulcol], errors='coerce')
            eps_solar = float(asplund[epscol])
            ulXH = (ul_eps_star - eps_solar).apply(lambda x: normal_round(x, precision=2))
            df[ulXFecol(epscol)] = (ulXH - FeH).apply(lambda x: normal_round(x, precision=2))

def llXFecol_from_ulfecol(df):
    """
    Converts the upper limit log(eps) columns to ul[X/Fe] columns
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        epscols = epscolnames(df)
        ulcols = ulcolnames(df)
        assert ('ulfe' in ulcols) and ('ulfe' in df.columns)
        asplund = get_solar(epscols)
        ulFeH = df['ulfe'].astype(float) - float(asplund['epsfe'])
        for epscol, ulcol in zip(epscols, ulcols):
            if ulcol=='ulfe': continue
            if llXFecol(epscol) in df: warnings.warn("{} already in DataFrame, replacing".format(llXFecol(ulcol)))
            eps_star = pd.to_numeric(df[epscol], errors='coerce')
            eps_solar = float(asplund[epscol])
            XH_star = eps_star - eps_solar
            llXFe = XH_star - ulFeH
            df[llXFecol(epscol)] = llXFe.apply(lambda x: normal_round(x, precision=2))
            
################################################################################
## Random Conversion Functions

def struct2array(x):
    """
    x : np.ndarray
        A structured array where all columns must have the same data type.

    Converts a structured array to a 2D array with the same data as the input, 
    but without columns names.
    """
    
    # Number of columns
    num_columns = len(x.dtype)
    
    # Data type of the first column
    first_col_type = x.dtype[0].type

    # Ensure all columns have the same data type
    all_same_type = np.all([x.dtype[i].type == first_col_type for i in range(num_columns)])
    assert all_same_type, "All columns in the structured array must have the same data type."

    # Convert to a regular 2D array
    return x.view(first_col_type).reshape((-1, num_columns))