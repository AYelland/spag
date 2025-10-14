# coding: utf-8

""" Element class for interactive shell """

from __future__ import (division, print_function, absolute_import, unicode_literals)


import os
from sys import exit
import sqlalchemy
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base

################################################################################
## Periodic Table & Dictionary
# ------------------------------------------------------------------------------
# The periodic table ('pt') defined below is a list of the elemental symbols in the
# periodic table. The list then used to generate a dictionary, 
# `pt_dict`. These can be useful when imported into other scripts
# for converting from atomic numbers to elements.
# 
# Example Import:
# import spag.periodic_table as pt
# from spag.periodic_table import pt
# from spag.periodic_table import pt_dict


pt_str = """H                                                  He
                        Li Be                               B  C  N  O  F  Ne
                        Na Mg                               Al Si P  S  Cl Ar
                        K  Ca Sc Ti V  Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr
                        Rb Sr Y  Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I  Xe
                        Cs Ba Lu Hf Ta W  Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn
                        Fr Ra Lr Rf""" # Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"""

lanthanoids    =   "La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb"
actinoids      =   "Ac Th Pa U  Np Pu Am Cm Bk Cf Es Fm Md No"

pt_str = pt_str.replace(" Ba ", " Ba " +lanthanoids+ " ") \
    .replace(" Ra ", " Ra " +actinoids+ " ")
del actinoids, lanthanoids

pt_list = pt_str.split() # list of elements in order of atomic number

## Periodic table dictionary, useful for converting atomic numbers to elements
pt_dict = dict()
for i, elem in enumerate(pt_list):
    pt_dict[i+1] = elem.strip()


################################################################################
## Interactive Shell Query
# ------------------------------------------------------------------------------
# This script is a simple interactive shell that allows users to query the
# periodic table database. The script uses SQLAlchemy to query the database and
# return the element information. The database is a SQLite database that
# contains the atomic number, symbol, name, and mass of each element.
#
# The script uses the 'Element' class to define the table schema and the
# 'element_query' function to query the database. The 'interactive_shell'
# function is used to run the interactive shell.
#
# The script can be run directly to start the interactive shell. The user can
# enter the atomic number, symbol, name, or mass of an element to receive more
# information about it. The user can type 'exit' or use ^C to exit the shell.
#
# The script can also be imported and used in other scripts. The 'element_query'
# function can be used to query the database and return the element information.
# This is specifically done in the 'utils_smh.py' script to query the database
# and return the element information for the 'getelem' function.


# Database Setup
spag_dir = os.path.abspath(os.path.dirname(__file__))
index = os.path.join(spag_dir, 'data/periodic_table/table.db')
engine = sqlalchemy.create_engine(f'sqlite:///{index}')
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()


# Element Class for SQLAlchemy
class Element(Base):
    __tablename__ = 'element'
    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True)
    symbol = sqlalchemy.Column(sqlalchemy.String)
    name = sqlalchemy.Column(sqlalchemy.String)
    atomic = sqlalchemy.Column(sqlalchemy.Integer)
    mass = sqlalchemy.Column(sqlalchemy.Float)

    def __repr__(self):
        return f"<Element(symbol='{self.symbol}', atomic={self.atomic})>"

# Function to determine input type
def type_(input_):
    try:
        value = float(input_)
        return int if value.is_integer() else float
    except ValueError:
        return str

# Function to query element
def element_query(_input):
    value = type_(_input)

    if value is int:
        return session.query(Element).filter(Element.atomic == _input, Element.atomic <= 92).first()
    if value is float:
        return session.query(Element).filter(Element.mass == _input, Element.atomic <= 92).first()

    _input = _input.capitalize()
    if len(_input) <= 2:
        return session.query(Element).filter(Element.symbol == _input, Element.atomic <= 92).first()
    return session.query(Element).filter(Element.name == _input, Element.atomic <= 92).first()


# Interactive Shell
def interactive_shell():
    attributes = ['id', 'atomic', 'symbol', 'name', 'mass']

    usage = "Enter the atomic number, symbol, name, or mass of an element to recieve more \ninformation about it. (use ^C or type 'exit' to exit.)"
    buff = '=' * 80
    table = """    H                                                  He
    Li Be                               B  C  N  O  F  Ne
    Na Mg                               Al Si P  S  Cl Ar
    K  Ca Sc Ti V  Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr
    Rb Sr Y  Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I  Xe
    Cs Ba    Hf Ta W  Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn
    Fr Ra    Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og
             La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb      <-- lanthanoids 
             Ac Th Pa U  Np Pu Am Cm Bk Cf Es Fm Md No      <-- actinoids"""
    print()
    print(buff)
    print()
    print(table)
    print()
    print(usage)

    while True:
        try:
            print(buff)
            query = input("> ")
            if query.lower() == 'exit':
                raise KeyboardInterrupt
        except KeyboardInterrupt:
            print("\nBye!")
            exit(0)

        element_ = element_query(query)
        if element_ is None:
            print(f"'{query}' is not valid!")
            print()
            print(usage)
            continue

        values = [f"{attribute}: {getattr(element_, attribute)}" for attribute in attributes]
        for line in values:
            print(line)

# Run the interactive shell when script is called directly
if __name__ == '__main__':
    interactive_shell()
