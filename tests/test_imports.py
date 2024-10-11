import spag

from spag import read_data as rd

print("spag version:", spag.__version__)
print("logeps(H) = ", rd.solar_logepsX_asplund09(Z=1))
