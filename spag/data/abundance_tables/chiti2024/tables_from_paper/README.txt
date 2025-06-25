Machine-readable data tables for the article, "Enrichment by Extragalactic First Stars in the Large Magellanic Cloud"
DOI: 10.1038/s41550-024-02223-w. 
Full descriptions of the tables are in the article, but we briefly summarize here.

Table 1 (Table1.csv):
    This table provides general information on the stars for which we obtained 
    long-exposure Magellan/MIKE data to derive their detailed elemental abundances. 
    Columns include names, coordinates, SkyMapper g magnitudes from the Gaia XP 
    spectra, and stellar parameters and metallicities with their respective random 
    uncertainties.

Table 2 (Table2.csv):
    This table provides names, radial velocities, metallicities, and selected 
    elemental abundances with random uncertainties for stars listed in Table 1. 
    [C/Fe]_c indicates carbon abundances that are corrected for the evolutionary 
    state of the star following Placco et al. (2014). Abundances that are upper 
    limits are flagged by the ul_[X/Fe] columns and have "nan" values for the 
    uncertainty.

Extended Data Table 1 (Extended_Data_Table1.csv):
    This table summarizes all of our observations, by providing names, coordinates, 
    SkyMapper g magnitudes from the Gaia XP spectra, exposure times, dates of 
    observation, and the instrument for these observations. This table includes 
    stars observed with MagE, those observed with short-exposures with MIKE for 
    just metallicities and carbon abundances, and those flagged as more 
    metal-rich upon initial exposure and hence, not further observed. 

Extended Data Table 2 (Extended_Data_Table2.csv):
    This table provides names, followed by stellar parameters, metallicities, and 
    carbon abundances, along with their respective uncertainties, for stars 
    observed with MagE or MIKE for short exposures to just obtain a metallicity 
    and carbon abundance. As in Table 2, abundances that are upper limits are 
    flagged by the ul_[C/Fe] column and "nan" entries for the uncertainty.

Supplementary Data 1 (Summary_Data_1.csv or Summary_Data_1.ascii):
    This table provides the suite of detailed element abundances and uncertainties 
    from the long-exposure MIKE spectra of the stars in Table 1. Columns include 
    the name, atomic number and ionization state of the element (element), the 
    number of features used to estimate the elemental abundance (N), the solar 
    abundance of that element (Solar), the absolute abundance (logeps), the 
    chemical abundance scaled by the solar abundance relative to hydrogen ([X/H]), 
    the ratio with respect to the iron abundance ([X/Fe]), the random uncertainty 
    ([X/H]_err) and an upper limit flag (ul), and errors from propagating the 
    uncertainties in the individual stellar parameters and the overall systematic 
    and total uncertainty ([X/H]_errteff, [X/H]_errlogg, [X/H]_errvt, [X/H]_errsys, 
    [X/H]_errtot). These are followed by the same columns, but with respect to iron 
    (e.g., [X/Fe]_errteff, [X/Fe]_errlogg). Abundances of the CH molecule are 
    indicated by 106.0 in the "element" column. This table is provided as a machine 
    readable csv file and as an ascii file, the latter for easier visual 
    readability.

Supplementary Data 2 (Summary_Data_2.csv or Summary_Data_2.ascii):
    This table summarizes the chemical abundances from individual absorption 
    features for the LMC stars with long-exposure MIKE spectra. The columns 
    include the star name, the atomic number and ionization state of the element 
    (species), the solar abundance of that element (Solar), followed by the 
    wavelength, excitation potential, and loggf of the line (wavelength, expot, 
    loggf), and then the measured equivalent width (EW), absolute abundance 
    (logeps), and a flag indicating whether the abundance is an upper limit (ul). 
    Abundances derived via spectral synthesis have "nan" entries for expot, loggf, 
    and EW, and abundances of the CH molecular band are indicated by 106.0. This 
    table is provided as a machine readable csv file and as an ascii file, the 
    latter for easier visual readability. 
