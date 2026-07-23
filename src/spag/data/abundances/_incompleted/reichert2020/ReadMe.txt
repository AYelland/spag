J/A+A/641/A127     13 dsph and ultra-faint galaxies analysis   (Reichert+, 2020)
================================================================================
Neutron-capture elements in dwarf galaxies.
III: A homogenized analysis of 13 dwarf spheroidal and ultra-faint galaxies.
    Reichert M., Hansen C.J., Hanke M., Skuladottir A., Arcones A., Grebel E.K.
    <Astron. Astrophys. 641, A127 (2020)>
    =2020A&A...641A.127R        (SIMBAD/NED BibCode)
================================================================================
ADC_Keywords: Galaxies, nearby ; Abundances
Keywords: galaxies: dwarf - galaxies: abundances - galaxies: evolution -
          catalogs - stars: abundances - stars: fundamental parameters

Abstract:
    We present a large homogeneous set of stellar parameters and
    abundances across a broad range of metallicities, involving 13
    classical dwarf spheroidal (dSph) and ultra-faint dSph (UFD) galaxies.
    In total, this study includes 380 stars in Fornax, Sagittarius,
    Sculptor, Sextans, Carina, Ursa Minor, Draco, Reticulum II, Bootes I,
    Ursa Major II, Leo I, Segue I, and Triangulum II. This sample
    represents the largest, homogeneous, high-resolution study of dSph
    galaxies to date.

    With our homogeneously derived catalog, we are able to search for
    similar and deviating trends across different galaxies. We investigate
    the mass dependence of the individual systems on the production of
    alpha-elements, but also try to shed light on the long-standing puzzle
    of the dominant production site of r-process elements.

    We used data from the Keck observatory archive and the ESO reduced
    archive to reanalyze stars from these 13 classical dSph and UFD
    galaxies. We automatized the step of obtaining stellar parameters, but
    ran a full spectrum synthesis (1D, local thermal equilibrium) to
    derive all abundances except for iron to which we applied nonlocal
    thermodynamic equilibrium corrections where possible.

    The homogenized set of abundances yielded the unique possibility of
    deriving a relation between the onset of type Ia supernovae and the
    stellar mass of the galaxy. Furthermore, we derived a formula to
    estimate the evolution of alpha-elements. This reveals a universal
    relation of these systems across a large range in mass. Finally, we
    show that between stellar masses of 2.1x10^7^M_{sun}_ and
    2.9x10^5^M_{sun}_ , there is no dependence of the production of heavy
    r-process elements on the stellar mass of the galaxy.

    Placing all abundances consistently on the same scale is crucial to
    answering questions about the chemical history of galaxies. By
    homogeneously analyzing Ba and Eu in the 13 systems, we have traced
    the onset of the s-process and found it to increase with metallicity
    as a function of the galaxy's stellar mass. Moreover, the r-process
    material correlates with the alpha-elements indicating some
    coproduction of these, which in turn would point toward rare
    core-collapse supernovae rather than binary neutron star mergers as a
    host for the r-process at low [Fe/H] in the investigated dSph systems.

Description:
    The following tables contain abundances of individual absorption
    lines, stellar parameters of the investigated stars, and averaged
    abundances together with their error.

File Summary:
--------------------------------------------------------------------------------
 FileName      Lrecl  Records   Explanations
--------------------------------------------------------------------------------
ReadMe            80        .   This file
tableo1.dat       67      380   Stellar parameters
tableo2.dat       57     4606   Absolute abundances per absorption feature
tableo3.dat      493      380   Absolute abundances and inferred errors
--------------------------------------------------------------------------------

See also:
 J/A+A/631/A171 : Neutron-capture elements in dwarf galaxies (Skuladottir+ 2019)

Byte-by-byte Description of file: tableo1.dat
--------------------------------------------------------------------------------
   Bytes Format Units     Label     Explanations
--------------------------------------------------------------------------------
   1- 30  A30   ---       ID        Object identifier
  32- 37  A6    ---       Galaxy    Galaxy identifier
  39- 42  I4    K         Teff      Adopted effective temperature
  44- 46  I3    K       e_Teff      Effective temperature error
  48- 52  F5.2  [-]       [Fe/H]    Metallicity
  54- 57  F4.2  [-]     e_[Fe/H]    Metallicity error
  59- 62  F4.2  [cm/s2]   logg      Adopted surface gravity
  64- 67  F4.2  [cm/s2] e_logg      Surface gravity error
--------------------------------------------------------------------------------

Byte-by-byte Description of file: tableo2.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label     Explanations
--------------------------------------------------------------------------------
   1- 30  A30   ---     ID        Object identifier
  32- 37  A6    ---     Galaxy    Galaxy identifier
  39- 40  A2    ---     El        Investigated element
  42- 43  A2    ---     Ion       Ionization state of the absorption feature
  45- 51  F7.2  0.1nm   lambda    Wavelength of the absorption feature
  53- 57  F5.2  [-]     log(eps)  Absolute abundance
--------------------------------------------------------------------------------

Byte-by-byte Description of file: tableo3.dat
--------------------------------------------------------------------------------
   Bytes Format Units  Label        Explanations
--------------------------------------------------------------------------------
   1- 30  A30   ---    ID           Object identifier
  32- 37  A6    ---    Galaxy       Galaxy identifier
  39- 43  F5.2  [-]    [Fe/H]       Metallicity
  45- 48  F4.2  [-]  e_[Fe/H]       Metallicity error
  50- 53  F4.2  [-]    logeps(Mg)   ? Absolute Mg abundance
  55- 58  F4.2  [-]  e_(tot)(Mg)    ? Total error on Mg
  60- 63  F4.2  [-]  e_(temp)(Mg)   ? Temperature error on Mg
  65- 68  F4.2  [-]  e_(logg)(Mg)   ? Surface gravity error on Mg
  70- 73  F4.2  [-]  e_([Fe/H])(Mg) ? Metallicity error on Mg
  75- 78  F4.2  [-]  e_(v)(Mg)      ? Microturbulence error on Mg
  80- 83  F4.2  [-]  e_(stat)(Mg)   ? Statistical error on Mg
  85- 88  F4.2  [-]  e_(noise)(Mg)  ? Error inferred through noise on Mg
  90- 94  F5.2  [-]    logeps(Sc)   ? Absolute Sc abundance
  96- 99  F4.2  [-]  e_(tot)(Sc)    ? Total error on Sc
 101-104  F4.2  [-]  e_(temp)(Sc)   ? Temperature error on Sc
 106-109  F4.2  [-]  e_(logg)(Sc)   ? Surface gravity error on Sc
 111-114  F4.2  [-]  e_([Fe/H])(Sc) ? Metallicity error on Sc
 116-119  F4.2  [-]  e_(v)(Sc)      ? Microturbulence error on Sc
 121-124  F4.2  [-]  e_(stat)(Sc)   ? Statistical error on Sc
 126-129  F4.2  [-]  e_(noise)(Sc)  ? Error inferred through noise on Sc
 131-134  F4.2  [-]    logeps(Ti)   ? Absolute Ti abundance
 136-139  F4.2  [-]  e_(tot)(Ti)    ? Total error on Ti
 141-144  F4.2  [-]  e_(temp)(Ti)   ? Temperature error on Ti
 146-149  F4.2  [-]  e_(logg)(Ti)   ? Surface gravity error on Ti
 151-154  F4.2  [-]  e_([Fe/H])(Ti) ? Metallicity error on Ti
 156-159  F4.2  [-]  e_(v)(Ti)      ? Microturbulence error on Ti
 161-164  F4.2  [-]  e_(stat)(Ti)   ? Statistical error on Ti
 166-169  F4.2  [-]  e_(noise)(Ti)  ? Error inferred through noise on Ti
 171-174  F4.2  [-]   logeps(Cr)    ? Absolute Cr abundance
 176-179  F4.2  [-]  e_(tot)(Cr)    ? Total error on Cr
 181-184  F4.2  [-]  e_(temp)(Cr)   ? Temperature error on Cr
 186-189  F4.2  [-]  e_(logg)(Cr)   ? Surface gravity error on Cr
 191-194  F4.2  [-]  e_([Fe/H])(Cr) ? Metallicity error on Cr
 196-199  F4.2  [-]  e_(v)(Cr)      ? Microturbulence error on Cr
 201-204  F4.2  [-]  e_(stat)(Cr)   ? Statistical error on Cr
 206-209  F4.2  [-]  e_(noise)(Cr)  ? Error inferred through noise on Cr
 211-214  F4.2  [-]    logeps(Mn)   ? Absolute Mn abundance
 216-219  F4.2  [-]  e_(tot)(Mn)    ? Total error on Mn
 221-224  F4.2  [-]  e_(temp)(Mn)   ? Temperature error on Mn
 226-229  F4.2  [-]  e_(logg)(Mn)   ? Surface gravity error on Mn
 231-234  F4.2  [-]  e_([Fe/H])(Mn) ? Metallicity error on Mn
 236-239  F4.2  [-]  e_(v)(Mn)      ? Microturbulence error on Mn
 241-244  F4.2  [-]  e_(stat)(Mn)   ? Statistical error on Mn
 246-249  F4.2  [-]  e_(noise)(Mn)  ? Error inferred through noise on Mn
 251-254  F4.2  [-]    logeps(Ni)   ? Absolute Ni abundance
 256-259  F4.2  [-]  e_(tot)(Ni)    ? Total error on Ni
 261-264  F4.2  [-]  e_(temp)(Ni)   ? Temperature error on Ni
 266-269  F4.2  [-]  e_(logg)(Ni)   ? Surface gravity error on Ni
 271-274  F4.2  [-]  e_([Fe/H])(Ni) ? Metallicity error on Ni
 276-279  F4.2  [-]  e_(v)(Ni)      ? Microturbulence error on Ni
 281-284  F4.2  [-]  e_(stat)(Ni)   ? Statistical error on Ni
 286-289  F4.2  [-]  e_(noise)(Ni)  ? Error inferred through noise on Ni
 291-294  F4.2  [-]    logeps(Zn)   ? Absolute Zn abundance
 296-299  F4.2  [-]  e_(tot)(Zn)    ? Total error on Zn
 301-304  F4.2  [-]  e_(temp)(Zn)   ? Temperature error on Zn
 306-309  F4.2  [-]  e_(logg)(Zn)   ? Surface gravity error on Zn
 311-314  F4.2  [-]  e_([Fe/H])(Zn) ? Metallicity error on Zn
 316-319  F4.2  [-]  e_(v)(Zn)      ? Microturbulence error on Zn
 321-324  F4.2  [-]  e_(stat)(Zn)   ? Statistical error on Zn
 326-329  F4.2  [-]  e_(noise)(Zn)  ? Error inferred through noise on Zn
 331-335  F5.2  [-]    logeps(Sr)   ? Absolute Sr abundance
 337-340  F4.2  [-]  e_(tot)(Sr)    ? Total error on Sr
 342-345  F4.2  [-]  e_(temp)(Sr)   ? Temperature error on Sr
 347-350  F4.2  [-]  e_(logg)(Sr)   ? Surface gravity error on Sr
 352-355  F4.2  [-]  e_([Fe/H])(Sr) ? Metallicity error on Sr
 357-360  F4.2  [-]  e_(v)(Sr)      ? Microturbulence error on Sr
 362-365  F4.2  [-]  e_(stat)(Sr)   ? Statistical error on Sr
 367-370  F4.2  [-]  e_(noise)(Sr)  ? Error inferred through noise on Sr
 372-376  F5.2  [-]    logeps(Y)    ? Absolute Y abundance
 378-381  F4.2  [-]  e_(tot)(Y)     ? Total error on Y
 383-386  F4.2  [-]  e_(temp)(Y)    ? Temperature error on Y
 388-391  F4.2  [-]  e_(logg)(Y)    ? Surface gravity error on Y
 393-396  F4.2  [-]  e_([Fe/H])(Y)  ? Metallicity error on Y
 398-401  F4.2  [-]  e_(v)(Y)       ? Microturbulence error on Y
 403-406  F4.2  [-]  e_(stat)(Y)    ? Statistical error on Y
 408-411  F4.2  [-]  e_(noise)(Y)   ? Error inferred through noise on Y
 413-417  F5.2  [-]    logeps(Ba)   ? Absolute Ba abundance
 419-422  F4.2  [-]  e_(tot)(Ba)    ? Total error on Ba
 424-427  F4.2  [-]  e_(temp)(Ba)   ? Temperature error on Ba
 429-432  F4.2  [-]  e_(logg)(Ba)   ? Surface gravity error on Ba
 434-437  F4.2  [-]  e_([Fe/H])(Ba) ? Metallicity error on Ba
 439-442  F4.2  [-]  e_(v)(Ba)      ? Microturbulence error on Ba
 444-447  F4.2  [-]  e_(stat)(Ba)   ? Statistical error on Ba
 449-452  F4.2  [-]  e_(noise)(Ba)  ? Error inferred through noise on Ba
 454-458  F5.2  [-]    logeps(Eu)   ? Absolute Eu abundance
 460-463  F4.2  [-]  e_(tot)(Eu)    ? Total error on Eu
 465-468  F4.2  [-]  e_(temp)(Eu)   ? Temperature error on Eu
 470-473  F4.2  [-]  e_(logg)(Eu)   ? Surface gravity error on Eu
 475-478  F4.2  [-]  e_([Fe/H])(Eu) ? Metallicity error on Eu
 480-483  F4.2  [-]  e_(v)(Eu)      ? Microturbulence error on Eu
 485-488  F4.2  [-]  e_(stat)(Eu)   ? Statistical error on Eu
 490-493  F4.2  [-]  e_(noise)(Eu)  ? Error inferred through noise on Eu
--------------------------------------------------------------------------------

Acknowledgements:
    Moritz Reichert, mreichert(at)theorie.ikp.physik.tu-darmstadt.de

References:
    Skuladottir et al.,  Paper I   2019A&A...631A.171S, Cat. J/A+A/631/A171
    Skuladottir et al.,  Paper II  2020A&A...634A..84S

================================================================================
(End)  Moritz Reichert [Darmstadt], Patricia Vannier [CDS]  13-Aug-2020
