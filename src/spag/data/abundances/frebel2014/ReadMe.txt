J/ApJ/786/74       EW measurements of 6 Segue 1 red giants       (Frebel+, 2014)
================================================================================
Segue 1: an unevolved fossil galaxy from the early universe.
    Frebel A., Simon J.D., Kirby E.N.
   <Astrophys. J., 786, 74 (2014)>
   =2014ApJ...786...74F    (SIMBAD/NED BibCode)
================================================================================
ADC_Keywords: Galaxies, nearby ; Equivalent widths ; Stars, population II ;
              Stars, giant ; Photometry, SDSS ; Abundances
Keywords: early universe - galaxies: dwarf - Galaxy: halo - Local Group -
          stars: abundances - stars: Population II

Abstract:
    We present Magellan/MIKE and Keck/HIRES high-resolution spectra of six
    red giant stars in the dwarf galaxy Segue 1. Including one additional
    Segue 1 star observed by Norris et al. (2010ApJ...722L.104N),
    high-resolution spectra have now been obtained for every red giant in
    Segue 1. Remarkably, three of these seven stars have metallicities
    below [Fe/H]=-3.5, suggesting that Segue 1 is the least chemically
    evolved galaxy known. We confirm previous medium-resolution analyses
    demonstrating that Segue 1 stars span a metallicity range of more than
    2 dex, from [Fe/H]=-1.4 to [Fe/H]=-3.8. All of the Segue 1 stars
    are {alpha}-enhanced, with [{alpha}/Fe]~0.5. High {alpha}-element
    abundances are typical for metal-poor stars, but in every previously
    studied galaxy [{alpha}/Fe] declines for more metal-rich stars, which
    is typically interpreted as iron enrichment from supernova Ia. The
    absence of this signature in Segue 1 indicates that it was enriched
    exclusively by massive stars. Other light element abundance ratios in
    Segue 1, including carbon enhancement in the three most metal-poor
    stars, closely resemble those of metal-poor halo stars. Finally, we
    classify the most metal-rich star as a CH star given its large
    overabundances of carbon and s-process elements. The other six stars
    show remarkably low neutron-capture element abundances of [Sr/H]<-4.9
    and [Ba/H]<-4.2, which are comparable to the lowest levels ever
    detected in halo stars. This suggests minimal neutron-capture
    enrichment, perhaps limited to a single r-process or weak s-process
    synthesizing event. Altogether, the chemical abundances of Segue 1
    indicate no substantial chemical evolution, supporting the idea that
    it may be a surviving first galaxy that experienced only one burst of
    star formation.

Description:
    We observed five of our six target stars with the MIKE spectrograph
    (Bernstein et al. 2003SPIE.4841.1694B) on the Magellan-Clay telescope
    in 2010 March and May, and 2011 March. Observing conditions during
    these runs were mostly clear, with an average seeing of 0.8" to 1.0".
    MIKE spectra have nearly full optical wavelength coverage from
    ~3500-9000 {AA}. A 1.0"x5" slit yields a spectral resolution of ~22000
    in the red and ~28000 in the blue wavelength regime. We observed the
    final star in the sample, SDSS J100742+160106, with the HIRES
    spectrograph (Vogt et al. 1994SPIE.2198..362V) on the Keck I telescope
    on 2010 April 1. The observations were obtained with a 1.15"x7" slit
    (providing a spectral resolution of 37500), the kv389 blocking filter,
    and a total integration time of 3.6 h.

Objects:
 -----------------------------------------------------
     RA   (ICRS)    DE        Designation(s)
 -----------------------------------------------------
  10 07 03.2    +16 04 25     Segue 1 = NAME Segue I
 -----------------------------------------------------

File Summary:
--------------------------------------------------------------------------------
 FileName      Lrecl  Records   Explanations
--------------------------------------------------------------------------------
ReadMe            80        .   This file
table1.dat       112        6   Observing Details
table2.dat       126      505   Equivalent width measurements of the Segue 1
                                stars
--------------------------------------------------------------------------------

See also:
 J/ApJ/692/1464 : Spectroscopy of Segue 1 (Geha+, 2009)

Byte-by-byte Description of file: table1.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label     Explanations
--------------------------------------------------------------------------------
   1- 19  A19   ---     Name      SDSS star's name (JHHMMSS+DDMMSS)
  21- 22  I2    h       RAh       Hour of Right Ascension (J2000)
  24- 25  I2    min     RAm       Minute of Right Ascension (J2000)
  27- 30  F4.1  s       RAs       Second of Right Ascension (J2000)
      32  A1    ---     DE-       Sign of the Declination (J2000)
  33- 34  I2    deg     DEd       Degree of Declination (J2000)
  36- 37  I2    arcmin  DEm       Arcminute of Declination (J2000)
  39- 42  F4.1  arcsec  DEs       Arcsecond of Declination (J2000)
  44- 76  A33   ---     Obs.date  UT Date(s) of the observation
  78- 81  F4.2  arcsec  Slit      Slit width
  83- 86  F4.1  h       Texp      Exposure time
  88- 92  F5.2  mag     gmag      SDSS g band magnitude
  94- 98  F5.3  mag     E(B-V)    Reddening
 100-101  I2    ---     S/N5300   Signal-to-noise ratio at 5300 {AA} (1)
 103-104  I2    ---     S/N6000   Signal-to-noise ratio 6000 {AA} (1)
 106-112  A7    ---     Name1     Star's name as written in table2.dat file
--------------------------------------------------------------------------------
Note (1): The S/N is measured per ~33 m{AA} pixel (MIKE spectra) and ~20 m{AA}
          pixel (HIRES spectrum).
--------------------------------------------------------------------------------

Byte-by-byte Description of file: table2.dat
--------------------------------------------------------------------------------
   Bytes Format Units    Label       Explanations
--------------------------------------------------------------------------------
   1-  4  A4    ---      ID          Element
   6- 13  F8.3  0.1nm    Wave        Wavelength; Angstroms
  15- 18  F4.2  eV       ExPot       ? Excitation potential
  20- 25  F6.3  [-]      log(gf)     ? Log oscillator strength
  27- 32  F6.2  10-13m   EW-J100714  ? Equivalent width of
                                      SDSS J100714+160154 (1)
      34  A1    ---    f_EW-J100714  [s] Flag indicating spectrum synthesis used
  36- 40  F5.2  [-]      eps-J100714 ? Log epsilon abundance of
                                      SDSS J100714+160154
  42- 47  F6.2  10-13m   EW-J100710  ? Equivalent width of
                                      SDSS J100702+155055 (1)
      49  A1    --     f_EW-J100710  [s] Flag indicating spectrum synthesis used
      51  A1    ---    l_eps-J100710 [<] Limit flag on eps-J100710
  52- 56  F5.2  [-]      eps-J100710 ? Log epsilon abundance of
                                      SDSS J100702+155055
  60- 65  F6.2  10-13m   EW-J100702  ? Equivalent width of
                                      SDSS J100702+155055 (1)
      67  A1    ---    f_EW-J100702  [s] Flag indicating spectrum synthesis used
      69  A1    ---    l_eps-J100702 [<] Limit flag on eps-J100702
  70- 75  F6.2  [-]      eps-J100702 ? Log epsilon abundance of
                                      SDSS J100702+155055
  77- 82  F6.2  10-13m   EW-J100742  ? Equivalent width of
                                      SDSS J100742+160106 (1)
      84  A1    ---    f_EW-J100742  [s] Flag indicating spectrum synthesis used
      86  A1    ---    l_eps-J100742 [<] Limit flag on eps-J100742
  87- 92  F6.2  [-]      eps-J100742 ? Log epsilon abundance of
                                      SDSS J100742+160106
  94- 99  F6.2  10-13m   EW-J100652  ? Equivalent width of
                                      SDSS J100652+160235 (1)
     101  A1    ---    f_EW-J100652  [s] Flag indicating spectrum synthesis used
     103  A1    ---    l_eps-J100652 [<] Limit flag on eps-J100652
 104-109  F6.2  [-]      eps-J100652 ? Log epsilon abundance of
                                      SDSS J100652+160235
 111-116  F6.2  10-13m   EW-J100639  ? Equivalent width of
                                      SDSS J100639+160008 (1)
     118  A1    ---    f_EW-J100639  [s] Flag indicating spectrum synthesis used
     120  A1    ---    l_eps-J100639 [<] Limit flag on eps-J100639
 121-126  F6.2  [-]      eps-J100639 ? Log epsilon abundance of
                                      SDSS J100639+160008
--------------------------------------------------------------------------------
Note (1): In units of milli-Angstroms.
--------------------------------------------------------------------------------

History:
    From electronic version of the journal

================================================================================
(End)             Prepared by [AAS], Tiphaine Pouvreau [CDS]         21-Jul-2017
