# Stellar Patterns and Abundances in Galaxies (SPAG)
SPAG is a collection of helper functions and datasets that can be useful for research in the field of Stellar Archaeology. This collection of data files, references, and code has been gathered and developed throughout the duration of my PhD. Much of the code has been inspired or modified from Dr. Alex Ji's similar software repository [`alexmods`](https://github.com/alexji/alexmods).

SPAG has been structured in a way such that it can be installed and imported on your local machine as python package. To install...

```zsh
cd /path/to/install/where/you/want/to/install/spag
https://github.com/AYelland/spag.git
cd spag
python setup.py develop
```
(`python setup.py install` could also work if `develop` is not working for you)

SPAG does have some dependence on packages that might warrent their own installation. If you have any additional questions, please reach out to Alexander Yelland (ayelland@mit.edu).

## Citations and references

If you use any of these tools in published work (especially data tables), please get in contact with me such that the proper sources are referenced for whichever parts of this tool you used. Below, I have tried to keep track of all references used in SPAG's development, though this might be incomplete.

### Solar Data
- Asplund et al. 2009, https://ui.adsabs.harvard.edu/abs/2009ARA%26A..47..481A/abstract
- Asplund et al. 2021, https://ui.adsabs.harvard.edu/abs/2021A%26A...653A.141A/abstract

### JINAbase
- Abohalima et al. 2018, https://ui.adsabs.harvard.edu/abs/2018ApJS..238...36A/abstract (https://jinabase.pythonanywhere.com)
- Mohammad Mardini GitHub, https://github.com/Mohammad-Mardini/JINAbase-updated
