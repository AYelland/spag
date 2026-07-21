import spag.read_data as rd

reference_loaders = {
    "cayrel2004" : rd.load_cayrel2004,
    "chiti2018a" : rd.load_chiti2018a,
    "chiti2018b" : rd.load_chiti2018b,
    "chiti2023" : rd.load_chiti2023,
    "chiti2025a" : rd.load_chiti2025a,
    "cowan2002" : rd.load_cowan2002,
    "francois2007" : rd.load_francois2007,
    "francois2016" : rd.load_francois2016,
    "feltzing2009" : rd.load_feltzing2009,
    "frebel2010a" : rd.load_frebel2010a,
    "frebel2010b" : rd.load_frebel2010b,
    "frebel2013c" : rd.load_frebel2013c,
    "frebel2014" : rd.load_frebel2014,
    "frebel2016" : rd.load_frebel2016,
    "gilmore2013a" : rd.load_gilmore2013a,
    "gull2021" : rd.load_gull2021,
    "hansent2017" : rd.load_hansent2017,
    "hansent2020a" : rd.load_hansent2020a,
    "hansent2024" : rd.load_hansent2024,
    "hughes2026" : rd.load_hughes2026,
    "ishigaki2014" : rd.load_ishigaki2014,
    "ito2013" : rd.load_ito2013,
    "ji2016a" : rd.load_ji2016a,
    "ji2016b" : rd.load_ji2016b,
    "ji2018" : rd.load_ji2018,
    "ji2019a" : rd.load_ji2019a,
    "ji2020a" : rd.load_ji2020a,
    "ji2020b" : rd.load_ji2020b,
    "ji2026" : rd.load_ji2026,
    "kirby2017b" : rd.load_kirby2017b,
    "koch2008c" : rd.load_koch2008c,
    "koch2013b" : rd.load_koch2013b,
    "lai2011b" : rd.load_lai2011b,
    "lemasle2012" : rd.load_lemasle2012,
    "lemasle2014" : rd.load_lemasle2014,
    "letarte2010" : rd.load_letarte2010,
    "limberg2025a" : rd.load_limberg2025a,
    "lucchesi2024" : rd.load_lucchesi2024,
    "lucey2026b" : rd.load_lucey2026b,
    "mardini2022b" : rd.load_mardini2022b,
    "mardini2024b" : rd.load_mardini2024b,
    "martin2022a" : rd.load_martin2022a,
    "marshall2019" : rd.load_marshall2019,
    "nagasawa2018" : rd.load_nagasawa2018,
    "nordlander2019" : rd.load_nordlander2019,
    "norris2010a" : rd.load_norris2010a,
    "norris2010b" : rd.load_norris2010b,
    "norris2010c" : rd.load_norris2010c,
    "norris2017b" : rd.load_norris2017b,
    "ou2024c" : rd.load_ou2024c,
    "ou2025" : rd.load_ou2025,
    "reggiani2021" : rd.load_reggiani2021,
    "roederer2010a" : rd.load_roederer2010a,
    "roederer2012a" : rd.load_roederer2012a,
    "roederer2012b" : rd.load_roederer2012b,
    "roederer2012c" : rd.load_roederer2012c,
    "roederer2012d" : rd.load_roederer2012d,
    "roederer2014a" : rd.load_roederer2014a,
    "roederer2014b" : rd.load_roederer2014b,
    "roederer2014c" : rd.load_roederer2014c,
    "roederer2014d" : rd.load_roederer2014d,
    "roederer2014e" : rd.load_roederer2014e,
    "roederer2016b" : rd.load_roederer2016b,
    "roederer2019d" : rd.load_roederer2019d,
    "roederer2023a" : rd.load_roederer2023a,
    "ruchti2011a" : rd.load_ruchti2011a,
    "sakari2018b" : rd.load_sakari2018b,
    "sbordone2007" : rd.load_sbordone2007,
    "sestito2024b" : rd.load_sestito2024b,
    "sestito2024d" : rd.load_sestito2024d,
    "shetrone2003" : rd.load_shetrone2003,
    "simon2010" : rd.load_simon2010,
    "spite2018" : rd.load_spite2018,
    "venn2012" : rd.load_venn2012,
    "waller2023" : rd.load_waller2023,
    "webber2023" : rd.load_webber2023,
}

passed, failed = 0, 0

for reference, loader in reference_loaders.items():
    print(f"  Loading {reference}...", end=" ", flush=True)  # flush=True forces immediate output
    try:
        result = loader()  # Call the function here
        print("OK")
        passed += 1
    except Exception as e:
        print(f"FAILED — {e}")
        failed += 1

print(f"\n{passed} passed, {failed} failed.")