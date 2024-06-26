# Forecasting Rubin Binary Supermassive Black Hole Detection Rates via Machine Learning

This project is a part of the [Time-Domain Needles in Rubin's Haystacks](https://rubin-anomalies-workshop.github.io/pages/about.html) Workshop held at the Harvard Center for Astrophysics from April 17-19th, 2024.

#### Participants: Meg Davis (UConn, megan.c.davis@uconn.edu), Shar Daniels, Szymon Nakoneczny

## Summary & Goals:
Periodic signatures in time-domain observations of quasars have been used to search for binary supermassive black holes (SMBHs). These searches, across existing time-domain surveys, have produced several hundred candidates. The general stochastic variability of quasars, however, can masquerade as a false-positive periodic signal, especially when monitoring cadence and duration are limited. As Rubin will observe millions of quasars, it will also open a new frontier for electromagnetic detection of binary SMBHs. In this Hack, we will explore the application of basic machine learning techniques to the binary detection problem by using thousands of synthetic Rubin observations of both binary and single quasars. The goals are as follows:
1. Identify pre-processing needs for light curve data.
2. Apply out-of-the-box, open-source ML classification algorithms from popular Python packages, such as [sci-kit learn](https://scikit-learn.org/stable/) and [keras](https://keras.io/examples/timeseries/timeseries_classification_from_scratch/).
3. Identify common pitfalls, analysis missteps, and general recommendations for applying this detection method to this type of data.
4. Stretch goal: Explore need for and generate more complex, homebrewed algorithms.
   
Keywords and relevant subfields: quasars, supermassive black holes, accretion physics, graviational wave sources, hierarchical structure formation

This work is expected to become a journal article led by M. Davis as a follow-up to [Davis et al. 2024](https://iopscience.iop.org/article/10.3847/1538-4357/ad276e).

## Data and data-related resources:
Relevant data paper: [Davis et al. 2024 ApJ 965 34](https://iopscience.iop.org/article/10.3847/1538-4357/ad276e)

[Sample i-band Rubin/LSST Single and Binary Quasar Light Curves](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/G0IOPJ)
- Four HDF5 files: Hackathon_2024_lcs_*.HDF5
   - small (full light curves, 10 datasets)
   - large (full light curves, 30 datasets)
   - WFD3 (3-day cadence and 5 month seasonal duration applied)
   - DDF (1-day cadence and 5 month seasonal duration applied)
 
- Each dataset within its corresponding HDF5 file has a 'time' column and 1001 light curve columns ['0','1','2','3'...]. 
- Examples on how the datasets are structured are in the Tutorials folder.
- The WFD3 and DDF data sets contain dummy values/flags to hold the place of a dropped observations (due to cadence, duration, seasonal gaps, weather). Please filter out the -999.0 floats. Insider info: they were all created with the same random seed, so they will have the same "bad weather" days.

## Tutorials
Please see the [Tutorials](./Tutorials/) folder within this repository. The first tutorial, "Interacting_with_the_data.ipynb', is intended to get you introduced to and started with the data.
