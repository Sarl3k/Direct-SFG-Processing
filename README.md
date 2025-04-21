# Direct-SFG-Processing
Python code used to process raw data files for vibrational sum frequency generation spectroscopy.

## Overview of the processing method
> [!NOTE]
> It is assumed that whoever is using this code is familiar with vibrational sum frequency generation spectroscopy.

`DirectSFG_Processing_main.py` is the script to run, but `DirectSFG_Processing_methods.py` is also needed since it contains the methods used to processing the data.

### 1. Collecting and sorting data files[^1]
[^1]: This step is performed via the `Datafiles` object and corresponding methods.

Data files are automatically collected. All `.csv` files in a specified directory are grabbed and must fit [a specified naming convention](#File-formatting-and-naming-convention).

From the file name, information about the measurement is extracted and stored into dictionaries. The information extracted currently include: the sample name, polarization combination, acquisition time in seconds, the file index, the type of data (i.e. reference, sample, background, calibration), and finally the data.

Finally, each signal file is matched with a background file, and each sample matched with a reference.

### 2. Processing the data

The included processing steps are:
1. Data cleaning _(optional)_\
An optional pre-processing of the data may be performed to remove data points corrupted by cosmic (gamma) rays hitting the CCD detector during acquisition.
For more details on cleaning cosmic rays, [see the dedicated section](Cosmic-ray-cleaning).

3. Averaging multiple frames\
The intensity is averaged over the multiple frames of each data file.

5. Background subtraction\
The background intensity is subtracted from the signal intensity.

7. Normalization\
The sample intensity is normalized using the reference intensity. This step also converts from the detected SFG wavelength (nm) to the corresponding IR resonant frequency (cm<sup>-1</sup>).

### 3. Plotting

## File formatting and naming convention

## Cosmic ray cleaning

## Future changes
List of features to be added or modified in the current code. Feel free to make a request or implement those changes yourself.
- [ ] Option to only keep highest aquisition time for reference used for normalization.
- [ ] Make it easier to only removed cosmic rays from data files (without having to process everything).
