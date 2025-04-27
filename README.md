# Direct-SFG-Processing
Python code used to process raw data files for vibrational sum frequency generation spectroscopy.

## Overview of the processing method
> [!NOTE]
> It is assumed that whoever is using this code is familiar with vibrational sum frequency generation spectroscopy.

`DirectSFG_Processing_main.py` is the script to run, but `DirectSFG_Processing_methods.py` is also needed since it contains the methods used to processing the data.

### 1. Collecting and sorting data files[^1]
[^1]: This step is performed via the `Datafiles` object and corresponding methods.

Data files are automatically collected. All `.csv` files in a specified directory are grabbed and must fit [a specified naming convention](#file-formatting-and-naming-convention).

From the file name, information about the measurement is extracted and stored into dictionaries. The information extracted currently include: the sample name, polarization combination, acquisition time in seconds, the file index, the type of data (i.e. reference, sample, background, calibration), and finally the data.

Finally, each signal file is matched with a background file, and each sample matched with a reference.

### 2. Processing the data

The included processing steps are:
1. Data cleaning _(optional)_\
An optional pre-processing of the data may be performed to remove data points corrupted by cosmic (gamma) rays hitting the CCD detector during acquisition.
For more details on cleaning cosmic rays, [see the dedicated section](#cosmic-ray-cleaning).

2. Averaging multiple frames\
The intensity is averaged over the multiple frames of each data file.

3. Background subtraction\
The background intensity is subtracted from the signal intensity.

4. Normalization\
The sample intensity is normalized using the reference intensity. This step also converts from the detected SFG wavelength (nm) to the corresponding IR resonant frequency (cm<sup>-1</sup>).

### 3. Plotting

After the data has been processed, it is plotted using `matplotlib.pyplot`.

First, the raw data is plotted based on its type as shown here:\
![Example Image](example%20output%201.png)\
Note the red `x`'s, they indicate data points that have been flagged as cosmic rays and will get removed.

Finally, the processed data is plotted. Samples are both plotted individually, and together.

## File formatting and naming convention

Documentation not yet implemented, see the python files.

## Cosmic ray cleaning

Documentation not yet implemented, see the python files.\
Additionally, I would like to indicate that the main cleaning algorithm originally came from [Nicolas Coca-Lopez, _Analytica Chimica Acta_ **1295**, 342312 (2024).](https://doi.org/10.1016/j.aca.2024.342312)

## Future changes
List of features to be added or modified in the current code. Feel free to make a request or implement those changes yourself.
- [ ] Option to only consider the highest acquisition time before selecting the reference file used for normalization.
- [x] Make it easier to only removed cosmic rays from data files (without having to process everything).
