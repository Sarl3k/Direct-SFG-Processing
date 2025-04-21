import re
import os
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.signal import find_peaks, peak_widths


class Datafiles:
    def __init__(
            self,
            directory: str,
            list_datafiles: list[str],
            ref: str,
            calibration_filename: str = 'polystyrene',
            naming_pattern: str = r"^([^_]+)(?:_\w+)*_([sp]{3})(?:_\w+)*_(\d+)s(?:_\w+)*_([0-9]{2}|[A-Za-z]{1})(?:_\w+)*\.csv$"
    ):
        """
        Creation of the DataFiles object, used for sorting and extracting information from indicated files.
        :param directory: directory for the datafiles
        :param list_datafiles: list with the filename of the datafiles
        :param ref: name of your reference as spelled in your filenames.
        :param calibration_filename: name of your calibration as spelled in your filenames.
        :param naming_pattern: pattern in which your files are named.
        Default naming convention is 'sample_pol_#s_##.csv'. If separated by underscores, additional words can
        be included anywhere EXCEPT before the sample (so not at the very beginning of the filename). Acquisition time
        (the _#s_) should be in seconds. The indexing of your filenames (the ##) must be two digits (e.g. 01, 02, ...) or a single letter.
        """
        self.ref: str = ref
        self.directory: str = directory
        self.list_datafiles_dict: list[dict] = []
        self.dict_datafiles = {'sample': [], 'ref': [], 'bg': [], 'calibration': []}
        self.extract_datafile_info(list_datafiles, naming_pattern, calibration_filename)
        self.sort_datafiles()
        for key, value in self.dict_datafiles.items():
            print(key, len(value))

    def extract_datafile_info(self, datafiles: list[str], naming_pattern: str, calibration: str) -> None:
        """
        From the name of a datafile, this method extract the data and information corresponding to the file.
        :param datafiles: list containing the name of all the files you want to process
        :param naming_pattern: pattern in which your files need to be named. see module re, and re.match() for more information
        :param calibration: name of the calibration you are using
        """
        for datafile in datafiles:
            naming_match = re.match(naming_pattern, datafile)
            try:
                if naming_match:
                    sample, polarization, acq_time, index = naming_match.groups()
                else:
                    raise ValueError(f'File name for {datafile} does not match the naming pattern')
                if 'bg' in datafile or 'bkg' in datafile:
                    data_type = 'bg'
                elif self.ref in datafile:
                    data_type = 'ref'
                else:
                    data_type = 'sample'
                datafile_info = {
                    'filename': datafile,
                    'sample': sample,
                    'polarization': polarization,
                    'acq_time': acq_time,
                    'index': index,
                    'data type': data_type
                }
            except ValueError:
                if calibration in datafile:
                    datafile_info: dict = {'filename': datafile, 'data type': 'calibration'}
                else:
                    raise ValueError(f'File name for {datafile} does not match the naming pattern')
            print(datafile_info)
            datafile_info['data'] = self.extract_data(datafile_info.get('filename'))
            self.list_datafiles_dict.append(datafile_info)

    def sort_datafiles(self) -> None:
        """
        Sort the datafiles into sample, reference, and background data.
        """
        for datafile in self.list_datafiles_dict:
            data_type: str = datafile.get('data type')
            self.dict_datafiles.get(data_type).append(datafile)

    def match_signal_to_bg(self) -> None:
        """
        For each sample and reference file, this method matches the corresponding background file.
        """
        for list_files in [self.dict_datafiles.get('sample'), self.dict_datafiles.get('ref')]:
            for datafile in list_files:
                keys = 'sample', 'polarization', 'acq_time'
                matching_bg = [i for i in self.dict_datafiles.get('bg')
                               if all(i.get(key) == datafile.get(key) for key in keys)]
                if len(matching_bg) > 1:
                    matching_bg = [i for i in matching_bg if i.get('index') == datafile.get('index')]
                    print(f'More than one possible background detected for {datafile.get('filename')},'
                          f' {matching_bg[0].get('filename')} is used')
                if len(matching_bg) == 1:
                    datafile['bg match'] = matching_bg[0].get('filename')
                else:
                    raise Exception(f'Could not select one and only one background match for {datafile.get('filename')}.'
                                     f' {len(matching_bg)} match found.')

    def match_sample_to_ref(self, ref_used: int = 1) -> None:
        """
        For each sample file, this method matches the corresponding reference file.
        :param ref_used: name of your reference as spelled in your filenames.
        """
        for datafile in self.dict_datafiles.get('sample'):
            try:
                datafile['ref'] = self.dict_datafiles.get('ref')[ref_used].get('filename')
            except IndexError:
                datafile['ref'] = self.dict_datafiles.get('ref')[0].get('filename')

    def extract_data(self, filename: str, delimiter: str = ',') -> pd.DataFrame:
        """
        Reads a datafile and stores the corresponding data in a pandas dataframe.
        :param filename: name of the file
        :param delimiter: delimiter
        :return: pandas dataframe containing the data from the datafile
        """
        return pd.read_csv(self.directory + filename, delimiter=delimiter, index_col=False)







class ProcessData:
    def __init__(self, directory: str, datafiles: dict[str|list[dict]], correction_wl: float):
        """
        Create the ProcessData object, used to process direct vSFG data files.
        :param directory:
        :param datafiles:
        :param correction_wl:
        """
        self.directory = directory
        self.datafiles = datafiles
        self.w1 = correction_wl

    def average_frames(self) -> None:
        """
        Average the multiple frames of a datafile.
        """
        for files_list in self.datafiles.values():
            for file in files_list:
                try:
                    df = file.get('data processed')
                    intensity = np.array(df['Intensity'])
                    intensity = np.reshape(intensity, (int(df['Frame'].max()), -1))
                    mean_intensity = np.mean(intensity, axis=0)
                    wavelength = np.array(df['Wavelength'])[:len(mean_intensity)]
                except TypeError:
                    df = file.get('data')
                    intensity = np.array(df['Intensity'])
                    intensity = np.reshape(intensity, (int(df['Frame'].max()), -1))
                    mean_intensity = np.mean(intensity, axis=0)
                    wavelength = np.array(df['Wavelength'])[:len(mean_intensity)]
                new_df = pd.DataFrame({'Wavelength': wavelength,
                                       'Intensity': mean_intensity})
                file['data processed'] = new_df


    def subtract_bg(self) -> None:
        """
        Subtract the background from signals
        """
        for files_list in [self.datafiles.get('sample'), self.datafiles.get('ref')]:
            for file in files_list:
                try:
                    signal = np.array(file.get('data processed')['Intensity'])
                    bg_file = [i for i in self.datafiles.get('bg') if i.get('filename') == file.get('bg match')][0]
                    bg = np.array(bg_file.get('data processed')['Intensity'])
                    wl = file.get('data processed')['Wavelength']
                except TypeError:
                    print(f"Warning: '{file.get('filename')}' does not seem to have been averaged")
                    signal = np.array(file.get('data')['Intensity'])
                    bg_file = [i for i in self.datafiles.get('bg') if i.get('filename') == file.get('bg match')][0]
                    bg = np.array(bg_file.get('data')['Intensity'])
                    wl = file.get('data')['Wavelength']
                new_df = pd.DataFrame({'Wavelength': wl, 'Intensity': signal - bg})
                file['data processed'] = new_df


    def normalize(self) -> None:
        """
        Normalize the data by dividing the sample signal by the reference signal.
        """
        def convert_to_wavenumber(wavelength) -> np.ndarray:
            """

            :param wavelength:
            :return:
            """
            return 1e7 / np.array(wavelength) - 1e7 / self.w1

        for file in self.datafiles.get('sample'):
            try:
                signal = np.array(file.get('data processed')['Intensity'])
                ref_file = [i for i in self.datafiles.get('ref') if i.get('filename') == file.get('ref')][0]
                ref = np.array(ref_file.get('data processed')['Intensity'])
                wn = convert_to_wavenumber(file.get('data processed')['Wavelength'])
            except TypeError:
                print(f"Warning: '{file.get('filename')}' does not seem to have been averaged or background subtracted")
                signal = np.array(file.get('data')['Intensity'])
                ref_file = [i for i in self.datafiles.get('ref') if i.get('filename') == file.get('ref')][0]
                ref = np.array(ref_file.get('data')['Intensity'])
                wn = convert_to_wavenumber(file.get('data')['Wavelength'])
            with np.errstate(divide='ignore'):
                new_df = pd.DataFrame({'Wavenumber': wn, 'Intensity': signal / ref})
            file['data processed'] = new_df

        for file in self.datafiles.get('calibration'):
            try:
                wn = convert_to_wavenumber(file.get('data processed')['Wavelength'])
                intensity = file.get('data processed')['Intensity']
            except IndexError:
                print(f"Warning: '{file.get('filename')}' does not seem to have been averaged or background subtracted")
                wn = convert_to_wavenumber(file.get('data')['Wavelength'])
                intensity = file.get('data')['Intensity']
            new_df = pd.DataFrame({'Wavenumber': wn, 'Intensity': intensity})
            file['data processed'] = new_df


    def save_processed_data(self) -> None:
        """
        Save the processed data as csv files and include informative text files.
        """
        print("Exporting processed data")
        for sample in self.datafiles.get('sample'):
            os.makedirs("../ProcessingOutput/ProcessedData/", exist_ok=True)
            os.makedirs("../ProcessingOutput/TextFiles/", exist_ok=True)
            df = sample.get('data')
            df.to_csv(f'../ProcessingOutput/ProcessedData/processed_{sample.get('filename')}', index=False)
            with open(f"../ProcessingOutput/TextFiles/INFO-processed_{sample.get('filename').replace('.csv', '.txt')}", "w") as txt:
                txt.write(
                    f"""
This text file contains the information used to obtain 'processed_{sample.get('filename')}' from its corresponding raw data file.
----------------------------------------------------------------
File processed: {sample.get('filename')}
Background file used: {sample.get('bg match')}
Normalized using: {sample.get('ref')}
-----
Calibration file used: 
calibration set to {self.w1} nm
-----
The data was processed using '{os.path.basename(__file__)}'
            """
                )
        print("Exportation (probably) successful ;)")


    def remove_cosmic_rays(self,
                           automatic: bool = True,
                           manual: list[dict] = None,
                           min_prominence: float = 1000,
                           rel_height: float = 0.5,
                           max_width: float = 3,
                           moving_average_window: int = 20,
                           interp_type: str ='linear'
                           ) -> None:
        """
        Remove peaks corresponding to cosmic (gamma) rays from the signal. Both automatic and manual options are possible.
        After the data points corresponding to cosmic rays are removed, new points are created by linear interpolation.
        :param automatic: Set to 'True' if you want rays to be flagged automatically.
        :param manual: Enter each peaks you would like to clean manually as a separate dictionary with the filename, frame, and wavelength range at which the peak occurs.
        :param min_prominence: For automatic detection only. Minimum prominence for a peak to be detected.
        :param rel_height: For automatic detection only. Relative height at which the peak width is measured.
        :param max_width: For automatic detection only. Maximum width (in terms of data points) a peak can have and be flagged as a cosmic ray.
        :param moving_average_window: For automatic detection only.
        :param interp_type: Type of interpolation performed, see scipy module for more information.
        """
        for files_list in self.datafiles.values():
            for file in files_list:
                if manual is not None or automatic:
                    x = np.array(file.get('data')['Wavelength'])
                    y = np.array(file.get('data')['Intensity'])
                    pts_selected = np.empty((0, 2))
                    y_out = y.copy()

                    if automatic:
                        print(f'cleaning {file.get('filename')}')
                        # Detect peaks based on prominence
                        peaks, _ = find_peaks(y, prominence=min_prominence)
                        # Create an array to flag spikes (1 = spike, 0 = normal)
                        spikes = np.zeros(len(y), dtype=int)
                        # Calculate peak widths
                        widths = peak_widths(y, peaks)[0]
                        # Extended width calculations
                        widths_ext_a = peak_widths(y, peaks, rel_height=rel_height)[2]
                        widths_ext_b = peak_widths(y, peaks, rel_height=rel_height)[3]
                        # Flag spikes based on width threshold
                        for width, ext_a, ext_b in zip(widths, widths_ext_a, widths_ext_b):
                            if width < max_width:
                                spikes[int(ext_a) - 1: int(ext_b) + 2] = 1
                        # Interpolation of corrupted points
                        for i, spike in enumerate(spikes):
                            if spike != 0:
                                pts_selected = np.vstack((pts_selected, [x[i], y[i]]))
                                window = np.arange(max(0, i - moving_average_window),
                                                   min(len(y), i + moving_average_window + 1))
                                window_exclude_spikes = window[spikes[window] == 0]
                                if len(window_exclude_spikes) > 1:
                                    interpolator = interpolate.interp1d(
                                        x[window_exclude_spikes], y[window_exclude_spikes], kind=interp_type,
                                        fill_value="extrapolate"
                                    )
                                    y_out[i] = interpolator(x[i])

                    if manual is not None:
                        for peak in manual:
                            if peak.get('filename') == file.get('filename'):
                                start_diff = np.abs(x - peak.get('range')[0])
                                start_idx = np.where(start_diff == np.min(start_diff))[0][int(peak.get('frame')-1)]
                                end_diff = np.abs(x - peak.get('range')[1])
                                end_idx = np.where(end_diff == np.min(end_diff))[0][int(peak.get('frame')-1)]
                                for i in range(start_idx, end_idx + 1):
                                    pts_selected = np.vstack((pts_selected, [x[i], y[i]]))
                                    window = np.concatenate((np.arange(0, start_idx), np.arange(end_idx + 1, len(x))))
                                    if len(window) > 1:
                                        interpolator = interpolate.interp1d(
                                            x[window], y[window], kind=interp_type,
                                            fill_value="extrapolate"
                                        )
                                        y_out[i] = interpolator(x[i])


                    print(f'{len(pts_selected)} ray points cleaned')
                    new_df = pd.DataFrame({'Frame': file.get('data')['Frame'], 'Wavelength': x, 'Intensity': y_out})
                    file['data processed'] = new_df
                    if len(pts_selected) > 0:
                        file['cleaned points'] = pts_selected
                    # print(file)


if __name__ == '__main__':
    print()