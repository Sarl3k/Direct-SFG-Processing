import re
import os
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.signal import find_peaks, peak_widths


class DataNotFound(Exception):
    """Custom exception raised when data is not found.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class File:
    def __init__(self, filename: str, directory: str):
        self.filename = filename
        self.directory = directory

        self.type: str | None = None

        self.sample: str | None = None
        self.acq_time: str | None = None
        self.polarization: str | None = None
        self.index: str | None = None

        self.raw_data: pd.DataFrame | None = None
        self.processed_data: pd.DataFrame | None = None
        self.pts_cleaned: np.ndarray = np.empty((0, 2))

    def extract_info(self, naming_pattern: str, calibration: str, ref: str):
        if calibration in self.filename:
            self.type = 'calibration'
        else:
            naming_match = re.match(naming_pattern, self.filename)
            if naming_match:
                self.sample, self.polarization, self.acq_time, self.index = naming_match.groups()
            else:
                raise ValueError(f'The name of {self.filename} does not match the naming pattern indicated')

            if 'bg' in self.filename or 'bkg' in self.filename:
                if ref in self.sample:
                    self.type = 'ref bg'
                else:
                    self.type = 'sample bg'
            elif ref in self.filename:
                self.type = 'ref sig'
            else:
                self.type = 'sample sig'

    def extract_data(self, delimiter: str):
        self.raw_data = pd.read_csv(self.directory + self.filename, delimiter=delimiter, index_col=False)

    def remove_cosmic_rays(self,
                           min_prominence: float = 1000,
                           rel_height: float = 0.5,
                           max_width: float = 3,
                           moving_average_window: int = 20,
                           interp_type: str = 'linear'
                           ) -> None:
        if self.processed_data is None:
            if self.raw_data is None:
                raise DataNotFound(f'Data to automatically clean not found for {self.filename}.')
            else:
                self.processed_data = self.raw_data.copy()
        df = self.processed_data
        x = np.array(df['Wavelength'])
        y = np.array(df['Intensity'])
        y_out = y.copy()

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
                self.pts_cleaned = np.vstack((self.pts_cleaned, [x[i], y[i]]))
                window = np.arange(max(0, i - moving_average_window),
                                   min(len(y), i + moving_average_window + 1))
                window_exclude_spikes = window[spikes[window] == 0]
                if len(window_exclude_spikes) > 1:
                    interpolator = interpolate.interp1d(
                        x[window_exclude_spikes], y[window_exclude_spikes], kind=interp_type,
                        fill_value="extrapolate"
                    )
                    y_out[i] = interpolator(x[i])
        df['Intensity'] = y_out

    def remove_pts(self, filename: str, frame: int, wl_range: tuple[float, float], interp_type: str = 'linear') -> None:
        if filename != self.filename:
            return

        if self.processed_data is None:
            if self.raw_data is None:
                raise DataNotFound(f'Data to manually clean not found for {self.filename}.')
            else:
                self.processed_data = self.raw_data.copy()
        df = self.processed_data
        x = np.array(df['Wavelength'])
        y = np.array(df['Intensity'])
        y_out = y.copy()

        start_diff = np.abs(x - wl_range[0])
        start_idx = np.where(start_diff == np.min(start_diff))[0][int(frame - 1)]
        end_diff = np.abs(x - wl_range[1])
        end_idx = np.where(end_diff == np.min(end_diff))[0][int(frame - 1)]
        for i in range(start_idx, end_idx + 1):
            pts_selected = np.vstack((pts_selected, [x[i], y[i]]))
            window = np.concatenate((np.arange(0, start_idx), np.arange(end_idx + 1, len(x))))
            if len(window) > 1:
                interpolator = interpolate.interp1d(
                    x[window], y[window], kind=interp_type,
                    fill_value="extrapolate"
                )
                y_out[i] = interpolator(x[i])

        df['Intensity'] = y_out

    def avg_frames(self):
        if self.processed_data is None:
            if self.raw_data is None:
                raise DataNotFound(f'Data not found for {self.filename} when averaging frames.')
            else:
                self.processed_data = self.raw_data.copy()
        df = self.processed_data
        intensity = np.array(df['Intensity'])
        intensity = np.reshape(intensity, (int(df['Frame'].max()), -1))
        mean_intensity = np.mean(intensity, axis=0)
        wavelength = np.array(df['Wavelength'])[:len(mean_intensity)]
        new_df = pd.DataFrame({'Wavelength': wavelength,
                               'Intensity': mean_intensity})
        self.processed_data = new_df

    def convert_to_wn(self, w1: float) -> None:
        """
        if also calling avg_frames(), make sure that convert_to_nm() is called after it!
        :param w1:
        """
        if self.processed_data is None:
            if self.raw_data is None:
                raise DataNotFound(f'Data not found for {self.filename} when converting to wn.')
            else:
                self.processed_data = self.raw_data.copy()
        df = self.processed_data
        df['Wavenumber'] = 1e7 / np.array(df['Wavelength']) - 1e7 / w1

    def save_data(self) -> None:
        raise ValueError('Function not implemented.')


def sort_files(files: list[File]) -> tuple[list[File], list[File], list[File], list[File], list[File]]:
    sample_sig = [f for f in files if f.type == 'sample sig']
    sample_bg = [f for f in files if f.type == 'sample bg']
    ref_sig = [f for f in files if f.type == 'ref sig']
    ref_bg = [f for f in files if f.type == 'ref bg']
    calibration = [f for f in files if f.type == 'calibration']
    return sample_sig, sample_bg, ref_sig, ref_bg, calibration


def subtract_bg(signals: list[File], bgs: list[File]) -> list[File]:
    bg_subtracted: list[File] = []
    for sig in signals:
        # find matching bg
        matching_bg = [bg for bg in bgs if (
                bg.sample == sig.sample and
                bg.acq_time == sig.acq_time and
                bg.polarization == sig.polarization
        )]
        if len(matching_bg) > 1:
            matching_bg = [bg for bg in matching_bg if bg.index == sig.index]
            print(f'More than one possible background detected for {sig.filename},'
                  f' {matching_bg[0].filename} is used')
        if len(matching_bg) != 1:
            raise Exception(
                f'Could not select one and only one background match for {sig.filename},'
                f' {len(matching_bg)} match found.')
        # subtract bg from signal
        for file in [sig, matching_bg[0]]:
            if file.processed_data is None:
                if file.raw_data is None:
                    raise DataNotFound(f'Data to manually clean not found for {file.filename}.')
                else:
                    file.processed_data = file.raw_data.copy()
        sig_int = np.array(sig.processed_data['Intensity'])
        bg_int = np.array(matching_bg[0].processed_data['Intensity'])
        sig.processed_data['Intensity'] = sig_int - bg_int
        bg_subtracted.append(sig)
    return bg_subtracted


def normalize(samples: list[File], ref: File) -> None:
    for file in samples + [ref]:
        if file.processed_data is None:
            if file.raw_data is None:
                raise DataNotFound(f'Data to manually clean not found for {file.filename}.')
            else:
                file.processed_data = file.raw_data.copy()
    ref_int = np.array(ref.processed_data['Intensity'])
    for spl in samples:
        spl_int = np.array(spl.processed_data['Intensity'])
        spl.processed_data['Intensity'] = spl_int / ref_int
