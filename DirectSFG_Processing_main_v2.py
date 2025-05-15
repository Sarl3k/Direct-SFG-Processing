import glob
import os
import matplotlib.pyplot as plt
from DirectSFG_Processing_methods_v2 import *


def main(
        w1_wavelength: float = 793.27,
        export: bool = False,
        directory: str = "example data/",
        x_lim: tuple[float, float] = (2800, 3800),
        y_lim: tuple[float, float] = (-0.02, 0.1),
) -> None:
    print("Let's start processing our SFG data ε> \n")

    # Grabbing the data files
    glob_list = glob.glob(os.path.join(directory, "*.csv"))
    filenames: list = [os.path.basename(file) for file in glob_list]
    print(filenames)
    print() # empty to skip a line

    files: list = []
    for filename in filenames:
        print(f'Processing {filename}')
        file = File(filename=filename, directory=directory)
        file.extract_info(
            naming_pattern=r"^([^_]+)(?:_\w+)*_([sp]{3})(?:_\w+)*_(\d+)s(?:_\w+)*_([0-9]{2}|[A-Za-z]{1})(?:_\w+)*\.csv$",
            calibration='polystyrene',
            ref='zqz')
        file.extract_data(delimiter=',')
        file.remove_cosmic_rays()
        # if file.filename == 'name':
        #     file.remove_pts()
        file.avg_frames()
        file.convert_to_wn(w1=w1_wavelength)
        files.append(file)

    print('\nSorting data\n')
    sample_sigs, sample_bgs, ref_sigs, ref_bgs, calibrations = sort_files(files)

    samples = subtract_bg(sample_sigs, sample_bgs)
    refs = subtract_bg(ref_sigs, ref_bgs)
    print('Subtracted background from signal')

    try:
        normalize(samples, refs[1])
        print(
            f'\nNormalized using reference: {refs[1].sample} {refs[1].polarization} {refs[1].acq_time}s {refs[1].index}')
    except IndexError:
        normalize(samples, refs[0])
        print(
            f'\nNormalized using reference: {refs[0].sample} {refs[0].polarization} {refs[0].acq_time}s {refs[0].index}')

    # if true, saves the processed data
    # if export:
    #     dat.save_processed_data(directory='processed_data', script_name=os.path.basename(__file__), w1=w1_wavelength)

    print('\n--- Plotting ---')
    # Plotting raw data
    print('Plotting raw data')
    fig1, ax1 = plt.subplots(nrows=2, ncols=3, figsize=(15, 8), dpi=100)

    for file in ref_sigs:
        ax1[0, 0].plot(file.raw_data['Wavelength'], file.raw_data['Intensity'],
                       label=file.filename, alpha=0.7, linewidth=0.75)
        ax1[0, 0].scatter(file.pts_cleaned[:, 0], file.pts_cleaned[:, 1], marker='x', color='red')
    for file in sample_sigs:
        ax1[0, 1].plot(file.raw_data['Wavelength'], file.raw_data['Intensity'],
                       label=file.filename, alpha=0.7, linewidth=0.75)
        ax1[0, 1].scatter(file.pts_cleaned[:, 0], file.pts_cleaned[:, 1], marker='x', color='red')
    for file in calibrations:
        ax1[0, 2].plot(file.raw_data['Wavelength'], file.raw_data['Intensity'],
                       label=file.filename, alpha=0.7, linewidth=0.75)
        ax1[0, 2].scatter(file.pts_cleaned[:, 0], file.pts_cleaned[:, 1], marker='x', color='red')
    for file in ref_bgs:
        ax1[1, 0].plot(file.raw_data['Wavelength'], file.raw_data['Intensity'],
                       label=file.filename, alpha=0.7, linewidth=0.75)
        ax1[1, 0].scatter(file.pts_cleaned[:, 0], file.pts_cleaned[:, 1], marker='x', color='red')
    for file in sample_bgs:
        ax1[1, 1].plot(file.raw_data['Wavelength'], file.raw_data['Intensity'],
                       label=file.filename, alpha=0.7, linewidth=0.75)
        ax1[1, 1].scatter(file.pts_cleaned[:, 0], file.pts_cleaned[:, 1], marker='x', color='red')
    for file in calibrations:
        ax1[1, 2].plot(file.processed_data['Wavelength'], file.processed_data['Intensity'],
                       label=file.filename, alpha=0.7, linewidth=0.75)

    ax1[0, 0].set_title('Ref raw data')
    ax1[0, 1].set_title('Sample raw data')
    ax1[0, 2].set_title('Calibration')
    ax1[1, 0].set_title('Background data')
    ax1[1, 1].set_title('sample bkg data')
    ax1[1, 2].set_title('Processed calibration')
    for ax in ax1.flat:
        ax.legend(fontsize=8)

    # Plotting processed data
    print('Plotting processed data')
    fig2, ax_2 = plt.subplots(figsize=(12, 8))
    ax_2.set_title('Processed samples')
    ax_2.set_prop_cycle('color', [plt.cm.plasma(i) for i in np.linspace(0, 0.8, len(samples))])
    ax_2.set_xlabel(r"Frequency (cm$^{-1}$)")
    ax_2.set_ylabel("Intensity (a.u.)")
    for file in samples:
        label = file.sample + ' ' + file.polarization + ' ' + file.index
        ax_2.plot(file.processed_data['Wavenumber'], file.processed_data['Intensity'], label=label, alpha=0.6)
        ax_2.set_xlim(x_lim)
        ax_2.set_ylim(y_lim)
        ax_2.legend(fontsize=10)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(file.processed_data['Wavenumber'], file.processed_data['Intensity'], color="0.2")
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_xlabel(r"Frequency (cm$^{-1}$)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_title(label)
        fig.tight_layout()

    print('\nProcessed finished! xoxo')

    fig1.tight_layout()
    fig2.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
