

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Cosmic Ray Cleaning""")
    return


@app.cell
def _():
    import marimo as mo
    import glob
    import os
    import numpy as np
    from matplotlib import pyplot as plt
    from DirectSFG_Processing_methods import Datafiles, ProcessData
    return Datafiles, ProcessData, glob, mo, os, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Import our data files""")
    return


@app.cell
def _(glob, os):
    directory: str = "example data/"
    glob_list = glob.glob(os.path.join(directory, "*.csv"))
    list_files = [os.path.basename(file) for file in glob_list]
    print(list_files)
    return directory, list_files


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Future title""")
    return


@app.cell
def _(Datafiles, directory, list_files):
    datafiles = Datafiles(directory, list_files, ref='zqz')
    return (datafiles,)


@app.cell
def _():
    manual_cleaning = [
        {'filename': 'water_ssp_600s_01.csv', 'range': (624.4, 625), 'frame': 2},
        {'filename': 'water_ssp_600s_01.csv', 'range': (635, 636), 'frame': 2},
    ]
    return (manual_cleaning,)


@app.cell
def _(ProcessData, datafiles, manual_cleaning):
    pdata = ProcessData(datafiles.dict_datafiles)
    pdata.remove_cosmic_rays(
        automatic=True,
        manual=manual_cleaning,
        min_prominence= 1000,
        rel_height= 0.5,
        max_width= 3,
        moving_average_window= 20,
        interp_type= 'linear'
    )
    return (pdata,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Plot the data""")
    return


@app.cell
def _(mo, pdata):
    # Get list of files
    files = pdata.datafiles.get("ref") + pdata.datafiles.get("sample") + pdata.datafiles.get("bg")
    file_options = [file.get("filename") for file in files]

    # Create dropdown
    selected_file = mo.ui.dropdown(options=file_options, label="Select a file to view")
    selected_file
    return files, selected_file


@app.cell
def _(files, mo, plt, selected_file):
    # Find the file that matches the selected filename
    file = next(f for f in files if f.get("filename") == selected_file.value)

    # Plot
    plt.figure(figsize=(11, 5))
    plt.plot(file.get('data processed')['Wavelength'], file.get('data processed')['Intensity'], color='magenta', label='clean data')
    plt.plot(file.get('data')['Wavelength'], file.get('data')['Intensity'], label='raw data', alpha=0.7)

    if 'cleaned points' in file:
        plt.scatter(file.get('cleaned points')[:, 0], file.get('cleaned points')[:, 1], marker='x', color='red', label='points removed')

    plt.title(file.get('filename'))
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity (cts.)')
    plt.legend()
    mo.mpl.interactive(plt.gcf())
    return


if __name__ == "__main__":
    app.run()
