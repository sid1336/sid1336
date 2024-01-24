import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.optimize import curve_fit

# For importing the Excel file,
# make sure you specify the location of the file depending on where your excel is kept on your PC
excel_file_name = 'sofia.xlsx'
downloads_path = os.path.join(os.path.expanduser("~"), 'Downloads', excel_file_name)

# This part reads the Excel file into a Pandas DataFrame
# Your sheet name must match here
df = pd.read_excel(downloads_path, sheet_name='sofia2', header=None)

column1_name = 'Data points for Green Light'
# column2_name = 'Column2'
column3_name = 'Data points for Brightness'
# column4_name = 'Column4'

# Extracting data
data_column1 = df[0]
data_column2 = df[1]
data_column3 = df[2]
data_column4 = df[3]


# Defining Gaussian function
def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


# Fitting Gaussian to data with initial parameter estimates
initial_params_column1 = [max(data_column2), np.mean(data_column1), np.std(data_column1)]
initial_params_column3 = [max(data_column4), np.mean(data_column3), np.std(data_column3)]

params_column1, _ = curve_fit(gaussian, data_column1, data_column2, p0=initial_params_column1)
params_column3, _ = curve_fit(gaussian, data_column3, data_column4, p0=initial_params_column3)

# Generating x values for plotting the fitted curves
x_fit_column1 = np.linspace(min(data_column1), max(data_column1), 100)
x_fit_column3 = np.linspace(min(data_column3), max(data_column3), 100)

# Calculating y values for the fitted curves
y_fit_column1 = gaussian(x_fit_column1, *params_column1)
y_fit_column3 = gaussian(x_fit_column3, *params_column3)

# Calculation for  Full-Width at Half-Maximum (FWHM)
fwhm_column1 = 2.3548 * params_column1[2]
fwhm_column3 = 2.3548 * params_column3[2]

# Plot size
plt.figure(figsize=(17, 15))

# Scatter plots
plt.scatter(data_column1, data_column2, label=f'{column1_name}')
plt.scatter(data_column3, data_column4, label=f'{column3_name}')

# Gaussian fits
plt.plot(x_fit_column1, y_fit_column1, 'g', linewidth=2, label=f'Gaussian fit for {column1_name}')
plt.plot(x_fit_column3, y_fit_column3, 'k', linewidth=2, label=f'Gaussian fit for {column3_name}')

# Full Width at Half Maximum Fit
plt.axvline(params_column1[1] - fwhm_column1 / 2, linestyle='--', color='g',
            label=f'FWHM for {column1_name}: {fwhm_column1:.4f}')
plt.axvline(params_column1[1] + fwhm_column1 / 2, linestyle='--', color='g')

plt.axvline(params_column3[1] - fwhm_column3 / 2, linestyle='--', color='k',
            label=f'FWHM for {column3_name}: {fwhm_column3:.4f}')
plt.axvline(params_column3[1] + fwhm_column3 / 2, linestyle='--', color='k')

# Labels and legend
plt.xlabel('Sample Data Points For Each Pixel', fontsize='xx-large')
plt.ylabel('Trace value', fontsize='xx-large')
plt.title(f'Scatter Plots with Gaussian Fits and FWHM', fontsize='xx-large')
plt.legend(fontsize='x-large')

plt.show()
