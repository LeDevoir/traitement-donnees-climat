import xarray as xr
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from shapely.ops import transform
import matplotlib.pyplot as plt

# Load the GRIB file using xarray
print("Retrieving data...")
grib_file_path = '/Users/sboumedda/Downloads/FEUX/API/oct.grib'
ds = xr.open_dataset(grib_file_path, engine='cfgrib')

# # Check the current longitudes
# print(ds.longitude)

print("Converting longitudes...")
# Convert from 0-360 to -180 to 180
ds = ds.assign_coords(longitude=(ds.longitude + 180) % 360 - 180)
ds = ds.roll(longitude=(ds.dims['longitude'] // 2), roll_coords=True)
print("Longitudes converted!")

# # Check the modified longitudes
# print(ds.longitude)

# Load the GeoDataFrame from a Shapefile
print("Opening Shapefile...")
shapefile_path = '/Users/sboumedda/Downloads/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp'
gdf = gpd.read_file(shapefile_path)

# Choose the country you want to export (e.g., "Canada")
country_name = "Canada"
print("Retrieving country geometry for: " + country_name)   

# Find the geometry for the chosen country
country_geometry = gdf[gdf['SOVEREIGNT'] == country_name].geometry.iloc[0]
 
# Get the transform and shape of the dataset
transform = rasterio.transform.from_origin(ds.longitude.values.min(), ds.latitude.values.max(),
                                           ds.longitude.values[1] - ds.longitude.values[0],
                                           ds.latitude.values[0] - ds.latitude.values[1])
shape = ds['cfire'].shape[-2:]

print("Obtaining emissions data for: " + country_name)

# Create a mask based on the geometry for the chosen country
mask = features.geometry_mask([country_geometry], out_shape=shape, transform=transform,
                              invert=True)

# # Check the geography of the selected country or state
# plt.imshow(mask, cmap='viridis')  # Choose a colormap that suits your data
# plt.colorbar()
# plt.show()

# Apply the mask to the emissions data
emissions_data = ds['cfire'].where(mask)

# Calculate the area for each latitude within the country's region
earth_radius = 6371000.
constant = (earth_radius**2) * 0.1 * (np.pi / 180.)
a_m2 = constant * (np.sin((ds['latitude'] + 0.05) * (np.pi / 180.))
                  - np.sin((ds['latitude'] - 0.05) * (np.pi / 180.)))

# Iterate through latitude and multiply corresponding elements
for lat_idx in range(emissions_data.shape[1]):
    emissions_data[:, lat_idx, :] *= a_m2[lat_idx].item()  # Convert to scalar

# Calculate total emissions for each timestamp
print("Calculating total emissions...")
total_emissions = np.nansum(emissions_data, axis=(1, 2))

# Calculate the cumulative sum of total emissions
print("Calculating cumulative emissions...")
cumulative_emissions = np.cumsum(total_emissions)

# Create a DataFrame with timestamp and total emissions data
country_df = pd.DataFrame({
    'Time': ds['time'].values,
    'Total Emissions (kg/s)': total_emissions,
    'Cumulative Emissions': cumulative_emissions
})

# Define the output CSV path for the chosen country
output_csv_path = f'/Users/sboumedda/Downloads/2023_{country_name}_emissions.csv'

# Add an additional column
print("Calculating emissions in Mt...")
country_df['Total Emissions (per day, in Mt)'] = total_emissions * 0.0000864
country_df['Cumulative Emissions (per day, in Mt)'] = cumulative_emissions * 0.0000864

# Save the DataFrame to a CSV file
country_df.to_csv(output_csv_path, index=False)
print("CSV file created!")

# Close the GRIB dataset
ds.close()
