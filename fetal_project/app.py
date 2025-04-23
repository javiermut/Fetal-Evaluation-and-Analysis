import argparse
from dash import Dash
from layouts import create_layout
from callbacks import register_callbacks
import argparse
import os
import pandas as pd
import SimpleITK as sitk
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import cv2
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.offline import plot
from ipywidgets import widgets
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from helpers import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from ipywidgets import interact
from skimage import measure, morphology

# Argument Parser
parser = argparse.ArgumentParser(description="Process some inputs.")
parser.add_argument("-a", "--patientid", type=str, required=True, help="Patient ID")
parser.add_argument("-b", "--studyfolder", type=str, required=True, help="Name of the folder of the studies")
parser.add_argument("-c", "--segmentationfolder", type=str, required=True, help="Name of the folder of the segmentations")
parser.add_argument("-d", "--t2start", type=str, required=True, help="Name of the folder of the t2* mapping")
parser.add_argument("-e", "--brain_t2_map", type=str, required=True, help="Name of the folder of the t2* mapping brain mapping")
parser.add_argument("-f", "--placenta_t2_map_segmentation", type=str, required=True, help="Name of the folder of the t2* mapping placenta segmentation")

args = parser.parse_args()

# Extract arguments
patient_id = args.patientid
study_folder = args.studyfolder
segmentation_folder = args.segmentationfolder
t2star_path = args.t2start
brain_mask_path = args.brain_t2_map
placenta_mask_path = args.placenta_t2_map_segmentation

# Check if the folder exists, if not, create it
output_folder_path = os.path.join(os.getcwd(),'fetal_project',("project_"+args.patientid))

if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

brain = sitk.ReadImage(args.studyfolder)

mask_brain = sitk.ReadImage(args.segmentationfolder)


brain = sitk.Cast(brain, sitk.sitkFloat32)
mask_brain = sitk.Cast(mask_brain, sitk.sitkUInt8)
brain = sitk.Mask(brain, mask_brain)


t2_image = sitk.ReadImage(t2star_path)
brain_mask_image = sitk.ReadImage(brain_mask_path)
placenta_mask_image = sitk.ReadImage(placenta_mask_path)

t2_np = sitk.GetArrayFromImage(t2_image)
brain_mask_np = sitk.GetArrayFromImage(brain_mask_image)
placenta_mask_np = sitk.GetArrayFromImage(placenta_mask_image)

t2_brain = t2_np * (brain_mask_np > 0)
t2_brain = t2_brain[0,:,:,:]
t2_placenta = t2_np * (placenta_mask_np > 0)
t2_placenta = t2_placenta[0,:,:,:]


measurements, _,_,_,_,_ = extract_brain_measurements(brain, mask_brain)


brain = resize_study(brain, target_size=(100, 128, 128))

brain = normalize_image(brain,method = 'minmax')

mask_brain = resize_study(mask_brain, target_size=(100, 128, 128))

_, max_area_slice_index,leftmost_x,rightmost_x,topmost_y,bottommost_y = extract_brain_measurements(brain, mask_brain)

def add_tumor(brain, mask_brain, center, radius, tumor_value):
    # Convert to numpy array for manipulation
    brain_np = sitk.GetArrayFromImage(brain)
    mask_brain_np = sitk.GetArrayFromImage(mask_brain)
    
    # Create a spherical mask for the tumor
    tumor_mask = np.zeros_like(brain_np, dtype=np.uint8)
    z, y, x = np.ogrid[:brain_np.shape[0], :brain_np.shape[1], :brain_np.shape[2]]
    dist_from_center = (z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2
    tumor_mask[dist_from_center <= radius**2] = 1
    
    # Add tumor only where mask_brain is present
    tumor_mask = np.logical_and(tumor_mask, mask_brain_np).astype(np.uint8)
    
    # Update brain with tumor value
    brain_np[tumor_mask == 1] = tumor_value
    
    # Convert back to SimpleITK image
    brain_with_tumor = sitk.GetImageFromArray(brain_np)
    brain_with_tumor.CopyInformation(brain)
    
    return brain_with_tumor

# Parameters for tumor
tumor_center = (60, 70, 80)  # Center of the tumor
tumor_radius = 10            # Radius of the tumor
tumor_value = 0.8             # Intensity value of the tumor

# Add tumor to the brain
# brain = add_tumor(brain, mask_brain, tumor_center, tumor_radius, tumor_value)

# Flatten the nested dictionary
flat_measurements = {}
for key, value in measurements.items():
    if isinstance(value, dict):
        for sub_key, sub_value in value.items():
            flat_measurements[f'{sub_key}'] = sub_value
    else:
        flat_measurements[key] = value

flat_measurements = pd.DataFrame(flat_measurements, index=[args.patientid])

studies_df = pd.read_csv('DB_consolidated.csv')

##### Linear Regression Model #####
features = ['Biparietal Diameter mm', 'Head Circumference mm', 'Transverse Cerebral Diameter mm', 
            'Total Brain Volume cm3']

X = studies_df[features]
y = studies_df['tag_ga']

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
# model = LinearRegression()
# model.fit(X_train, y_train)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
residuals = y_test - y_pred
n = len(y_test)
p = X.shape[1]
mse = np.mean(residuals**2)
se = np.sqrt(mse)

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Extract the required measurements
pd_prediction = flat_measurements[['Biparietal Diameter mm', 'Head Circumference mm', 'Transverse Cerebral Diameter mm', 'Total Brain Volume cm3']]
subject_to_study_values = pd_prediction[features].values
predicted_tag_ga = model.predict(subject_to_study_values)

flat_measurements['tag_ga'] = predicted_tag_ga[0]
flat_measurements['study'] = args.patientid
flat_measurements['study_id'] = args.patientid
flat_measurements['study_name'] = args.studyfolder
flat_measurements['mask_name'] = args.segmentationfolder
studies_df = pd.concat([studies_df, flat_measurements], ignore_index=True)

uncertainty = se
# Display the result with uncertainty
print(f'Predicted Gestational Age: {predicted_tag_ga[0]:.2f} ± {uncertainty:.2f}')


if predicted_tag_ga < 26:
    print('Prediction less than 26')
    ckpts_dir ='/home/hpc/mfdp/mfdp104h/Thesis/saved/Ckpts_2025_02_04-13:45:23-Best25-26'
elif predicted_tag_ga < 28:
    print('Prediction less than 28')
    ckpts_dir ='/home/hpc/mfdp/mfdp104h/Thesis/saved/Ckpts_2025_02_04-13:50:08-Best27-28'
elif predicted_tag_ga < 30:
    print('Prediction less than 30')
    ckpts_dir ='/home/hpc/mfdp/mfdp104h/Thesis/saved/Ckpts_2025_02_04-13:54:47-Best29-30'
elif predicted_tag_ga < 32:
    print('Prediction less than 32')
    ckpts_dir ='/home/hpc/mfdp/mfdp104h/Thesis/saved/Ckpts_2025_02_04-13:58:54-Best31-32'
elif predicted_tag_ga < 34:
    print('Prediction less than 34')
    ckpts_dir ='/home/hpc/mfdp/mfdp104h/Thesis/saved/Ckpts_2025_02_04-14:06:13-Best33-34'
elif predicted_tag_ga < 36:
    print('Prediction less than 36')
    ckpts_dir ='/home/hpc/mfdp/mfdp104h/Thesis/saved/Ckpts_2025_02_04-14:09:19-Best35-36'
elif predicted_tag_ga < 38:
    print('Prediction less than 38')
    ckpts_dir ='/home/hpc/mfdp/mfdp104h/Thesis/saved/Ckpts_2025_02_04-14:12:17-Best37-38'
else:
    print('Prediction more than 40')
    ckpts_dir ='/home/hpc/mfdp/mfdp104h/Thesis/saved/Ckpts_2025_02_04-14:15:48-Best39-40'

brain = brain - 0.5
trained_vqvae_model = keras.models.load_model(ckpts_dir+'/vqvae_model.h5', custom_objects={'VectorQuantizer': VectorQuantizer})
brain = sitk.GetArrayFromImage(brain)
brain = np.expand_dims(brain, axis=-1)  # Add channel dimension if needed
reconstructed_brain = trained_vqvae_model.predict(brain)
print(f'Original brain shape: {brain.shape}')
print(f'Reconstructed brain shape: {reconstructed_brain.shape}')

brain = np.squeeze(brain) + 0.5
reconstructed_brain = np.squeeze(reconstructed_brain)+0.5
reconstructed_brain[reconstructed_brain < 0] = 0

probability_maps = np.abs(reconstructed_brain-brain)
probability_maps[probability_maps < 0.4] = 0

# Read the brain labels CSV file

brain_labels = pd.read_csv('/home/hpc/mfdp/mfdp104h/Thesis/fetal_project/brain_labels.csv')

outliers_df = studies_df.copy()

volume_measurements = [
    'Volume eCSF Left cm3', 'Volume eCSF Right cm3', 'Volume Cortical GM Left cm3',
    'Volume Cortical GM Right cm3', 'Volume Fetal WM Left cm3', 'Volume Fetal WM Right cm3',
    'Volume Lateral Ventricle Left cm3', 'Volume Lateral Ventricle Right cm3',
    'Volume Cavum septum pellucidum cm3', 'Volume Brainstem cm3', 'Volume Cerebellum Left cm3',
    'Volume Cerebellum Right cm3', 'Volume Cerebellar Vermis cm3', 'Volume Basal Ganglia Left cm3',
    'Volume Basal Ganglia Right cm3', 'Volume Thalamus Left cm3', 'Volume Thalamus Right cm3',
    'Volume Third Ventricle cm3', 'Volume Fourth Ventricle cm3','Cortical Gray Matter cm3','Parenchyma cm3',
    'Cerebellum cm3','Percent Ventricular Asymmetry','Volume Ratio Ventricles/Parenchyma cm3'
]


outliers_df[volume_measurements] = np.nan


# Initialize the Dash app
app = Dash(__name__)

# Set the app title
app.title = "Fetal MRI"

# Register layout
app.layout = create_layout(patient_id, studies_df, outliers_df,volume_measurements, predicted_tag_ga, uncertainty,max_area_slice_index)

# Register callbacks
register_callbacks(app, brain, reconstructed_brain, brain_labels, mask_brain, probability_maps, max_area_slice_index,
                       leftmost_x, rightmost_x, topmost_y, bottommost_y, studies_df, 
                       outliers_df, patient_id, t2_brain, t2_placenta)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
