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

from helpers import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from ipywidgets import interact

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy.interpolate import griddata


def main():
    # Check if the folder exists, if not, create it
    output_folder_path = os.path.join(os.getcwd(),'fetal_project',("project_"+args.patientid))
    
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    brain = sitk.ReadImage(args.studyfolder)
    mask_brain = sitk.ReadImage(args.segmentationfolder)

    brain = sitk.Cast(brain, sitk.sitkFloat32)
    mask_brain = sitk.Cast(mask_brain, sitk.sitkUInt8)
    brain = sitk.Mask(brain, mask_brain)

    brain = resize_study(brain, target_size=(100, 128, 128))
    brain = normalize_image(brain,method = 'minmax')
    mask_brain = resize_study(mask_brain, target_size=(100, 128, 128))
    measurements, max_area_slice_index,leftmost_x,rightmost_x,topmost_y,bottommost_y = extract_brain_measurements(brain, mask_brain)

    # Flatten the nested dictionary
    flat_measurements = {}
    for key, value in measurements.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat_measurements[f'{sub_key}'] = sub_value
        else:
            flat_measurements[key] = value

    flat_measurements = pd.DataFrame(flat_measurements, index=[args.patientid])

    studies_pd = pd.read_csv('DB_consolidated.csv')
    
    ##### Linear Regression Model #####
    features = ['Biparietal Diameter mm', 'Head Circumference mm', 'Transverse Cerebral Diameter mm', 
                'Total Brain Volume cm3']

    X = studies_pd[features]
    y = studies_pd['tag_ga']

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Extract the required measurements
    pd_prediction = flat_measurements[['Biparietal Diameter mm', 'Head Circumference mm', 'Transverse Cerebral Diameter mm', 'Total Brain Volume cm3']]
    subject_to_study_values = pd_prediction[features].values
    predicted_tag_ga = model.predict(subject_to_study_values)
    ##### END Linear Regression Model #####

    #### Add the predicted gestational age to the DataFrame ####

    flat_measurements['tag_ga'] = predicted_tag_ga[0]
    flat_measurements['study'] = args.patientid
    flat_measurements['study_name'] = args.studyfolder
    flat_measurements['mask_name'] = args.segmentationfolder
    studies_pd = pd.concat([studies_pd, flat_measurements], ignore_index=True)
    # print(studies_pd)
    uncertainty = y_pred.std()

    # Display the result with uncertainty
    print(f'Predicted Gestational Age: {predicted_tag_ga[0]:.2f} ± {uncertainty:.2f}')

    ################################## Plotting the correlation ##################################
    fig = sp.make_subplots(
        rows=5,
        cols=5,
        vertical_spacing=0.02, 
        horizontal_spacing=0.02
    )

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

    for index, measure in enumerate(volume_measurements):
        i = index
        row = i // 5 + 1
        col = i % 5 + 1
        # if index==0:
        #     brain = sitk.GetArrayFromImage(brain)
        #     mask_brain = sitk.GetArrayFromImage(mask_brain)
        #     largest_slice_tissue = brain[max_area_slice_index, :, :]
        #     largest_slice_binary = mask_brain[max_area_slice_index, :, :]
        #     fig.add_trace(
        #         go.Heatmap(
        #             z=largest_slice_tissue,
        #             colorscale='Viridis',
        #             showscale=False
        #         ),
        #         row=row, col=col
        #     )
        #     fig.add_trace(
        #         go.Heatmap(
        #             z=largest_slice_binary,
        #             colorscale='Gray',
        #             opacity=0.2,
        #             showscale=False
        #         ),
        #         row=row, col=col
        #     )
        #     fig.add_shape(
        #         type="line",
        #         x0=leftmost_x, y0=topmost_y, x1=rightmost_x, y1=topmost_y,
        #         line=dict(color="red"),
        #         row=row, col=col
        #     )
        #     fig.add_shape(
        #         type="line",
        #         x0=leftmost_x, y0=bottommost_y, x1=rightmost_x, y1=bottommost_y,
        #         line=dict(color="red"),
        #         row=row, col=col
        #     )
        #     fig.add_shape(
        #         type="line",
        #         x0=leftmost_x, y0=topmost_y, x1=leftmost_x, y1=bottommost_y,
        #         line=dict(color="red"),
        #         row=row, col=col
        #     )
        #     fig.add_shape(
        #         type="line",
        #         x0=rightmost_x, y0=topmost_y, x1=rightmost_x, y1=bottommost_y,
        #         line=dict(color="red"),
        #         row=row, col=col
        #     )
        add_measure_plot(fig, studies_pd, measure, row, col, args)

    fig.update_layout(
        title="Correlation of Various Volume Measurements with Gestational Age and Linear Regression",
        height=2000,
        width=2000,
        showlegend=False
    )

    plot(fig, filename=os.path.join(output_folder_path,'2_correlation_plot.html'), auto_open=False)


    ################################## Plotting the study ##################################
    fig1 = sp.make_subplots(
        rows=2,
        cols=3,
        vertical_spacing=0.05, 
        horizontal_spacing=0.05
    )

    # create_interactive_slice_plot_in_subplot(fig1, study_masked, row=1, col=2)


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


    # Parameters for the sphere
    center = (90, 50)  # center of the slice
    radius = 8  # adjust the radius of the sphere

    # Create a meshgrid for the coordinates
    x = np.arange(0, 128)
    y = np.arange(0, 128)
    xv, yv = np.meshgrid(x, y)

    # Function to create a spherical mask
    spherical_mask = (xv - center[0])**2 + (yv - center[1])**2 <= radius**2
    # Invert the spherical mask
    spherical_mask = np.logical_not(spherical_mask)
    # Convert the logical values in spherical_mask to 1 and 0
    spherical_mask = spherical_mask.astype(np.float32)

    # Apply the spherical mask to the slices 12, 13, and 14
    for slice_index in [49]:
        # Note: Adjust `slice_index` based on your actual data indexing
        brain[slice_index] = spherical_mask*brain[slice_index]
    
    brain = brain - 0.5
    trained_vqvae_model = keras.models.load_model(ckpts_dir+'/vqvae_model.h5', custom_objects={'VectorQuantizer': VectorQuantizer})
    reconstructed_brain = trained_vqvae_model.predict(brain)
    brain = brain + 0.5
    
    reconstructed_brain = np.squeeze(reconstructed_brain)+0.5
    reconstructed_brain[reconstructed_brain < 0] = 0

    difference = np.abs(reconstructed_brain-brain)
    add_volume_slice(fig1, brain, colormap = 'balance', row=1, col=1, axis = 'x', title='Original Brain')
    add_volume_slice(fig1, difference, colormap = 'balance', row=1, col=2, axis = 'x', title='Reconstructed Brain')
    # Save the figure as an HTML file
    fig1.update_layout(
    title=dict(
        text="Fetal EvaluaTion and AnaLysis",
        font=dict(size=25),
        x=0.5,
        xanchor='center'
    ),
    annotations=[
        dict(
            text=f"Patient ID: {args.patientid}, Estimated Gestational Age: {predicted_tag_ga[0]:.2f} ± {uncertainty:.2f} weeks",
            x=0.1,
            y=1.05,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=20)
        )
    ]
    )
    plot(fig1, filename=os.path.join(output_folder_path,'1_study.html'), auto_open=False)

    ################################## Plotting the 3D Volume Difference ##################################
    fig2 = go.Figure()

    # Create a 3D surface plot for the difference volume
    x, y, z = np.where(difference > 0.1)  # Threshold to highlight significant differences
    values = difference[x, y, z]

    # Create a meshgrid for the coordinates
    xi, yi, zi = np.meshgrid(np.unique(x), np.unique(y), np.unique(z), indexing='ij')

    # Interpolate the values to create a surface
    grid_values = griddata((x, y, z), values, (xi, yi, zi), method='linear')

    fig2.add_trace(go.Surface(
        x=xi[:, :, 0],
        y=yi[:, :, 0],
        z=zi[:, :, 0],
        surfacecolor=grid_values[:, :, 0],
        colorscale='Viridis',
        opacity=0.8
    ))

    fig2.update_layout(
        title="3D Volume Difference",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )

    plot(fig2, filename=os.path.join(output_folder_path, '3d_volume_difference.html'), auto_open=False)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some inputs.')

    parser.add_argument('-a', '--patientid', type=str, required=True, help='patient id')
    parser.add_argument('-b', '--studyfolder', type=str, required=True, help='Name of the folder of the segmentations')
    parser.add_argument('-c', '--segmentationfolder', type=str, required=True, help='Name of the folder of the studies')

    args = parser.parse_args()

    main()