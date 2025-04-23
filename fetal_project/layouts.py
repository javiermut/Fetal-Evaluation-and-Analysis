from dash import html, dcc
import statsmodels.api as sm
import numpy as np
import random
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
import pickle

image_list = [
    "/assets/image1.jpeg",
    "/assets/image2.jpeg",
    "/assets/image3.jpeg"
]

# Font sizes dictionary remains the same
font_sizes = {
    "header": "40px",
    "subheader": "26px",
    "subject_info": "26px",
    "checklist": "25px",
    "measurements": "26px",
    "measurement_values": "25px",
    "table_header": "25px",
    "study_details": "20px",
    "table_body": "25px",
    "slider_marks": "15px",
    "graph_title": "25px"
}

def create_layout(patient_id, studies_df, outliers_df, volume_measurements, predicted_tag_ga, uncertainty,max_area_slice_index):

    for selected_measurement in volume_measurements:
        x = studies_df['tag_ga']
        y = studies_df[selected_measurement]
        # Add a constant for the intercept (for the linear model)
        x_with_const = sm.add_constant(x)
        
        # Fit a linear regression model
        model = sm.OLS(y, x_with_const).fit()
        
        # Predict values and calculate residuals
        y_pred = model.predict(x_with_const)
        residuals = y - y_pred

        # Identify outliers (residuals > 2 standard deviations)
        threshold = 2 * np.std(residuals)
        outliers = studies_df[np.abs(residuals) > threshold]

        # If the current study is an outlier, add it to the outliers_df
        for outlier_id in outliers['study']:
            outliers_df.loc[outliers_df['study'] == outlier_id, selected_measurement] = studies_df.loc[studies_df['study'] == outlier_id, selected_measurement].values[0]
    
    patient_outlier_measurements = outliers_df[outliers_df['study'] == patient_id]
    if patient_outlier_measurements['Volume Lateral Ventricle Right cm3'].notna().any() or patient_outlier_measurements['Volume Lateral Ventricle Left cm3'].notna().any():
        ventriculomegaly_flag = "Possible Case"
        print("Possible ventriculomegaly detected")
    else:
        ventriculomegaly_flag = "Not Detected"
        print("No ventriculomegaly detected")






    # Read the CSV
    df = pd.read_csv("preeclampsia_db_placenta.csv")

    # Suppose this is your list of patient names that you want to filter on
    patient_names = [patient_id]

    # Filter the DataFrame for only the rows whose 'Patient Name' is in the list
    filtered_df = df[df["Patient Name"].isin(patient_names)]

    # # If you only want the first occurrence of each patient, drop subsequent duplicates
    # filtered_df = filtered_df.drop_duplicates(subset="Patient Name", keep="first")
    # filtered_df = filtered_df.drop(columns=['ci', 'cw', 'sys', 'dias', 'adc', 't2s', 'kurt', 'skew','LV CO', 'HR', 'volume','LVEDV', 'LVESV','carp_id', 'fa', 'LVSV','Number','CI','heart_curve','BSA',])
    # features_df = filtered_df.drop(columns=['Patient Name', 'cohort'])
    # features_df = features_df.reindex(columns=['Brain Mask','File Root','Placenta Mask'], fill_value=0)
    selected_features = [
    'original_shape_Elongation',
    'original_shape_Flatness',
    'original_shape_LeastAxisLength',
    'original_shape_MajorAxisLength',
    'original_shape_Maximum2DDiameterColumn',
    'original_shape_Maximum2DDiameterRow',
    'original_shape_Maximum2DDiameterSlice',
    'original_shape_Maximum3DDiameter',
    'original_shape_MeshVolume',
    'original_shape_MinorAxisLength',
    'original_shape_Sphericity',
    'original_shape_SurfaceArea',
    'original_shape_SurfaceVolumeRatio',
    'original_shape_VoxelVolume',
    'original_firstorder_10Percentile',
    'original_firstorder_90Percentile',
    'original_firstorder_Energy',
    'original_firstorder_Entropy',
    'original_firstorder_InterquartileRange',
    'original_firstorder_Kurtosis',
    'original_firstorder_Maximum',
    'original_firstorder_MeanAbsoluteDeviation',
    'original_firstorder_Mean',
    'original_firstorder_Median',
    'original_firstorder_Minimum',
    'original_firstorder_Range',
    'original_firstorder_RobustMeanAbsoluteDeviation',
    'original_firstorder_RootMeanSquared',
    'original_firstorder_Skewness',
    'original_firstorder_TotalEnergy',
    'original_firstorder_Uniformity',
    'original_firstorder_Variance',
    'original_glcm_Autocorrelation',
    'original_glcm_ClusterProminence',
    'original_glcm_ClusterShade',
    'original_glcm_ClusterTendency',
    'original_glcm_Contrast',
    'original_glcm_Correlation',
    'original_glcm_DifferenceAverage',
    'original_glcm_DifferenceEntropy',
    'original_glcm_DifferenceVariance',
    'original_glcm_Id',
    'original_glcm_Idm',
    'original_glcm_Idmn',
    'original_glcm_Idn',
    'original_glcm_Imc1',
    'original_glcm_Imc2',
    'original_glcm_InverseVariance',
    'original_glcm_JointAverage',
    'original_glcm_JointEnergy',
    'original_glcm_JointEntropy',
    'original_glcm_MCC',
    'original_glcm_MaximumProbability',
    'original_glcm_SumAverage',
    'original_glcm_SumEntropy',
    'original_glcm_SumSquares',
    'original_gldm_DependenceEntropy',
    'original_gldm_DependenceNonUniformity',
    'original_gldm_DependenceNonUniformityNormalized',
    'original_gldm_DependenceVariance',
    'original_gldm_GrayLevelNonUniformity',
    'original_gldm_GrayLevelVariance',
    'original_gldm_HighGrayLevelEmphasis',
    'original_gldm_LargeDependenceEmphasis',
    'original_gldm_LargeDependenceHighGrayLevelEmphasis',
    'original_gldm_LargeDependenceLowGrayLevelEmphasis',
    'original_gldm_LowGrayLevelEmphasis',
    'original_gldm_SmallDependenceEmphasis',
    'original_gldm_SmallDependenceHighGrayLevelEmphasis',
    'original_gldm_SmallDependenceLowGrayLevelEmphasis',
    'original_glrlm_GrayLevelNonUniformity',
    'original_glrlm_GrayLevelNonUniformityNormalized',
    'original_glrlm_GrayLevelVariance',
    'original_glrlm_HighGrayLevelRunEmphasis',
    'original_glrlm_LongRunEmphasis',
    'original_glrlm_LongRunHighGrayLevelEmphasis',
    'original_glrlm_LongRunLowGrayLevelEmphasis',
    'original_glrlm_LowGrayLevelRunEmphasis',
    'original_glrlm_RunEntropy',
    'original_glrlm_RunLengthNonUniformity',
    'original_glrlm_RunLengthNonUniformityNormalized',
    'original_glrlm_RunPercentage',
    'original_glrlm_RunVariance',
    'original_glrlm_ShortRunEmphasis',
    'original_glrlm_ShortRunHighGrayLevelEmphasis',
    'original_glrlm_ShortRunLowGrayLevelEmphasis',
    'original_glszm_GrayLevelNonUniformity',
    'original_glszm_GrayLevelNonUniformityNormalized',
    'original_glszm_GrayLevelVariance',
    'original_glszm_HighGrayLevelZoneEmphasis',
    'original_glszm_LargeAreaEmphasis',
    'original_glszm_LargeAreaHighGrayLevelEmphasis',
    'original_glszm_LargeAreaLowGrayLevelEmphasis',
    'original_glszm_LowGrayLevelZoneEmphasis',
    'original_glszm_SizeZoneNonUniformity',
    'original_glszm_SizeZoneNonUniformityNormalized',
    'original_glszm_SmallAreaEmphasis',
    'original_glszm_SmallAreaHighGrayLevelEmphasis',
    'original_glszm_SmallAreaLowGrayLevelEmphasis',
    'original_glszm_ZoneEntropy',
    'original_glszm_ZonePercentage',
    'original_glszm_ZoneVariance',
    'original_ngtdm_Busyness',
    'original_ngtdm_Coarseness',
    'original_ngtdm_Complexity',
    'original_ngtdm_Contrast',
    'original_ngtdm_Strength', 'ga']

    features_df = filtered_df[selected_features]


    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
        # print(scaler.feature_names_in_)
    
    
    features_scaled = scaler.transform(features_df)
    input_dim = features_scaled.shape[1]
    num_classes = 3  # change this number as needed

    def create_nn_model(input_dim, num_classes):
        model = Sequential([
            Dense(64, activation='relu', input_shape=(input_dim,)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        return model

    # Create and compile the model (compilation is optional for pure prediction, but we do it here)
    model_inference = create_nn_model(input_dim, num_classes)
    model_inference.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Load the model weights (previously saved as 'bayesian_nn_weights.h5')
    model_inference.load_weights('bayesian_nn_weights.h5')


    predictions = model_inference.predict(features_scaled)
    cohort_prob_columns = [f"Cohort {i}" for i in range(num_classes)]
    probs_df = pd.DataFrame(predictions, columns=cohort_prob_columns).iloc[0]
    results_str = "\n".join([f"{col}: {val * 100:.4f}% --" for col, val in probs_df.items()])
    print(results_str)


    return html.Div([
        # Store component
        dcc.Store(id='selected-measurement'),
        
        # Header section remains the same
        html.Div([
            html.Div([
                html.H1("Welcome to Fetal EvaluaTion and AnaLysis", style={
                    "margin": "0",
                    "padding": "10px",
                    "textAlign": "left",
                    "color": "white",
                    "fontSize": font_sizes["header"],
                    "marginLeft": "20px"
                }),
                html.H3("Exploration of fetal self-assessment", style={
                    "margin": "0",
                    "padding": "0 10px 5px 20px",
                    "textAlign": "left",
                    "color": "white",
                    "fontSize": font_sizes["subheader"]
                }),
            ], style={"display": "inline-block", "width": "50%", "verticalAlign": "top"}),

            html.Div([
                html.H4("Subject Information", style={
                    "textAlign": "left",
                    "color": "white",
                    "fontSize": font_sizes["subject_info"],
                    "margin": "0",
                    "padding": "5px 0"
                }),
                html.P(f"Subject ID: {patient_id}", style={
                    "textAlign": "left",
                    "color": "white",
                    "fontSize": font_sizes["subject_info"],
                    "margin": "0",
                    "padding": "5px 0"
                }),
                html.P(f"Predicted GA: {predicted_tag_ga[0]:.2f} ± {uncertainty:.2f} weeks", style={
                    "textAlign": "left",
                    "color": "white",
                    "fontSize": font_sizes["subject_info"],
                    "margin": "0",
                    "padding": "5px 0"
                }),
                html.P(f"Ventriculomegaly: {ventriculomegaly_flag}", style={
                    "textAlign": "left",
                    "color": "white",
                    "fontSize": font_sizes["subject_info"],
                    "margin": "0",
                    "padding": "5px 0"
                }),
                    html.P(f"Pre-eclampsia: {results_str}", style={
                    "textAlign": "left",
                    "color": "white",
                    "fontSize": font_sizes["subject_info"],
                    "margin": "0",
                    "padding": "5px 0"
                }),
            ], style={"display": "inline-block", "width": "35%", "verticalAlign": "top"}),
            

            html.Div([
                html.Img(src=random.choice(image_list), style={
                    "width": "200px",
                    "height": "auto",
                    "borderRadius": "10px",
                    "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.2)",
                })
            ], style={"display": "inline-block", "width": "15%", "textAlign": "center", "verticalAlign": "top"})
        ], style={
            "backgroundColor": "#0056b3",
            "padding": "20px",
            "borderBottom": "2px solid #cccccc",
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "space-between",
        }),

        # Main content container
        html.Div([
            # Toggles and Measurements section
            html.Div([
                # Toggles
                html.Div([
                    dcc.Checklist(
                        id='heatmap-toggle',
                        options=[{'label': 'Show Abnormalities Heatmap', 'value': 'show'}],
                        value=[],
                        className="custom-checkbox"
                    ),
                    dcc.Checklist(
                        id='segmentation-toggle',
                        options=[{'label': 'Show Segmentation', 'value': 'show'}],
                        value=[],
                        className="custom-checkbox"
                    )
                ], style={"width": "30%", "display": "inline-block", "fontSize": font_sizes["checklist"]}),
                
                # Measurements grid
                html.Div([
                    html.H4("Measurement Estimations", style={
                        "fontSize": font_sizes["measurements"],
                        "margin": "0 0 10px 0"
                    }),
                    html.Div([
                        create_measurement_box("Biparietal Diameter", 
                            f"{studies_df.loc[studies_df['study'] == patient_id, 'Biparietal Diameter mm'].values[0]:.2f}", "mm"),
                        create_measurement_box("Head Circumference", 
                            f"{studies_df.loc[studies_df['study'] == patient_id, 'Head Circumference mm'].values[0]:.2f}", "mm"),
                        create_measurement_box("Transverse Cerebral Diameter", 
                            f"{studies_df.loc[studies_df['study'] == patient_id, 'Transverse Cerebral Diameter mm'].values[0]:.2f}", "mm"),
                        create_measurement_box("Total Brain Volume", 
                            f"{studies_df.loc[studies_df['study'] == patient_id, 'Total Brain Volume cm3'].values[0]:.2f}", "cm³"),
                    ], style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "10px"})
                ], style={"width": "70%", "display": "inline-block"})
            ], style={"marginBottom": "10px"}),

            # Views section - now in a 4-column grid
            html.Div([
                create_view_container('axial-view', 'axial-slice-slider', 'axial-graph',max_area_slice_index),
                create_view_container('sagittal-view', 'sagittal-slice-slider', 'sagittal-graph',max_area_slice_index),
                create_view_container('coronal-view', 'coronal-slice-slider', 'coronal-graph',max_area_slice_index),
                # 3D plot container
                html.Div([
                    dcc.Graph(id='3d-plot', style={"height": "80vh"}),
                    html.Button(id='dummy-button', style={'display': 'none'}),
                ], style={"width": "50%"})
            ], style={
                "display": "grid",
                "gridTemplateColumns": "repeat(4, 1fr)",
                "gap": "10px",
                "marginBottom": "10px"
            }),

            # Study details and plots
            html.Div([
                html.Div([
                    html.H4("Study Details", style={"fontSize": font_sizes["study_details"]}),
                    create_study_table(patient_id, studies_df, outliers_df, font_sizes)
                ], style={"width": "35%"}),
                
                html.Div(id='measurement-plot-container', style={"width": "50%"}),
            ], style={"display": "flex", "justifyContent": "space-between", "marginTop": "20px","alignItems": "center"}),

            # T2* and IVIM section
            # html.Div([
            #     html.H3("Placenta T2* and Brain T2*", style={"textAlign": "center", "fontSize": font_sizes["graph_title"]}),
            #     html.Div([
            #         dcc.Graph(id='t2-star-placenta', style={"width": "45%"}),
            #         dcc.Graph(id='t2-star-brain', style={"width": "45%"})
            #     ], style={"display": "flex", "justifyContent": "space-between"})
            # ], style={"marginTop": "20px"})
        ], style={
            "maxWidth": "1800px",
            "margin": "0 auto",
            "padding": "0px"
        })
    ])

# Helper functions remain exactly the same
def create_measurement_box(label, value, unit):
    return html.Div([
        html.P(label, style={
            "margin": "0",
            "fontSize": font_sizes["measurements"]
        }),
        html.P(f"{value} {unit}", style={
            "margin": "0",
            "color": "blue",
            "fontSize": font_sizes["measurement_values"]
        })
    ], style={
        "padding": "1px",
        "backgroundColor": "#f5f5f5",
        "borderRadius": "5px"
    })

def create_view_container(graph_id, slider_id, second_graph_id,max_area_slice_index):
    return html.Div([
        dcc.Graph(id=graph_id, config={"scrollZoom": True}, style={"height": "35vh"}),
        dcc.Slider(
            id=slider_id,
            min=0,
            max=99,
            step=1,
            value=max_area_slice_index,
            tooltip={"always_visible": True},
            marks={i: {'label': str(i), 'style': {'fontSize': font_sizes["slider_marks"]}} 
                   for i in range(0, 100, 10)}
        ),
        dcc.Graph(id=second_graph_id, config={"scrollZoom": True}, style={"height": "35vh"})
    ], style={"width": "100%"})

def create_study_table(patient_id, studies_df, outliers_df, font_sizes):
    return html.Table([
        html.Tr([
            html.Th("Measurement"),
            html.Th("Value")
        ], style={"fontSize": font_sizes["table_header"], "textAlign": "left"}),
        html.Tbody([
            html.Tr(
                [
                    html.Td(col, style={
                        "fontSize": font_sizes["table_body"],
                        "color": "red" if val in outliers_df.values else "green",
                        "paddingLeft": "10px",
                        "textAlign": "left"
                    }),
                    html.Td(f"{val:.2f}", style={
                        "fontSize": font_sizes["table_body"],
                        "color": "red" if val in outliers_df.values else "green",
                        "paddingLeft": "10px",
                        "textAlign": "left"
                    })
                ],
                id={'type': 'table-row', 'index': col},
                style={
                    "cursor": "pointer",
                    "backgroundColor": "#f9f9f9" if idx % 2 == 0 else "#e9e9e9",
                    "borderBottom": "1px solid #ddd"
                }
            )
            for idx, (col, val) in enumerate(studies_df[studies_df['study'] == patient_id].iloc[0, 11:-1].items())
            if col not in ["Biparietal Diameter mm", "Head Circumference mm", 
                          "Transverse Cerebral Diameter mm", "Total Brain Volume cm3"]
        ])
    ], style={
        "width": "100%",
        "borderCollapse": "collapse",
        "textAlign": "left"
    })

