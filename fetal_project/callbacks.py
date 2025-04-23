from dash import Input, Output, callback_context, dcc, html
from dash.dependencies import ALL
import plotly.graph_objects as go
import statsmodels.api as sm
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
from skimage import measure
import SimpleITK as sitk
import plotly.graph_objects as go
import numpy as np
import dash


def generate_hover_text(mask, brain_labels):
    """
    Utility function to generate hover text for segmentation overlays.
    """
    hover_text = np.empty(mask.shape, dtype=object)
    for value, label in brain_labels.items():
        hover_text[mask == value] = label
    return hover_text


def register_callbacks(app, brain, reconstructed_brain, brain_labels, mask_brain, probability_maps, max_area_slice_index,
                       leftmost_x, rightmost_x, topmost_y, bottommost_y, studies_df, 
                       outliers_df, patient_id, t2_brain, t2_placenta):
    """
    Register all Dash callbacks.
    """
    # Define colorscale for segmentation
    custom_colors = [
        "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#9edae5", "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", 
        "#c5b0d5", "#f7b6d2", "#c49c94", "#f7b6d2", "#dbdb8d","#ffff00"
    ]
    colorscale = [(i / 19, color) for i, color in enumerate(custom_colors)]
    spacing = mask_brain.GetSpacing()
    mask_brain = sitk.GetArrayFromImage(mask_brain)
    # === Slice View Callbacks ===
    @app.callback(
        [Output('axial-view', 'figure'),
         Output('axial-graph', 'figure')],
        [Input('axial-slice-slider', 'value'),
         Input('heatmap-toggle', 'value'),
         Input('segmentation-toggle', 'value')]
    )
    def update_axial_view(slice_idx, heatmap_toggle, segmentation_toggle):
        fig_axial = go.Figure()

        # Base heatmap
        fig_axial.add_trace(
            go.Heatmap(
                z=brain[slice_idx, :, :],
                colorscale="gray",
                zmin=np.min(brain),
                zmax=np.max(brain),
                showscale='show' not in segmentation_toggle and 'show' not in heatmap_toggle
            )
        )

        # Highlight bounding box
        if slice_idx == max_area_slice_index:
            fig_axial.add_trace(
                go.Scatter(
                    x=[leftmost_x, rightmost_x, rightmost_x, leftmost_x, leftmost_x],
                    y=[topmost_y, topmost_y, bottommost_y, bottommost_y, topmost_y],
                    mode='lines',
                    line=dict(color='blue')
                )
            )

        # Overlay heatmap probability_mapss
        if 'show' in heatmap_toggle:
            fig_axial.add_trace(
                go.Heatmap(
                    z=probability_maps[slice_idx, :, :],
                    colorscale="Blues", #Blues
                    zmin=np.min(probability_maps),
                    zmax=np.max(probability_maps),
                    opacity=1,
                    showscale='show' not in segmentation_toggle
                )
            )

        # Overlay segmentation
        if 'show' in segmentation_toggle:
            fig_axial.add_trace(
                go.Heatmap(
                    z=mask_brain[slice_idx, :, :],
                    colorscale=colorscale,
                    showscale=True,
                    opacity=0.5
                )
            )
        fig_mse = go.Figure()

        # Add a scatter plot for MSE values
        mse_values = mse_values = [np.mean((brain[i, :, :] - reconstructed_brain[i, :, :])**2) for i in range(brain.shape[0])]
        fig_mse.add_trace(go.Scatter(x=list(range(brain.shape[0])), y=mse_values, mode='lines+markers', name='MSE Values'))

        # Update layout for MSE plot
        fig_mse.update_layout(
            title='MSE Values Across Slices',
            xaxis_title='Slice Index',
            yaxis_title='MSE Value'
        )

        fig_axial.update_layout(title="Axial Slice View", margin=dict(l=0, r=0, t=30, b=0))
        return fig_axial, fig_mse

    @app.callback(
        [Output('sagittal-view', 'figure'),
        Output('sagittal-graph', 'figure')],
        [Input('sagittal-slice-slider', 'value'),
         Input('heatmap-toggle', 'value'),
         Input('segmentation-toggle', 'value')]
    )
    def update_sagital_view(slice_idx, heatmap_toggle, segmentation_toggle):
        fig_sagital = go.Figure()

        # Base heatmap
        fig_sagital.add_trace(
            go.Heatmap(
                z=brain[:, :, slice_idx],
                colorscale="gray",
                zmin=np.min(brain),
                zmax=np.max(brain),
                showscale='show' not in segmentation_toggle and 'show' not in heatmap_toggle
            )
        )

        # Overlay heatmap probability_mapss
        if 'show' in heatmap_toggle:
            fig_sagital.add_trace(
                go.Heatmap(
                    z=probability_maps[:, :, slice_idx],
                    colorscale="Blues",
                    zmin=np.min(probability_maps),
                    zmax=np.max(probability_maps),
                    opacity=1,
                    showscale='show' not in segmentation_toggle
                )
            )

        # Overlay segmentation
        if 'show' in segmentation_toggle:
            fig_sagital.add_trace(
                go.Heatmap(
                    z=mask_brain[:, :, slice_idx],
                    colorscale=colorscale,
                    showscale=True,
                    opacity=0.5
                )
            )
        fig_mse = go.Figure()
        # Add a scatter plot for MSE values
        mse_values = mse_values = [np.mean((brain[:, :, i] - reconstructed_brain[:, :, i])**2) for i in range(brain.shape[2])]
        fig_mse.add_trace(go.Scatter(x=list(range(brain.shape[2])), y=mse_values, mode='lines+markers', name='MSE Values'))

        # Update layout for MSE plot
        fig_mse.update_layout(
            title='MSE Values Across Slices',
            xaxis_title='Slice Index',
            yaxis_title='MSE Value'
        )

        fig_sagital.update_layout(title="Sagital Slice View", margin=dict(l=0, r=0, t=30, b=0))
        return fig_sagital, fig_mse

    @app.callback(
        [Output('coronal-view', 'figure'),
        Output('coronal-graph', 'figure')],
        [Input('coronal-slice-slider', 'value'),
         Input('heatmap-toggle', 'value'),
         Input('segmentation-toggle', 'value')]
    )
    def update_coronal_view(slice_idx, heatmap_toggle, segmentation_toggle):
        fig_coronal = go.Figure()

        # Base heatmap
        fig_coronal.add_trace(
            go.Heatmap(
                z=brain[:, slice_idx, :],
                colorscale="gray",
                zmin=np.min(brain),
                zmax=np.max(brain),
                showscale='show' not in segmentation_toggle and 'show' not in heatmap_toggle
            )
        )

        # Overlay heatmap probability_mapss
        if 'show' in heatmap_toggle:
            fig_coronal.add_trace(
                go.Heatmap(
                    z=probability_maps[:, slice_idx, :],
                    colorscale="Blues",
                    zmin=np.min(probability_maps),
                    zmax=np.max(probability_maps),
                    opacity=1,
                    showscale='show' not in segmentation_toggle
                )
            )

        # Overlay segmentation
        if 'show' in segmentation_toggle:
            fig_coronal.add_trace(
                go.Heatmap(
                    z=mask_brain[:, slice_idx, :],
                    colorscale=colorscale,
                    showscale=True,
                    opacity=0.5
                )
            )

        fig_mse = go.Figure()
        mse_values = mse_values = [np.mean((brain[:, i, :] - reconstructed_brain[:, i, :])**2) for i in range(brain.shape[1])]
        fig_mse.add_trace(go.Scatter(x=list(range(brain.shape[1])), y=mse_values, mode='lines+markers', name='MSE Values'))

        # Update layout for MSE plot
        fig_mse.update_layout(
            title='MSE Values Across Slices',
            xaxis_title='Slice Index',
            yaxis_title='MSE Value'
        )

        fig_coronal.update_layout(title="Coronal Slice View", margin=dict(l=0, r=0, t=30, b=0))
        return fig_coronal, fig_mse

    # === Measurement Interaction Callbacks ===
    @app.callback(
        Output('selected-measurement', 'data'),
        Input({'type': 'table-row', 'index': ALL}, 'n_clicks'),
        prevent_initial_call=True
    )
    def update_selected_measurement(n_clicks):
        ctx = callback_context
        if not ctx.triggered:
            return None
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        return eval(triggered_id)['index']  # Extract the measurement name

    @app.callback(
        Output('measurement-plot-container', 'children'),
        Input('selected-measurement', 'data'),
        prevent_initial_call=True
    )
    def update_plot(selected_measurement):
        x = studies_df['tag_ga']
        
        y = studies_df[selected_measurement]

        if x.empty or y.empty:
            return f"No data available for {selected_measurement}."

        x_with_const = sm.add_constant(x)
        model = sm.OLS(y, x_with_const).fit()
        y_pred = model.predict(x_with_const)
        residuals = y - y_pred
        threshold = 2 * np.std(residuals)

        outliers = studies_df[np.abs(residuals) > threshold]
        for outlier_id in outliers['study']:
            outliers_df.loc[outliers_df['study'] == outlier_id, selected_measurement] = studies_df.loc[studies_df['study'] == outlier_id, selected_measurement].values[0]


        x_vals = np.linspace(x.min(), x.max(), 100)
        x_vals_with_const = sm.add_constant(x_vals)
        y_vals = model.predict(x_vals_with_const)
        upper_bound = y_vals + threshold
        lower_bound = y_vals - threshold

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name="Data", marker=dict(color='gray', size=8)))
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name="Regression Line", line=dict(color='blue', dash='dash')))
        fig.add_trace(go.Scatter(x=x_vals, y=upper_bound, mode='lines', name="Upper Bound", line=dict(color='red', dash='dot')))
        fig.add_trace(go.Scatter(x=x_vals, y=lower_bound, mode='lines', name="Lower Bound", line=dict(color='red', dash='dot')))

        fig.add_trace(go.Scatter(
            x=outliers['tag_ga'],
            y=outliers[selected_measurement],
            mode='markers',
            name="Outliers",
            marker=dict(color='orange', size=10)
        ))

        current_study_x = studies_df.loc[studies_df['study'] == patient_id, 'tag_ga'].values[0]
        current_study_y = studies_df.loc[studies_df['study'] == patient_id, selected_measurement].values[0]
        fig.add_trace(go.Scatter(
            x=[current_study_x],
            y=[current_study_y],
            mode='markers',
            name="Current Study",
            marker=dict(
                color='red' if not pd.isna(outliers_df.loc[outliers_df['study'] == patient_id, selected_measurement].values[0]) else 'green',
                size=30,
            )
        ))

        fig.update_layout(
            title=f'{selected_measurement} vs Gestational Age',
            title_font=dict(size=30),  # Increase title font size
            xaxis_title='Gestational Age (weeks)',
            yaxis_title=selected_measurement,
            xaxis_title_font=dict(size=25),  # Increase X axis title font size
            yaxis_title_font=dict(size=25),  # Increase Y axis title font size
            xaxis_tickfont=dict(size=20),  # Increase X axis tick font size
            yaxis_tickfont=dict(size=20),  # Increase Y axis tick font size
            legend=dict(font=dict(size=25)),  # Increase legend font size),
            margin=dict(l=5, r=0, t=70, b=0)
        )
        return dcc.Graph(figure=fig, id='measurement-plot')
    


    @app.callback(
        Output('3d-plot', 'figure'),
        Input('dummy-button', 'n_clicks')
    )
    def update_plot(n_clicks):
        # Ensure probability_maps is properly loaded and in the correct format
        # Assuming probability_maps is your 3D numpy array
        
        # Create coordinates for each axis
        patient_outlier_measurements = outliers_df[outliers_df['study'] == patient_id]
        if patient_outlier_measurements['Volume Lateral Ventricle Right cm3'].notna().any() or patient_outlier_measurements['Volume Lateral Ventricle Left cm3'].notna().any():
            probability_maps[mask_brain == 7] = 0.5
            probability_maps[mask_brain == 8] = 0.5


        X, Y, Z = np.mgrid[0:probability_maps.shape[0], 
                        0:probability_maps.shape[1], 
                        0:probability_maps.shape[2]]

        # Create the 3D volume plot
        fig = go.Figure(data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=probability_maps.flatten(),
            isomin=0.01,  # Lower threshold to see more data
            isomax=1.0,
            opacity=0.9,
            opacityscale=[
                [0, 0],
                [0.05, 0.2],
                [0.25, 0.5],
                [1.0, 0.8]
            ],
            surface_count=15,  # Increased surface count for better resolution
            colorscale='Blues',
            caps=dict(
                x_show=False,
                y_show=False,
                z_show=False
            ),
            showscale=True  # Show colorbar
        ))

        # Update the layout
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    title='Sagittal',
                    range=[0, probability_maps.shape[0]],  # Set proper axis range
                    gridcolor='white',
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)'
                ),
                yaxis=dict(
                    title='Coronal',
                    range=[0, probability_maps.shape[1]],
                    gridcolor='white',
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)'
                ),
                zaxis=dict(
                    title='Axial',
                    range=[0, probability_maps.shape[2]],
                    gridcolor='white',
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)'
                ),
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                aspectmode='data'  # This ensures proper scaling
            ),
            title='3D Brain Probability Map',
            width=800,
            height=800,
            margin=dict(t=40, b=0, l=0, r=0)
        )

        # View buttons
        fig.update_layout(
            updatemenus=[dict(
                type='buttons',
                showactive=False,
                buttons=[
                    dict(
                        label='Top View',
                        method='relayout',
                        args=['scene.camera', dict(
                            up=dict(x=0, y=1, z=0),
                            center=dict(x=0, y=0, z=0),
                            eye=dict(x=0, y=0, z=2)
                        )]
                    ),
                    dict(
                        label='Side View',
                        method='relayout',
                        args=['scene.camera', dict(
                            up=dict(x=0, y=0, z=1),
                            center=dict(x=0, y=0, z=0),
                            eye=dict(x=2, y=0, z=0)
                        )]
                    ),
                    dict(
                        label='Front View',
                        method='relayout',
                        args=['scene.camera', dict(
                            up=dict(x=0, y=0, z=1),
                            center=dict(x=0, y=0, z=0),
                            eye=dict(x=0, y=2, z=0)
                        )]
                    ),
                    dict(
                        label='Isometric View',
                        method='relayout',
                        args=['scene.camera', dict(
                            up=dict(x=0, y=0, z=1),
                            center=dict(x=0, y=0, z=0),
                            eye=dict(x=1.5, y=1.5, z=1.5)
                        )]
                    )
                ],
                x=0.9,
                y=1.1,
                xanchor='right',
                yanchor='top'
            )]
        )

        return fig
    