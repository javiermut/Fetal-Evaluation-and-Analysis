import numpy as np
import SimpleITK as sitk
import cv2
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot
import statsmodels.api as sm
from tensorflow.keras import layers
import tensorflow as tf


def normalize_image(image, method, window_range=None):
    """
    Normalize a medical image using various methods.
    
    Args:
        image (SimpleITK.Image): Input image
        method (str): Normalization method ('minmax', 'zscore', 'window')
        window_range (tuple): Optional (min, max) range for windowing
    
    Returns:
        SimpleITK.Image: Normalized image
    """
    # Convert to numpy array for easier manipulation
    array = sitk.GetArrayFromImage(image)
    
    if method == 'minmax':
        # Min-max normalization to [0, 1] range
        min_val = array.min()
        max_val = array.max()
        if max_val - min_val != 0:
            normalized = (array - min_val) / (max_val - min_val)
        else:
            normalized = array
            
    elif method == 'zscore':
        # Z-score normalization (standardization)
        mean = array.mean()
        std = array.std()
        if std != 0:
            normalized = (array - mean) / std
        else:
            normalized = array
            
    elif method == 'window':
        # Intensity windowing (useful for CT images)
        if window_range is None:
            window_range = (array.mean() - array.std(), array.mean() + array.std())
        
        normalized = np.clip(array, window_range[0], window_range[1])
        normalized = (normalized - window_range[0]) / (window_range[1] - window_range[0])
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    

    # Convert back to SimpleITK Image
    normalized_image = sitk.GetImageFromArray(normalized)
    normalized_image.CopyInformation(image)  # Preserve metadata

    return normalized_image


#################### DATA ANALYSIS FUNCTIONS ####################

def extract_biparietal_diameter(x_indices, spacing):
    """
    Extracts the biparietal diameter (BPD) from given x-coordinates and spacing.

    Parameters:
    x_indices (array-like): Array of x-coordinates.
    spacing (array-like): Array containing the spacing values in each direction. 
                          spacing[2] should correspond to the spacing in the x-direction.

    Returns:
    float: The biparietal diameter in millimeters.
    float: The leftmost x-coordinate.
    float: The rightmost x-coordinate.
    """
    leftmost_x = np.min(x_indices)
    rightmost_x = np.max(x_indices)
    biparietal_diameter = rightmost_x - leftmost_x
    biparietal_diameter_mm = biparietal_diameter * spacing[2]  # spacing[2] is for x-direction
    return biparietal_diameter_mm,leftmost_x,rightmost_x

def extract_head_circumference(edges, spacing):
    """
    Calculate the head circumference from the edge pixels of an image.

    Parameters:
    edges (numpy.ndarray): A binary image where the edges are marked with non-zero values.
    spacing (tuple): A tuple containing the spacing of the image in millimeters (x, y, z).

    Returns:
    float: The head circumference in millimeters.
    """
    perimeter_pixels = np.sum(edges > 0)
    head_circumference_mm = perimeter_pixels * spacing[2]  # Convert to millimeters
    return head_circumference_mm

def extract_transverse_diameter(edges, y_indices, spacing):
    """
    Extracts the transverse cerebral diameter from given edge indices and spacing.

    Parameters:
    edges (ndarray): The edges of the image (not used in the function but kept for consistency).
    y_indices (ndarray): The y-coordinates of the edge points.
    spacing (tuple): The spacing of the image in each dimension (spacing[1] is for y-direction).

    Returns:
    tuple: A tuple containing:
        - transverse_cerebral_diameter_mm (float): The transverse cerebral diameter in millimeters.
        - topmost_y (int): The topmost y-coordinate of the edge points.
        - bottommost_y (int): The bottommost y-coordinate of the edge points.
    """
    topmost_y = np.min(y_indices)
    bottommost_y = np.max(y_indices)
    transverse_cerebral_diameter = bottommost_y - topmost_y
    transverse_cerebral_diameter_mm = transverse_cerebral_diameter * spacing[1]  # spacing[1] is for y-direction
    return transverse_cerebral_diameter_mm,topmost_y,bottommost_y

def calculate_percent_ventricular_asymmetry(volumes):
    """
    Calculate the percent ventricular symmetry between the left and right lateral ventricles.

    Parameters:
    volumes (dict): A dictionary containing the necessary brain region volumes with keys:
        - 'Volume Lateral Ventricle Left cm3'
        - 'Volume Lateral Ventricle Right cm3'

    Returns:
    float: The percent ventricular asymmetry.
    """
    left_volume = volumes['Volume Lateral Ventricle Left cm3']
    right_volume = volumes['Volume Lateral Ventricle Right cm3']
    largest = max(left_volume, right_volume)
    smaller = min(left_volume, right_volume)
    
    asymmetry = ((largest/smaller)-1) * 100
    return asymmetry

def calculate_volume_ratio(parenchyma, volumes):
    """
    Calculate the ratio of ventricular volume to hemispheric parenchyma volume.

    Args:
    volumes (dict): A dictionary containing the necessary brain region volumes with keys:
        - 'Volume Lateral Ventricle Left cm3'
        - 'Volume Lateral Ventricle Right cm3'
        - 'Volume Third Ventricle cm3'
        - 'Volume Fourth Ventricle cm3'

    Returns:
    float: The ratio of ventricular volume to hemispheric parenchyma volume.
    """

    required_keys = [
        'Volume Lateral Ventricle Left cm3',
        'Volume Lateral Ventricle Right cm3',
        'Volume Third Ventricle cm3',
        'Volume Fourth Ventricle cm3',
    ]
    
    for key in required_keys:
        if key not in volumes:
            print(f"{key} not found in volumes. Setting to 0.")
            volumes[key] = 0
    
    # Calculate ventricular volume
    ventricular_volume = (
        volumes['Volume Lateral Ventricle Left cm3'] +
        volumes['Volume Lateral Ventricle Right cm3'] +
        volumes['Volume Third Ventricle cm3'] +
        volumes['Volume Fourth Ventricle cm3']
    )

    # Calculate the ratio
    if parenchyma == 0:
        raise ValueError("Parenchyma volume cannot be zero.")
    
    volume_ratio = ventricular_volume / parenchyma
    return volume_ratio

def extract_volume(labels_arr, spacing):
    """
    Calculate the brain volume in cubic centimeters from a labeled array and voxel spacing.

    Parameters:
    labels_arr (numpy.ndarray): A 3D array where non-zero values indicate brain voxels.
    spacing (tuple or list of float): The spacing of the voxels in each dimension (x, y, z) in millimeters.

    Returns:
    dict: A dictionary containing the volumes of different brain regions in cubic centimeters.
    """
    voxel_volume = spacing[0] * spacing[1] * spacing[2]  # Volume of a single voxel in cubic millimeters
    brain_volume_mm3 = np.sum(labels_arr > 0) * voxel_volume  # Total brain volume in cubic millimeters
    brain_volume_cm3 = brain_volume_mm3 / 1000  # Convert to cubic centimeters

    label_dict = {
        1: 'Volume eCSF Left cm3',
        2: 'Volume eCSF Right cm3',
        3: 'Volume Cortical GM Left cm3',
        4: 'Volume Cortical GM Right cm3',
        5: 'Volume Fetal WM Left cm3',
        6: 'Volume Fetal WM Right cm3',
        7: 'Volume Lateral Ventricle Left cm3',
        8: 'Volume Lateral Ventricle Right cm3',
        9: 'Volume Cavum septum pellucidum cm3',
        10: 'Volume Brainstem cm3',
        11: 'Volume Cerebellum Left cm3',
        12: 'Volume Cerebellum Right cm3',
        13: 'Volume Cerebellar Vermis cm3',
        14: 'Volume Basal Ganglia Left cm3',
        15: 'Volume Basal Ganglia Right cm3',
        16: 'Volume Thalamus Left cm3',
        17: 'Volume Thalamus Right cm3',
        18: 'Volume Third Ventricle cm3',
        19: 'Volume Fourth Ventricle cm3'
    }

    volumes = {'Total Brain Volume cm3': brain_volume_cm3}
    for label in np.unique(labels_arr):
        if label in label_dict:
            label_volume_mm3 = np.sum(labels_arr == label) * voxel_volume  # Volume for the current label in cubic millimeters
            label_volume_cm3 = label_volume_mm3 / 1000  # Convert to cubic centimeters
            volumes[label_dict[label]] = label_volume_cm3

    return volumes

def extract_brain_measurements(image, mask):
    mask_arr = sitk.GetArrayFromImage(mask)

    # Select area with the largest sum of non-zero values
    axial_sums = [np.sum(slice) for slice in mask_arr]
    max_area_slice_index = np.argmax(axial_sums)

    largest_slice_labels = mask_arr[max_area_slice_index]

    largest_slice_binary = (largest_slice_labels > 0).astype(np.uint8) * 255
    edges = cv2.Canny(largest_slice_binary, threshold1=30, threshold2=100)
    y_indices, x_indices = np.where(edges > 0)

    spacing = image.GetSpacing()  # Retrieves (spacing_z, spacing_y, spacing_x)

    biparietal_diameter_mm, leftmost_x, rightmost_x = extract_biparietal_diameter(x_indices, spacing)
    head_circumference_mm = extract_head_circumference(edges, spacing)
    transverse_cerebral_diameter_mm, topmost_y, bottommost_y = extract_transverse_diameter(edges, y_indices, spacing)
    volumes = extract_volume(mask_arr, spacing)
    try:
        cortical_gm = volumes['Volume Cortical GM Left cm3'] + volumes['Volume Cortical GM Right cm3']
    except KeyError:
        print("Volume Cortical GM Left cm3 or Volume Cortical GM Right cm3 not found in volumes. Setting to 0.")
        cortical_gm = 0

    try:
        parenchyma = volumes['Volume Cortical GM Left cm3']+volumes['Volume Cortical GM Right cm3']+volumes['Volume Fetal WM Left cm3']+volumes['Volume Fetal WM Right cm3'] + volumes['Volume Basal Ganglia Left cm3']+volumes['Volume Basal Ganglia Right cm3']+volumes['Volume Thalamus Left cm3'] +volumes['Volume Thalamus Right cm3']
    except KeyError:
        print("Volume Fetal WM Left cm3 or Volume Fetal WM Right cm3 not found in volumes. Setting to 0.")
        parenchyma = 0

    try:
        cerebellum = volumes['Volume Cerebellum Left cm3'] + volumes['Volume Cerebellum Right cm3'] + volumes['Volume Cerebellar Vermis cm3']
    except KeyError:
        print("Volume Cerebellum Left cm3, Volume Cerebellum Right cm3, or Volume Cerebellar Vermis cm3 not found in volumes. Setting to 0.")
        cerebellum = 0
        
    percent_ventricular_symmetry = calculate_percent_ventricular_asymmetry(volumes)
    volume_ratio = calculate_volume_ratio(parenchyma,volumes)

    measurements = {
        'Biparietal Diameter mm': biparietal_diameter_mm,
        'Head Circumference mm': head_circumference_mm,
        'Transverse Cerebral Diameter mm': transverse_cerebral_diameter_mm,
        'Volumes cm3': volumes,
        'Cortical Gray Matter cm3': cortical_gm,
        'Parenchyma cm3': parenchyma,
        'Cerebellum cm3': cerebellum,
        'Percent Ventricular Asymmetry': percent_ventricular_symmetry,
        'Volume Ratio Ventricles/Parenchyma cm3': volume_ratio,
    }

    return measurements, max_area_slice_index,leftmost_x,rightmost_x,topmost_y,bottommost_y

#################### VISUALIZATION FUNCTIONS ####################
def add_volume_slice(fig, volume, colormap, row, col, axis='z', title=None):
    """
    Add a 2D slice view of a 3D volume to a subplot with a slider control.
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        The main figure with subplots
    volume : numpy.ndarray
        3D volume data (shape: [x, y, z])
    colormap : str
        Name of the colormap to use (e.g., 'viridis', 'plasma')
    row : int
        Row index of the subplot
    col : int
        Column index of the subplot
    axis : str
        Axis along which to slice ('x', 'y', or 'z')
    title : str, optional
        Title for the subplot
    """
    import plotly.graph_objects as go
    import numpy as np
    
    # Determine slice orientation and initial slice
    axis_dict = {'x': 0, 'y': 1, 'z': 2}
    axis_idx = axis_dict[axis.lower()]
    
    # Get initial middle slice
    slice_idx = volume.shape[axis_idx] // 2
    
    if axis == 'z':
        initial_slice = volume[:, :, slice_idx]
        slider_dir = 'z'
    elif axis == 'y':
        initial_slice = volume[:, slice_idx, :]
        slider_dir = 'y'
    else:  # axis == 'x'
        initial_slice = volume[slice_idx, :, :]
        slider_dir = 'x'
    
    # Add heatmap
    heatmap = go.Heatmap(
        z=initial_slice,
        colorscale=colormap,
        showscale=True,
    )
    
    fig.add_trace(heatmap, row=row, col=col)
    
    # Update layout for this subplot
    fig.update_xaxes(title_text=f'{slider_dir}', row=row, col=col)
    fig.update_yaxes(title_text=f'{slider_dir}', row=row, col=col)
    
    if title:
        fig.update_xaxes(title_text=title, row=row, col=col)
    
    # Add slider
    steps = []
    for i in range(volume.shape[axis_idx]):
        if axis == 'z':
            slice_data = volume[:, :, i]
        elif axis == 'y':
            slice_data = volume[:, i, :]
        else:  # axis == 'x'
            slice_data = volume[i, :, :]
            
        step = dict(
            method="restyle",
            args=[{"z": [slice_data]}],
            label=f"{slider_dir}={i}"
        )
        steps.append(step)
    
    sliders = [dict(
        active=slice_idx,
        currentvalue={"prefix": f"{slider_dir}-slice: "},
        pad={"t": 50},
        steps=steps
    )]
    
    # Update layout to include slider
    fig.update_layout(
        sliders=sliders
    )
    
    return fig

def add_measure_plot(fig, studies_pd, measure, row, col, args):
    # Get the x and y values from the dataframe
    x = studies_pd['tag_ga']
    y = studies_pd[measure]
    
    # Add a constant for the intercept (for the linear model)
    x_with_const = sm.add_constant(x)
    
    # Fit a linear regression model
    model = sm.OLS(y, x_with_const).fit()
    
    # Predict values and calculate residuals
    y_pred = model.predict(x_with_const)
    residuals = y - y_pred
    
    # Identify outliers (residuals > 2 standard deviations)
    threshold = 2 * np.std(residuals)
    outliers = studies_pd[np.abs(residuals) > threshold]
    
    # Generate regression line with thresholds
    x_vals = np.linspace(x.min(), x.max(), 100)
    x_vals_with_const = sm.add_constant(x_vals)
    y_vals = model.predict(x_vals_with_const)
    upper_bound = y_vals + threshold
    lower_bound = y_vals - threshold
    
    # Create scatter plot trace
    trace = go.Scatter(
        x=x, 
        y=y, 
        mode='markers', 
        name=f"Data: {measure}",
        marker=dict(color='gray', size=8, opacity=0.6)
    )
    
    # Add regression line trace
    reg_line_trace = go.Scatter(
        x=x_vals, 
        y=y_vals, 
        mode='lines', 
        name=f"Regression Line: {measure}",
        line=dict(color='blue', dash='dash')
    )
    
    # Add upper and lower bound traces
    upper_bound_trace = go.Scatter(
        x=x_vals, 
        y=upper_bound, 
        mode='lines', 
        name=f"Upper Bound: {measure}",
        line=dict(color='red', dash='dot')
    )
    
    lower_bound_trace = go.Scatter(
        x=x_vals, 
        y=lower_bound, 
        mode='lines', 
        name=f"Lower Bound: {measure}",
        line=dict(color='red', dash='dot')
    )
    
    # Add traces for outliers
    outliers_trace = go.Scatter(
        x=outliers['tag_ga'], 
        y=outliers[measure], 
        mode='markers', 
        name=f"Outliers: {measure}",
        marker=dict(color='orange', size=10)
    )
    
    # Add all traces to the subplot
    fig.add_trace(trace, row=row, col=col)
    fig.add_trace(reg_line_trace, row=row, col=col)
    fig.add_trace(upper_bound_trace, row=row, col=col)
    fig.add_trace(lower_bound_trace, row=row, col=col)
    fig.add_trace(outliers_trace, row=row, col=col)
    
    # Add trace for the current study
    current_study_trace = go.Scatter(
        x=[studies_pd.loc[studies_pd['study'] == args.patientid, 'tag_ga'].values[0]],
        y=[studies_pd.loc[studies_pd['study'] == args.patientid, measure].values[0]],
        mode='markers',
        name=f"Current Study: {measure}",
        marker=dict(
            color='green' if args.patientid not in outliers['study'].values else 'red',
            size=20,
        )
    )
    fig.update_xaxes(title_text='Gestational Age (weeks)', row=row, col=col)
    fig.update_yaxes(title_text=measure, row=row, col=col)
    fig.update_layout(title_text=f'{measure} vs Gestational Age', title_x=0.5)
    fig.add_trace(current_study_trace, row=row, col=col)


def resize_study(image, target_size=(32, 128, 128)):

    """
    Resize a 3D medical image to target dimensions using SimpleITK.
    
    Args:
        image (SimpleITK.Image): Input 3D image
        target_size (tuple): Desired output size as (depth, height, width)
    
    Returns:
        SimpleITK.Image: Resized image
    """
    # Get current size
    target_size = (target_size[1], target_size[2], target_size[0])
    current_size = image.GetSize()
    # Calculate scaling factors
    scale = [float(t)/float(c) for t, c in zip(target_size, current_size)]
    
    # Create resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(target_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    
    # Adjust spacing based on scale factors
    original_spacing = image.GetSpacing()
    new_spacing = [orig_spacing/scale_factor for orig_spacing, scale_factor in zip(original_spacing, scale)]
    resampler.SetOutputSpacing(new_spacing)
    
    # Use linear interpolation
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkLinear)
    
    # Perform the resizing
    resized_image = resampler.Execute(image)
    
    return resized_image

class VectorQuantizer(layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        # The `beta` parameter is best kept between [0.25, 2] as per the paper.
        self.beta = beta

        # Initialize the embeddings which we will quantize.
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
            ),
            trainable=True,
            name="embeddings_vqvae",
        )

    def call(self, x):
        # Calculate the input shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact.
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Quantization.
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)

        # Reshape the quantized values back to the original input shape
        quantized = tf.reshape(quantized, input_shape)

        # Calculate vector quantization loss and add that to the layer. You can learn more
        # about adding losses to different layers here:
        # https://keras.io/guides/making_new_layers_and_models_via_subclassing/. Check
        # the original paper to get a handle on the formulation of the loss function.
        commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(self.beta * commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized

    def get_code_indices(self, flattened_inputs):
        # Calculate L2-normalized distance between the inputs and the codes.
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0)
            - 2 * similarity
        )

        # Derive the indices for minimum distances.
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices
    
    def get_config(self):
        # Get the configuration of the layer, including the arguments passed to __init__
        config = super().get_config()
        config.update({
            "num_embeddings": self.num_embeddings,
            "embedding_dim": self.embedding_dim,
            "beta": self.beta
        })
        return config
