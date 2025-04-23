import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
# import tensorflow_probability as tfp
import tensorflow as tf
from plot_keras_history import plot_history

from datetime import datetime
import time
import os
from helpers1 import *
import SimpleITK as sitk
from sklearn.model_selection import train_test_split
from tensorflow.image import ssim, psnr
import tensorflow_addons as tfa
import csv
import seaborn as sns; sns.set_theme()

def vq(batch_size,num_embeddings,latent_dim):
    
    slices = 100
    image_size = 128
    num_channels = 1
    latent_dim = latent_dim
    num_embeddings = num_embeddings
    batch_size = batch_size
    epochs = 200

    saved_dir = './saved/'

    date = datetime. now(). strftime("%Y_%m_%d-%H:%M:%S")
    ckpts_dir = os.path.join(saved_dir, f'Ckpts_{date}')
    os.makedirs(ckpts_dir)
    fig_path = os.path.join(ckpts_dir, 'History_plot.png')

    ############ VQ-VAE Model ############
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


    def get_encoder(latent_dim=16):
        encoder_inputs = keras.Input(shape=(image_size, image_size, num_channels))

        x = layers.Conv2D(32 , 5, activation=layers.ReLU(), strides=2, padding="same")(encoder_inputs)
        x = layers.Conv2D(64 , 5, activation=layers.ReLU(), strides=2, padding="same")(x)
        x = layers.Conv2D(128, 5, activation=layers.ReLU(), strides=2, padding="same")(x)    
        x = layers.Conv2D(128, 5, activation=layers.ReLU(), strides=2, padding="same")(x)  
        encoder_outputs = layers.Conv2D(latent_dim, 1, padding="same")(x)
        return keras.Model(encoder_inputs, encoder_outputs, name="encoder")


    def get_decoder(latent_dim=16):
        latent_inputs = keras.Input(shape=get_encoder(latent_dim).output.shape[1:])
        x = layers.Conv2D(128, 1, strides=1, activation=layers.ReLU(), padding="same")(latent_inputs)    
        x = layers.Conv2DTranspose(128, 5, strides=2, activation=layers.ReLU(), padding="same")(x) 
        x = layers.Conv2DTranspose(64 , 5, strides=2, activation=layers.ReLU(), padding="same")(x)
        x = layers.Conv2DTranspose(32 , 5, strides=2, activation=layers.ReLU(), padding="same")(x)
        x = layers.Conv2DTranspose(32 , 5, strides=2, activation=layers.ReLU(), padding="same")(x)
        decoder_outputs = layers.Conv2DTranspose(1, 3, padding="same")(x)
        return keras.Model(latent_inputs, decoder_outputs, name="decoder")


    def get_vqvae(latent_dim=16, num_embeddings=64):
        vq_layer = VectorQuantizer(num_embeddings, latent_dim, name="vector_quantizer")
        encoder = get_encoder(latent_dim)
        decoder = get_decoder(latent_dim)
        inputs = keras.Input(shape=(image_size, image_size, num_channels))
        encoder_outputs = encoder(inputs)
        quantized_latents = vq_layer(encoder_outputs)
        reconstructions = decoder(quantized_latents)
        return keras.Model(inputs, reconstructions, name="vq_vae")


    get_vqvae().summary()


    ############ VQ-VAE Trainer ############

    class VQVAETrainer(keras.models.Model):
        def __init__(self, train_variance, latent_dim=latent_dim, num_embeddings=num_embeddings, **kwargs):
            super().__init__(**kwargs)
            self.train_variance = train_variance
            self.latent_dim = latent_dim
            self.num_embeddings = num_embeddings

            self.vqvae = get_vqvae(self.latent_dim, self.num_embeddings)

            # Loss trackers for training
            self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
            self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
            self.vq_loss_tracker = keras.metrics.Mean(name="vq_loss")

            # Metric trackers for training
            self.mse_tracker = keras.metrics.Mean(name="mse")
            self.psnr_tracker = keras.metrics.Mean(name="psnr")
            self.ssim_tracker = keras.metrics.Mean(name="ssim")

            # Loss trackers for validation
            self.val_total_loss_tracker = keras.metrics.Mean(name="val_total_loss")
            self.val_reconstruction_loss_tracker = keras.metrics.Mean(name="val_reconstruction_loss")
            self.val_vq_loss_tracker = keras.metrics.Mean(name="val_vq_loss")

            # Metric trackers for validation
            self.val_mse_tracker = keras.metrics.Mean(name="val_mse")
            self.val_psnr_tracker = keras.metrics.Mean(name="val_psnr")
            self.val_ssim_tracker = keras.metrics.Mean(name="val_ssim")

        @property
        def metrics(self):
            # Include validation metrics in the list returned by this property
            return [
                self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.vq_loss_tracker,
                self.mse_tracker,
                self.psnr_tracker,
                self.ssim_tracker,
            ]

        def train_step(self, x):
            with tf.GradientTape() as tape:
                # Forward pass through the VQ-VAE
                reconstructions = self.vqvae(x)

                # Calculate losses
                reconstruction_loss = (
                    tf.reduce_mean((x - reconstructions) ** 2) / self.train_variance
                )
                total_loss = reconstruction_loss + sum(self.vqvae.losses)

            # Backpropagation
            grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

            # Compute SSIM and PSNR
            batch_ssim = tf.reduce_mean(ssim(x, reconstructions, max_val=1.0))  # Normalized inputs
            batch_psnr = tf.reduce_mean(psnr(x, reconstructions, max_val=1.0))  # Normalized inputs

            # Loss tracking
            self.total_loss_tracker.update_state(total_loss)
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            self.vq_loss_tracker.update_state(sum(self.vqvae.losses))

            # Metric tracking
            self.mse_tracker.update_state(reconstruction_loss)
            self.psnr_tracker.update_state(batch_psnr)
            self.ssim_tracker.update_state(batch_ssim)

            # Log results
            return {
                "loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "vqvae_loss": self.vq_loss_tracker.result(),
                "mse": self.mse_tracker.result(),
                "psnr": self.psnr_tracker.result(),
                "ssim": self.ssim_tracker.result(),
            }


    ############## Data ##############

    DATA_DIR = '/home/vault/mfdp/mfdp104h/Thesis/DB'
    STUDIES_PATH = os.path.join(DATA_DIR, 'brain_studies')
    SEGMENTATIONS_PATH = os.path.join(DATA_DIR, 'brain_segmentations')
    INFO_PATH = os.path.join(DATA_DIR,'gestational_ages.csv')


    studies_pd = load_and_merge_data(STUDIES_PATH, INFO_PATH)
    studies_pd = studies_pd[(studies_pd['tag_ga'] >= 24) & (studies_pd['tag_ga'] <= 35)]
    # print(studies_pd)

    def load_nifti_files(studies_pd, studies_path,target_size = (32, 128, 128)):
        images = []
        for study_name in studies_pd['study_name']:
            file_path = os.path.join(studies_path, study_name)
            if os.path.exists(file_path):
                image = sitk.ReadImage(file_path)
                image = resize_study(normalize_image(image),target_size = target_size)
                image_array = sitk.GetArrayFromImage(image)
                images.append(image_array)
            else:
                print(f"File {file_path} does not exist.")
        return np.array(images)

    segmentations = load_nifti_files(studies_pd, STUDIES_PATH,target_size = (slices, image_size, image_size))
    segmentations = segmentations.reshape(-1, image_size, image_size)

    # Split the data into training and temporary sets (80% training, 20% temporary)
    x_train, x_temp = train_test_split(segmentations, test_size=0.05, random_state=42)

    # Split the temporary set into validation and test sets (50% validation, 50% test of the temporary set)
    x_val, x_test = train_test_split(x_temp, test_size=0.5, random_state=42)

    # Calculate the data variance
    data_variance = np.var(x_train)
    # Expand dimensions for training, validation, and test data
    x_train = np.expand_dims(x_train, -1)
    x_val = np.expand_dims(x_val, -1)
    x_test = np.expand_dims(x_test, -1)

    # Scale the data
    x_train_scaled = x_train - 0.5
    x_val_scaled = x_val - 0.5
    x_test_scaled = x_test - 0.5


    print(f'Training data shape: {x_train.shape}')
    print(f'Testing data shape: {x_test.shape}')

    data_variance = np.var(x_train)

    ############ Training ############

    vqvae_trainer = VQVAETrainer(data_variance, latent_dim=latent_dim, num_embeddings=num_embeddings)
    vqvae_trainer.compile(optimizer=keras.optimizers.Adam())
    early_stopping = keras.callbacks.EarlyStopping(monitor='loss',patience=5,verbose=1,restore_best_weights=True)

    history = vqvae_trainer.fit(x_train_scaled,epochs=epochs, batch_size=batch_size,verbose=1,callbacks=[early_stopping])

    plot_history(history, path=fig_path)
    plt.close()
    time.sleep(2)


    ### PLOTS AND SAVING ###
    def show_subplot(original, reconstructed):
        plt.subplot(1, 2, 1)
        plt.imshow(original.squeeze() + 0.5,cmap='gray')
        plt.title("Original")
        plt.grid(False)

        plt.subplot(1, 2, 2)
        plt.imshow(reconstructed.squeeze() + 0.5,cmap='gray')
        plt.title("Reconstructed")
        plt.grid(False)
        plt.show()


    trained_vqvae_model = vqvae_trainer.vqvae
    idx = np.random.choice(len(x_test_scaled), 10)
    test_images = x_test_scaled[idx]
    reconstructions_test = trained_vqvae_model.predict(test_images)

    for test_image, reconstructed_image in zip(test_images, reconstructions_test):
        show_subplot(test_image, reconstructed_image)
        for i, (test_image, reconstructed_image) in enumerate(zip(test_images, reconstructions_test)):
            plt.figure()
            show_subplot(test_image, reconstructed_image)
            plt.savefig(os.path.join(ckpts_dir, f'reconstruction_{i}.jpg'))
            plt.close()

    # Save the trained VQ-VAE model
    model_save_path = os.path.join(ckpts_dir, 'vqvae_model.h5')
    trained_vqvae_model.save(model_save_path)
    print(f"Model saved to {model_save_path}")


    ############ Testing ############

    test_image_path = '/home/vault/mfdp/mfdp104h/Thesis/DB/brain_studies/fm0218_t2_recon_2.nii.gz'
    test_image = sitk.ReadImage(test_image_path)
    test_image_array = resize_study(test_image, target_size=(slices, image_size, image_size))
    test_image_array = normalize_image(test_image_array,method = 'minmax')
    test_image_array = sitk.GetArrayFromImage(test_image_array)


    test_image_array = test_image_array.reshape(-1, image_size, image_size, num_channels) - 0.5

    # Make prediction
    # Load the trained VQ-VAE model
    trained_vqvae_model = keras.models.load_model(ckpts_dir+'/vqvae_model.h5', custom_objects={'VectorQuantizer': VectorQuantizer})
    print(f"Model loaded from {ckpts_dir}")
    reconstructed_test_image = trained_vqvae_model.predict(test_image_array)

    # Reshape and save the reconstructed image
    reconstructed_test_image = reconstructed_test_image + 0.5
    reconstructed_test_image = reconstructed_test_image.reshape(slices, image_size, image_size)
    reconstructed_test_image_sitk = sitk.GetImageFromArray(reconstructed_test_image)
    sitk.WriteImage(reconstructed_test_image_sitk, os.path.join(ckpts_dir, 'reconstructed_test.nii.gz'))
    print(f"Reconstructed image saved to {os.path.join(ckpts_dir, 'reconstructed_test.nii.gz')}")


    # Calculate MS-SSIM and SSIM values
    reconstructed_test_image = tf.expand_dims(reconstructed_test_image, axis=-1)
    test_image_array = tf.convert_to_tensor(test_image_array+0.5, dtype=tf.float32)
    reconstructed_test_image = tf.convert_to_tensor(reconstructed_test_image, dtype=tf.float32)


    ssim_value = SSIMLoss(test_image_array, reconstructed_test_image)
    # Calculate DICE coefficient
    def dice_coefficient(y_true, y_pred, smooth=1e-6):
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    dice_value = dice_coefficient(test_image_array, reconstructed_test_image)

    spnr_value = psnr(test_image_array, reconstructed_test_image, max_val=1.0)

    # Save variables and their values to a text file
    variables = {
        "slices": slices,
        "image_size": image_size,
        "num_channels": num_channels,
        "latent_dim": latent_dim,
        "num_embeddings": num_embeddings,
        "batch_size": batch_size,
        "epochs": epochs,
        "saved_dir": saved_dir,
        "ckpts_dir": ckpts_dir,
        "fig_path": fig_path,
        "DATA_DIR": DATA_DIR,
        "STUDIES_PATH": STUDIES_PATH,
        "SEGMENTATIONS_PATH": SEGMENTATIONS_PATH,
        "INFO_PATH": INFO_PATH,
        "data_variance": data_variance,
        "last_loss": vqvae_trainer.total_loss_tracker.result(),
        "last_reconstruction_loss": vqvae_trainer.reconstruction_loss_tracker.result(),
        "last_vqvae_loss": vqvae_trainer.vq_loss_tracker.result(),
        "last_mse": vqvae_trainer.mse_tracker.result(),
        "last_ssim": vqvae_trainer.ssim_tracker.result(),
        "last_psnr": vqvae_trainer.psnr_tracker.result(),
        "DICE value": dice_value,
        "Min SSIM value": np.min(ssim_value),
        "Max SSIM value": np.max(ssim_value),
        "Average SSIM value": np.mean(ssim_value),
        "Min SPNR value": np.min(spnr_value),
        "Max SPNR value": np.max(spnr_value),
        "Average SPNR value": np.mean(spnr_value),
    }

    variables_file_path = os.path.join(ckpts_dir, 'variables.txt')
    with open(variables_file_path, 'w') as f:
        for key, value in variables.items():
            f.write(f"{key}: {value}\n")

    print(f"Variables saved to {variables_file_path}")

    return variables
    

batch_sizes = [100, 50, 10, 1]
num_embeddings_list = [64, 128, 256]
latent_dims = [16, 32, 64]

results = []
saved_dir = './'
for batch_size in batch_sizes:
    for num_embeddings in num_embeddings_list:
        for latent_dim in latent_dims:
            print(f"Training with batch_size={batch_size}, num_embeddings={num_embeddings}, latent_dim={latent_dim}")
            variables = vq(batch_size, num_embeddings, latent_dim)
            csv_file_path = os.path.join(saved_dir, 'results.csv')
            fieldnames = [
                "slices", "image_size", "num_channels", "latent_dim", "num_embeddings", 
                "batch_size", "epochs", "ckpts_dir", "last_loss", "last_reconstruction_loss", 
                "last_vqvae_loss", "last_mse", "last_ssim", "last_psnr", "DICE value", 
                "Average SSIM value", "Average SPNR value"
            ]

            # Check if the CSV file exists, if not, create it and write the header
            if not os.path.exists(csv_file_path):
                with open(csv_file_path, mode='w', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=fieldnames)
                    writer.writeheader()

            # Write the current variables to the CSV file
            with open(csv_file_path, mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writerow({
                    "slices": variables["slices"],
                    "image_size": variables["image_size"],
                    "num_channels": variables["num_channels"],
                    "latent_dim": variables["latent_dim"],
                    "num_embeddings": variables["num_embeddings"],
                    "batch_size": variables["batch_size"],
                    "epochs": variables["epochs"],
                    "ckpts_dir": variables["ckpts_dir"],
                    "last_loss": variables["last_loss"].numpy(),
                    "last_reconstruction_loss": variables["last_reconstruction_loss"].numpy(),
                    "last_vqvae_loss": variables["last_vqvae_loss"].numpy(),
                    "last_mse": variables["last_mse"].numpy(),
                    "last_ssim": variables["last_ssim"].numpy(),
                    "last_psnr": variables["last_psnr"].numpy(),
                    "DICE value": variables["DICE value"].numpy(),
                    "Average SSIM value": variables["Average SSIM value"],
                    "Average SPNR value": variables["Average SPNR value"]
                })
            results.append(variables)
