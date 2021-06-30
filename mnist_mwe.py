import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop

tfd = tfp.distributions
tfpl = tfp.layers

# Function to load training and testing data, with labels in integer and one-hot form
def load_data(name):
    data_dir = os.path.join("data", name)
    x_train = 1 - np.load(os.path.join(data_dir, "x_train.npy")) / 255.0
    x_train = x_train.astype(np.float32)
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    y_train_oh = tf.keras.utils.to_categorical(y_train)
    x_test = 1 - np.load(os.path.join(data_dir, "x_test.npy")) / 255.0
    x_test = x_test.astype(np.float32)
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    y_test_oh = tf.keras.utils.to_categorical(y_test)

    return (x_train, y_train, y_train_oh), (x_test, y_test, y_test_oh)


# Function to inspect dataset digits
def inspect_images(data, num_images):
    fig, ax = plt.subplots(nrows=1, ncols=num_images, figsize=(2 * num_images, 2))
    for i in range(num_images):
        ax[i].imshow(data[i, ..., 0], cmap="gray")
        ax[i].axis("off")
    plt.show()


# Load and inspect the MNIST dataset
(x_train, y_train, y_train_oh), (x_test, y_test, y_test_oh) = load_data("MNIST")
inspect_images(data=x_train, num_images=8)


# Load and inspect the MNIST-C dataset
(x_c_train, y_c_train, y_c_train_oh), (x_c_test, y_c_test, y_c_test_oh) = load_data(
    "MNIST_corrupted"
)
inspect_images(data=x_c_train, num_images=8)


def get_deterministic_model(input_shape, loss, optimizer, metrics):
    """
    This function should build and compile a CNN model according to the above specification.
    The function takes input_shape, loss, optimizer and metrics as arguments, which should be
    used to define and compile the model.
    Your function should return the compiled model.
    """
    model = Sequential(
        [
            Conv2D(
                input_shape=input_shape,
                filters=8,
                kernel_size=(5, 5),
                activation="relu",
                padding="valid",
            ),
            MaxPooling2D(pool_size=(6, 5)),
            Flatten(),
            Dense(10, activation="softmax"),
        ]
    )
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


# Run your function to get the benchmark model
tf.random.set_seed(0)
deterministic_model = get_deterministic_model(
    input_shape=(28, 28, 1),
    loss=SparseCategoricalCrossentropy(),
    optimizer=RMSprop(),
    metrics=["accuracy"],
)


# Print the model summary
deterministic_model.summary()


# Train the model
deterministic_model.fit(x_train, y_train, epochs=5)


# Evaluate the model
print(
    "Accuracy on MNIST test set: ",
    str(deterministic_model.evaluate(x_test, y_test, verbose=False)[1]),
)
print(
    "Accuracy on corrupted MNIST test set: ",
    str(deterministic_model.evaluate(x_c_test, y_c_test, verbose=False)[1]),
)


def nll(y_true, y_pred):
    """
    This function should return the negative log-likelihood of each sample
    in y_true given the predicted distribution y_pred. If y_true is of shape
    [B, E] and y_pred has batch shape [B] and event_shape [E], the output
    should be a Tensor of shape [B].
    """
    return -y_pred.log_prob(y_true)


def get_probabilistic_model(input_shape, loss, optimizer, metrics):
    """
    This function should return the probabilistic model according to the
    above specification.
    The function takes input_shape, loss, optimizer and metrics as arguments, which should be
    used to define and compile the model.
    Your function should return the compiled model.
    """
    model = Sequential(
        [
            Conv2D(
                input_shape=input_shape,
                filters=8,
                kernel_size=(5, 5),
                activation="relu",
                padding="valid",
            ),
            MaxPooling2D(pool_size=(6, 5)),
            Flatten(),
            Dense(tfpl.OneHotCategorical.params_size(10)),
            tfpl.OneHotCategorical(10, convert_to_tensor_fn=tfd.Distribution.mode),
        ]
    )
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


tf.random.set_seed(0)
probabilistic_model = get_probabilistic_model(
    input_shape=(28, 28, 1), loss=nll, optimizer=RMSprop(), metrics=["accuracy"]
)


probabilistic_model.summary()

# Train the model
probabilistic_model.fit(x_train, y_train_oh, epochs=5)


# Evaluate the model

print(
    "Accuracy on MNIST test set: ",
    str(probabilistic_model.evaluate(x_test, y_test_oh, verbose=False)[1]),
)
print(
    "Accuracy on corrupted MNIST test set: ",
    str(probabilistic_model.evaluate(x_c_test, y_c_test_oh, verbose=False)[1]),
)


for deterministic_variable, probabilistic_variable in zip(
    deterministic_model.weights, probabilistic_model.weights
):
    print(np.allclose(deterministic_variable.numpy(), probabilistic_variable.numpy()))

# Function to make plots of the probabilities that the model estimates for an image


def analyse_model_prediction(data, true_labels, model, image_num, run_ensemble=False):
    if run_ensemble:
        ensemble_size = 200
    else:
        ensemble_size = 1
    image = data[image_num]
    true_label = true_labels[image_num, 0]
    predicted_probabilities = np.empty(shape=(ensemble_size, 10))
    for i in range(ensemble_size):
        predicted_probabilities[i] = model(image[np.newaxis, :]).mean().numpy()[0]
    model_prediction = model(image[np.newaxis, :])
    fig, (ax1, ax2) = plt.subplots(
        nrows=1, ncols=2, figsize=(10, 2), gridspec_kw={"width_ratios": [2, 4]}
    )

    # Show the image and the true label
    ax1.imshow(image[..., 0], cmap="gray")
    ax1.axis("off")
    ax1.set_title("True label: {}".format(str(true_label)))

    # Show a 95% prediction interval of model predicted probabilities
    pct_2p5 = np.array(
        [np.percentile(predicted_probabilities[:, i], 2.5) for i in range(10)]
    )
    pct_97p5 = np.array(
        [np.percentile(predicted_probabilities[:, i], 97.5) for i in range(10)]
    )
    bar = ax2.bar(np.arange(10), pct_97p5, color="red")
    bar[int(true_label)].set_color("green")
    ax2.bar(
        np.arange(10), pct_2p5 - 0.02, color="white", linewidth=1, edgecolor="white"
    )
    ax2.set_xticks(np.arange(10))
    ax2.set_ylim([0, 1])
    ax2.set_ylabel("Probability")
    ax2.set_title("Model estimated probabilities")
    plt.show()


# Prediction examples on MNIST

for i in [0, 1577]:
    analyse_model_prediction(x_test, y_test, probabilistic_model, i)


# Prediction examples on MNIST-C

for i in [0, 3710]:
    analyse_model_prediction(x_c_test, y_c_test, probabilistic_model, i)


# Prediction examples from both datasets

for i in [9241]:
    analyse_model_prediction(x_test, y_test, probabilistic_model, i)
    analyse_model_prediction(x_c_test, y_c_test, probabilistic_model, i)


# Functions to plot the distribution of the information entropy across samples,
# split into whether the model prediction is correct or incorrect


def get_correct_indices(model, x, labels):
    y_model = model(x)
    correct = np.argmax(y_model.mean(), axis=1) == np.squeeze(labels)
    correct_indices = [i for i in range(x.shape[0]) if correct[i]]
    incorrect_indices = [i for i in range(x.shape[0]) if not correct[i]]
    return correct_indices, incorrect_indices


def plot_entropy_distribution(model, x, labels):
    probs = model(x).mean().numpy()
    entropy = -np.sum(probs * np.log2(probs), axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, category in zip(range(2), ["Correct", "Incorrect"]):
        entropy_category = entropy[get_correct_indices(model, x, labels)[i]]
        mean_entropy = np.mean(entropy_category)
        num_samples = entropy_category.shape[0]
        title = category + "ly labelled ({:.1f}% of total)".format(
            num_samples / x.shape[0] * 100
        )
        axes[i].hist(entropy_category, weights=(1 / num_samples) * np.ones(num_samples))
        axes[i].annotate(
            "Mean: {:.3f} bits".format(mean_entropy), (0.4, 0.9), ha="center"
        )
        axes[i].set_xlabel("Entropy (bits)")
        axes[i].set_ylim([0, 1])
        axes[i].set_ylabel("Probability")
        axes[i].set_title(title)
    plt.show()


# Entropy plots for the MNIST dataset

print("MNIST test set:")
plot_entropy_distribution(probabilistic_model, x_test, y_test)


# Entropy plots for the MNIST-C dataset

print("Corrupted MNIST test set:")
plot_entropy_distribution(probabilistic_model, x_c_test, y_c_test)


def get_convolutional_reparameterization_layer(input_shape, divergence_fn):
    """
    This function should create an instance of a Convolution2DReparameterization
    layer according to the above specification.
    The function takes the input_shape and divergence_fn as arguments, which should
    be used to define the layer.
    Your function should then return the layer instance.
    """
    return tfpl.Convolution2DReparameterization(
        input_shape=input_shape,
        filters=8,
        kernel_size=(5, 5),
        activation="relu",
        padding="valid",
        kernel_divergence_fn=divergence_fn,
        bias_divergence_fn=divergence_fn,
    )


def spike_and_slab(event_shape, dtype):
    distribution = tfd.Mixture(
        cat=tfd.Categorical(probs=[0.5, 0.5]),
        components=[
            tfd.Independent(
                tfd.Normal(
                    loc=tf.zeros(event_shape, dtype=dtype),
                    scale=1.0 * tf.ones(event_shape, dtype=dtype),
                ),
                reinterpreted_batch_ndims=1,
            ),
            tfd.Independent(
                tfd.Normal(
                    loc=tf.zeros(event_shape, dtype=dtype),
                    scale=10.0 * tf.ones(event_shape, dtype=dtype),
                ),
                reinterpreted_batch_ndims=1,
            ),
        ],
        name="spike_and_slab",
    )
    return distribution


# Plot the spike and slab distribution pdf

x_plot = np.linspace(-5, 5, 1000)[:, np.newaxis]
plt.plot(
    x_plot,
    tfd.Normal(loc=0, scale=1).prob(x_plot).numpy(),
    label="unit normal",
    linestyle="--",
)
plt.plot(
    x_plot,
    spike_and_slab(1, dtype=tf.float32).prob(x_plot).numpy(),
    label="spike and slab",
)
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()
plt.show()


def get_prior(kernel_size, bias_size, dtype=None):
    """
    This function should create the prior distribution, consisting of the
    "spike and slab" distribution that is described above.
    The distribution should be created using the kernel_size, bias_size and dtype
    function arguments above.
    The function should then return a callable, that returns the prior distribution.
    """
    n = kernel_size + bias_size
    return lambda t: spike_and_slab(n, dtype=dtype)


def get_posterior(kernel_size, bias_size, dtype=None):
    """
    This function should create the posterior distribution as specified above.
    The distribution should be created using the kernel_size, bias_size and dtype
    function arguments above.
    The function should then return a callable, that returns the posterior distribution.
    """
    n = kernel_size + bias_size
    return Sequential(
        [
            tfpl.VariableLayer(tfpl.IndependentNormal.params_size(n), dtype=dtype),
            tfpl.IndependentNormal(n),
        ]
    )


def get_dense_variational_layer(prior_fn, posterior_fn, kl_weight):
    """
    This function should create an instance of a DenseVariational layer according
    to the above specification.
    The function takes the prior_fn, posterior_fn and kl_weight as arguments, which should
    be used to define the layer.
    Your function should then return the layer instance.
    """
    return tfpl.DenseVariational(
        10, make_prior_fn=prior_fn, make_posterior_fn=posterior_fn, kl_weight=kl_weight
    )


# Create the layers

tf.random.set_seed(0)
divergence_fn = lambda q, p, _: tfd.kl_divergence(q, p) / x_train.shape[0]
convolutional_reparameterization_layer = get_convolutional_reparameterization_layer(
    input_shape=(28, 28, 1), divergence_fn=divergence_fn
)
dense_variational_layer = get_dense_variational_layer(
    get_prior, get_posterior, kl_weight=1 / x_train.shape[0]
)


# Build and compile the Bayesian CNN model

bayesian_model = Sequential(
    [
        convolutional_reparameterization_layer,
        MaxPooling2D(pool_size=(6, 6)),
        Flatten(),
        dense_variational_layer,
        tfpl.OneHotCategorical(10, convert_to_tensor_fn=tfd.Distribution.mode),
    ]
)
bayesian_model.compile(
    loss=nll,
    optimizer=RMSprop(),
    metrics=["accuracy"],
    experimental_run_tf_function=False,
)


# Print the model summary

bayesian_model.summary()


# Train the model

bayesian_model.fit(x=x_train, y=y_train_oh, epochs=10, verbose=True)


# Evaluate the model

print(
    "Accuracy on MNIST test set: ",
    str(bayesian_model.evaluate(x_test, y_test_oh, verbose=False)[1]),
)
print(
    "Accuracy on corrupted MNIST test set: ",
    str(bayesian_model.evaluate(x_c_test, y_c_test_oh, verbose=False)[1]),
)


# Prediction examples on MNIST

for i in [0, 1577]:
    analyse_model_prediction(x_test, y_test, bayesian_model, i, run_ensemble=True)


# Prediction examples on MNIST-C

for i in [0, 3710]:
    analyse_model_prediction(x_c_test, y_c_test, bayesian_model, i, run_ensemble=True)


# Prediction examples from both datasets

for i in [9241]:
    analyse_model_prediction(x_test, y_test, bayesian_model, i, run_ensemble=True)
    analyse_model_prediction(x_c_test, y_c_test, bayesian_model, i, run_ensemble=True)


# Entropy plots for the MNIST dataset

print("MNIST test set:")
plot_entropy_distribution(bayesian_model, x_test, y_test)


# Entropy plots for the MNIST-C dataset

print("Corrupted MNIST test set:")
plot_entropy_distribution(bayesian_model, x_c_test, y_c_test)


# Congratulations on completing this programming assignment! In the next week of the course we will look at the bijectors module and normalising flows.
