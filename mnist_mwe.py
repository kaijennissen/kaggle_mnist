import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.python.ops.gen_array_ops import Reshape

tfd = tfp.distributions
tfpl = tfp.layers


def load_data(path: Path):
    df = pd.read_csv(path)
    if "label" in df.columns:
        y = df.pop("label")
    else:
        y = np.zeros(df.shape[0])
    x = 1 - df.values.reshape(-1, 28, 28) / 255
    return x, y


# Function to inspect dataset digits
def inspect_images(data, num_images):
    fig, ax = plt.subplots(nrows=1, ncols=num_images, figsize=(2 * num_images, 2))
    for i in range(num_images):
        ax[i].imshow(data[i, ..., 0], cmap="gray")
        ax[i].axis("off")
    plt.show()


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
                filters=10,
                kernel_size=(5, 5),
                activation="relu",
                padding="valid",
                data_format="channels_last",
            ),
            # BatchNormalization(),
            tfa.layers.GroupNormalization(groups=5, axis=3),
            MaxPooling2D(pool_size=(5, 5)),
            Flatten(),
            Dense(10, activation="softmax"),
        ]
    )
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


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
                filters=10,
                kernel_size=(5, 5),
                activation=tf.keras.layers.LeakyReLU(0.03),
                padding="valid",
            ),
            BatchNormalization(),
            # tfa.layers.GroupNormalization(groups=5, axis=3),
            MaxPooling2D(pool_size=(5, 5)),
            Flatten(),
            Dense(tfpl.OneHotCategorical.params_size(10)),
            tfpl.OneHotCategorical(10, convert_to_tensor_fn=tfd.Distribution.mode),
        ]
    )
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


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


# Build and compile the Bayesian CNN model
def get_bayesian_model(
    n, input_shape=(28, 28, 1), loss=nll, optimizer=RMSprop(), metrics=["accuracy"]
):
    # Create the layers
    divergence_fn = lambda q, p, _: tfd.kl_divergence(q, p) / n
    convolutional_reparameterization_layer = get_convolutional_reparameterization_layer(
        input_shape=input_shape, divergence_fn=divergence_fn
    )
    dense_variational_layer = get_dense_variational_layer(
        get_prior, get_posterior, kl_weight=1 / n
    )

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
        optimizer=optimizer,
        metrics=metrics,
        experimental_run_tf_function=False,
    )
    return bayesian_model


def get_model(model_str: str, n: int):
    if model_str.lower() == "deterministic":
        return get_deterministic_model(
            input_shape=(28, 28, 1),
            loss=SparseCategoricalCrossentropy(),
            optimizer=RMSprop(),
            metrics=["accuracy"],
        )
    elif model_str.lower() == "probabilistic":
        return get_probabilistic_model(
            input_shape=(28, 28, 1), loss=nll, optimizer=RMSprop(), metrics=["accuracy"]
        )

    elif model_str.lower() == "bayesian":
        return get_bayesian_model(
            n=n,
            get_input_shape=(28, 28, 1),
            loss=nll,
            optimizer=RMSprop(),
            metrics=["accuracy"],
        )

    else:
        raise ValueError(f"{model_str} is unknown.")


def main(model_str: str, predict: bool = False):

    x, y = load_data("data/train.csv")
    x = x[..., np.newaxis]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, shuffle=True, stratify=y
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x, y, test_size=0.5, shuffle=True, stratify=y
    )

    y_train_oh = tf.keras.utils.to_categorical(y_train)
    y_val_oh = tf.keras.utils.to_categorical(y_val)

    if model_str.lower() == "deterministic":
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_ds = train_ds.shuffle(buffer_size=1024).batch(64)

        val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_ds = val_ds.batch(64)
    else:
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train_oh))
        train_ds = train_ds.shuffle(buffer_size=1024).batch(64)

        val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val_oh))
        val_ds = val_ds.batch(64)

    # Create the model
    model = get_model(model_str=model_str, n=x_train.shape[0])

    # Print the model summary
    model.summary()

    # Train the model
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=10, min_lr=0.00001
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=12, restore_best_weights=True
    )
    callbacks = [reduce_lr, early_stop]
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=100,
        batch_size=64,
        callbacks=callbacks,
    )

    # Evaluate the model
    print(
        "Accuracy on MNIST test set: ",
        str(model.evaluate(x_test, y_test, verbose=False)[1]),
    )

    if predict:
        x_test, y_test = load_data("data/test.csv")
        x_test = x_test[..., np.newaxis]
        y_test = model(x_test)
        y_test = y_test.mode().numpy().argmax(axis=-1)
        df = pd.DataFrame(
            {
                "ImageId": [i + 1 for i in range(y_test.shape[0])],
                "Label": y_test,
            }
        )
        df.to_csv("data/predictions_{model_str}.csv", index=False)
    # print("MNIST test set:")
    # plot_entropy_distribution(model, x_test, y_test)
    # # Prediction examples on MNIST

    # for i in [0, 1577]:
    #     analyse_model_prediction(x_test, y_test, model, i)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--predict", action="store_true")
    args = parser.parse_args()

    main(model_str=args.model, predict=args.predict)
