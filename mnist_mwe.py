import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    GlobalAvgPool2D,
    MaxPooling2D,
    SpatialDropout2D,
)
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.python.keras import activations
from tensorflow.python.ops.gen_array_ops import Reshape
from tensorflow_addons.optimizers import Lookahead, RectifiedAdam


class GradientCentralization(RectifiedAdam):
    def get_gradients(self, loss, params):
        grads = []
        gradients = super().get_gradients()
        for grad in gradients:
            grad_len = len(grad.shape)
            if grad_len > 1:
                axis = list(range(grad_len - 1))
                grad -= tf.reduce_mean(grad, axis=axis, keep_dims=True)
            grads.append(grad)
        return grads


tfd = tfp.distributions
tfpl = tfp.layers


def load_data(path: Path):
    df = pd.read_csv(path)
    if "label" in df.columns:
        y = df.pop("label").values
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
            tfa.layers.WeightNormalization(
                Conv2D(
                    input_shape=input_shape,
                    filters=32,
                    kernel_size=(3, 3),
                    activation="relu",
                    padding="valid",
                    data_format="channels_last",
                )
            ),
            BatchNormalization(scale=False),
            MaxPooling2D(pool_size=(2, 2)),
            tfa.layers.WeightNormalization(
                Conv2D(
                    filters=64,
                    kernel_size=(3, 3),
                    activation="relu",
                    padding="valid",
                    data_format="channels_last",
                )
            ),
            BatchNormalization(scale=False),
            MaxPooling2D(pool_size=(2, 2)),
            tfa.layers.WeightNormalization(
                Conv2D(
                    filters=64,
                    kernel_size=(3, 3),
                    activation="relu",
                    padding="valid",
                    data_format="channels_last",
                )
            ),
            BatchNormalization(scale=False),
            Flatten(),
            tfa.layers.WeightNormalization(Dense(128, activation="relu")),
            tfa.layers.WeightNormalization(Dense(64, activation="relu")),
            tfa.layers.WeightNormalization(Dense(32, activation="relu")),
            tfa.layers.WeightNormalization(Dense(10, activation="softmax")),
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
            # tf.keras.layers.RandomRotation(0.2),
            # tf.keras.layers.RandomZoom(
            #     height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1)
            # ),
            tfa.layers.WeightNormalization(
                Conv2D(
                    input_shape=input_shape,
                    filters=32,
                    kernel_size=(3, 3),
                    activation="relu",
                    padding="valid",
                    data_format="channels_last",
                )
            ),
            BatchNormalization(scale=False),
            MaxPooling2D(pool_size=(2, 2)),
            tfa.layers.WeightNormalization(
                Conv2D(
                    filters=64,
                    kernel_size=(3, 3),
                    activation="relu",
                    padding="valid",
                    data_format="channels_last",
                )
            ),
            BatchNormalization(scale=False),
            MaxPooling2D(pool_size=(2, 2)),
            tfa.layers.WeightNormalization(
                Conv2D(
                    filters=64,
                    kernel_size=(3, 3),
                    activation="relu",
                    padding="valid",
                    data_format="channels_last",
                )
            ),
            BatchNormalization(scale=False),
            Flatten(),
            tfa.layers.WeightNormalization(Dense(128, activation="relu")),
            tfa.layers.WeightNormalization(Dense(64, activation="relu")),
            tfa.layers.WeightNormalization(Dense(32, activation="relu")),
            tfa.layers.WeightNormalization(
                Dense(tfpl.OneHotCategorical.params_size(10))
            ),
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


def get_convolutional_reparameterization_layer(
    filters, kernel_size, input_shape, divergence_fn, activation="relu"
):
    """
    This function should create an instance of a Convolution2DReparameterization
    layer according to the above specification.
    The function takes the input_shape and divergence_fn as arguments, which should
    be used to define the layer.
    Your function should then return the layer instance.
    """
    return tfpl.Convolution2DReparameterization(
        input_shape=input_shape,
        filters=filters,
        kernel_size=kernel_size,
        activation=activation,
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


def get_dense_variational_layer(
    units, prior_fn, posterior_fn, kl_weight, activation=None
):
    """
    This function should create an instance of a DenseVariational layer according
    to the above specification.
    The function takes the prior_fn, posterior_fn and kl_weight as arguments, which should
    be used to define the layer.
    Your function should then return the layer instance.
    """
    return tfpl.DenseVariational(
        units,
        make_prior_fn=prior_fn,
        make_posterior_fn=posterior_fn,
        kl_weight=kl_weight,
        activation=activation,
    )


# Build and compile the Bayesian CNN model
def get_bayesian_model(
    n, input_shape=(28, 28, 1), loss=nll, optimizer=RMSprop(), metrics=["accuracy"]
):
    # Create the layers
    divergence_fn = lambda q, p, _: tfd.kl_divergence(q, p) / n
    convolutional_reparameterization_layer = (
        lambda units, kernel, activation: get_convolutional_reparameterization_layer(
            units,
            kernel,
            input_shape=input_shape,
            divergence_fn=divergence_fn,
            activation=activation,
        )
    )
    dense_variational_layer = lambda units, activation: get_dense_variational_layer(
        units, get_prior, get_posterior, kl_weight=1 / n, activation=activation
    )

    bayesian_model = Sequential(
        [
            convolutional_reparameterization_layer(32, 2, "relu"),
            # BatchNormalization(),
            # tfa.layers.GroupNormalization(groups=8, axis=3),
            MaxPooling2D(pool_size=(6, 6)),
            convolutional_reparameterization_layer(32, 2, "relu"),
            # BatchNormalization(),
            # tfa.layers.GroupNormalization(groups=8, axis=3),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            # dense_variational_layer(32, "relu"),
            dense_variational_layer(10, None),
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


def get_model(model_str: str, n: int, optimizer):
    if model_str.lower() == "deterministic":
        return get_deterministic_model(
            input_shape=(28, 28, 1),
            loss=SparseCategoricalCrossentropy(),
            optimizer=optimizer,
            metrics=["accuracy"],
        )
    elif model_str.lower() == "probabilistic":
        return get_probabilistic_model(
            input_shape=(28, 28, 1), loss=nll, optimizer=optimizer, metrics=["accuracy"]
        )

    elif model_str.lower() == "bayesian":
        return get_bayesian_model(
            n=n,
            input_shape=(28, 28, 1),
            loss=nll,
            optimizer=optimizer,
            metrics=["accuracy"],
        )

    else:
        raise ValueError(f"{model_str} is unknown.")


def main(model_str: str, predict: bool = False):

    BATCH_SIZE = 128

    x, y = load_data("data/train.csv")
    x = x[..., np.newaxis]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.40, shuffle=True, stratify=y
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_test, y_test, test_size=0.5, shuffle=True, stratify=y_test
    )

    y_train_oh = tf.keras.utils.to_categorical(y_train)
    y_val_oh = tf.keras.utils.to_categorical(y_val)

    if model_str.lower() == "deterministic":
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_ds = train_ds.shuffle(buffer_size=50000).batch(BATCH_SIZE)

        val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_ds = val_ds.batch(BATCH_SIZE)
    else:
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train_oh))
        train_ds = train_ds.shuffle(buffer_size=50000).batch(BATCH_SIZE)

        val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val_oh))
        val_ds = val_ds.batch(BATCH_SIZE)

    steps_per_epoch = len(x_train) // BATCH_SIZE

    clr = tfa.optimizers.CyclicalLearningRate(
        initial_learning_rate=1e-4,
        maximal_learning_rate=1e-2,
        scale_fn=lambda x: 1 / (2.0 ** (x - 1)),
        step_size=2 * steps_per_epoch,
    )
    optimizer = tf.keras.optimizers.SGD(clr, momentum=0.99, nesterov=True)

    # option 1: no gradient centralization (gcz)
    # opt = Adam(learning_rate=1e-4)

    # option 2: with gradient centralization (gcz)
    # opt = GradientCentralization(learning_rate=1e-4)

    # option 3: with gcz + lookahead
    # gcz = GradientCentralization(learning_rate=1e-4)
    # opt = Lookahead(gcz, sync_period=6, slow_step_size=0.5)

    # sgd = tf.keras.optimizers.SGD(0.01)
    # moving_avg_sgd = tfa.optimizers.MovingAverage(sgd)
    # stocastic_avg_sgd = tfa.optimizers.SWA(sgd)

    # Create the model
    model = get_model(model_str=model_str, n=x_train.shape[0], optimizer=RMSprop())

    # Print the model summary
    # model.summary()

    # Callbacks
    # checkpoint_path = "./training/cp-{epoch:04d}.ckpt"
    # checkpoint_dir = os.path.dirname(checkpoint_path)
    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    #     monitor="val_loss", factor=0.1, patience=5, min_lr=0.00001
    # )
    # # Callback
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=checkpoint_dir, save_weights_only=True, verbose=1
    # )
    # avg_callback = tfa.callbacks.AverageModelCheckpoint(
    #     filepath=checkpoint_dir, update_weights=True
    # )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    callbacks = [
        early_stop,
        # reduce_lr,
    ]
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=200,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=True,
    )

    hist_df = pd.DataFrame(hist.history)
    hist_df.plot(figsize=(12, 8), subplots=True)
    plt.show()

    # Evaluate the model
    if model_str.lower() == "bayesian":
        y_pred_sum = np.zeros((y_test.shape[0], 10, 100))
        for i in range(100):
            y_pred = model(x_test)
            y_pred_sum[:, :, i] = y_pred.mode().numpy()
        y_hat = np.mean(y_pred_sum, axis=-1)
        y_hat = y_hat.argmax(axis=-1)
        equality = tf.math.equal(y_hat, y_test)
        accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))
        print("Accuracy on MNIST test set: ", accuracy.numpy())
    elif model_str.lower() == "probabilistic":
        y_hat = tf.math.reduce_max(model(x_test), axis=-1)
        equality = tf.math.equal(y_hat, y_test)
        accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))
        print("Accuracy on MNIST test set: ", accuracy.numpy())
    else:
        print(
            "Accuracy on MNIST test set: ",
            str(model.evaluate(x_test, y_test, verbose=False)[1]),
        )

    if predict:

        x_test, y_test = load_data("data/test.csv")
        x_test = x_test[..., np.newaxis]

        if model_str.lower() == "bayesian":
            y_test_sum = np.zeros((x_test.shape[0], 10, 100))
            for i in range(100):
                y_test = model(x_test)
                y_test_sum[:, :, i] = y_test.mode().numpy()
            y_test = np.mean(y_test_sum, axis=-1)
            y_test = y_test.argmax(axis=-1)
        elif model_str.lower() == "probabilistic":
            y_test = model(x_test)
            y_test = y_test.mode().numpy().argmax(axis=-1)
        else:
            y_test = model(x_test).numpy().argmax(axis=-1)

        df = pd.DataFrame(
            {
                "ImageId": [i + 1 for i in range(y_test.shape[0])],
                "Label": y_test,
            }
        )
        df.to_csv(f"data/predictions_{model_str}.csv", index=False)
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
