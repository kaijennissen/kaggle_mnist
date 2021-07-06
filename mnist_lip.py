import numpy as np
from deel.lip.activations import GroupSort
from deel.lip.layers import (
    FrobeniusDense,
    ScaledL2NormPooling2D,
    SpectralConv2D,
    SpectralDense,
)
from deel.lip.losses import HKR_multiclass_loss
from deel.lip.model import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Sequential (resp Model) from deel.model has the same properties as any lipschitz model.
# It act only as a container, with features specific to lipschitz
# functions (condensation, vanilla_exportation...)
model = Sequential(
    [
        Input(shape=(28, 28, 1)),
        # Lipschitz layers preserve the API of their superclass ( here Conv2D )
        # an optional param is available: k_coef_lip which control the lipschitz
        # constant of the layer
        SpectralConv2D(
            filters=16,
            kernel_size=(3, 3),
            activation=GroupSort(2),
            use_bias=True,
            kernel_initializer="orthogonal",
        ),
        # usual pooling layer are implemented (avg, max...), but new layers are also available
        ScaledL2NormPooling2D(pool_size=(2, 2), data_format="channels_last"),
        SpectralConv2D(
            filters=16,
            kernel_size=(3, 3),
            activation=GroupSort(2),
            use_bias=True,
            kernel_initializer="orthogonal",
        ),
        ScaledL2NormPooling2D(pool_size=(2, 2), data_format="channels_last"),
        # our layers are fully interoperable with existing keras layers
        Flatten(),
        SpectralDense(
            32,
            activation=GroupSort(2),
            use_bias=True,
            kernel_initializer="orthogonal",
        ),
        FrobeniusDense(
            10, activation=None, use_bias=False, kernel_initializer="orthogonal"
        ),
    ],
    # similary model has a parameter to set the lipschitz constant
    # to set automatically the constant of each layer
    k_coef_lip=1.0,
    name="hkr_model",
)

# HKR (Hinge-Krantorovich-Rubinstein) optimize robustness along with accuracy
model.compile(
    # decreasing alpha and increasing min_margin improve robustness (at the cost of accuracy)
    # note also in the case of lipschitz networks, more robustness require more parameters.
    loss=HKR_multiclass_loss(alpha=25, min_margin=0.25),
    optimizer=Adam(lr=0.005),
    metrics=["accuracy"],
)

model.summary()

# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# standardize and reshape the data
x_train = np.expand_dims(x_train, -1)
mean = x_train.mean()
std = x_train.std()
x_train = (x_train - mean) / std
x_test = np.expand_dims(x_test, -1)
x_test = (x_test - mean) / std
# one hot encode the labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# fit the model
model.fit(
    x_train,
    y_train,
    batch_size=256,
    epochs=15,
    validation_data=(x_test, y_test),
    shuffle=True,
)
