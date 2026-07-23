# ==================================================================================
# Brain-to-voice model
# Maitreyee Wairagkar, 2025
# ==================================================================================

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@tf.function
def positional_encoding(
    sequence_length: int,
    embed_dim: int,
) -> np.ndarray:
    """
    Compute sinusoidal positional encodings for transformer input sequences

    Args:
        sequence_length : number of time steps in the input sequence
        embed_dim       : embedding dimensionality of the input features

    Returns:
        position_encodings : sinusoidal positional encoding array of shape
                            (sequence_length, embed_dim) in float32
    """
    position_encodings = np.empty(shape=[sequence_length, embed_dim])

    for pos in range(sequence_length):
        for i in range(embed_dim):

            if i % 2 == 0: # even feature num
                position_encodings[pos, i] = np.sin(
                    pos / (10000 ** (2 * i / embed_dim))
                )
            else:
                position_encodings[pos, i] = np.cos(
                    pos / (10000 ** (2 * i / embed_dim))
                )

    return position_encodings.astype("float32")


def transformer_encoder(
    inputs: tf.Tensor,
    head_size: int,
    n_heads: int,
    dropout: float = 0,
) -> tf.Tensor:
    """
    Single transformer encoder block with multi-head self-attention, a
    feed-forward network, residual connections, and layer normalisation

    Args:
        inputs     : input tensor of shape (batch, time, features)
        head_size  : dimensionality of each attention head
        n_heads    : number of parallel attention heads
        dropout    : dropout rate applied after attention and dense layers

    Returns:
        x : encoded output tensor of same shape as inputs
    """
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)  # Pre-layer norm 1

    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=n_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)

    res = x + inputs  # residual connection 1
    x = layers.LayerNormalization(epsilon=1e-6)(res)  # layer norm 2

    # Feed forward network
    x = layers.Dense(head_size * 2, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(head_size, activation="relu")(x)
    x = layers.Dropout(dropout)(x)

    x = x + res  # residual connection 2
    x = layers.LayerNormalization(epsilon=1e-6)(x)  # Post-layer norm 3

    return x


def day_embedding_model(
    input_shape: tuple[int, int],
    l1_shape: int,
    l2_shape: int,
) -> tf.keras.Sequential:
    """
    Build a session-specific day embedding network that transforms neural
    inputs into a common latent space before the base transformer

    Args:
        input_shape : shape of the input tensor excluding batch dimension
        l1_shape    : number of units in the first dense layer
        l2_shape    : number of units in the second dense layer

    Returns:
        day_model : trainable two-layer dense sequential model
    """

    day_model = tf.keras.Sequential()
    day_model.add(keras.Input(shape=input_shape))
    day_model.add(layers.Dense(l1_shape, activation="relu"))
    day_model.add(layers.Dense(l2_shape, activation="relu"))
    day_model.trainable = True

    return day_model


def base_transformer(
    transformer_input_shape: tuple[int, ...],
    output_dim: int,
    head_size: int,
    n_heads: int,
    n_transformer_blocks: int,
    mlp_units: list[int],
    dropout: float = 0,
) -> tf.keras.Model:
    """
    Build the base transformer model with positional encoding, stacked encoder
    blocks with cross-block residual connections, and a final MLP head

    Args:
        transformer_input_shape : shape of the input tensor excluding batch dimension
        output_dim              : number of output features (LPCNet feature dimensions)
        head_size               : dimensionality of each attention head
        n_heads                 : number of parallel attention heads
        n_transformer_blocks    : number of stacked transformer encoder blocks
        mlp_units               : list of units for each dense layer in the MLP head
        dropout                 : dropout rate applied throughout the model

    Returns:
        model : compiled Keras functional model mapping input to output_dim predictions
    """

    inputs = keras.Input(shape=transformer_input_shape)
    x = inputs

    # Add positional encoding
    x = x + positional_encoding(
        sequence_length=transformer_input_shape[0], embed_dim=x.shape[-1]
    )

    # Transformer encoder
    block_outputs = [x]  # initialise list of block outputs for residual connection
    for i in range(n_transformer_blocks):
        if i > 0:
            x = x + block_outputs[i - 1]  # residual connection from previous encoder block output
        x = transformer_encoder(x, head_size, n_heads, dropout)
        block_outputs.append(x)
        
    # Norm layer after all transformer blocks
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    # Dense layers at the end
    x = tf.squeeze(
        tf.nn.avg_pool1d(
            x, transformer_input_shape[0], transformer_input_shape[0], padding="VALID"
        ),
        axis=1,
    )

    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)

    outputs = layers.Dense(output_dim)(x)

    return keras.Model(inputs, outputs)


class Voiceformer(keras.Model):
    """
    Brain-to-voice decoder model combining session-specific day embedding layers
    with a shared base transformer to predict LPCNet speech features from neural data
    """

    def __init__(
        self,
        base_transformer: tf.keras.Model,
        n_sessions: int,
    ):
        """
        Initialise the Voiceformer with one day embedding layer per session
        and a shared base transformer

        Args:
            base_transformer        : shared transformer model applied after day-specific embedding
            n_sessions              : total number of recording sessions, one embedding layer per session
        """
        super().__init__()

        self.n_sessions = n_sessions

        # Add day embedding layers
        self.day_embedding = []
        for i in range(self.n_sessions):
            self.day_embedding.append(
                day_embedding_model(input_shape=(30, 512), l1_shape=512, l2_shape=128)
            )  # embedding_layer)
            self.day_embedding[i].summary()

        # The base transformer
        self.base_transformer = base_transformer
        self.base_transformer.summary()

        # define a list of all trainable variables for optimization
        self.list_embedding_trainable_vars = []
        for i in range(self.n_sessions):
            self.list_embedding_trainable_vars.extend(
                self.day_embedding[i].trainable_variables
            )

        self.base_trans_trainable_vars = self.base_transformer.trainable_variables

    def call(
        self,
        inputs: list[tf.Tensor],
        training: bool = False,
    ) -> tf.Tensor:
        """
        Forward pass through the Voiceformer. Selects the appropriate day embedding
        layer based on session number, then passes through the base transformer.

        Args:
            inputs : list of [neural_inputs, session_num] where neural_inputs has shape
                    (batch, time, channels) and session_num is a scalar session index
            training : whether to run in training mode

        Returns:
            output : predicted LPCNet features of shape (batch, output_dim)
        """

        x = inputs
        neural_inputs, session_num = x[0], x[1][0, 0]

        inputs = tf.nn.avg_pool1d(neural_inputs, ksize=2, strides=2, padding="VALID")

        # input network selector for transforming data for each day
        inputTransformSelector = {
            i: lambda i=i: self.day_embedding[i](inputs, training=False)
            for i in range(self.n_sessions)
        }

        # day-specific transform and the base transformer
        inputTransformedFeatures = tf.switch_case(session_num, inputTransformSelector)

        return self.base_transformer(inputTransformedFeatures, training=training)


    def train_step(
        self,
        data: tuple[list[tf.Tensor], tf.Tensor],
    ) -> dict[str, tf.Tensor]:
        """
        Custom training step with separate gradient updates for the base transformer
        and session-specific day embedding layers. Loss is upweighted for high
        frequency LPCNet features (indices 4-10).

        Args:
            data : tuple of (x, y) where x is [neural_inputs, session_num] and
                y is the target LPCNet feature array of shape (batch, output_dim)

        Returns:
            metrics : dict mapping metric names to their current values for this step
        """

        # Unpack the data (x, y and sess_num)
        x, y = data
        neural_inputs, session_num = x[0], x[1][0, 0]

        inputs = tf.nn.avg_pool1d(neural_inputs, ksize=2, strides=2, padding="VALID")

        # input network selector for transforming data for each day
        inputTransformSelector = {}
        for i in range(self.n_sessions):
            train = tf.cond(
                tf.constant(i, dtype=tf.int32) == session_num,
                lambda: True,
                lambda: False,
            )
            inputTransformSelector[i] = lambda i=i: self.day_embedding[i](
                inputs, training=train
            )

        with tf.GradientTape() as tape:

            # day-specific transform and the base transformer
            inputTransformedFeatures = tf.switch_case(
                session_num, inputTransformSelector
            )
            y_pred = self.base_transformer(inputTransformedFeatures, training=True)

            # Compute the loss value (upweight high frequency features). Weights are determined based on hyperparameter tuning. 
            loss = self.compute_loss(y=y, y_pred=y_pred) + 1.5 * self.compute_loss(
                y=y[:, 4:10], y_pred=y_pred[:, 4:10]
            )

        # Compute gradients
        trainable_vars = []
        trainable_vars.extend(self.base_trans_trainable_vars)
        trainable_vars.extend(self.list_embedding_trainable_vars)

        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(
            (grd, var)
            for (grd, var) in zip(gradients, trainable_vars)
            if grd is not None
        )

        # Update metrics
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
                print(f"{metric.name}: {metric.result()}")

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
