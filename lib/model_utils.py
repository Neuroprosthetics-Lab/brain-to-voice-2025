# ==================================================================================
# Model utilities for brain-to-voice
# Maitreyee Wairagkar, 2025
# ==================================================================================

import os
from typing import Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam

from models import brain2voice as b2v


def build_model(
    cfg: dict, token_batch_X, token_batch_Y, token_sess_num, n_sessions: int
):
    """
    Initialise and compile the Voiceformer model inside a MirroredStrategy
    scope for distributed training

    Args:
        cfg            : full parsed config dict containing 'model' and 'training' sections
        token_batch_X  : sample input batch used to infer input shape and warm up sub-layers
        token_batch_Y  : sample target batch used to infer output dimensionality
        token_sess_num : sample session number array used for the warm-up forward pass
        n_sessions     : total number of recording sessions, one day embedding layer per session

    Returns:
        b2voice_model : compiled Voiceformer model ready for training
    """
    mdl_cfg = cfg["model"]
    train_cfg = cfg["training"]

    mirrored_strategy = tf.distribute.MirroredStrategy()

    with mirrored_strategy.scope():

        base_transformer_model = b2v.base_transformer(
            transformer_input_shape=tuple(mdl_cfg["transformer_input_shape"]),
            output_dim=token_batch_Y.shape[-1],
            head_size=mdl_cfg["head_size"],
            n_heads=mdl_cfg["n_heads"],
            n_transformer_blocks=mdl_cfg["n_transformer_blocks"],
            mlp_units=mdl_cfg["mlp_units"],
            dropout=mdl_cfg["dropout"],
        )
        base_transformer_model.trainable = True

        adam = Adam(
            beta_1=train_cfg["adam_beta_1"],
            beta_2=train_cfg["adam_beta_2"],
            epsilon=train_cfg["adam_epsilon"],
            learning_rate=train_cfg["learning_rate"],
        )
        acc = tf.keras.metrics.Accuracy()

        b2voice_model = b2v.Voiceformer(
            base_transformer_model, n_sessions=n_sessions
        )

        # Warm-up call to build all sub-layers
        _ = b2voice_model([token_batch_X, token_sess_num])

    b2voice_model.compile(
        optimizer=adam,
        loss=tf.keras.losses.Huber(delta=train_cfg["huber_delta"]),
        metrics=[acc],
    )
    b2voice_model.summary()

    return b2voice_model


def load_model_weights(b2voice_model: keras.models.Model, load_model_path: Optional[str]):
    """
    Copy weights from a previously saved model into the current model.
    If load_model_path is None, training starts from scratch. New day
    embedding layers beyond those in the loaded model are initialized
    from the last available embedding layer.

    Args:
        b2voice_model   : current Voiceformer model to receive the copied weights
        load_model_path : path to a previously saved model in TF SavedModel format,
                        or None to skip weight loading
    """

    if not load_model_path:
        print("No previous model specified - training from scratch")
        return

    print(f"Loading initial weights from: {load_model_path}")
    load_mod = tf.keras.models.load_model(load_model_path)
    load_mod.summary()

    b2voice_model.base_transformer.set_weights(load_mod.base_transformer.get_weights())

    load_mod_len = len(load_mod.day_embedding)
    curr_mod_len = len(b2voice_model.day_embedding)
    for i in range(curr_mod_len):
        if i < load_mod_len:
            b2voice_model.day_embedding[i].set_weights(
                load_mod.day_embedding[i].get_weights()
            )
        else:
            print(f"  Initialising new day embedding layer {i} from last layer of previous model.")
            b2voice_model.day_embedding[i].set_weights(
                load_mod.day_embedding[-1].get_weights()
            )


def save_inference_model(b2voice_model, token_batch_X, save_path_base: str, day_embed_layer: int = -1):
    """
    Build and save a lightweight inference model with the given day embedding
    layer pre-selected, eliminating the need to pass a session number at
    inference time

    Args:
        b2voice_model   : trained Voiceformer model containing day embeddings and base transformer
        token_batch_X   : sample input batch used to infer the input shape
        save_path_base  : base path for saving the training model; the inference
                          model is saved to the same path with a .h5 extension
        day_embed_layer : index of the day embedding layer to pre-select for inference (default is the last session layer)
    """

    inp = keras.Input(shape=token_batch_X.shape[1:])
    x = tf.nn.avg_pool1d(inp, ksize=2, strides=2, padding="VALID")
    x = b2voice_model.day_embedding[day_embed_layer](x)
    x = b2voice_model.base_transformer(x)
    inference_model = keras.Model(inp, x)
    inference_model.summary()

    h5_path = os.path.splitext(save_path_base)[0] + ".h5"
    inference_model.save(h5_path)
    print(f"Inference model saved to: {h5_path}")