#!/usr/bin/env python3
"""
Quantization-Aware Training (QAT) for GhostFaceNet.

This script performs QAT to improve INT8 quantization precision for GhostFaceNet.
Instead of using ArcFace loss (which requires identity labels), we use a simpler
knowledge distillation approach:

Strategy: Embedding Distillation QAT
- Teacher: Float32 GhostFaceNet (original)
- Student: QAT-enabled GhostFaceNet
- Loss: MSE between teacher and student embeddings
- Goal: Make quantized embeddings match float32 embeddings

This approach:
1. Doesn't require identity labels (only face images)
2. Directly optimizes for quantization consistency
3. Uses the same calibration images as PTQ

Usage:
    # Basic training
    python qat_ghostfacenet.py

    # With more epochs
    python qat_ghostfacenet.py --epochs 10

    # Skip Vela compilation
    python qat_ghostfacenet.py --skip-vela

Requirements:
    - tensorflow-model-optimization
    - Calibration images in calibration_data/emb_112/
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

print("=" * 60)
print("GhostFaceNet Quantization-Aware Training (QAT)")
print("=" * 60)

# Check dependencies
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
except ImportError:
    print("Please install tensorflow: pip install tensorflow")
    sys.exit(1)

try:
    import tensorflow_model_optimization as tfmot
    print(f"TF-MOT version: {tfmot.__version__}")
except ImportError:
    print("Installing tensorflow-model-optimization...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow-model-optimization"])
    import tensorflow_model_optimization as tfmot

try:
    import cv2
except ImportError:
    print("Please install opencv-python: pip install opencv-python")
    sys.exit(1)


# Default paths
DEFAULT_H5 = "GN_W0.5_S2_ArcFace_epoch16.h5"
DEFAULT_CALIB_DIR = "calibration_data/qat_112"  # Use QAT-specific data directory
FALLBACK_CALIB_DIR = "calibration_data/emb_112"  # Fallback if QAT data not prepared


def load_teacher_model(h5_path: str) -> tf.keras.Model:
    """
    Load the float32 teacher model.

    Args:
        h5_path: Path to GhostFaceNet H5 model

    Returns:
        Float32 Keras model
    """
    print(f"\nLoading teacher model: {h5_path}")

    # Set float32 policy
    tf.keras.mixed_precision.set_global_policy('float32')

    # Load model
    model = tf.keras.models.load_model(h5_path, compile=False)

    # Convert to float32 if needed
    config = model.get_config()

    def clean_config(cfg):
        if isinstance(cfg, dict):
            if 'dtype' in cfg:
                cfg['dtype'] = 'float32'
            if 'dtype_policy' in cfg:
                cfg['dtype_policy'] = 'float32'
            for v in cfg.values():
                clean_config(v)
        elif isinstance(cfg, list):
            for item in cfg:
                clean_config(item)

    clean_config(config)

    new_model = tf.keras.Model.from_config(config)
    new_model.set_weights([w.astype(np.float32) for w in model.get_weights()])

    print(f"  Input shape: {new_model.input_shape}")
    print(f"  Output shape: {new_model.output_shape}")
    print(f"  Parameters: {new_model.count_params():,}")

    return new_model


class PReLUQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    """Custom quantization config for PReLU layer."""

    def get_weights_and_quantizers(self, layer):
        # Quantize the alpha parameter
        return [(layer.alpha, tfmot.quantization.keras.quantizers.LastValueQuantizer(
            num_bits=8, symmetric=True, narrow_range=False, per_axis=False))]

    def get_activations_and_quantizers(self, layer):
        return []

    def set_quantize_weights(self, layer, quantize_weights):
        layer.alpha = quantize_weights[0]

    def set_quantize_activations(self, layer, quantize_activations):
        pass

    def get_output_quantizers(self, layer):
        return [tfmot.quantization.keras.quantizers.MovingAverageQuantizer(
            num_bits=8, symmetric=False, narrow_range=False, per_axis=False)]

    def get_config(self):
        return {}


class NoOpQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    """Config that doesn't quantize the layer but adds output quantizers."""

    def get_weights_and_quantizers(self, layer):
        return []

    def get_activations_and_quantizers(self, layer):
        return []

    def set_quantize_weights(self, layer, quantize_weights):
        pass

    def set_quantize_activations(self, layer, quantize_activations):
        pass

    def get_output_quantizers(self, layer):
        return [tfmot.quantization.keras.quantizers.MovingAverageQuantizer(
            num_bits=8, symmetric=False, narrow_range=False, per_axis=False)]

    def get_config(self):
        return {}


def create_qat_model(model: tf.keras.Model) -> tf.keras.Model:
    """
    Create a QAT-enabled model from the float32 model.

    Args:
        model: Float32 Keras model

    Returns:
        QAT-enabled model with fake quantization nodes
    """
    print("\nCreating QAT model...")

    # Try annotating all layers including PReLU with custom config
    quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer
    quantize_apply = tfmot.quantization.keras.quantize_apply
    quantize_scope = tfmot.quantization.keras.quantize_scope

    def apply_quantization(layer):
        # Quantize all key layers
        if isinstance(layer, (tf.keras.layers.Conv2D,
                              tf.keras.layers.Dense,
                              tf.keras.layers.DepthwiseConv2D)):
            return quantize_annotate_layer(layer)
        # Use custom config for PReLU
        elif isinstance(layer, tf.keras.layers.PReLU):
            return quantize_annotate_layer(layer, quantize_config=PReLUQuantizeConfig())
        # Add output quantizers for BatchNormalization and Add
        elif isinstance(layer, (tf.keras.layers.BatchNormalization,
                                tf.keras.layers.Add,
                                tf.keras.layers.Multiply)):
            return quantize_annotate_layer(layer, quantize_config=NoOpQuantizeConfig())
        return layer

    print("  Annotating layers for QAT...")
    annotated_model = tf.keras.models.clone_model(
        model,
        clone_function=apply_quantization
    )

    # Apply quantization with custom scope
    with quantize_scope({
        'PReLUQuantizeConfig': PReLUQuantizeConfig,
        'NoOpQuantizeConfig': NoOpQuantizeConfig,
    }):
        qat_model = quantize_apply(annotated_model)

    print("  QAT model created with custom PReLU quantization")

    # Compile for training
    qat_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='mse'  # Will use embedding MSE loss
    )

    print(f"  QAT model parameters: {qat_model.count_params():,}")

    return qat_model


def load_training_images(calib_dir: str, max_images: int = 13000) -> np.ndarray:
    """
    Load training images for QAT.

    Args:
        calib_dir: Directory containing calibration images
        max_images: Maximum number of images to load

    Returns:
        Numpy array of images, normalized to [-1, 1]
    """
    calib_path = Path(calib_dir)

    if not calib_path.exists():
        # Try fallback directory
        if calib_dir == DEFAULT_CALIB_DIR:
            fallback_path = Path(FALLBACK_CALIB_DIR)
            if fallback_path.exists():
                print(f"QAT data directory not found, using fallback: {FALLBACK_CALIB_DIR}")
                print(f"  Tip: Run 'python prepare_calibration_data.py --qat' for more training data")
                calib_path = fallback_path
            else:
                print(f"ERROR: No calibration directory found!")
                print(f"  Expected: {calib_dir} or {FALLBACK_CALIB_DIR}")
                print(f"  Run: python prepare_calibration_data.py --qat")
                sys.exit(1)
        else:
            print(f"ERROR: Calibration directory not found: {calib_dir}")
            sys.exit(1)

    image_files = sorted(calib_path.glob("*.jpg"))[:max_images]

    if len(image_files) < 100:
        print(f"ERROR: Not enough training images ({len(image_files)} found, need 100+)")
        sys.exit(1)

    print(f"\nLoading {len(image_files)} training images...")

    images = []
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[:2] != (112, 112):
            img = cv2.resize(img, (112, 112))
        # ArcFace normalization: [-1, 1]
        img = (img.astype(np.float32) - 127.5) / 128.0
        images.append(img)

    images = np.array(images, dtype=np.float32)
    print(f"  Loaded {len(images)} images")
    print(f"  Shape: {images.shape}")
    print(f"  Range: [{images.min():.2f}, {images.max():.2f}]")

    return images


def augment_image(image):
    """
    Apply moderate data augmentation for QAT training.

    Moderate augmentation is preferred for knowledge distillation because:
    - Goal is to match teacher output, not learn new features
    - Strong augmentation may produce unstable teacher outputs
    """
    # Horizontal flip - safe for faces (symmetric)
    image = tf.image.random_flip_left_right(image)
    # Slight brightness variation (±10%)
    image = tf.image.random_brightness(image, 0.1)
    # Slight contrast variation (0.9-1.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    # Keep values in valid range [-1, 1]
    image = tf.clip_by_value(image, -1.0, 1.0)
    return image


def get_activation_layers(model: tf.keras.Model):
    """
    Get list of layers to monitor for activation range.

    Returns list of layer names that should be penalized for large activations.
    """
    monitored_layers = []

    for layer in model.layers:
        # Monitor activation layers (including PReLU), conv outputs, and add layers
        layer_type = layer.__class__.__name__
        layer_name = layer.name.lower()

        # Check for activation-related layers
        if any(x in layer_type.lower() for x in ['activation', 'prelu', 'relu']):
            monitored_layers.append(layer.name)
        # Also monitor depthwise conv outputs (these have large ranges in the original model)
        elif 'depthwise' in layer_name:
            monitored_layers.append(layer.name)
        # Monitor layers with 'quant_activation' in name (QAT wrapped layers)
        elif 'quant_activation' in layer_name or 'quant_prelu' in layer_name:
            monitored_layers.append(layer.name)

    return monitored_layers


def create_activation_extractor(model: tf.keras.Model, layer_names: list):
    """
    Create a function that extracts activations from specified layers.

    For QAT models, we need to carefully extract layer outputs without
    creating a new model graph (which causes disconnected graph issues).

    Returns:
        A function that takes input and returns list of activation tensors
    """
    # Build a dict mapping layer name to layer for faster lookup
    layer_dict = {layer.name: layer for layer in model.layers}

    # Find output tensors for monitored layers
    outputs = []
    valid_names = []

    for name in layer_names:
        if name in layer_dict:
            try:
                layer = layer_dict[name]
                # Try to get the output tensor
                if hasattr(layer, 'output'):
                    outputs.append(layer.output)
                    valid_names.append(name)
            except Exception:
                pass

    return outputs, valid_names


def train_qat(
    teacher_model: tf.keras.Model,
    qat_model: tf.keras.Model,
    train_images: np.ndarray,
    epochs: int = 50,
    batch_size: int = 64,
    warmup_epochs: int = 3,
    patience: int = 5,
    use_augmentation: bool = True,
    activation_penalty_weight: float = 0.001,
    activation_limit: float = 30.0
) -> tf.keras.Model:
    """
    Train QAT model using knowledge distillation with cosine similarity loss.

    NEW: Adds activation range constraint to penalize large activation values,
    which improves INT8 quantization precision.

    Args:
        teacher_model: Float32 teacher model
        qat_model: QAT student model
        train_images: Training images
        epochs: Number of training epochs
        batch_size: Batch size
        warmup_epochs: Number of warmup epochs
        patience: Early stopping patience
        use_augmentation: Whether to use data augmentation
        activation_penalty_weight: Weight for activation range penalty (default: 0.001)
        activation_limit: Soft limit for activation values (default: 30.0)

    Returns:
        Trained QAT model
    """
    print(f"\nStarting QAT training...")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Warmup epochs: {warmup_epochs}")
    print(f"  Early stopping patience: {patience}")
    print(f"  Data augmentation: {use_augmentation}")
    print(f"  Training samples: {len(train_images)}")
    print(f"  Activation penalty weight: {activation_penalty_weight}")
    print(f"  Activation limit: {activation_limit}")

    # Create dataset with full shuffle
    # NOTE: For augmentation, we compute teacher embeddings dynamically per batch
    # This ensures teacher and student see the SAME augmented image
    dataset = tf.data.Dataset.from_tensor_slices(train_images)
    dataset = dataset.shuffle(buffer_size=len(train_images))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # Learning rate schedule: warmup + cosine decay
    initial_lr = 5e-5      # Starting LR (during warmup)
    peak_lr = 2e-4         # Peak LR after warmup
    final_lr = 1e-6        # Final LR

    # Use legacy optimizer for M1/M2 Mac compatibility
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=initial_lr)

    # Embedding range penalty
    # Instead of monitoring all internal layers (which is complex with QAT),
    # we penalize the output embedding values directly.
    # This indirectly encourages the network to keep internal activations bounded.
    print("\n  Setting up embedding range penalty...")
    print(f"  Embedding limit: {activation_limit}")
    print(f"  Penalty weight: {activation_penalty_weight}")
    use_embedding_penalty = activation_penalty_weight > 0

    # Convert limits to tensors for tf.function
    emb_limit = tf.constant(activation_limit, dtype=tf.float32)
    emb_penalty_weight = tf.constant(activation_penalty_weight, dtype=tf.float32)

    # Augmentation function for tf.data.Dataset
    @tf.function
    def apply_augmentation(images):
        """Apply augmentation to a batch of images."""
        return tf.map_fn(augment_image, images)

    @tf.function
    def compute_embedding_penalty(embeddings):
        """
        Compute penalty for embedding values exceeding the soft limit.

        Uses a smooth penalty: penalty = mean(max(0, |embedding| - limit)^2)
        This penalizes embedding values outside [-limit, limit] quadratically.

        By penalizing large embedding values, we indirectly encourage
        the network to keep internal activations bounded, which helps
        with INT8 quantization.
        """
        # Penalty for values exceeding the limit
        excess = tf.maximum(0.0, tf.abs(embeddings) - emb_limit)
        penalty = tf.reduce_mean(tf.square(excess))
        return penalty

    @tf.function
    def train_step_with_penalty(images, targets):
        with tf.GradientTape() as tape:
            predictions = qat_model(images, training=True)

            # L2 normalize embeddings for similarity computation
            pred_norm = tf.nn.l2_normalize(predictions, axis=-1)
            target_norm = tf.nn.l2_normalize(targets, axis=-1)

            # Pure Cosine Similarity Loss: 1 - cosine_similarity
            cosine_sim = tf.reduce_sum(pred_norm * target_norm, axis=-1)
            cosine_loss = tf.reduce_mean(1.0 - cosine_sim)

            # Compute embedding range penalty (on raw predictions)
            embedding_penalty = compute_embedding_penalty(predictions)

            # Loss: Pure Cosine Similarity + embedding penalty
            loss = cosine_loss + emb_penalty_weight * embedding_penalty

        gradients = tape.gradient(loss, qat_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, qat_model.trainable_variables))
        return loss, tf.reduce_mean(cosine_sim), embedding_penalty

    @tf.function
    def train_step_no_penalty(images, targets):
        with tf.GradientTape() as tape:
            predictions = qat_model(images, training=True)

            # L2 normalize embeddings
            pred_norm = tf.nn.l2_normalize(predictions, axis=-1)
            target_norm = tf.nn.l2_normalize(targets, axis=-1)

            # Pure Cosine Similarity Loss: 1 - cosine_similarity
            cosine_sim = tf.reduce_sum(pred_norm * target_norm, axis=-1)
            cosine_loss = tf.reduce_mean(1.0 - cosine_sim)

            # Loss: Pure Cosine Similarity
            loss = cosine_loss

        gradients = tape.gradient(loss, qat_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, qat_model.trainable_variables))
        return loss, tf.reduce_mean(cosine_sim), tf.constant(0.0)

    # Choose train step function based on whether we use embedding penalty
    train_step = train_step_with_penalty if use_embedding_penalty else train_step_no_penalty

    # Training state for early stopping
    best_similarity = 0.0
    best_weights = None
    patience_counter = 0
    min_improvement = 0.005

    # Training with warmup + cosine decay
    for epoch in range(epochs):
        # Learning rate schedule: warmup then cosine decay
        if epoch < warmup_epochs:
            # Linear warmup
            current_lr = initial_lr + (peak_lr - initial_lr) * (epoch / warmup_epochs)
        else:
            # Cosine decay after warmup
            decay_epochs = epochs - warmup_epochs
            decay_progress = (epoch - warmup_epochs) / max(decay_epochs - 1, 1)
            current_lr = final_lr + 0.5 * (peak_lr - final_lr) * (1 + np.cos(np.pi * decay_progress))
        optimizer.learning_rate.assign(current_lr)

        print(f"\nEpoch {epoch + 1}/{epochs} (lr={current_lr:.2e})")
        epoch_losses = []
        epoch_similarities = []
        epoch_act_penalties = []

        for step, images in enumerate(dataset):
            # Apply augmentation if enabled
            if use_augmentation:
                images_aug = apply_augmentation(images)
            else:
                images_aug = images

            # Compute teacher embeddings for this batch (with augmentation)
            # This ensures teacher and student see the SAME augmented images
            targets = teacher_model(images_aug, training=False)

            loss, cosine_sim, act_penalty = train_step(images_aug, targets)
            epoch_losses.append(float(loss))
            epoch_similarities.append(float(cosine_sim))
            epoch_act_penalties.append(float(act_penalty))

            if (step + 1) % 50 == 0:
                avg_sim = np.mean(epoch_similarities[-50:])
                avg_penalty = np.mean(epoch_act_penalties[-50:])
                print(f"  Step {step + 1}, Loss: {loss:.6f}, Sim: {avg_sim:.4f}, EmbPenalty: {avg_penalty:.4f}")

        avg_loss = np.mean(epoch_losses)
        avg_similarity = np.mean(epoch_similarities)
        avg_act_penalty = np.mean(epoch_act_penalties)
        print(f"  Epoch {epoch + 1} - Loss: {avg_loss:.6f}, Similarity: {avg_similarity:.4f}, EmbPenalty: {avg_act_penalty:.4f}")

        # Early stopping check based on similarity
        if avg_similarity > best_similarity + min_improvement:
            best_similarity = avg_similarity
            best_weights = qat_model.get_weights()
            patience_counter = 0
            print(f"  [BEST] New best similarity: {best_similarity:.4f}")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epoch(s)")

            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    # Restore best weights
    if best_weights is not None:
        print(f"\nRestoring best weights (similarity: {best_similarity:.4f})")
        qat_model.set_weights(best_weights)

    print("\nQAT training complete!")

    return qat_model


def evaluate_qat_model(
    teacher_model: tf.keras.Model,
    qat_model: tf.keras.Model,
    test_images: np.ndarray
) -> float:
    """
    Evaluate QAT model quality.

    Args:
        teacher_model: Float32 teacher model
        qat_model: Trained QAT model
        test_images: Test images

    Returns:
        Average cosine similarity between teacher and QAT embeddings
    """
    print("\nEvaluating QAT model...")

    # Get embeddings
    teacher_emb = teacher_model.predict(test_images, verbose=0)
    qat_emb = qat_model.predict(test_images, verbose=0)

    # Compute cosine similarity
    similarities = []
    for t, q in zip(teacher_emb, qat_emb):
        t_norm = t / (np.linalg.norm(t) + 1e-8)
        q_norm = q / (np.linalg.norm(q) + 1e-8)
        sim = np.dot(t_norm, q_norm)
        similarities.append(sim)

    avg_sim = np.mean(similarities)
    print(f"  QAT vs Float32 cosine similarity: {avg_sim:.4f}")
    print(f"  Min: {np.min(similarities):.4f}, Max: {np.max(similarities):.4f}")

    return avg_sim


def convert_to_tflite_int8(
    qat_model: tf.keras.Model,
    output_path: str,
    calibration_images: np.ndarray
) -> bool:
    """
    Convert QAT model to TFLite INT8.

    IMPORTANT: For QAT models, the FakeQuant nodes already contain learned
    quantization parameters. We should NOT use representative_dataset for
    internal layers, only for input/output if needed.

    Args:
        qat_model: Trained QAT model with FakeQuant nodes
        output_path: Output TFLite path
        calibration_images: Calibration images (for I/O quantization only)

    Returns:
        True if conversion succeeded
    """
    print(f"\nConverting QAT model to TFLite INT8...")

    # For QAT models, the recommended approach is:
    # 1. Convert directly from keras model (FakeQuant nodes are recognized)
    # 2. Use DEFAULT optimization (respects FakeQuant)
    # 3. Only use representative_dataset for I/O quantization

    # Method 1: Direct QAT conversion (recommended by TensorFlow)
    print("  Method 1: Direct QAT conversion from Keras model...")
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)

        # DEFAULT optimization recognizes FakeQuant nodes from QAT
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # For full INT8 I/O, we need representative dataset
        # But this only affects I/O tensors, not internal layers (which use FakeQuant params)
        num_calib = min(len(calibration_images), 200)
        print(f"  Using {num_calib} samples for I/O quantization...")

        def representative_dataset():
            for img in calibration_images[:num_calib]:
                yield [np.expand_dims(img, axis=0).astype(np.float32)]

        converter.representative_dataset = representative_dataset

        # Target full INT8
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        print("  Converting...")
        tflite_model = converter.convert()
        print("  Method 1 successful!")

        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        print(f"  Saved: {output_path} ({len(tflite_model) / 1024:.1f} KB)")
        return True

    except Exception as e:
        print(f"  Method 1 failed: {e}")

    # Method 2: Concrete function with fixed shape (for NPU compatibility)
    print("\n  Method 2: Concrete function conversion...")
    try:
        @tf.function(input_signature=[tf.TensorSpec([1, 112, 112, 3], tf.float32)])
        def serving_fn(x):
            return qat_model(x, training=False)

        concrete_func = serving_fn.get_concrete_function()

        converter = tf.lite.TFLiteConverter.from_concrete_functions(
            [concrete_func], qat_model
        )
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        def representative_dataset():
            for img in calibration_images[:num_calib]:
                yield [np.expand_dims(img, axis=0).astype(np.float32)]

        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        print("  Converting...")
        tflite_model = converter.convert()
        print("  Method 2 successful!")

        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        print(f"  Saved: {output_path} ({len(tflite_model) / 1024:.1f} KB)")
        return True

    except Exception as e:
        print(f"  Method 2 failed: {e}")

    # Method 3: Without INT8 I/O constraint (float32 I/O, int8 internal)
    print("\n  Method 3: INT8 internal with float32 I/O...")
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Don't force INT8 I/O - let converter decide based on FakeQuant
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS  # Allow float ops for I/O
        ]

        print("  Converting...")
        tflite_model = converter.convert()
        print("  Method 3 successful!")

        # Save as separate file to compare
        f32io_path = output_path.replace('.tflite', '_f32io.tflite')
        with open(f32io_path, 'wb') as f:
            f.write(tflite_model)
        print(f"  Saved: {f32io_path} ({len(tflite_model) / 1024:.1f} KB)")

        # Now try to make full INT8 version
        converter2 = tf.lite.TFLiteConverter.from_keras_model(qat_model)
        converter2.optimizations = [tf.lite.Optimize.DEFAULT]
        converter2.representative_dataset = representative_dataset
        converter2.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter2.inference_input_type = tf.int8
        converter2.inference_output_type = tf.int8

        tflite_model_int8 = converter2.convert()
        with open(output_path, 'wb') as f:
            f.write(tflite_model_int8)
        print(f"  Saved: {output_path} ({len(tflite_model_int8) / 1024:.1f} KB)")
        return True

    except Exception as e:
        print(f"  Method 3 failed: {e}")
        return False


def validate_tflite_int8(
    model_path: str,
    teacher_model: tf.keras.Model,
    test_images: np.ndarray
) -> float:
    """
    Validate INT8 TFLite model against float32 teacher.

    Args:
        model_path: Path to TFLite model
        teacher_model: Float32 teacher model
        test_images: Test images

    Returns:
        Average cosine similarity
    """
    print(f"\nValidating INT8 model: {model_path}")

    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    inp_qp = input_details.get('quantization_parameters', {})
    inp_scale = inp_qp.get('scales', [1.0/128.0])[0]
    inp_zp = inp_qp.get('zero_points', [0])[0]

    out_qp = output_details.get('quantization_parameters', {})
    out_scale = out_qp.get('scales', [1.0])[0]
    out_zp = out_qp.get('zero_points', [0])[0]

    print(f"  Input quant: scale={inp_scale:.6f}, zp={inp_zp}")
    print(f"  Output quant: scale={out_scale:.6f}, zp={out_zp}")

    # Get teacher embeddings
    teacher_emb = teacher_model.predict(test_images[:50], verbose=0)

    # Get INT8 embeddings
    similarities = []
    for i, img in enumerate(test_images[:50]):
        # Quantize input
        int8_input = (img / inp_scale + inp_zp).astype(np.int8)
        interpreter.set_tensor(input_details['index'], np.expand_dims(int8_input, 0))
        interpreter.invoke()

        # Dequantize output
        int8_output = interpreter.get_tensor(output_details['index'])[0]
        float_output = (int8_output.astype(np.float32) - out_zp) * out_scale

        # Compute cosine similarity
        t_norm = teacher_emb[i] / (np.linalg.norm(teacher_emb[i]) + 1e-8)
        q_norm = float_output / (np.linalg.norm(float_output) + 1e-8)
        sim = np.dot(t_norm, q_norm)
        similarities.append(sim)

    avg_sim = np.mean(similarities)
    print(f"\n  INT8 vs Float32 cosine similarity: {avg_sim:.4f}")
    print(f"  Min: {np.min(similarities):.4f}, Max: {np.max(similarities):.4f}")

    return avg_sim


def compile_with_vela(input_path: str, output_dir: str = ".") -> str:
    """Compile with Vela for Ethos-U55 NPU."""
    import subprocess
    import shutil

    print(f"\nCompiling with Vela...")

    vela_path = shutil.which("vela")
    if not vela_path:
        print("  Vela not found. Skipping NPU compilation.")
        return None

    cmd = [
        "vela", input_path,
        "--accelerator-config", "ethos-u55-64",
        "--optimise", "Performance",
        "--output-dir", output_dir
    ]

    print(f"  Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  Vela failed: {result.stderr}")
        return None

    # Print summary
    output_lines = result.stdout.strip().split('\n')
    for line in output_lines:
        if 'NPU' in line or 'CPU' in line or 'operators' in line.lower():
            print(f"  {line}")

    base = Path(input_path).stem
    output = os.path.join(output_dir, f"{base}_vela.tflite")

    if os.path.exists(output):
        print(f"  Output: {output}")
        return output

    return None


def main():
    parser = argparse.ArgumentParser(
        description='QAT for GhostFaceNet INT8 quantization',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--h5', type=str, default=DEFAULT_H5,
                        help=f'Path to GhostFaceNet H5 model (default: {DEFAULT_H5})')
    parser.add_argument('--calib-dir', type=str, default=DEFAULT_CALIB_DIR,
                        help='Calibration images directory')
    parser.add_argument('--output', type=str, default='ghostfacenet_qat_int8.tflite',
                        help='Output TFLite model path')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--warmup-epochs', type=int, default=3,
                        help='Warmup epochs (default: 3)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience (default: 5)')
    parser.add_argument('--augmentation', action='store_true', default=True,
                        help='Enable data augmentation (default: True)')
    parser.add_argument('--no-augmentation', dest='augmentation', action='store_false',
                        help='Disable data augmentation')
    parser.add_argument('--activation-penalty', type=float, default=0.001,
                        help='Activation range penalty weight (default: 0.001)')
    parser.add_argument('--activation-limit', type=float, default=30.0,
                        help='Soft limit for activation values (default: 30.0)')
    parser.add_argument('--skip-vela', action='store_true',
                        help='Skip Vela compilation')
    args = parser.parse_args()

    print(f"\nConfiguration:")
    print(f"  H5 model: {args.h5}")
    print(f"  Calibration: {args.calib_dir}")
    print(f"  Output: {args.output}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Warmup epochs: {args.warmup_epochs}")
    print(f"  Early stopping patience: {args.patience}")
    print(f"  Data augmentation: {args.augmentation}")
    print(f"  Activation penalty: {args.activation_penalty}")
    print(f"  Activation limit: {args.activation_limit}")

    # Check model file
    if not os.path.exists(args.h5):
        print(f"\nERROR: Model file not found: {args.h5}")
        return 1

    # Step 1: Load teacher model
    teacher_model = load_teacher_model(args.h5)

    # Step 2: Load training images
    train_images = load_training_images(args.calib_dir)

    # Step 3: Create QAT model
    qat_model = create_qat_model(teacher_model)

    # Step 4: Train QAT model
    qat_model = train_qat(
        teacher_model=teacher_model,
        qat_model=qat_model,
        train_images=train_images,
        epochs=args.epochs,
        batch_size=args.batch_size,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        use_augmentation=args.augmentation,
        activation_penalty_weight=args.activation_penalty,
        activation_limit=args.activation_limit
    )

    # Step 5: Evaluate QAT model (before conversion)
    qat_sim = evaluate_qat_model(teacher_model, qat_model, train_images[:100])

    # Step 6: Convert to TFLite INT8
    if not convert_to_tflite_int8(qat_model, args.output, train_images):
        print("\nFailed to convert QAT model to TFLite!")
        return 1

    # Step 7: Validate INT8 model
    int8_sim = validate_tflite_int8(args.output, teacher_model, train_images)

    # Step 8: Compile with Vela
    if not args.skip_vela:
        vela_output = compile_with_vela(args.output)
        if vela_output:
            print(f"\nVela model: {vela_output}")

    # Summary
    print("\n" + "=" * 60)
    print("QAT Training Complete!")
    print("=" * 60)
    print(f"\nResults:")
    print(f"  QAT vs Float32 similarity: {qat_sim:.4f}")
    print(f"  INT8 vs Float32 similarity: {int8_sim:.4f}")
    print(f"\nOutput: {args.output}")

    if int8_sim > 0.85:
        print("\n  QAT SUCCESS! INT8 model has good precision.")
    elif int8_sim > 0.7:
        print("\n  QAT PARTIAL SUCCESS. Consider more training epochs.")
    else:
        print("\n  QAT needs more work. Try increasing epochs or data.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
