"""
Grad-CAM utilities for Brain Hemorrhage Detection.

Generates a heatmap highlighting image regions that most influenced
the model's prediction.

Usage (from app.py):
    from gradcam import generate_gradcam
    generate_gradcam(model, image_path, output_path)
"""

import os
from typing import Optional

import cv2
import numpy as np
import tensorflow as tf
import keras
from keras.models import Model

from magicsauce import preprocess_jpg


def _find_last_conv_layer_name(model: keras.Model) -> str:
    """
    Automatically detect the last convolutional-like layer in the model.

    Handles models with:
    - ResNet50/VGG16 base wrapped in Sequential
    - GlobalAveragePooling2D after conv base (need to go into base model)
    - Direct conv layers

    Returns the name of the last layer with 4D output (batch, H, W, C).
    """
    # Strategy: Walk backwards through layers
    # If we find a Sequential/Model layer (like ResNet50 base), look inside it
    for layer in reversed(model.layers):
        try:
            # Check if this layer itself has 4D output
            output_shape = layer.output_shape
            if isinstance(output_shape, tuple) and len(output_shape) == 4:
                return layer.name
            
            # If this is a nested model (like ResNet50 base), search inside it
            if hasattr(layer, 'layers') and len(layer.layers) > 0:
                # This is likely a base model (ResNet50, VGG16, etc.)
                # Search backwards through its layers for the last 4D output
                for sub_layer in reversed(layer.layers):
                    try:
                        sub_output_shape = sub_layer.output_shape
                        if isinstance(sub_output_shape, tuple) and len(sub_output_shape) == 4:
                            # Return the full path: base_layer_name.sub_layer_name
                            return f"{layer.name}.{sub_layer.name}"
                    except Exception:
                        continue
        except Exception:
            continue

    # Fallback: try to find any layer with 'conv' in the name that has 4D output
    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower():
            try:
                output_shape = layer.output_shape
                if isinstance(output_shape, tuple) and len(output_shape) == 4:
                    return layer.name
            except Exception:
                continue

    raise ValueError(
        f"Grad-CAM: Could not find a 4D convolutional layer. "
        f"Model layers: {[l.name for l in model.layers]}"
    )


def generate_gradcam(model: keras.Model, image_path: str, output_path: str) -> Optional[str]:
    """
    Generate a Grad-CAM heatmap and save the overlayed image.

    Parameters
    ----------
    model : keras.Model
        Trained Keras model (binary classifier with sigmoid output).
    image_path : str
        Path to the original input image (JPG/PNG).
    output_path : str
        Where to save the Grad-CAM overlay image (e.g. static/gradcam_output.jpg).

    Returns
    -------
    Optional[str]
        The output_path on success, or None if generation failed.
    """
    if model is None:
        raise ValueError("Model is None. Load the trained model before calling generate_gradcam.")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image for Grad-CAM not found: {image_path}")

    # Read original image (BGR, OpenCV default) to preserve original size
    original = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if original is None:
        raise ValueError(f"Could not read image at path: {image_path}")

    # Preprocess image for model input (resize + normalization)
    # This will produce tensor of shape (1, H, W, 3) matching training.
    img_tensor = preprocess_jpg(image_path)

    # Detect last convolutional layer and build grad_model
    # We must build a single connected graph so gradients flow from prediction to conv_outputs.
    try:
        first_layer = model.layers[0]
        if hasattr(first_layer, 'layers') and len(first_layer.layers) > 0:
            # Base model (ResNet50, VGG16, etc.) wrapped in Sequential
            print(f"Grad-CAM: Detected base model '{first_layer.name}'")
            base_model = first_layer

            # Find the last layer with 4D output within the base model
            last_conv_layer = None
            for layer in reversed(base_model.layers):
                try:
                    output_shape = layer.output_shape
                    if isinstance(output_shape, tuple) and len(output_shape) == 4:
                        last_conv_layer = layer
                        print(f"Grad-CAM: Found last conv layer: '{layer.name}'")
                        break
                except Exception:
                    continue

            if last_conv_layer is None:
                conv_output_ref = base_model.output
                print(f"Grad-CAM: Using base model output as conv features")
            else:
                conv_output_ref = last_conv_layer.output

            # Single forward pass: base_input -> [conv_output, base_output] -> final_output
            # This keeps the graph connected so gradients flow.
            base_input = base_model.input
            base_model_2out = Model(
                inputs=base_model.input,
                outputs=[conv_output_ref, base_model.output],
            )
            conv_output_tensor, base_output_tensor = base_model_2out(base_input)

            remaining_layers = model.layers[1:]
            x = base_output_tensor
            for layer in remaining_layers:
                x = layer(x)
            final_output = x

            grad_model = Model(
                inputs=base_input,
                outputs=[conv_output_tensor, final_output],
            )
            print(f"Grad-CAM: Successfully built grad_model with single forward pass")
        else:
            # No nested base model
            last_conv_layer_name = _find_last_conv_layer_name(model)
            print(f"Grad-CAM: Using layer '{last_conv_layer_name}' for visualization")
            if '.' in last_conv_layer_name:
                base_name, sub_layer_name = last_conv_layer_name.split('.', 1)
                base_layer = model.get_layer(base_name)
                last_conv_layer = base_layer.get_layer(sub_layer_name)
            else:
                last_conv_layer = model.get_layer(last_conv_layer_name)
            grad_model = Model(
                inputs=model.inputs,
                outputs=[last_conv_layer.output, model.output],
            )
    except Exception as e:
        print(f"Grad-CAM: Model building failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    # GradientTape: only TF operations, no predict(), no argmax, no numpy before gradients
    try:
        with tf.GradientTape() as tape:
            # Keep as tf tensor; same dtype as model expects
            inputs = tf.constant(img_tensor, dtype=tf.float32)
            conv_outputs, predictions = grad_model(inputs)

            # Binary classification: target = predictions[:, 0] (differentiable)
            if len(predictions.shape) == 2 and predictions.shape[-1] == 1:
                target = predictions[:, 0]
            elif len(predictions.shape) == 1:
                target = predictions[0]
            else:
                # Multi-class: use sum of (class prob * one-hot) so gradient flows
                target = tf.reduce_sum(predictions)

        grads = tape.gradient(target, conv_outputs)
        if grads is None:
            print("Grad-CAM: Failed to compute gradients (grads is None).")
            print(f"  predictions shape: {predictions.shape}, conv_outputs shape: {conv_outputs.shape}, target shape: {target.shape}")
            return None
        print(f"Grad-CAM: grads shape: {grads.shape}, conv_outputs shape: {conv_outputs.shape}")
    except Exception as e:
        print(f"Grad-CAM: Gradient computation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Global average pooling on the gradients (all TF ops; numpy only after this)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # shape: (channels,)

    # We only need the feature maps for the first (and only) image in the batch
    conv_outputs = conv_outputs[0]  # shape: (H, W, channels)

    # Weight the feature maps by the pooled gradients
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # Convert to numpy and normalize between 0 and 1
    heatmap = heatmap.numpy()
    heatmap = np.maximum(heatmap, 0)
    max_val = heatmap.max() if heatmap.max() != 0 else 1e-8
    heatmap /= max_val

    # Resize heatmap to match original image size
    heatmap_resized = cv2.resize(
        heatmap,
        (original.shape[1], original.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )

    # Convert to RGB heatmap using a colormap
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Superimpose the heatmap on the original image
    # Both are BGR (OpenCV), so we can blend them directly.
    alpha = 0.4  # heatmap intensity
    superimposed_img = cv2.addWeighted(heatmap_color, alpha, original, 1 - alpha, 0)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create if there's actually a directory path
        os.makedirs(output_dir, exist_ok=True)

    # Save result
    print(f"Grad-CAM: Saving heatmap to {output_path}")
    success = cv2.imwrite(output_path, superimposed_img)
    if not success:
        print(f"Grad-CAM: Failed to write output image to {output_path}")
        print(f"  Output directory exists: {os.path.exists(output_dir) if output_dir else 'N/A'}")
        return None

    # Verify file was created
    if os.path.exists(output_path):
        print(f"Grad-CAM: Successfully saved heatmap ({os.path.getsize(output_path)} bytes)")
    else:
        print(f"Grad-CAM: Warning - file not found after save: {output_path}")
        return None

    return output_path

