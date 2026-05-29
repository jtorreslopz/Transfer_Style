#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              NEURAL STYLE TRANSFER — VGG19 Implementation                  ║
║                                                                              ║
║  Transfers the artistic style of a painting onto a content photograph       ║
║  using a pretrained VGG-19 network as a feature extractor.                 ║
║                                                                              ║
║  Based on: Gatys et al. (2015) — "A Neural Algorithm of Artistic Style"    ║
║  Framework: TensorFlow / Keras                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
 
ALGORITHM OVERVIEW
──────────────────
  1. Extract style features  → Gram matrices from 5 convolutional layers
  2. Extract content features → Activation maps from block4_conv2
  3. Optimize image pixels to minimize:
        L_total = α · L_style + β · L_content + TVW · L_tv
 
USAGE
─────
  Set the image paths and hyperparameters in Section 4, then run all cells.
  Output images and loss logs are saved to a dedicated experiment folder.
"""
 
# ──────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
 
import os
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 1. IMAGE LOADING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
 
def load_img(path_to_img: str) -> tf.Tensor:
    """
    Load an image from disk, normalize it to [0, 1] and resize so that
    the longest dimension equals MAX_DIM, preserving the aspect ratio.
 
    Args:
        path_to_img: Path to the image file (JPEG or PNG).
 
    Returns:
        A float32 tensor of shape (1, H, W, 3) ready for VGG-19 inference.
    """
    MAX_DIM = 512
 
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)   # Scale to [0, 1]
 
    # Compute new dimensions keeping aspect ratio
    shape     = tf.cast(tf.shape(img)[:-1], tf.float32)
    scale     = MAX_DIM / max(shape)
    new_shape = tf.cast(shape * scale, tf.int32)
 
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]                               # Add batch dimension
    return img
 
 
def imshow(image: tf.Tensor, title: str = None) -> None:
    """Display a tensor image using matplotlib."""
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 2. VGG-19 FEATURE EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════
 
# Style layers capture texture, colour and brushstroke at increasing scales
STYLE_LAYERS = [
    'block1_conv1',   # Ultra-fine textures and micro-colours
    'block2_conv1',   # Basic linear patterns
    'block3_conv1',   # Intermediate artistic structures
    'block4_conv1',   # Complex brushstroke shapes
    'block5_conv1',   # Abstract large-scale patterns
]
 
# Content layer captures object geometry and identity
CONTENT_LAYERS = ['block4_conv2']
 
 
def build_vgg_extractor(layer_names: list) -> tf.keras.Model:
    """
    Build a Keras sub-model that returns the activations of the requested
    VGG-19 layers. The network weights are frozen (non-trainable).
 
    Args:
        layer_names: List of VGG-19 layer names to expose as outputs.
 
    Returns:
        A tf.keras.Model with one output per requested layer.
    """
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    return tf.keras.Model([vgg.input], outputs)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 3. GRAM MATRIX & STYLE-CONTENT MODEL
# ══════════════════════════════════════════════════════════════════════════════
 
def gram_matrix(input_tensor: tf.Tensor) -> tf.Tensor:
    """
    Compute the Gram matrix of a feature map.
 
    The Gram matrix G_mn = Σ_j F_mj · F_nj captures the cross-channel
    correlations (co-occurrences) of the feature maps, encoding style
    information independently of spatial position.
 
    Args:
        input_tensor: Feature map tensor of shape (batch, H, W, C).
 
    Returns:
        Gram matrix of shape (batch, C, C).
    """
    # Einstein summation: b=batch, i=height, j=width, c/d=channels
    return tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
 
 
class StyleContentModel(tf.keras.models.Model):
    """
    Wrapper around a frozen VGG-19 that returns:
      - style  : Gram matrices for each style layer
      - content: Raw activation maps for each content layer
    """
 
    def __init__(self, style_layers: list, content_layers: list):
        super().__init__()
        self.vgg              = build_vgg_extractor(style_layers + content_layers)
        self.style_layers     = style_layers
        self.content_layers   = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable    = False
 
    def call(self, inputs: tf.Tensor) -> dict:
        # VGG-19 expects pixel values in [0, 255] with ImageNet mean subtracted
        inputs             = inputs * 255.0
        preprocessed       = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs            = self.vgg(preprocessed)
 
        style_outputs, content_outputs = (
            outputs[:self.num_style_layers],
            outputs[self.num_style_layers:]
        )
 
        # Convert style feature maps to Gram matrices
        style_outputs = [gram_matrix(out) for out in style_outputs]
 
        content_dict = dict(zip(self.content_layers, content_outputs))
        style_dict   = dict(zip(self.style_layers,   style_outputs))
 
        return {'content': content_dict, 'style': style_dict}
 
 
# Instantiate the global extractor
extractor = StyleContentModel(STYLE_LAYERS, CONTENT_LAYERS)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 4. EXPERIMENT CONFIGURATION  ← Edit hyperparameters here
# ══════════════════════════════════════════════════════════════════════════════
 
# ── Image paths ───────────────────────────────────────────────────────────────
STYLE_PATH   = 'Estilo_VG1.jpg'    # Source image for style extraction
CONTENT_PATH = 'Cont_Perro.jpg'    # Source image for content preservation
 
# ── Per-layer style weights (must sum to 1.0) ─────────────────────────────────
#   [block1, block2, block3, block4, block5]
#   Equal weights  → balanced aesthetic (Gatys et al. default)
#   Front-loaded   → fine textures and brushstrokes (e.g. Pointillism)
#   Back-loaded    → abstract deformations (e.g. Cubism)
style_layer_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
 
# ── Global loss weights ───────────────────────────────────────────────────────
#   Only the RATIO α/β matters (Adam normalises the gradient scale).
#   Increasing α/β → more artistic style, less content fidelity.
style_weight           = 0.01            # α  — style importance
content_weight         = 100_000_000_000 # β  — content importance  (α/β = 10⁻¹³)
total_variation_weight = 30              # TVW — spatial smoothing regulariser
 
# ── Optimiser ─────────────────────────────────────────────────────────────────
opt = tf.optimizers.Adam(learning_rate=0.02)
 
# ── Load images and extract fixed targets ─────────────────────────────────────
style_image   = load_img(STYLE_PATH)
content_image = load_img(CONTENT_PATH)
 
style_targets   = extractor(style_image)['style']
content_targets = extractor(content_image)['content']
 
# Preview the two input images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1); imshow(content_image, 'Content')
plt.subplot(1, 2, 2); imshow(style_image,   'Style')
plt.tight_layout(); plt.show()
 
# The image to optimise — initialised as a copy of the content image
image = tf.Variable(content_image)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAINING STEP (GRADIENT DESCENT ON PIXELS)
# ══════════════════════════════════════════════════════════════════════════════
 
def train_step(
    img_variable: tf.Variable,
    s_w:  float,
    c_w:  float,
    tv_w: float
) -> tuple:
    """
    Perform one optimisation step:
      1. Forward pass through VGG-19 to get current style & content features.
      2. Compute L_style, L_content and L_tv.
      3. Back-propagate gradients to the image pixels.
      4. Clip pixel values to [0, 1].
 
    Args:
        img_variable : The mutable image tensor being optimised.
        s_w          : Global style weight (α).
        c_w          : Global content weight (β).
        tv_w         : Total variation weight (TVW).
 
    Returns:
        Tuple (layer_loss_matrix, global_loss_matrix) for logging.
    """
    with tf.GradientTape() as tape:
        outputs = extractor(img_variable)
 
        # ── Style loss ────────────────────────────────────────────────────────
        layer_loss_log  = {}
        style_loss_base = 0.0
        style_loss_wtd  = 0.0
        style_outputs   = outputs['style']
 
        for i, name in enumerate(STYLE_LAYERS):
            gram_shape   = tf.shape(style_outputs[name])
            n_elements   = tf.cast(gram_shape[1] * gram_shape[2], tf.float32)
            sq_diff      = tf.reduce_sum((style_outputs[name] - style_targets[name]) ** 2)
 
            loss_raw     = sq_diff / n_elements                       # Normalised MSE
            loss_wtd     = style_layer_weights[i] * loss_raw          # Weighted by wₖ
 
            style_loss_base += loss_raw
            style_loss_wtd  += loss_wtd
 
            layer_loss_log[name] = {
                'antes_internal':   loss_raw.numpy(),
                'despues_internal': loss_wtd.numpy(),
            }
 
        layer_loss_log['SUMA_TOTAL'] = {
            'antes_internal':   style_loss_base.numpy(),
            'despues_internal': style_loss_wtd.numpy(),
        }
 
        # ── Content loss ──────────────────────────────────────────────────────
        content_loss    = 0.0
        content_outputs = outputs['content']
 
        for name in content_outputs:
            shape       = tf.shape(content_outputs[name])
            n_el        = tf.cast(shape[1] * shape[2] * shape[3], tf.float32)
            sq_diff     = tf.reduce_sum((content_outputs[name] - content_targets[name]) ** 2)
            content_loss += 0.5 * (sq_diff / n_el)                   # Gatys et al. ½ MSE
 
        # ── Total Variation loss ──────────────────────────────────────────────
        tv_loss = tf.reduce_sum(tf.image.total_variation(img_variable))
 
        # ── Global loss matrix (for logging) ─────────────────────────────────
        global_loss_log = {
            'Estilo': {
                'antes_global':   style_loss_wtd.numpy(),
                'despues_global': (style_loss_wtd * s_w).numpy(),
            },
            'Contenido': {
                'antes_global':   content_loss.numpy(),
                'despues_global': (content_loss * c_w).numpy(),
            },
            'Var_Total (TV)': {
                'antes_global':   tv_loss.numpy(),
                'despues_global': (tv_loss * tv_w).numpy(),
            },
        }
 
        # ── Total loss: L = α·L_style + β·L_content + TVW·L_tv ───────────────
        total_loss = (style_loss_wtd * s_w) + (content_loss * c_w) + (tv_loss * tv_w)
 
    # Back-propagate to pixels
    grad = tape.gradient(total_loss, img_variable)
    opt.apply_gradients([(grad, img_variable)])
 
    # Clamp pixels to valid range [0, 1]
    img_variable.assign(tf.clip_by_value(img_variable, 0.0, 1.0))
 
    return layer_loss_log, global_loss_log
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 6. OUTPUT DIRECTORIES & LOG FILE INITIALISATION
# ══════════════════════════════════════════════════════════════════════════════
 
os.makedirs('resultados', exist_ok=True)
 
# Build a unique, descriptive folder name for this experiment
_content_name  = os.path.splitext(os.path.basename(CONTENT_PATH))[0]
_style_name    = os.path.splitext(os.path.basename(STYLE_PATH))[0]
_weights_str   = "-".join(str(w) for w in style_layer_weights)
 
experiment_dir = (
    f"resultados/Exp_{_content_name}_vs_{_style_name}_"
    f"alpha{style_weight}_beta{content_weight}_"
    f"pesos_{_weights_str}_tvw{total_variation_weight}"
)
os.makedirs(experiment_dir, exist_ok=True)
 
# Initialise the log file
log_path = os.path.join(experiment_dir, "configuracion.txt")
with open(log_path, "w", encoding="utf-8") as f:
    f.write("=== CONFIGURACIÓN DEL EXPERIMENTO ===\n")
    f.write(f"Fecha y Hora de inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"Style Weight (SW): {style_weight}\n")
    f.write(f"Content Weight (CW): {content_weight}\n")
    f.write(f"Total Variation Weight (TVW): {total_variation_weight}\n")
    f.write(f"Pesos específicos por capa de estilo: {style_layer_weights}\n")
 
print(f"✓ Experiment folder : {experiment_dir}")
print(f"✓ Log file          : {log_path}")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 7. TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════
 
EPOCHS          = 100   # Number of display/save checkpoints
STEPS_PER_EPOCH = 100   # Optimisation steps per checkpoint  (total = 10 000)
 
start  = time.time()
step   = 0
layer_log, global_log = {}, {}
 
for epoch in range(EPOCHS):
 
    # ── Inner optimisation loop ───────────────────────────────────────────────
    for _ in range(STEPS_PER_EPOCH):
        step += 1
        layer_log, global_log = train_step(
            image, style_weight, content_weight, total_variation_weight
        )
        print('.', end='', flush=True)
 
    # ── Save intermediate image ───────────────────────────────────────────────
    img_path = os.path.join(experiment_dir, f"estilo_paso_{step}.png")
    img_save = tf.squeeze(image.read_value(), axis=0) * 255.0
    tf.keras.utils.save_img(img_path, img_save)
 
    # ── Build report block ────────────────────────────────────────────────────
    lines = [f"\n\n{'='*50} PASO {step} {'='*50}"]
 
    lines.append("\n[MATRIZ 1] DESGLOSE DE CAPAS DE ESTILO")
    lines.append(f"{'Capa Convolucional':<18} | {'Antes de Pesos Internos':<25} | {'Después de Pesos Internos':<25}")
    lines.append("-" * 75)
    for layer, vals in layer_log.items():
        if layer == 'SUMA_TOTAL':
            lines.append("-" * 75)
        lines.append(
            f"{layer:<18} | {vals['antes_internal']:<25.6f} | {vals['despues_internal']:<25.6f}"
        )
 
    lines.append("\n[MATRIZ 2] BALANCE GLOBAL DEL EXPERIMENTO")
    lines.append(f"{'Tipo de Pérdida':<15} | {'Antes de Peso Global (Base)':<30} | {'Después de Peso Global (Ponderado)':<35}")
    lines.append("-" * 85)
    for loss_type, vals in global_log.items():
        lines.append(
            f"{loss_type:<15} | {vals['antes_global']:<30.6f} | {vals['despues_global']:<35.2f}"
        )
    lines.append("=" * 50)
 
    report = "\n".join(lines)
 
    print(report)
    print(f"    → Image saved: {img_path}")
 
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(report)
 
    # ── Live preview ──────────────────────────────────────────────────────────
    plt.figure(figsize=(5, 5))
    imshow(image.read_value(), title=f"Step {step}")
    plt.show()
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 8. EXPERIMENT SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
 
elapsed = time.time() - start
 
with open(log_path, "a", encoding="utf-8") as f:
    f.write(f"\n\n=== Experimento Finalizado con Éxito en {elapsed:.1f} segundos ===")
 
print(f"\n✓ Finished in {elapsed:.1f} s")
print(f"✓ Full log saved to: {log_path}")