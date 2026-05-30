# Neural Style Transfer — VGG19

Implementation of the Neural Style Transfer algorithm from **Gatys et al. (2015) — "A Neural Algorithm of Artistic Style"**, using a pretrained VGG-19 network as a feature extractor.

The algorithm transfers the artistic style of a painting onto a content photograph by optimising the image pixels directly, minimising a combined loss:

```
L_total = α · L_style + β · L_content + TVW · L_tv
```

---

## How it works

| Component | Description |
|---|---|
| **Feature extractor** | Frozen VGG-19 pretrained on ImageNet |
| **Style representation** | Gram matrices from 5 convolutional layers (block1–block5) |
| **Content representation** | Activation maps from `block4_conv2` |
| **Optimiser** | Adam — gradient descent on image pixels |
| **Regularisation** | Total Variation loss to reduce noise |

---

## Results

Below are some examples obtained with this implementation.

| Content | Style | Result |
|---|---|---|
| `images/content/Cont_Perro.jpg` | `images/style/Estilo_VG1.jpg` | `resultados/...` |

Full experiment logs and intermediate images are available in the `resultados/` folder.

---

## Project structure

```
TransferStyle/
├── TransferStyleScript.py    ← main script
├── requirements.txt          ← dependencies
├── README.md
├── images/
│   ├── content/              ← content images (what to preserve)
│   └── style/                ← style images (artistic reference)
└── resultados/               ← experiment outputs
 
```

---

## Installation

```bash
git clone https://github.com/jtorreslopz/TransferStyle
cd TransferStyle
pip install -r requirements.txt
```

---

## Usage

**1. Set your image paths** in Section 4 of the script:

```python
STYLE_PATH   = 'images/style/your_style.jpg'
CONTENT_PATH = 'images/content/your_content.jpg'
```

**2. Adjust the hyperparameters** (optional):

```python
style_weight           = 0.01             # α — more → stronger style
content_weight         = 100_000_000_000  # β — more → preserves content
total_variation_weight = 30               # TVW — more → smoother result
style_layer_weights    = [0.2, 0.2, 0.2, 0.2, 0.2]  # per-layer weights
```

**3. Run the script:**

```bash
python TransferStyleScript.py
```

Results are saved automatically to a uniquely named folder inside `resultados/`.

---

## Hyperparameter guide

Only the **ratio α/β** matters — Adam normalises the gradient scale.

| Goal | Suggestion |
|---|---|
| More artistic style | Increase α or decrease β |
| Preserve content better | Increase β or decrease α |
| Smoother result | Increase TVW |
| Fine textures (Pointillism) | Front-load `style_layer_weights` e.g. `[0.4, 0.3, 0.2, 0.05, 0.05]` |
| Abstract deformation (Cubism) | Back-load e.g. `[0.05, 0.05, 0.2, 0.3, 0.4]` |

---

## Output

Each experiment generates a dedicated folder with:
- **PNG images** saved every 100 optimisation steps
- **`configuracion.txt`** — full log with two loss matrices per checkpoint:
  - *Matrix 1*: per-layer style loss before and after internal weights
  - *Matrix 2*: global balance of style, content and TV losses

---

## Reference

> Gatys, L. A., Ecker, A. S., & Bethge, M. (2015).
> *A Neural Algorithm of Artistic Style*.
> arXiv:1508.06576. https://arxiv.org/abs/1508.06576

---

## Author

Torres López, J. (2025). Neural Style Transfer — VGG19 Implementation. GitHub.
https://github.com/jtorreslopz/TransferStyle
