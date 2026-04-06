## Disentangling Grasp–Object Representations in the Latent Space

This project contains the code used to analyse the latent space of a Conditional Variational Autoencoder (CVAE) trained on hand–object interaction data, and to visualise this space using both PCA and CEBRA-style embeddings. The experiments correspond to the work described in `paper.pdf` and `LM2025.pdf` and are inspired by CEBRA ([nature article](https://www.nature.com/articles/s41586-023-06031-6) and [official repo](https://github.com/AdaptiveMotorControlLab/CEBRA)).

### Project structure

- **`src/cvae_model.py`**: CVAE model definition (object encoder, hand encoder, prior/posterior, decoder).
- **`src/cebra.py`**: End-to-end analysis script:
  - loads the preprocessed dataset,
  - computes CVAE latent vectors for hand poses,
  - runs PCA and visualises 2D/3D projections,
  - trains CEBRA embeddings:
    - using object identity as discrete labels,
    - using time as a continuous variable,
  - creates 2D/3D visualisations (Plotly + Matplotlib).
- **`dataset/hand_object_data.pkl`**: Preprocessed HO-3D\_v3 hand–object data (pickled dict).
- **`dataset/scalers.pkl`**: Optional scalers used during preprocessing (not required for running `src/cebra.py`).
- **`requirements.txt`**: Exact Python package versions used.
- **`paper.pdf`, `LM2025.pdf`, `CEBRA.pdf`**: Manuscripts / reference documents.

### Environment setup

1. **Create and activate a virtual environment (recommended)**

   ```bash
   cd cvae_cebra
   python -m venv env
   # Windows PowerShell
   .\env\Scripts\Activate.ps1
   # or Windows CMD
   .\env\Scripts\activate.bat
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   This installs PyTorch, scikit-learn, matplotlib, Plotly, CEBRA, and other required libraries.

### Data format

The main script expects a preprocessed pickle file at `dataset/hand_object_data.pkl` with (at least) the following keys:

- **`hand_train`, `hand_val`, `hand_test`**: Hand pose features.
- **`obj_train`, `obj_val`, `obj_test`**: Object features.
- **`obj_names`**: List/array of object names.
- **`train_indices`, `val_indices`, `test_indices`**: Indices into `obj_names` for each split.

These are typically derived from the HO-3D\_v3 dataset after preprocessing. The file already exists in this project; no extra steps are needed unless you want to regenerate it.

### Pretrained CVAE weights

The script uses a pretrained CVAE checkpoint:

- **`src/cvae_weight.pth`**

By default, `src/cebra.py` points to this file via a hard-coded path. If you move the project or rename files, update the `model_weight` and `dataset_path` variables near the bottom of `src/cebra.py`.

### How to run the analysis

After activating the environment and installing dependencies:

```bash
cd cvae_cebra
python src/cebra.py
```

What this does:

- Loads the dataset from `dataset/hand_object_data.pkl`.
- Loads the pretrained CVAE from `src/cvae_weight.pth`.
- Computes latent vectors (posterior means) for the test set.
- Runs PCA and shows:
  - cumulative explained variance plot,
  - 3D PCA scatter (selected PCs),
  - 2D PCA scatter by object.
- Trains CEBRA embeddings:
  - **Object-based CEBRA**: discrete labels = object names.
  - **Time-based CEBRA**: continuous labels = sample index (time).
- Visualises CEBRA embeddings:
  - 3D/2D coloured by object,
  - 3D/2D coloured by time,
  - optional CEBRA built-in plots (`cebra.plot_embedding`).

The plots open in interactive windows (Matplotlib and Plotly). Make sure your environment supports GUI plotting (e.g., run from a local Python session or an IDE like PyCharm).

### Customisation tips

- **Latent dimensionality**: Change `latent_dim` in `src/cebra.py` and retrain the CVAE if needed.
- **PCA components**: Adjust which PCs are visualised in `plot_3Dpca` / `plot_2Dpca`.
- **CEBRA hyperparameters**: Modify `output_dimension`, `max_iterations`, `batch_size`, `learning_rate`, etc., in:
  - `compute_cebra_embeddings`,
  - `compute_cebra_embeddings_with_time`.
- **GPU vs CPU**: CEBRA automatically uses CUDA if available; otherwise it falls back to CPU.

This README should be enough to recreate the environment and run the latent space visualisation pipeline end-to-end.



