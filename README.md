# Synthetic Financial Time Series Generation with GANs

This repository contains the code and experiment notebooks for my thesis project on **synthetic financial time series generation with GANs**.  
It implements four baseline GAN models for return generation and a proposed **regime-controllable** extension (**P-RSQGAN**), together with a unified set of diagnostic plots and evaluation utilities.

## Project Structure

### Notebooks (NDX / Gold)
Each model is provided as a standalone Jupyter notebook for two datasets (NASDAQ-100 index returns and gold returns).

- **Baseline GAN (LSTM-GAN style)**
  - `GAN_ndx.ipynb`
  - `GAN_gold.ipynb`

- **TimeGAN**
  - `TimeGAN_ndx.ipynb`
  - `TimeGAN_gold.ipynb`

- **WGAN-GP**
  - `WGAN_ndx.ipynb`
  - `WGAN_gold.ipynb`

- **QuantGAN (TCN-based)**
  - `QuantGAN_ndx.ipynb`
  - `QuantGAN_gold.ipynb`

- **Proposed: P-RSQGAN (Projection + Regime Conditioning + WGAN-GP objective)**
  - `P-RSQGAN_ndx.ipynb`
  - `P-RSQGAN_gold.ipynb`

### Data
- `dataset/`  
  Contains raw or processed data files used by the notebooks (prices and/or returns).  
  Each notebook loads data from this folder and performs preprocessing (log-return conversion, normalization, sliding windows).

### Thesis
- `Synthetic Financial Time Series Generation with GANs.pdf`  
  The full thesis document describing motivation, methodology, evaluation framework, and results.

## What This Repo Covers

### Tasks
- Learn a generative model for **financial return sequences**
- Produce synthetic return windows and reconstruct price/index paths
- Compare models under a unified evaluation protocol
- Add **regime controllability** (low-vol vs high-vol) using P-RSQGAN

### Evaluation Diagnostics (as implemented in notebooks)
- Training curves (generator / discriminator or critic loss)
- Real vs generated return comparison (sample paths)
- Empirical PDF / KDE comparison
- Q–Q plots (tail diagnostics)
- ACF of returns and ACF of |returns| (temporal dependence & volatility clustering)
- Rolling volatility (volatility clustering in time domain)
- Fan charts (path-wise uncertainty bands / scenario dispersion)
- Regime controllability + regime consistency (for P-RSQGAN; post-hoc regime inspection for unconditional models)
