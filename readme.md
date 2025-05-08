# Quantum Communication Analysis

## Overview

This project analyzes quantum communication protocols, focusing on the BB84 protocol and the GG02 protocol. It explores the influence of noise tolerance and transmission distance on the performance of quantum key distribution (QKD) systems. The code includes simulations that demonstrate the impacts of different quantum states on secure key rates.

## Features

- **Gamma Matrix Construction**: Builds the gamma matrix before entering the quantum channel.
- **Secure Key Rate Calculation**: Calculates secure key rates for various protocols including squeezed and coherent states.
- **Symplectic Eigenvalue Calculation**: Computes the symplectic eigenvalues used in the analysis.
- **Visualization**: Provides plots to visualize the relationships between distance and excess noise in different quantum communication scenarios.

## Requirements

- Python 3.11
- NumPy
- SciPy
- Matplotlib

You can install the required packages using pip:

```bash
pip install numpy scipy matplotlib  
```