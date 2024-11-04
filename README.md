# StreamerNet: Deep Learning for Streamer Discharge Simulation

StreamerNet is a cutting-edge deep learning framework designed to model and predict the complex dynamics of streamer discharges using state-of-the-art neural network architectures.

## Key Features

- **Multiple Neural Network Architectures**
  - U-Net: Specialized for spatial feature extraction
  - Fourier Neural Operator (FNO2D): Leverages spectral methods for efficient computation
  - Cylindrical Symmetric FNO (CSFNO2D): Custom architecture optimized for cylindrical symmetry

- **Advanced Data Processing**
  - Handles complex HDF5 datasets
  - Supports multi-dimensional time series data
  - Built-in data normalization and preprocessing
  - Efficient batch processing for large-scale simulations

- **Comprehensive Analysis Tools**
  - Fourier transform analysis capabilities
  - Real-time visualization of predictions
  - Model performance comparison utilities
  - Detailed logging and monitoring through Weights & Biases

## Results

Our models have demonstrated remarkable success in predicting streamer discharge evolution:
- Accurate prediction of electron density distributions
- Precise electric field calculations
- Efficient computation compared to traditional numerical methods
- Validated against high-fidelity physical simulations

## Technical Specifications

### Data Requirements
- Input format: HDF5
- Supported features: electron density ('e'), electric field
- Configurable time steps for input and prediction

### Model Configuration
- Customizable network architectures
- Flexible hyperparameter tuning
- Support for various activation functions
- Adaptable for different physical domains

## License

This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.

## Acknowledgments

This research was supported by Ramón Areces Foundation. Special thanks to Jannis Teunissen for his valuable feedback and support.
