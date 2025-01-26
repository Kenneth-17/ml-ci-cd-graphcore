# Automated ML Model Testing Pipeline with CI/CD

[![CI/CD Pipeline](https://github.com/<your-username>/ml-ci-cd-graphcore/actions/workflows/main.yml/badge.svg)](https://github.com/<your-username>/ml-ci-cd-graphcore/actions)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

End-to-end CI/CD pipeline for ML model testing with Graphcore IPU integration

## Project Overview
![Pipeline Architecture](docs/pipeline_architecture.png)

## Features
- PyTorch/JAX image classifier
- Automated numerical accuracy tests
- IPU hardware performance benchmarking
- GitHub Actions CI/CD integration
- TensorBoard logging
- Kubernetes deployment templates (optional)

## Installation

### Prerequisites
- Python 3.8+
- Graphcore Poplar SDK (v3.0+)
- Docker (for IPU emulation)

```bash
# Clone repository
git clone https://github.com/<your-username>/ml-ci-cd-graphcore.git
cd ml-ci-cd-graphcore

# Install dependencies
pip install -r requirements.txt

# Install Graphcore PopTorch
pip install poptorch