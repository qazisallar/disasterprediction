# Disaster Risk Modeling with Satellite Imagery

This project implements a deep learning pipeline to detect flood events using satellite imagery. Developed through NVIDIA DLI in collaboration with UNOSAT (United Nations Satellite Centre), the solution applies transfer learning and hardware-accelerated inference for near real-time disaster response.

## Key Features
- Semantic segmentation of flooded regions from multi-band satellite imagery
- Transfer learning using NVIDIA TAO Toolkit
- Inference accelerated via TensorRT and deployed using Triton Inference Server
- Image preprocessing with NVIDIA DALI for high-throughput pipelines

## Impact
The pipeline enables cost-effective and scalable flood detection, supporting governments and NGOs in disaster management scenarios.

## Tools & Libraries
- PyTorch, NVIDIA TAO Toolkit
- TensorRT, Triton Inference Server
- DALI, OpenCV, NumPy
