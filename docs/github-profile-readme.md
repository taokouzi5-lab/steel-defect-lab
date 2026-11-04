# Hi, I'm Chunqi Sun

I build carbon-aware machine learning systems for manufacturing — measure the
footprint first, then engineer it down.

<!-- DEMO_LINK: replace after ModelScope deployment -->

**Live demo:** the DefectLens demo is being deployed to ModelScope 创空间;
this line is the placeholder for the URL and will be updated after deployment.

## What I'm building

Two repositories that form one story: a measurement framework, then an
engineering validation of it.

### 1. [MACE-F](https://github.com/taokouzi5-lab/mace-f) — measure first

A manufacturing AI carbon emission framework: 26 experiments across 10 model
families, five model tiers (T1–T5) and seven methods (A–G), measured with
CodeCarbon 3.2.7 and calibrated against external power-meter methodology.
It answers *how much carbon does this model choice actually cost?* across
training, inference, deployment location and facility topology.

### 2. [DefectLens](https://github.com/taokouzi5-lab/steel-defect-lab) — engineering validation

An end-to-end steel surface defect detection platform that puts the framework
to work: ResNet-18/50 training on NEU-DET, accuracy-gated INT8 quantization,
hand-written Grad-CAM explanations, a FastAPI + React single-container
deployment, and a per-inference carbon estimate disclosed in every API
response. It shows the MACE-F constants and decision logic running inside a
real service — including the negative result that INT8 bought 4× smaller
artifacts but no reliable CPU latency win.

## Stack

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU-EE4C2C?logo=pytorch&logoColor=white)
![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-INT8-005CED)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)
![TypeScript](https://img.shields.io/badge/TypeScript-5%2B-3178C6?logo=typescript&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-single%20container-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Links

- MACE-F — [github.com/taokouzi5-lab/mace-f](https://github.com/taokouzi5-lab/mace-f)
- DefectLens — [github.com/taokouzi5-lab/steel-defect-lab](https://github.com/taokouzi5-lab/steel-defect-lab)
- DefectLens demo — placeholder, to be replaced after ModelScope deployment
