# ML-Projects

Machine learning projects by Hugo Simonet: anomaly detection, recommendation, graph neural networks, federated learning, computer vision, NLP, reinforcement learning and MLOps.

Portfolio: [hugo-simonet.vercel.app](https://hugo-simonet.vercel.app)

## Projects with results

| Project | What it does | Headline result |
|---|---|---|
| [Anomaly Detection System](anomaly_detection_system) | Autoencoders, VAE, Isolation Forest, One-Class SVM and LOF behind one evaluation framework | **F1 0.989, AUC 1.000** on manufacturing defects. The other benchmarks (credit-card fraud, network intrusion, IoT sensors) and their caveats are in [UNDERSTANDING_RESULTS.md](anomaly_detection_system/docs/UNDERSTANDING_RESULTS.md) |
| [Federated Learning](Federated-Learning) | FedAvg across 5 simulated clients, with differential-privacy support | **99.33%** global test accuracy on MNIST after 5 rounds; 56 tests pass |
| [Recommendation System](Recommendation-System) | User/item collaborative filtering, content-based, SVD, ALS and neural CF, served by a FastAPI API | **SVD: RMSE 0.950, MAE 0.744** on MovieLens 100K ([results](Recommendation-System/RESULTS.md)) |
| [Graph Neural Networks](Graph-Neural-Networks) | GCN, GAT and GraphSAGE for node classification | **GAT: 74.6% accuracy, F1 0.742** on Cora (GraphSAGE 74.4%, GCN 72.8%) |
| [CIFAR-10](CIFAR-10) | CNN image classifier with training curves and error analysis | [Results and plots](CIFAR-10/RESULTS.md) |
| [License Plate Recognition](ALPR) | Classical computer vision to find the plate, plus an SVC to read each character | [Pipeline and results](ALPR/RESULTS.md) |
| [MNIST Neural Network](MNIST-NN) | A neural network written from scratch in NumPy: initialization, forward pass, backpropagation | Notebook walkthrough |

## Code complete, results not yet published

| Project | What it does | Status |
|---|---|---|
| [MLOps Pipeline](MLOps-Pipeline) | Training, deployment, monitoring and A/B testing, with Docker and Kubernetes configs | 10 tests pass |
| [RL Trading Agent](RL-Trading-Agent) | DQN, PPO and A3C trading agents with risk management and backtesting | Imports cleanly; no published backtest yet |
| [GAN Data Augmentation](GAN-Data-Augmentation) | GANs that generate synthetic training data | Imports cleanly; no published results yet |
| [Medical CV Diagnosis](Medical-CV-Diagnosis) | Medical image classification with uncertainty and explainability | Needs `opencv-python`; no published results yet |
| [NLP with LLMs](NLP-LLM-Applications) | Fine-tuning and prompting GPT-2, BERT and T5 | Needs `transformers`; early results only |
| [Time Series Anomaly Detection](Anomaly-Detection) | Transformer forecasting with STL and Isolation Forest anomaly detection | Imports cleanly; no published results yet |

## In progress

| Project | Status |
|---|---|
| [Time Series Transformers](Time-Series-Transformers) | Informer and Autoformer implementations; the benchmark isn't finished |
| [Multi-Modal VLM](Multi-Modal-VLM) | Captioning, VQA and retrieval; the `utils/checkpoint` module still needs writing |

---

Status reflects an audit on 2026-11-23: every file compiles, modules import, and tests pass where noted. The `.gitignore` keeps new datasets and model weights out of the repo while tracking the source code in `data/` and `models/` folders.
