A university research project exploring the trade-off between weight sparsity and classification accuracy in convolutional neural networks. The MLP classifier head uses a custom SparseDense layer that applies a fixed binary mask at initialisation, zeroing a configurable fraction of weights before training begins.
Key features:

Full sparsity sweep (0% → 100%) with multiple seeded runs per level for statistically robust results
He-uniform initialisation re-scaled to account for reduced effective fan-in
Linear warmup + cosine decay LR schedule with early stopping
Data augmentation (random flip + translation) applied only during training
Outputs: accuracy/loss plots, per-sparsity learning curves, confusion matrix, and a CSV results table

Built with TensorFlow/Keras, NumPy, pandas, and Matplotlib.
