# AI--Learning
Learning AI tokenization with LangChain and HuggingFace transformers

# MNIST GAN — Generating Handwritten Digits from Noise

A simple Generative Adversarial Network (GAN) trained on MNIST,
built from scratch in PyTorch. Trained in ~2 minutes on Google Colab's
free T4 GPU.

## What it does

Starting from random Gaussian noise, the trained Generator produces
recognizable handwritten digits.

| Before training (random noise) | After 10 epochs |
|---|---|
| ![noise](images/epoch_0.png) | ![digits](images/epoch_10.png) |

## The math behind it

GANs play a **minimax game** between two networks:

```
min_G max_D  E[log D(x)]  +  E[log(1 - D(G(z)))]
```

- **Generator (G):** Maps random noise z ~ N(0, I) → fake image G(z)
- **Discriminator (D):** Classifies images as real or fake

At equilibrium, the Generator produces images so realistic that the
Discriminator can't distinguish them from real MNIST digits.

(Theoretically equivalent to minimizing Jensen-Shannon divergence
between real and generated distributions.)

## Architecture

- **Generator:** MLP — 100 → 256 → 512 → 1024 → 784, LeakyReLU + Tanh
- **Discriminator:** MLP — 784 → 1024 → 512 → 256 → 1, LeakyReLU + Dropout + Sigmoid
- **Loss:** Binary Cross-Entropy
- **Optimizer:** Adam (lr=2e-4, betas=(0.5, 0.999))
- **Batch size:** 128
- **Epochs:** 10

## Observations

- **Training dynamics:** 3 phases clearly visible — D dominance, G recovery, equilibrium.
- **Partial mode collapse:** Generator favors simpler digits (1, 7, 4) over harder ones (0, 5, 8).
- **Training time:** 2.3 min on Tesla T4 GPU.

## How to run

1. Open `gan_mnist.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Set Runtime → Change runtime type → T4 GPU
3. Run all cells

## What I learned

Built this as part of my AI/ML self-study (Section 3 of a 9-section roadmap).
Watching real failure modes (vanishing gradients in early training, mode bias
in final samples) made the GAN theory click in a way that reading papers never did.

## Built with

Python • PyTorch • torchvision • matplotlib • Google Colab
