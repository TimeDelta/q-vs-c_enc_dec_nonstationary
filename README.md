# Quantum vs Classical, Autoregressive vs Reconstructive Encoder/Decoder Architectures on Non-Stationary Time-Series: Convergence, Entropy, and Complexity with and without Recurrence
## Introduction
Time-series data often exhibit complex, non-stationary patterns that pose challenges for modeling and compression. Autoencoders (AEs) have long been used for unsupervised learning of low-dimensional representations (embeddings) of data (Hinton, 2006). A classical AE is typically trained to reconstruct its input after compressing it through a low-dimensional “bottleneck” layer in order to capture the most salient features of the input distribution. In the autoregressive version (herein called a “transition encoder” — TE), the same model architecture takes the current state as input and is trained to predict the next state of a time series, effectively learning to compress the underlying temporal dynamics.

Quantum equivalents have also emerged (Romero, 2017), exploiting quantum superposition and entanglement to achieve more efficient compression or to learn patterns inaccessible to classical networks. In principle, quantum models can represent certain complex transformations with fewer parameters or different capacities than classical neural networks, although their training dynamics can differ significantly due to quantum effects (e.g., interference, measurement noise). This raises questions about how quantum vs. classical architectures compare when applied to learning non-stationary time-series patterns.

Signal complexity metrics – such as Lempel-Ziv complexity, Hurst exponent, fractal dimension and differential entropy – provide quantitative measures of the unpredictability, self-similarity, or information content of time-series data. For example, Lempel-Ziv complexity (Lempel, 1976) measures the number of unique symbols in a sequence, the Hurst exponent gauges long-range dependence in time-series (with H=0.5 for random walk, H>0.5 indicating persistent long-term correlations, etc.), and the Higuchi fractal dimension captures the fractal scaling behavior of a signal while the differential entropy extends Shannon entropy (unpredictability) to the continuous realm. This paper hypothesizes that an effective encoder of a time-series should reflect the complexity characteristics of the time series on which it was trained in its bottleneck (the ratio between the metrics should negatively correlate to the loss). In particular, the above-mentioned complexity metrics are expected to correlate between features extracted from the model bottlenecks over its validation series and the validation series themselves from the same dataset on which the model was trained. A high complexity signal might require the encoder to use a similarly high-complexity latent representation to faithfully capture the signal’s variability, especially in a highly non-stationary regime.

In this work, I conduct a comprehensive comparison of eight model variants spanning all combinations of:
- Quantum vs. Classical architectures,
- Autoencoder vs. Transition Encoder (Autoregressive) objectives
- Minimalist Recurrent vs. Non-Recurrent models

By evaluating these models on the same set of time-series data, the aim is to test several key hypotheses about their performance and internal dynamics. Specifically, the central hypothesis is that transition encoders will converge faster during training, achieve lower loss (error), and produce latent representations that better align with established complexity metrics of the data, compared to traditional autoencoders. The intuition is that because transition encoders are explicitly trained on the temporal relationships (the state-to-state transitions), they intrinsically have better access to the underlying non-stationary patterns and dynamics of the sequence. This should make it easier for them to learn structure that a stationary autoencoder might miss if the distribution shifts over time. An additional hypothesis is that recurrence being added to the autoencoder objective will not be enough to outcompete the autoregressive versions without recursion, especially since the implemented recursion adds only a single extra parameter.

Furthermore, I investigate expected differences between quantum and classical variants. Quantum models, implemented via parametrized quantum circuits, often exhibit highly non-linear parameter landscapes. Despite attempting to increase the similarity of the classical and quantum loss landscapes by mixing the effects of the classical parameters in the linear layers [(details)](#quantum_vs_classical), I expect that the quantum variants will still have more rugged loss landscapes, characterized by larger first and second derivatives and more high-frequency content in the derivatives spectrum, compared to their classical analogues. This expectation comes mostly from parameter entanglement causing sharpness.

## Model Architectures and Methods
### Quantum vs Classical
To make a fair comparison between classical and quantum approaches, the classical network architecture is designed to mirror the quantum one in terms of structure and parameter count as closely as possible. Both are essentially encoder-decoder pairs with one or more layers (blocks) of transformation in the encoder and a corresponding inverse transformation in the decoder. The key difference is that the quantum encoder/decoder applies a unitary quantum circuit, whereas the classical encoder/decoder applies an orthogonal linear transform.


## Abbreviations
- AE = Auto-encoder
- TE = Transition Encoder (autoregressive encoder / decoder)

## References (BibTex)
1. @article{
doi:10.1126/science.1127647,
author = {G. E. Hinton  and R. R. Salakhutdinov },
title = {Reducing the Dimensionality of Data with Neural Networks},
journal = {Science},
volume = {313},
number = {5786},
pages = {504-507},
year = {2006},
doi = {10.1126/science.1127647},
URL = {https://www.science.org/doi/abs/10.1126/science.1127647},
eprint = {https://www.science.org/doi/pdf/10.1126/science.1127647},
abstract = {High-dimensional data can be converted to low-dimensional codes by training a multilayer neural network with a small central layer to reconstruct high-dimensional input vectors. Gradient descent can be used for fine-tuning the weights in such “autoencoder” networks, but this works well only if the initial weights are close to a good solution. We describe an effective way of initializing the weights that allows deep autoencoder networks to learn low-dimensional codes that work much better than principal components analysis as a tool to reduce the dimensionality of data.}}
1. @article{Romero_2017,
title={Quantum autoencoders for efficient compression of quantum data},
volume={2},
ISSN={2058-9565},
url={http://dx.doi.org/10.1088/2058-9565/aa8072},
DOI={10.1088/2058-9565/aa8072},
number={4},
journal={Quantum Science and Technology},
publisher={IOP Publishing},
author={Romero, Jonathan and Olson, Jonathan P and Aspuru-Guzik, Alan},
year={2017},
month=aug, pages={045001} }
1. @article{1055501,
author={Lempel, A. and Ziv, J.},
journal={IEEE Transactions on Information Theory},
title={On the Complexity of Finite Sequences},
year={1976},
volume={22},
number={1},
pages={75-81},
keywords={},
doi={10.1109/TIT.1976.1055501}}

## Useful Commands
To determine how many datasets need models trained over them based on the saved grid files ("series_cell_..."), use `find generated_datasets.4qubits/ -iname series_cell_\*dataset\*.npy | sed -E 's/.*series_cell.*dataset(.*)\.npy/\1/' | sort | uniq | wc -l`