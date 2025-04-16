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
To make a fair comparison between classical and quantum approaches, the classical network architecture is designed to mirror the quantum one in terms of structure and parameter count as closely as possible. Both are essentially encoder-decoder pairs with one or more layers (blocks) of transformation in the encoder and a corresponding inverse transformation in the decoder. All models have their latent dimensionality enforced via a cost function term - heretoforth referred to as the "bottleneck trash feature penalty" (BTFP) loss term. The key differences are in the way that correlationary coupling is introduced, the size of the space in which each fetaure's rotations can happen (the Hilbert space of the quantum architecture being exponentially larger) and the exact function used to calculate the BTFP.

For simplicity in creating a classical analogue, the quantum architecture restricts each block to a layer with single-qubit rotation gates followed by an entanglement layer (always of the same topology determined by hyperparameter optimization) and embeds each feature into its own qubit. It's BTFP is calculated by sorting the marginal probability of the bottleneck density matrix (between the encoder and decoder) and then summing the (number of qubits - bottleneck size) lowest marginals.

In the classical architecture, each block is a linear layer with a specified even number of free parameters equal to both the input and output dimension. To achieve this, they are split in half between an two smaller matrices, each with half dimensions. One is an orthogonal rotation matrix created via a Cayley transform and the other is a diagonal scaling matrix. The dimensionality of this core transformation is then increased by repeatedly adding (with overlap) the lower- dimensional result along the diagonal of the higher dimensional matrix so that each individual node has a unique rotation while still forcing the free parameters to be "coupled" across the layer. While this is not a perfect analogue to the quantum architecture — since in a quantum system the qubits themselves are inherently correlated — it does allow a type of interference between the effects of the rotations. This engineered coupling mimics, to some extent, the way local gate parameters interact in quantum circuits, though it does not reproduce the full complexity of quantum entanglement. In the Cayley transform parameterization, the coupling is a consequence of the mapping process itself. In entanglement, the coupling arises due to the physical evolution governed by the Hamiltonian of an interacting system and the tensor product structure of the Hilbert space.
- In a quantum system, the non-separability of the state (entanglement) is a fundamental property with deep implications—for example, violating Bell inequalities and enabling non-classical information processing. These correlations are intrinsic to the quantum state and are subject to rules of quantum mechanics.
- In the classical layer used here, the free parameters are used in a structured manner (overlapping diagonal addition), which means that the effect of a rotation in one node influences the outputs at nearby nodes. This creates a kind of coupling of the effects of the rotations, but it is not the same as having correlated quantum states. It is a controlled and engineered correlation on the level of the transformation, rather than on the level of the data (or state vectors).

The BTFP for the classical architecture is the sum of the lowest activation values in the bottleneck.

## Abbreviations
- AE = Auto-encoder
- BTFP = Bottleneck Trash Feature Penalty
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