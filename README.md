# Quantum vs Classical, Autoregressive vs Reconstructive Encoder/Decoder Architectures on Non-Stationary Time-Series: Convergence, Entropy, and Complexity with and without Recurrence
## Contents
- [Introduction](#introduction)
  - [Hypotheses](#hypotheses)
- [Model Architectures and Methods](#model-architectures-and-methods)
  - [Data Generation](#data-generation)
  - [Hyperparameter Optimizaiton](#hyperparameter-optimization)
- [Abbreviations](#abbreviations)
- [References](#references)
- [Useful Commands](#useful-commands)
## Introduction
Time-series data often exhibit complex, non-stationary patterns that pose challenges for modeling and compression. Autoencoders (AEs) have long been used for unsupervised learning of low-dimensional representations (embeddings) of data (Hinton, 2006). A classical AE is typically trained to reconstruct its input after compressing it through a low-dimensional “bottleneck” layer in order to capture the most salient features of the input distribution. In the autoregressive version (herein called a “transition encoder” — TE), the same model architecture takes the current state as input and is trained to predict the next state of a time series, effectively learning to compress the underlying temporal dynamics.

Quantum equivalents have also emerged (Romero, 2017), exploiting quantum superposition and entanglement to achieve more efficient compression or to learn patterns inaccessible to classical networks. In principle, quantum models can represent certain complex transformations with fewer parameters or different capacities than classical neural networks, although their training dynamics can differ significantly due to quantum effects (e.g., interference, measurement noise). This raises questions about how quantum vs. classical architectures compare when applied to learning non-stationary time-series patterns.

Signal complexity metrics – such as Lempel-Ziv complexity (LZC), Hurst exponent (HE), Higuchi fractal dimension (HFD) and differential entropy (DE) – provide quantitative measures of the unpredictability, self-similarity, or information content of time-series data. For example, LZC (Lempel, 1976) measures the number of unique symbols in a sequence, the HE gauges long-range dependence in time-series (with H=0.5 for random walk, H>0.5 indicating persistent long-term correlations, etc.), and the HFD captures the fractal scaling behavior of a signal while the DE extends Shannon entropy (unpredictability) to the continuous realm. This paper hypothesizes that an effective encoder of a time-series should reflect the complexity characteristics of the time series on which it was trained in its bottleneck (the ratio between the metrics should negatively correlate to the loss). In particular, the above-mentioned complexity metrics are expected to correlate between features extracted from the model bottlenecks over its validation series and the validation series themselves from the same dataset on which the model was trained. A high complexity signal might require the encoder to use a similarly high-complexity latent representation to faithfully capture the signal’s variability, especially in a highly non-stationary regime. For the differential entropy in particular, the quantum versions being able to maintain superpositions and mixtures of basis states in the bottleneck that retain uncertainty (entropy), whereas a classical bottleneck might collapse information more deterministically into a few active features, especially given the linear nature of the model architectures. For further analysis in the quantum realm only, correlations are made between each model's validation series complexity metrics and the mean entanglement and mean full VonNeumann entropy (EE and VNE) of its bottleneck states when going through each series.

In this work, I conduct a comprehensive comparison of eight model variants spanning all combinations of:
- Quantum vs. Classical architectures,
- Autoencoder vs. Transition Encoder (Autoregressive) objectives
- Minimalist Recurrent vs. Non-Recurrent models

By evaluating these models on the same set of time-series data, the aim is to test several key hypotheses about their performance and internal dynamics. Specifically, the central hypothesis is that transition encoders will converge faster during training, achieve lower loss (error), and produce latent representations that better align with established complexity metrics of the data, compared to traditional autoencoders. The intuition is that because transition encoders are explicitly trained on the temporal relationships (the state-to-state transitions), they intrinsically have better access to the underlying non-stationary patterns and dynamics of the sequence. This should make it easier for them to learn structure that a stationary autoencoder might miss if the distribution shifts over time. An additional hypothesis is that recurrence being added to the autoencoder objective will not be enough to outcompete the autoregressive versions without recursion, especially since the implemented recursion adds only a single extra parameter.

Furthermore, I investigate expected differences between quantum and classical variants. Quantum models, implemented via parametrized quantum circuits, often exhibit highly non-linear loss landscapes. Despite attempting to increase the similarity of the classical and quantum loss landscapes by mixing the effects of the classical parameters in the linear layers [(details)](#model-architectures-and-methods), I expect that the quantum variants will still have more rugged loss landscapes, characterized by larger first and second derivatives and more high-frequency content in the derivatives spectrum, compared to their classical analogues. This expectation comes mostly from parameter entanglement causing sharpness.

In summary, this paper drafts a systematic study of how learning objective (reconstruction vs. prediction), sequence modeling capability (recurrent vs. not), and computational paradigm (quantum vs. classical) affect performance when trained on complex, highly nonstationary time-series. The experimental setup and initial expectations are outlined below, and provide a structured framework for results and analysis.

### Hypotheses
This study evaluates all 2×2×2 combinations of the above factors, and is, to my knowledge, the first to directly compare quantum and classical sequence encoder-decoder (ENC-DEC) architectures on non-stationary data. The key hypotheses examined include:
- **Prediction vs Reconstruction Objective:** Autoregressive tasks (predicting the next state) will converge faster and attain lower final cost than reconstruction on the same highly non-stationary data. I posit that, in such a highly non-stationary regime, predicting the next state is an easier and more informative task for capturing the dynamics relevant to the task than reconstructing the input is. A transition-based training signal directly emphasizes learning the evolving pattern, which should also translate into better alignment with the dataset’s complexity (e.g., chaotic or highly complex sequences should force the model to utilize its capacity more effectively). This paper will also assess if transition encoders indeed show improved training efficiency (more negative slope to the reconstruction/prediction loss history). Additionally, we will examine whether the learned latent representations from transition models correlate more strongly with known complexity measures (higher latent entropy for higher complexity data), supporting the idea that they “encode complexity” more faithfully than an ENC-DEC trained on reconstruction.
- **Recurrent vs Feedforward:**
- **Quantum vs Classical:**


## Model Architectures and Methods
To make a fair comparison between classical and quantum approaches, the classical network architecture is designed to mirror the quantum one in terms of structure and parameter count as closely as possible. Both are essentially encoder-decoder pairs with one or more layers (blocks) of transformation in the encoder and a corresponding inverse transformation in the decoder. All models have their latent dimensionality enforced via a cost function term - heretoforth referred to as the "bottleneck trash feature penalty" (BTFP) loss term. The key differences are in the way that correlationary coupling is introduced, the size of the space in which each fetaure's rotations can happen (the Hilbert space of the quantum architecture being exponentially larger) and the exact function used to calculate the BTFP. For simiplicity, all architectures are restricted to a linear regime. Even the recurrence is linear, being implemented as minimalistically as possible with a single scalar parameter that is transformed to be a value between 0 and 1 using a sigmoid function. This effective free parameter range enforcement is necessary for the quantum bottleneck state to maintain validity after the recurrence operation and is copied to the classical version for parity. Other than this, parameter count parity is maintained across all models. However this single extra parameter does constitute an entire 1/8 extra parameters over the non-recurrent models because the parameter counts were kept low in order to keep the training time per model low enough that 8 * (number of datasets) models could be trained in time. During training, the weighting for the trash feature information penalty starts just above 0 and is linearly increased each epoch until it reaches a desired maximum. This helps prevent mode collapse while still allowing time to learn the encoding before having to learn to compress.

For simplicity in creating a classical analogue, the quantum architecture restricts each block to a layer with single-qubit rotation gates followed by a linear entanglement layer and embeds each feature into its own qubit. It's BTFP is calculated by sorting the marginal probability of the bottleneck density matrix (between the encoder and decoder) and then summing the (number of qubits - bottleneck size) lowest marginals. All quantum circuits were simulated on classical hardware in order to rule out noise as a complicating factor for the conclusions and due to the difficulty in implementing a dynamic trash qubit index determination without the use of a simulator.

In the classical architecture, each block is a linear layer with a specified even number of free parameters equal to both the input and output dimension. To achieve this, they are split in half between an two smaller matrices, each with half dimensions. One is an orthogonal rotation matrix created via a Cayley transform and the other is a diagonal scaling matrix. The core transformation is a scaled rotation. The dimensionality of this core transformation is then increased by repeatedly adding (with overlap) the lower-dimensional result along the diagonal of the higher dimensional matrix so that each individual node has a unique rotation while still forcing the free parameters to be "coupled" across the layer. In the figure below, only the 4x4 part in the dark black border is kept. The third rotation of the core matrix when increasing the dimensionality is to ensure that each feature gets a unique rotation. ![figure x](./images/core-linear-layer-rotating-diagonal-sum-dimensionality-increase.png)

While this is not a perfect analogue to the quantum architecture — since in a quantum system the qubits themselves are inherently correlated — it does allow a type of correlation between the effects of each feature's rotations. This engineered coupling mimics, to some extent, the way local gate parameters interact in quantum circuits, though it does not reproduce the full complexity of quantum entanglement. In the Cayley transform parameterization, the coupling is a consequence of the mapping process itself. In entanglement, the coupling arises due to the physical evolution governed by the Hamiltonian of an interacting system and the tensor product structure of the Hilbert space.
- In a quantum system, the non-separability of the state (entanglement) is a fundamental property with deep implications—for example, violating Bell inequalities and enabling non-classical information processing. These correlations are intrinsic to the quantum state and are subject to rules of quantum mechanics.
- In the classical layer used here, the free parameters are used in a structured manner (overlapping diagonal addition), which means that the effect of a rotation in one node influences the outputs at nearby nodes. This creates a kind of coupling of the effects of the rotations, but it is not the same as having correlated quantum states. It is a controlled and engineered correlation on the level of the transformation, rather than on the level of the data (or state vectors).
- The classical version of "coupling" used is most similar to linear entanglement topology so use that for quantum version.

!TODO! Compare Lie Algebras

The BTFP for the classical architecture is the sum of the lowest activation values in the bottleneck.

### Data Generation
Multivariate time series are synthesized by concatenating blocks where each feature is a separate fractional Brownian motion (FBM) series, which is a zero‑mean Gaussian process characterized by a target HE to control long‑range dependence. The mean and variance of the block are then set to different values per feature and change between each consecutive block to induce nonstationarity based on another FBM sequence. Based on the dataset index, its generated series progressively include fewer unique blocks with tiling enforcing a fixed length, which slowly decreases the maximum possible Lempel-Ziv complexity value. Sequences are then randomly shuffled and then representative sequences are then selected via 3D binning in the space of LZC, HE, and Higuchi fractal dimension (HFD). These sample sequences are ensured to be in the validation set so that there is a good spread of metric values to use when looking at relationships during the analysis. In order to ensure a reasonable amount of training data for each dataset, the grid was limited to choosing at most a third of the series in each dataset. The unchosen sequences are then split between each dataset's training and validation partitions as close as possible to a desired split ratio and the size of each validation partition is then standardized to the maximum validation partition size. For the experimental results in this paper, a ratio of 2/3 training to 1/3 validation is used.

### Hyperparameter Optimization
- hyperband
- one example from training and one example from validation partitions of each dataset
- ignore configurations that lead to obvious overfitting when number of trained epochs is ≥ 10 by returning infinite cost for any configuration whose validation cost is higher than its final training cost by more than 50% of its training cost

## Results

## Discussion

## Conclusion

## Future Work
- Multiple blocks per model half
- More features w/ different entanglement topologies and equivalent classical "coupling" mechanisms
- Add nonlinearities
- Attempt to use data generation that has opposing trends for complexity metric targets (target HE)

## Abbreviations
- AE = Auto-encoder
- BTFP = Bottleneck Trash Feature Penalty
- DE = Differential Entropy
- EE = Entanglement entropy
- ENC-DEC = Encoder-Decoder
- FBM = Fractional Brownian Motion
- HE = Hurst Exponent
- HFD = Higuchi Fractal Dimension
- LZC = Lempel-Ziv Complexity
- TE = Transition Encoder (autoregressive encoder / decoder)
- VNE = VonNeumann Entropy

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