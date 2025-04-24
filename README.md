# Quantum vs Classical, Autoregressive vs Reconstructive Encoder/Decoder Architectures on Non-Stationary Time-Series: Loss Landscapes and Latent Complexities with and without Recurrence
## Contents
- [Introduction](#introduction)
  - [Hypotheses](#hypotheses)
- [Methods](#methods)
  - [Model Architectures](#model-architectures)
  - [Data Generation](#data-generation)
  - [Hyperparameter Optimization](#hyperparameter-optimization)
  - [Training](#training)
  - [Experimental Environment](#experimental-environment)
  - [Analysis](#analysis)
    - [Quantization Methods](#quantization-methods)
    - [Metric Normalization](#metric-normalization)
- [Results](#results)
  - [Classical vs Quantum](#classical-vs-quantum)
    - [Loss Landscape Similarity](#loss-landscape-similarity)
  - [Prediction vs Reconstruction](#prediction-vs-reconstruction)
  - [Recurrence](#recurrence)
- [Discussion](#discussion)
  - [Sources of Error](#sources-of-error)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [Abbreviations](#abbreviations)
- [References](#references)
- [Useful Commands](#useful-commands)
## Introduction
Time-series data often exhibit complex, non-stationary patterns that pose challenges for modeling and compression.
Autoencoders (AEs) have long been used for unsupervised learning of low-dimensional representations (embeddings) of data (Hinton, Salakhutdinov; 2006).
A classical AE is typically trained to reconstruct its input after compressing it through a low-dimensional “bottleneck” layer in order to capture the most salient features of the input distribution.
In the autoregressive (AR) version, the same model architecture takes the current state as input and is trained to predict the next state of a time series, effectively learning to compress the underlying temporal dynamics.

Quantum equivalents have also emerged (Romero, Olsen, Aspuru-Guzik; 2017), exploiting quantum superposition and entanglement to achieve more efficient compression or to learn patterns inaccessible to classical networks.
In principle, quantum models can represent certain complex transformations with fewer parameters or different capacities than classical neural networks, although their training dynamics can differ significantly due to quantum effects (e.g., interference, measurement noise).
This raises questions about how quantum vs. classical AE architectures compare when applied to learning compression of non-stationary time-series.

This work conducts a comprehensive comparison on non-stationary data of eight sequence encoder-decoder (ENC-DEC) variants spanning all 2×2×2 combinations of:
- Quantum vs. Classical architectures
- Prediction (AR) vs. Reconstruction (standard AE) objectives
- Minimalist Recurrent vs. Non-Recurrent models

By evaluating these models on the same sets of time-series data, the aim is to test several key hypotheses about their performance and internal dynamics.

Four main metrics are used to quantify time-series complexity:
- **Lempel‑Ziv Complexity (LZC):** Number of unique substrings needed to span a discrete sequence (Lempel, Ziv; 1976).
- **Hurst Exponent (HE):** Measures long-range dependence (H=0.5 for random walk; H > 0.5 indicates persistence) (Hurst; 1951).
- **Optimized Multiscale Permutation Entropy (MPE):** Extends classic permutation entropy by computing the Shannon entropy of ordinal-pattern distributions over multiple coarse-grained versions of the signal—thereby capturing scale-dependent structural complexity without any amplitude binning (Want et al.; 2019).
- **Differential Entropy (DE):** Extension of Shannon entropy to continuous-valued data.

In summary, this paper drafts a systematic study of how learning objective (reconstruction vs. prediction), sequence modeling capability (recurrent vs. not), and computational paradigm (quantum vs. classical) affect performance when trained on complex, highly nonstationary time-series data.
The experimental setup and initial expectations are outlined below, and provide a structured framework for results and analysis.

### Hypotheses
The key hypotheses examined include:
> - ***Latent Complexity Matching:*** An effective encoder of a time-series is hypothesized to reflect the complexity characteristics of the time series on which it was trained in its bottleneck.
Specifically, the absolute squared difference between each metric over each model's validation data and over the model's latent representations from the validation data will positively correlate to the loss.
The intuition behind this prediction is an expectation that a high complexity signal should require the encoder to use a similarly high-complexity latent representation to faithfully capture the signal's variability, especially in a highly non-stationary regime where the variability is volatile.
> - ***Prediction vs Reconstruction Objective:*** AR tasks (predicting the next state) are expected to converge faster and attain lower final cost than reconstruction on the same highly non-stationary data.
In such a highly non-stationary regime, predicting the next state is an easier and more informative task for capturing the dynamics relevant to the task than reconstructing the input is.
A transition-based training signal directly emphasizes learning the evolving pattern, which should produce a closer match (lower mean squared difference) between the dataset's complexity metrics and those of the model's bottleneck representations since the chosen metrics (except for DE) are based on state ordering, which is something the prediction objective emphasizes but the reconstruction objective does not.
Because it does not directly rely on the ordering of states but instead focuses on the global distribution, an additional hypothesis is that the advantage seen by the prediction-trained models will be less pronounced for DE (see [metric normalization](#metric-normalization)).
To assess if ENC-DEC models trained on the predictive task indeed show improved training efficiency, a check is done for more negative mean slope to the prediction vs reconstruction loss history as well as for less area under the loss history curve.
Another hypothesis this paper proposes is that the learned latent representations from prediction-trained models correlate more strongly with the chosen complexity metrics than the reconstruction-trained ones, supporting the idea that they “encode complexity” more faithfully than an ENC-DEC trained on reconstruction.
Specifically, a check will be made for higher positive slope for first-order line of best fit between each complexity metric calculated over the validation series as well as for a lower mean squared difference between the metric over each series and the metric over the model's bottleneck states.
> - ***Recurrent vs Feedforward:*** Recurrent models will exacerbate the ruggedness of the landscapes (increased steepness, curvature, etc.) but ultimately outperform non-recurrent models in final loss because the hidden state contains temporal information relevant to the nonstationarity.
However, the difference will be smaller in the quantum architecture because of the facts that its latent state is exponentially larger than its classical counterpart despite having a single qubit per feature and that the quantum embedding only uses single-qubit rotation gates, thus preventing full utilization of the embedding space.
Moreover, recurrent encoders will exhibit different bottleneck characteristics, potentially higher entropy, since the hidden state provides an additional pathway to carry information.
Another hypothesis is that recurrence being added to the autoencoder objective will not be enough to outcompete the autoregressive versions without recursion, especially since the implemented recursion adds only a single extra parameter.
> - ***Quantum vs Classical:*** In their cost histories, quantum models will have higher mean absolute first and second derivatives during training because they often exhibit highly non-linear loss landscapes (Holzer, Turkalj; 2024) despite attempting to increase the similarity of the classical and quantum loss landscapes by mixing the effects of the classical parameters in the linear layers [(details)](#model-architectures).
Upon analyzing the power spectrum of the loss gradients over training iterations, an additional expectation is to see more high-frequency fluctuations for quantum models, reflecting parameter interference effects and more rapid changes as the quantum circuit parameters navigate a more rugged loss landscape.
For quantum models, an additional hypothesis is that both the Meyer-Wallach global entanglement (MWGE) and the VonNeumann entropy (VNE) of their bottleneck representations should positively correlate to the main complexity metrics (LZC, HE, MPE, DE).
This is expected for DE in particular because the quantum versions being able to maintain superpositions and mixtures of basis states in the bottleneck that retain uncertainty (entropy), whereas a classical bottleneck might collapse information more consistently into a few active features, especially given the linear nature of the model architectures.
A final expectation is that the quantum models will generalize more poorly (higher ratio of normalized validation cost: normalized training cost) due to the expected higher loss landscape complexity.
To rule out scale differences between the quantum and classical BTFP functions during analysis, their values are normalized by the initial trash cost.

## Methods
### Model Architectures
To make a fair comparison between classical and quantum approaches, the [classical network architecture](./models.py#L212) is designed to mirror the [quantum one](./models.py#L14) in terms of structure and parameter count as closely as possible.
Both are essentially ENC-DEC pairs with one or more layers (blocks) of transformation in the encoder and a corresponding inverse transformation in the decoder.
All models have their latent dimensionality enforced via a cost function term - heretoforth referred to as the "bottleneck trash feature penalty" (BTFP) loss term.
During training, the weighting for the trash feature information penalty starts just above 0 and is linearly increased each epoch until it reaches a desired maximum.
This helps prevent mode collapse while still allowing time to learn the encoding before having to learn to compress.
The key differences are in the way that correlationary coupling is introduced, the size of the space in which each fetaure's rotations can happen (the Hilbert space of the quantum architecture being exponentially larger) and the exact function used to calculate the BTFP.
For simiplicity, all architectures are restricted to a linear regime.
Even the recurrence is linear, being implemented as minimalistically as possible with a single scalar parameter that is transformed to be a value between 0 and 1 using a sigmoid function.
This effective free parameter range enforcement is necessary for the quantum bottleneck state to maintain validity after the recurrence operation and is copied to the classical version for parity.
Other than this, parameter count parity is maintained across all models.
However this single extra parameter does constitute an entire 1/8 extra parameters over the non-recurrent models because the parameter counts were kept low in order to keep the training time per model low enough that 8 * (number of datasets) models could be trained in time as the experiment is performed on an 8-year-old laptop.

For simplicity in creating a classical analogue, the quantum architecture restricts each block to a layer with single-qubit rotation gates followed by a ring entanglement layer and embeds each feature into its own qubit.
[It's BTFP](./loss.py#L27) is calculated by sorting the marginal probability of the bottleneck density matrix (between the encoder and decoder) and then summing the (number of qubits - bottleneck size) lowest marginals.
All quantum circuits were simulated on classical hardware in order to rule out noise as a complicating factor for the conclusions and due to the difficulty in implementing a dynamic trash qubit index determination without the use of a simulator.

In the classical architecture, each block is a [RingGivensRotationLayer](./models.py#L178) with a number of free parameters equal to both the input and output dimension.
Unfortunately, due to dimensionality constraints, it is not possible to maintain perfect transformational parity with the quantum ansatz.
This experiment breaks that parity by not enforcing the special orthogonal (SO) Lie group for the per-feature rotations but still maintains global SO adherance for the final weight matrix.
SO is a subgroup of the unitary (U) Lie group (lies inside it's manifold since the complex part is 0).
It is well documented that Givens matrices have the U Lie group (Givens; 1958).
This layer construction uses each free parameter as an angle in a n/2 × n/2 Givens matrix and constructs unique rotations for each feature (as in the quantum ansatz).
Importantly, the SO group is preserved under matrix multiplication (Golub, Van Loan; 2013).
This allows the use of individual n/2×n/2 SO matrices (one per feature) embedded along the diagonal of the full-dimensional identity matrix in a banded fashion, which most closely resembles ring entanglement in the final linear layer weight matrix (see [definition of planes in a ring](./models.py#L192) and Figure figure_num_a below) to be multiplied together while still restricting to the SO group at the global level.
Unfortunately, this necessarily breaks the enforcement of unitarity on each individual rotation due the the afore-mentioned dimensionality constraints.
![Figure figure_num_b: Construction of a Single Givens Rotation Matrix Embedded in a Larger Matrix](./images/givens-rotation-matrix-construction.png)
*Figure figure_num_b: Construction of a Single Givens Rotation Matrix Embedded in a Larger Matrix.*
*Source: Stanford Exploration Project, “Givens Rotations,” Stanford University, December 18, 1997. [https://sepwww.stanford.edu/data/media/public/docs/sep70/steve2/paper_html/node2.html](https://sepwww.stanford.edu/data/media/public/docs/sep70/steve2/paper_html/node2.html)*
![Figure figure_num_a: Influence of Each Free Angle Parameter in the Final Weight Matrix](./images/givens-rotation-param-influence.png)

While this "coupling" is not a perfect analogue to the quantum architecture even beyond the forced break in parity — since in a quantum system the qubits themselves are inherently correlated — it does allow a type of correlation between the effects of each feature's rotations.
This engineered coupling mimics, to some extent, the way local gate parameters interact in quantum circuits, though it does not reproduce the full complexity of quantum entanglement.
In the Banded givens rotations parameterization, the coupling is a consequence of the mapping process itself.
In quantum entanglement, the coupling arises due to the physical evolution governed by the Hamiltonian of an interacting system and the tensor product structure of the Hilbert space.
In a quantum system, the non-separability of the state (entanglement) is a fundamental property with deep implications—for example, violating Bell inequalities and enabling non-classical information processing.
These correlations are intrinsic to the quantum state and are subject to rules of quantum mechanics.

The [BTFP for the classical architecture](./loss.py#L60) is the sum of the lowest activation values in the bottleneck.

### Data Generation
Multivariate time series are synthesized by concatenating blocks where each feature is a separate fractional Brownian motion (FBM) series, which is a nonstationary zero‑mean Gaussian process.
This was done via the Davies-Harte method (Davies, Harte; 1987), which is characterized by a target HE to control long‑range dependence, allowing the creation of and sampling from series with a wide range of values for this metric.
The mean and variance of the block are then set to different values per feature and change between each consecutive block to induce nonstationarity based on another FBM sequence that gets passed through a sine to introduce nonlinearity and control amplitude.
Based on the dataset index, its generated series progressively include fewer unique blocks with tiling enforcing a fixed length, which slowly decreases the maximum possible LZC value.
Sequences are then randomly shuffled and representative sequences are selected via 3D binning in the space of LZC, HE, and MPE (first experiment run was Higuchi Fractal Dimension instead of MPE).
These sample sequences are ensured to be in the validation set so that there is a good spread of metric values to use when looking at relationships during the analysis.
In order to ensure a reasonable amount of training data for each dataset, the grid was limited to choosing at most a third of the series in each dataset.
The unchosen sequences are then split between each dataset's training and validation partitions as close as possible to a desired split ratio and the size of each validation partition is then standardized to the maximum validation partition size.
For the experimental results in this paper, a ratio of 2/3 training to 1/3 validation is used.

### Hyperparameter Optimization
Hyperparameter optimization was done over the same validation set for each config, following the standard definition of hyperparameter tuning.
Hyperband was used to efficiently search this space, which allocates the number of training epochs as a resource and uses successive halving to prune underperforming configurations (Lisha et al.; 2018).
In the implementation used, a series of “brackets” are created based on the maximum number of training epochs and a reduction factor, iterating from the most exploratory (many configurations, few epochs) to the most exploitative (few configurations, many epochs) phases ([`hyperband_search(...)`](./optimize_hyperparams.py#L63)).
- **Data Sampling:** One training and one validation series are randomly sampled from each dataset partition inside the [`get_best_config(...)`](./optimize_hyperparams.py#L111) routine, providing a representative but lightweight evaluation set.
- **Configuration Sampling:** The [`sample_hyperparameters(...)`](./optimize_hyperparams.py#L11) function draws candidate settings by:
  - Sampling the learning rate uniformly in log₁₀-space between 10⁻⁴ and 10⁻¹
  - Setting `bottleneck_size` to half the number of features
  - Selecting `num_blocks` uniformly from {1,…,`MAX_NUM_BLOCKS`} (here, forced to 1 for time constraints)
  - Fixing `max_penalty_weight` at 2.0
  - Randomly choosing `entanglement_gate`, `embedding_gate`, and `block_gate` from predefined lists.
- **Random‑Search Principle:** This stochastic sampling approach is grounded in empirical evidence that random search is more efficient than grid search in high‑dimensional hyperparameter spaces (Bergstra, 2012).
- **Model Evaluation:** Each sampled configuration is evaluated across all eight model variants (quantum/classical × reconstruction/prediction × recurrent/non‑recurrent) by training with the ADAM optimizer ([`train_adam(...)`](./training.py#L23)) and computing the mean training loss from the final `cost_history` entries and validation loss by summing up the per-series values and normalizing each by the size of its partition.
- **Overfitting Detection:** Configurations for which the validation loss exceeds 150% of the training loss after at least 10 epochs are penalized with an infinite cost via, effectively discarding overfitting settings.
- **Complexity Scaling:** To account for model capacity, the raw loss is scaled by the ratio of `num_blocks` to `MAX_NUM_BLOCKS`, penalizing more complex configurations proportionally.
- **Successive Halving:** Within each Hyperband bracket, the candidate pool is reduced by the `reduction_factor` (default 4 due to time constraints) each round while the epoch budget per surviving configuration increases, ensuring that the best-performing settings are progressively refined and ultimately selected based on minimum scaled loss.

This Hyperband‑based strategy efficiently balances the exploration of diverse hyperparameter regions with the exploitation of promising configurations, providing a single optimal set of hyperparameters for all model types in the experiment.
One potential negative consequence of using a single hyperparameter configuration for all datasets, however, is that the hyperparameters could get tuned to be more effective on the data with complexity metric values in higher density areas.
Final configuration (just the non-forced values) = {'learning_rate': 0.021450664374153845, 'entanglement_gate': 'cz', 'embedding_gate': 'rz', 'block_gate': 'rz'}

### Training
All models are trained using an ADAM optimizer together with finite‑difference gradient estimation to accommodate the non‑differentiable quantum circuits.
A truncated version of finite differences (only single, shared, calculation for first point in all gradient calculations for that epoch) was used to increase training speed.
For reproducability, permanent teacher forcing is used inside the cost functions (instead of using a linearly increasing probability of dealing with their own noise).
Early‑stage flexibility is attained via a linearly increasing BTFP weight based on percentage of total epochs elapsed.
Trainable parameters are initialized from a uniform distribution in \([-π, π]\), and the first- and second-moment vectors (moment1, moment2) are set to zero.
Finally, `np.random.seed(RANDOM_SEED)` is set before each model training to ensure consistent initialization across runs.

### Experimental Environment
All quantum circuits are simulated on classical hardware to isolate algorithmic performance from hardware noise.
Experiment ran on a 2017 MacBook Pro (3.1GHz quad‑core i7, 16GB RAM) without GPU acceleration.

### Analysis
- Scaling factor is removed from BTFP cost history for analysis in order to get a clear understanding of the BTFP itself over time.
#### Metric Parameter Choices
- **MPE**
  - For the order parameter, values of 2 and 3 are used and the mean value is taken.
This maintains the guideline of sequence length (100) ≫ order! suggested by (Cuesta-Frau et al.; 2019).
  - Multiple time delays are used and the mean result is taken.
The center point for the time delay range is is set to the zero crossing of the signal autocorrelation because it is a key indicator of multi-scale patterns (Kafashi et al.; 2008).
The minimum time delay is calculated as `max(1, min(center_point - 1/20 * sequence_length, 9/10 * center_point))`
The maximum time delay is calculated as `max(center_point + 1/20 * sequence_length, 11/10 * center_point)`
  - The scale parameter is set to 3 in order to ensure that (Wang et al.; 2019).
  - See [implementation](./analysis.py#L229) for further details.
#### Metric Normalization
For comparison across metrics with different scales, the normalization used is specific to each metric:
- HE: No normalization necessary because it's already limited to [0,1] by definition.
- MPE: Theoretical maximum normalization is used, which divides by `log2(order!)`.
- LZC: Theoretical maximum normalization is used, which calculates the maximum as the sequence length divided by the log of (the sequence length) in base (number of unique symbols).
- DE: Theoretical maximum normalization is used, which simply divides by the log of the number of unique symbols in the discretized signal.
#### Quantization Methods
When extending a discrete metric to cover continuous data (LZC and DE), multiple quantization methods are used to improve conclusion robustness.
Specifically, the included methods are:
  - [Equal Bin-Width](./analysis.py#L80): Partitions each feature's range into equal‑width bins based on a fixed symbol count (here, 1/10 of the sequence length) then turn each state into a symbol via equal-base mixed-radix encoding.
Fails to adapt to skewed or multimodal distributions.
  - [Bayesian Blocks by (Scargle et al.; 2013)](./analysis.py#L109): Finds an optimal segmentation of one‑dimensional data (quantizes each feature separately) by maximizing a fitness function under a Poisson (event) model.
This yields non‑uniform bin widths that adapt to local data density by creating edges where the statistical properties change, yielding finer resolution in regions of high statistical property fluctuation and coarser bins elsewhere.
An adaptive histogram approach such as this better captures multimodal structure by placing narrow bins around abrupt changes in density and wider bins elsewhere.
This prevents the smoothing over of sharp, localized peaks that uniform binning introduces.
The final symbols are then obtained via mixed-radix encoding with each feature's base being equal to the number of symbols (bins) found for that feature.
  - [HDBSCAN](./analysis.py#L136): Assigns symbols via hierarchical density‑based clustering, uncovering clusters of varying shapes and densities without requiring preset bin counts for each feature.
As per standard practice, the `cluster_selection_epsilon` parameter is set to the mean plus the standard deviation of the interpoint distances between each pair of nearest neighbors.
The chosen `cluster_selection_method` is `'leaf'` for improved granularity.
The `min_cluster_size` is set at 2 to minimize labeling points as noise.

## Results
### Classical vs Quantum
- Quantum bottleneck features used are each qubit's marginal probability of |0> (for analysis only).
- For further analysis in the quantum realm only, correlations are made between each model's mean validation series complexity metrics and the mean MWGE as well as the mean VNE of its bottleneck states when going through each series in that set.
- All quantum models consistently chose the first two qubits as the trash feature indices at every state in each validation series as seen in the [BTFP index histograms](./images/results/btfp_histograms/).
- The classical models in contrast showed much more flexibility in their chosen trash indices.
### Prediction vs Reconstruction
### Reccurence
### Loss Landscape Similarities
- Gradient Norms
  - Due to an bug with saving the gradient norm history during training, these results are not available
- Prediction Loss
  - All pairwise Pearson correlation coefficients for raw histories are 1.0
  - Mean Absolute 1st Derivative per Model Type:
    - QAE:  0.0000000000
    - QRAE: 0.0295603101
    - QTE:  0.0000000000
    - QRTE: 0.0292647070
    - CAE:  0.4579354477
    - CRAE: 0.1798117561
    - CTE:  0.4775164051
    - CRTE: 0.2251897169
  - Mean Absolute 2nd Derivative per Model Type:
    - QAE:  0.0000000000
    - QRAE: 0.0148292331
    - QTE:  0.0000000000
    - QRTE: 0.0146809408
    - CAE:  0.0025985003
    - CRAE: 0.1104035420
    - CTE:  0.0034203822
    - CRTE: 0.1363251462
  - Pairwise Max Normalized Cross-Correlation of Mean-Centered Raw Histories are all 1.0
  - ... 1st Derivatives:
  |    | QAE| QRAE | QTE| QRTE | CAE  | CRAE | CTE  | CRTE |
  |QAE | nan| nan  | nan| nan  | nan  | nan  | nan  | nan  |
  |QRAE| nan| 1.000| nan| 1.000| 1.000| 1.000| 1.000| 1.000|
  |QTE | nan| nan  | nan| nan  | nan  | nan  | nan  | nan  |
  |QRTE| nan| 1.000| nan| 1.000| 1.000| 1.000| 1.000| 1.000|
  |CAE | nan| 1.000| nan| 1.000| 1.000| 1.000| 1.000| 1.000|
  |CRAE| nan| 1.000| nan| 1.000| 1.000| 1.000| 1.000| 1.000|
  |CTE | nan| 1.000| nan| 1.000| 1.000| 1.000| 1.000| 1.000|
  |CRTE| nan| 1.000| nan| 1.000| 1.000| 1.000| 1.000| 1.000|
  - ... 2nd Derivatives: all 1.0
  - High Frequency First Derivative Threshold (based on 95% energy cutoff ratio): 0.0625
    - High Frequency Energy Ratio
      - QAE:  nan
      - QRAE: nan
      - QTE:  nan
      - QRTE: nan
      - CAE:  0.0300
      - CRAE: 0.6422
      - CTE:  0.0303
      - CRTE: 0.6843
  - High Frequency Second Derivative Threshold (based on 95% energy cutoff ratio): 0.2812
    - High Frequency Energy Ratio
      - QAE:  nan
      - QRAE: nan
      - QTE:  nan
      - QRTE: nan
      - CAE:  0.0003
      - CRAE: 0.0431
      - CTE:  0.0006
      - CRTE: 0.0428
- BTFP
  - All pairwise Pearson correlation coefficients for raw histories are 1.0
  - Mean Absolute 1st Derivative per Model Type:
    - QAE:  0.0000000000
    - QRAE: 0.0147310770
    - QTE:  0.0000000000
    - QRTE: 0.0145837662
    - CAE:  0.4562267551
    - CRAE: 0.0694082142
    - CTE:  0.4757796760
    - CRTE: 0.0888645707
  - Mean Absolute 2nd Derivative per Model Type:
    - QAE:  0.0000000000
    - QRAE: 0.0148239043
    - QTE:  0.0000000000
    - QRTE: 0.0146756653
    - CAE:  0.0013651153
    - CRAE: 0.1076974426
    - CTE:  0.0023848520
    - CRTE: 0.1334352733
  - Pairwise Max Normalized Cross-Correlation of Mean-Centered Raw Histories, 1st and 2nd derivatives are all 1.0
  - High Frequency First Derivative Threshold (based on 95% energy cutoff ratio): 0.1250
    - High Frequency Energy Ratios
      - QAE:  0.9072
      - QRAE: 0.5235
      - QTE:  0.9061
      - QRTE: 0.5235
      - CAE:  0.0110
      - CRAE: 0.5168
      - CTE:  0.0109
      - CRTE: 0.4585
  High Frequency Second Derivative Threshold (based on 95% energy cutoff ratio): 0.2812
    - High Frequency Energy Ratios
      - QAE:  0.1366
      - QRAE: 0.0391
      - QTE:  0.1157
      - QRTE: 0.0391
      - CAE:  0.0010
      - CRAE: 0.0393
      - CTE:  0.0009
      - CRTE: 0.0384

## Discussion
### Sources of Error
- A logical error in the LZC calculation that allowed for overlap of phrases was found after data generation (see lzc_corrections.py from commit 1b51cf870c7df4a98eeb8bf26c07eb09cf77c24f) with the following statistics for their differences: mean=1.04; median=1; max=5; std dev=0.9429.
The correct value was always higher due to this because allowing overlap means a phrase that has already be seen can be used.
The minimum correct value for any series in the generated data was 33 for a maximum effect of 15.15% and both a mean and median effect of around 1/33 (3%).
The corrected values are used in analysis, however, so the effect of this is infinitesimal being limited only to how much variety there was in the complexity metrics of the series chosen for the comparison grid.
- Since the original data generation grid choices and data partitions were done with 3D binning based on the Higuchi fractal dimension instead of MPE (before the replacement), the variation in MPE values is much lower than desired.

## Conclusion

## Future Work
- Give the reconstruction objective two consecutive points and ask it to reconstruct them both to see if it learns temporal dynamics better than single state reconstruction
- When time allows, each block (including embedding) should have at least rotation gates for each quantum axis per block (w/ 3 layers per block for classical as well), eliminating the gate choices other than entanglement from hyperparam search
- Many blocks per model half
  - Look at total information content in each block along encoder and decoder
    - Would expect encoder to *slowly decrease* and decoder to *quickly increase*
- More features w/ different entanglement topologies and equivalent classical "coupling" mechanisms
- Add nonlinearities
- Attempt to use data generation that has opposing trends for complexity metric targets (target HE vs target LZC)
- Chaotic time series
- Composite regimes (same randomly sampled sequence of generators for each series in dataset and different sequence per dataset)
- Try without pre-specifying a bottleneck size by extending the trash cost functions to just sum across all features to get something like total n-dimensional distance from 0 at the bottleneck
- Try adding back the linearly increased probabilistic use of their own noise (see commits `b134b50` and `20c5cad`)

## Abbreviations
- AE = Auto-encoder
- AR = Autoregressive
- BTFP = Bottleneck Trash Feature Penalty
- DE = Differential Entropy
- ENC-DEC = Encoder-Decoder
- FBM = Fractional Brownian Motion
- HE = Hurst Exponent
- LZC = Lempel-Ziv Complexity
- MPE = Optimized Multiscale Permutation Entropy
- MWGE = Meyer-Wallach Global Entanglement
- SO = Special Orthogonal
- U = Unitary
- VNE = VonNeumann Entropy

## References
(BibTex format)
1. @article{doi:10.1126/science.1127647,
  author   = {G. E. Hinton  and R. R. Salakhutdinov },
  title    = {Reducing the Dimensionality of Data with Neural Networks},
  year     = {2006},
  journal  = {Science},
  volume   = {313},
  number   = {5786},
  pages    = {504-507},
  doi      = {10.1126/science.1127647},
  URL      = { https://www.science.org/doi/abs/10.1126/science.1127647 },
  eprint   = { https://www.science.org/doi/pdf/10.1126/science.1127647 },
  abstract = {High-dimensional data can be converted to low-dimensional codes by training a multilayer neural network with a small central layer to reconstruct high-dimensional input vectors. Gradient descent can be used for fine-tuning the weights in such “autoencoder” networks, but this works well only if the initial weights are close to a good solution. We describe an effective way of initializing the weights that allows deep autoencoder networks to learn low-dimensional codes that work much better than principal components analysis as a tool to reduce the dimensionality of data.}
}
1. @article{Romero_2017,
  author    = {Romero, Jonathan and Olson, Jonathan P and Aspuru-Guzik, Alan},
  title     = {Quantum autoencoders for efficient compression of quantum data},
  year      = {2017},
  month     = aug,
  ISSN      = {2058-9565},
  url       = { http://dx.doi.org/10.1088/2058-9565/aa8072 },
  DOI       = {10.1088/2058-9565/aa8072},
  number    = {4},
  journal   = {Quantum Science and Technology},
  volume    = {2},
  publisher = {IOP Publishing},
  pages     = {045001}
}
1. @article{1055501,
  author   = {Lempel, A. and Ziv, J.},
  title    = {On the Complexity of Finite Sequences},
  year     = {1976},
  journal  = {IEEE Transactions on Information Theory},
  volume   = {22},
  number   = {1},
  pages    = {75-81},
  keywords = {},
  doi      = {10.1109/TIT.1976.1055501}
}
1. @misc{holzer2024spectralinvariancemaximalityproperties,
  author        = {Patrick Holzer and Ivica Turkalj},
  title         = {Spectral invariance and maximality properties of the frequency spectrum of quantum neural networks},
  year          = {2024},
  eprint        = {2402.14515},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph},
  url           = { https://arxiv.org/abs/2402.14515 },
}
1. @article{doi:10.1137/0106004,
  author  = {Givens, Wallace},
  title   = {Computation of Plain Unitary Rotations Transforming a General Matrix to Triangular Form},
  year    = {1958},
  journal = {Journal of the Society for Industrial and Applied Mathematics},
  volume  = {6},
  number  = {1},
  pages   = {26-50},
  doi     = {10.1137/0106004},
  URL     = { https://doi.org/10.1137/0106004 },
  eprint  = { https://doi.org/10.1137/0106004 }
}
1. @book{golub2013matrix,
  author    = {Golub, Gene H. and Van Loan, Charles F.},
  title     = {Matrix Computations},
  edition   = {4},
  year      = {2013},
  publisher = {Johns Hopkins University Press},
  address   = {Baltimore, MD},
  isbn      = {978-1421407944},
}
1. @article{JMLR:v18:16-558,
  author  = {Lisha Li and Kevin Jamieson and Giulia DeSalvo and Afshin Rostamizadeh and Ameet Talwalkar},
  title   = {Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization},
  year    = {2018},
  journal = {Journal of Machine Learning Research},
  volume  = {18},
  number  = {185},
  pages   = {1--52},
  url     = { http://jmlr.org/papers/v18/16-558.html }
}
1. @article{JMLR:v13:bergstra12a,
  author  = {James Bergstra and Yoshua Bengio},
  title   = {Random Search for Hyper-Parameter Optimization},
  year    = {2012},
  journal = {Journal of Machine Learning Research},
  volume  = {13},
  number  = {10},
  pages   = {281--305},
  url     = { http://jmlr.org/papers/v13/bergstra12a.html }
}
1. @article{2013ApJ...764..167S,
  author        = {{Scargle}, Jeffrey D. and {Norris}, Jay P. and {Jackson}, Brad and {Chiang}, James},
  title         = "{Studies in Astronomical Time Series Analysis. VI. Bayesian Block Representations}",
  year          = 2013,
  month         = feb,
  journal       = {\apj},
  keywords      = {methods: data analysis, methods: statistical, Astrophysics - Instrumentation and Methods for Astrophysics, Mathematics - Statistics Theory, G.3},
  volume        = {764},
  number        = {2},
  eid           = {167},
  pages         = {167},
  doi           = {10.1088/0004-637X/764/2/167},
  archivePrefix = {arXiv},
  eprint        = {1207.5578},
  primaryClass  = {astro-ph.IM},
  adsurl        = { https://ui.adsabs.harvard.edu/abs/2013ApJ...764..167S },
  adsnote       = {Provided by the SAO/NASA Astrophysics Data System}
}
1. @article{doi:10.1061/TACEAT.0006518,
  author   = {H. E. Hurst },
  title    = {Long-Term Storage Capacity of Reservoirs},
  year     = {1951},
  journal  = {Transactions of the American Society of Civil Engineers},
  volume   = {116},
  number   = {1},
  pages    = {770-799},
  doi      = {10.1061/TACEAT.0006518},
  URL      = { https://ascelibrary.org/doi/abs/10.1061/TACEAT.0006518 },
  eprint   = { https://ascelibrary.org/doi/pdf/10.1061/TACEAT.0006518 },
  abstract = { A solution of the problem of determining the reservoir storage required on a given stream, to guarantee a given draft, is presented in this paper. For example, if a long-time record of annual total discharges from the stream is available, the storage required to yield the average flow, each year, is obtained by computing the cumulative sums of the departures of the annual totals from the mean annual total discharge. The range from the maximum to the minimum of these cumulative totals is taken as the required stornge. }
}
1. @article{10.1093/biomet/74.1.95,
  author   = {DAVIES, R. B. and HARTE, D. S.},
  title    = {Tests for Hurst effect},
  year     = {1987},
  month    = {03},
  journal  = {Biometrika},
  volume   = {74},
  number   = {1},
  pages    = {95-101},
  abstract = {We consider the power of tests for distinguishing between fractional Gaussian noise and white noise of a first-order autoregressive process. Our tests are based on the beta-optimal principle (Davies, 1969), local optimality and the rescaled range test.},
  issn     = {0006-3444},
  doi      = {10.1093/biomet/74.1.95},
  url      = { https://doi.org/10.1093/biomet/74.1.95 },
  eprint   = { https://academic.oup.com/biomet/article-pdf/74/1/95/578634/74-1-95.pdf },
}
1. ***!TODO!  READ THIS PAPER:*** @misc{dijk2018deep,
  author = {David van Dijk and Scott Gigante and Alexander Strzalkowski and Guy Wolf and Smita Krishnaswamy},
  title  = {Deep Transition-Encoding Networks for Learning Dynamics},
  year   = {2018},
  url    = { https://openreview.net/forum?id=S1VG0F1Dz }
}
1. @article{article,
  author  = {Kaffashi, Farhad and Foglyano, Ryan and Wilson, Christopher and Loparo, Kenneth},
  title   = {The effect of time delay on Approximate & Sample Entropy calculations},
  year    = {2008},
  month   = {12},
  pages   = {3069-3074},
  volume  = {237},
  journal = {Physica D: Nonlinear Phenomena},
  doi     = {10.1016/j.physd.2008.06.005}
}
1. @article{Cuesta-Frau2019,
  author  = {Cuesta-Frau, David and Murillo-Escobar, Juan Pablo and Orrego, Diana Alexandra and Delgado-Trejos, Edilson},
  title   = {Embedded Dimension and Time Series Length. Practical Influence on Permutation Entropy and Its Applications},
  year    = {2019},
  journal = {Entropy},
  volume  = {21},
  number  = {4},
  pages   = {385},
  doi     = {10.3390/e21040385}
}
1. @Article{e21020170,
  AUTHOR         = {Wang, Xianzhi and Si, Shubin and Wei, Yu and Li, Yongbo},
  TITLE          = {The Optimized Multi-Scale Permutation Entropy and Its Application in Compound Fault Diagnosis of Rotating Machinery},
  YEAR           = {2019},
  JOURNAL        = {Entropy},
  VOLUME         = {21},
  NUMBER         = {2},
  ARTICLE-NUMBER = {170},
  URL            = { https://www.mdpi.com/1099-4300/21/2/170 },
  PubMedID       = {33266886},
  ISSN           = {1099-4300},
  ABSTRACT       = {Multi-scale permutation entropy (MPE) is a statistic indicator to detect nonlinear dynamic changes in time series, which has merits of high calculation efficiency, good robust ability, and independence from prior knowledge, etc. However, the performance of MPE is dependent on the parameter selection of embedding dimension and time delay. To complete the automatic parameter selection of MPE, a novel parameter optimization strategy of MPE is proposed, namely optimized multi-scale permutation entropy (OMPE). In the OMPE method, an improved Cao method is proposed to adaptively select the embedding dimension. Meanwhile, the time delay is determined based on mutual information. To verify the effectiveness of OMPE method, a simulated signal and two experimental signals are used for validation. Results demonstrate that the proposed OMPE method has a better feature extraction ability comparing with existing MPE methods.},
  DOI = {10.3390/e21020170}
}


## Useful Commands
- To determine how many datasets need models trained over them based on the saved grid files ("series_cell_..."), use `find generated_datasets.4qubits/ -iname series_cell_\*dataset\*.npy | sed -E 's/.*series_cell.*dataset(.*)\.npy/\1/' | sort | uniq | wc -l`