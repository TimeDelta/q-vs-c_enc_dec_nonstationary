# Manuscript draft and publication status

The [manuscript](manuscript.md) and [rendered PDF](manuscript.pdf) present the corrected evidence as an initialization study and a bounded forecasting benchmark. The [LaTeX source](manuscript.tex), [editable prose template](manuscript_template.md) and [bibliography](references.bib) are included. Numerical descriptor results and forecast tables are populated from checked-in evidence by `scripts/build_manuscript.py`.

The paper's supported findings are:

- An existing decoder-parameter initialization correction reduces an avoidable reset-qubit output penalty in paired training/validation controls.
- Reduced-rank regression has lower native shifted forecast error in all eight fresh realizations of the specified eight-epoch procedure.
- Validation descriptor distance adds no positive primary predictive evidence beyond validation distortion and model identity in the exploratory grouped analysis.
- Coordinatewise descriptor differences can change under equivalent rotated representations with unchanged linear-probe predictions.

The fresh-data forecast comparison was specified before fitting and test generation. The descriptor regression is exploratory after those test results were inspected. The manuscript preserves this distinction and does not infer quantum advantage, a barren plateau, an MDL principle or converged model rankings.

## Before submission

This is an evidence-backed draft for author review, with authorship and institutional details intentionally absent until supplied by the author. No submission or journal selection has been made.

Author review must establish the author list and contributions, affiliations, contact information, funding and competing-interest declarations. These facts cannot be inferred from a repository handle. Any use of automated assistance should be described according to the eventual venue's policy.

The main scientific priority is a fixed extension covering multiple initialization seeds and convergence diagnostics. The current result evaluates a short training procedure, so architecture-superiority conclusions are unsupported. An additional generator with substantially different nonlinear dynamics would strengthen external validity. A new untouched replication is needed before treating descriptor-error relationships as confirmatory. Increasing related sequence rows without adding independent realizations would not resolve this issue.

A broader related-work review remains necessary before making novelty claims. The draft distinguishes latent generator-complexity work and entropy-guided classification from the present diagnostic. EntroTS bibliographic metadata is verified against an author's publication page, but its publisher full text was inaccessible during this preparation; detailed comparison of its mechanism and evaluation remains pending. The LZ76 implementation and normalization are checked against AntroPy 0.2.2 documentation, with the original paper cited for attribution. No borrowed abstracts or figures are incorporated.

## Build and reproduce

Install the pinned CPU experiment dependencies and `requirements-test.txt` to run regression checks and report rendering. Manuscript rendering additionally requires Pandoc with citeproc, pdfLaTeX, the `microtype` LaTeX package and Poppler for visual inspection. These are authoring tools, not experiment dependencies.

```bash
python scripts/build_manuscript.py
```

Edit the prose template and bibliography to update the manuscript, then rebuild. The historical manuscript remains archived separately. Existing experiment reports contain full configurations, exact source identities, selection hashes, sequence measurements and checkpoint archives.
