# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

<!-- insertion marker -->
## v1.26.5 (2026-09-04)

### Feat

- masking modes, reliability pipeline overhaul, single-pass padded inference, multi-npz training
- **predict**: support --prophage, --save-embedding, --save-nmd with --chunk-size
- **predict**: add native --chunk-size option for large FASTAs
- **nnlib**: add ValidationConfusionMatrix callback for per-epoch validation metrics
- **postprocess**: tRNA-aware prophage boundaries, interactive plots, genbank support
- **dataops**: --balance-classes for optimize-data
- **train**: crop_mode sample/range for multi-crop training
- **nnlib**: MaskedLastPooling for causal models (pooling: last)
- **nnlib**: mask-aware norm_type on residual_block (config-selectable normalization)
- **nnlib**: mask_mode on MaskedConv1D, default 'any'
- **nnlib**: hyena filter_normalize + kernel_regularizer options
- **train**: EMA + cosine LR schedule options for seed-variance reduction
- **nnlib**: improve hyena layers (stability, SIREN, output projection)
- **predict**: experimental linear-chain CRF (Viterbi) window decoding
- **scripts**: add class distribution and N-filter CSV analysis scripts
- **train**: add N-stretch OOD perturbation to reliability data generation
- **predict**: add --no-dustmask flag to skip pydustmasker during inference
- **postprocess**: refine prophage boundaries with pyrodigal-gv
- **nnlib**: add MaskedGlobalMaxPooling layer and masked_max pooler
- **train**: tune reliability threshold after reliability training
- **nnlib**: add binary_f1 metric for single-logit reliability head
- **train**: add projection_train_steps and projection_validation_steps config keys
- **hyena**: register hyena_block in DynamicModelBuilder
- **hyena**: add Hyena operator and block layers
- **hyena**: add implicit filter generator
- **hyena**: add causal FFT convolution helper
- **train**: scripts to add big eukaryote fragments while capping eukarya count
- **masking**: wire --masking/--no-masking through layers and train command
- **reliability**: speed up classifier inference, add logits/header, add --masking flag
- **predict**: optional short-contig classification with --min-len (two-pass)
- **predict**: dynamic stride for short contigs
- **predict**: add post-hoc refinement layer via --refine
- **reliability**: store real sequence IDs in prediction CSVs via polars
- **reliability**: write prediction CSVs with integer seq_id/label columns and auto-upgrade legacy files
- **reliability**: include seq_id and original label in prediction CSVs
- **reliability**: use existing prediction CSVs for ID/OOD selection
- **reliability**: do not overwrite existing prediction CSVs
- **reliability**: support multiple shuffle modes in perturbations
- **reliability**: derive prediction CSV names from raw input basenames
- **reliability**: stream data generation to avoid OOM
- **reliability**: write classifier train/val probabilities to CSV
- **predict**: make embedding/nmd saves opt-in via flags
- **cli**: add --save-embedding and --save-nmd flags to predict
- **train**: allow self-supervised pretraining + head-only fine-tune in one run
- **slurm**: add Zeus job for RF 127 baseline
- **slurm**: add Zeus job for multi-scale RF experiment
- **configs**: add multi-scale and RF 127 baseline generator
- **builder**: add parallel_branches pseudo-layer for multi-scale features
- **metrics**: add macro F1 score metric configurable via training config
- **utils**: treat axial/length attention as full-sequence in receptive field
- **layers**: add ignore_mask option to MaskedBiLSTM for cuDNN support
- **layers**: add MaskedBiLSTM layer support
- **utils**: add receptive field utility and display during training
- **train**: support per-branch batch sizes in config
- **train**: allow XLA for pretrain with fp32/bf16, warn and disable for fp16
- **train**: add --precision fp32/fp16/bf16 flag and deprecate --mixed_precision
- **train**: honor --ignore_convergence and log skipped branches
- **builder**: read converged.json in checkpoint scanner
- **train**: write convergence marker on early stop
- **train**: add --ignore_convergence flag
- **layers**: expose alpha_init for MaskedDYT in ResidualBlock and AxialAttention
- **layers**: add norm_type support to AxialAttention
- **nnlib**: add MaskedDYT layer and wire norm_type in ResidualBlock
- **builder**: add nmd_plus_signals reliability mode with OODSignalLayer
- **layers**: add OODSignalLayer
- **train**: wire grouped batching into NumPy classifier and reliability branches
- **train**: add grouped batching helper
- **loaders**: support pad_to_max=False for object-array runtime crops
- **loaders**: support pad_to_max=False for dense runtime crops
- **loaders**: forward pad_to_max through _load_numpy_dataset
- **brain**: add brain-specific config and 4xA100 multi-GPU Slurm pipeline
- **slurm**: enable XLA for Zeus pipeline
- **zeus**: add Zeus slurm pipeline and config for 1500bp nmd merge 6-class
- **reliability**: add mix/chimera perturbation across distinct classes
- **seqops**: add apply_mix chimeric sequence helper
- generate synthetic OOD from validation sequences for reliability validation set
- crop raw CSV sequences to reliability crop_size before classifier inference
- support units (nuc/codon) for reliability_data_generation crop_size
- add --units flag to optimize-data for codon/nucleotide crop sizing
- allow crop_size override in reliability_data_generation config
- reuse existing reliability NPZs instead of regenerating
- support separate train/val CSVs for reliability data generation
- **pipeline**: full Jaeger training-to-evaluation shell script
- configurable reliability data generation
- **train**: post-classifier reliability data generation
- **train**: unload classifier data before loading reliability data
- **optimize-data**: store integer features in smallest fitting dtype
- **loaders**: runtime multi-crop slicing for NumPy loader
- **train**: do not cache ragged numpy datasets
- **loaders**: support ragged object-array NPZ files
- **cli**: expose --max-memory-mb and --pad in optimize-data
- **convert**: choose streaming vs fast path based on memory budget
- **convert**: implement memory-bounded streaming NPZ converter
- **convert**: expose pad parameter in legacy fast path
- **convert**: add batch finalization helper for padded and unpadded output
- **convert**: add per-row output memory estimate helper
- **cli**: add --overlap option to optimize-data
- **scripts**: add variable-length config generator
- **scripts**: add NPZ merger for variable-length training
- **scripts**: add benchmark plotting script
- **scripts**: add benchmark evaluation runner
- **scripts**: save confusion matrix in evaluate_saved_model.py
- **scripts**: accept --lengths in prepare_length_csvs.py
- **config**: add DVF-style example config
- **builder**: wire branched representation learner and classifier
- **builder**: add _build_branched_block helper
- **builder**: add 1D pooling and activation aliases
- unify numpy loader for new NPZ schema and restrict training data_format to csv/numpy
- refactor CLI optimize-data for new NPZ formats
- update convert_dataset dispatcher and optimize_data_core to new NPZ interface
- implement _convert_to_npz with nucleotide, translated, and dicodon encoding
- add crop generation and padding helpers
- add numba nucleotide batch encoder
- add codon and nucleotide map helpers
- validate and wire merged NMD to reliability head
- collect and merge multiple NMD tensors
- register nmd layer in builder
- add NMDMerge layer
- add standalone NMDLayer
- **nnlib**: add short-fragment architecture layers and config (#35)
- support for more layers

### Fix

- **tests**: move hyena_test.yaml fixture into tests/pytest
- **predict**: report actual processed count with --chunk-size
- **predict**: suppress Jaeger banner in chunk subprocesses
- **predict**: show per-chunk logs and aggregate stats with --chunk-size
- **postprocess**: stride-aware prophage coordinates
- **nnlib**: rely on Keras auto-masking in ResidualBlock; tolerate return_nmd: false in non-batchnorm norms
- **nnlib**: fully-masked samples pool to zeros instead of -1e9 sentinel
- **nnlib**: mask-aware default pooling for padded-batch inference
- **health**: save test SavedModel to a temp dir instead of the package dir
- **train**: write convergence marker for all completed stages
- **train**: resume from the most advanced stage checkpoint (reliability > classifier > projection)
- **train**: resume projection training from the correct checkpoint
- **train**: require --force to overwrite existing model on --save_model
- **commands**: guard TF thread config in predict and taxonomy against pre-initialized runtime
- **train**: tolerate preset TF threading when runtime is already initialized
- **predict**: handle empty passes in two-pass concat; overhaul train pipeline script
- **scripts**: pair predictions with labels by prefix and merge on contig_id
- **seqops**: handle input_type nucleotide in crop resolution
- **seqops**: canonicalize crop length to codons (3*codons+5)
- **reliability**: scale reliability data generation and fix macro_f1 on binary heads
- **predict**: avoid kwargs collision when calling _write_prediction_outputs
- **train**: allow SSL pretraining to start and resume correctly
- **train**: save and restore ArcFace loss weights with projection checkpoints
- **hyena**: make HyenaFilter mixed-precision safe
- **train**: skip classifier checkpoint load during self-supervised pretraining
- **predict**: write outputs after two-pass inference
- **predict**: respect seq_onehot in padded-batch specs for short contigs
- **reliability**: use file handle for appending prediction CSV rows
- **predict**: remove leftover inline numpy import
- **nn**: use Keras built-in gradient accumulation
- **nn**: set optimizer distribution strategy before flush
- **nn**: wrap gradient-accumulation flush in tf.function
- **nn**: report log gradient norm and compute it before resetting accumulators
- **nn**: keep gradient-accumulation variables on GPU for XLA
- **data**: honour crop_sizes in sharded manifest NPZ loader
- **builder**: coerce regularizer weights to float to handle string/scientific notation values
- **configs**: point Zeus training data_dir to beegfs numpy path
- **slurm**: quote bind mounts correctly so PROJECT_ROOT expands inside srun
- **slurm**: correct variable expansion and add pre-flight checks in multiscale_rf_zeus.slurm
- **configs**: add alpha_init to generated residual blocks
- **builder,configs**: inherit channel count in branches and match projection input shape
- **builder**: handle pooling=null and unique pooler names for parallel branches
- **layers**: disable cuDNN in MaskedBiLSTM to support arbitrary masks
- **nnlib**: stabilize projection pretraining under mixed precision
- **train**: disable XLA for projection pretraining to avoid NaN gradients under mixed precision
- **train**: disable XLA for projection pretraining to avoid NaN gradients under mixed precision
- **predict**: select classification model when an embedding graph is also present
- **builder**: scan classifier, reliability, and projection checkpoint dirs for resume
- **train**: fall back to crop_size when crop_sizes is missing
- **data**: avoid densifying huge variable-length object arrays
- preserve YAML key order in config serialization
- **layers**: correct masked layer normalization statistics
- **builder**: keep reliability head when no training data is configured
- **builder**: support npz class counts and defer reliability bias until OOD data is generated
- **layers**: restrict return_nmd to masked_batchnorm and add ResidualBlock norm_type tests
- **layers**: handle plain logits input in OODSignalLayer
- **train**: correct log format string in reliability data path check
- **train**: correct grouped batching fallback and default validation
- **train**: grouped batching helper edge cases and feature_key
- **train,slurm**: use reliability validation for final test, bind source in containers, disable XLA on Brain multi-GPU
- **container,slurm**: expose nvidia pip libs to loader and force overwrite stale checkpoints on Zeus
- **zeus**: point config to numpy subdir and existing raw CSVs
- **zeus**: bind /mnt/beegfs in apptainer runs
- **zeus**: replace jq/singularity with python3/apptainer in pipeline script
- **zeus**: pin slurm pipeline to node030
- **singularity**: use --no-same-owner when extracting mmseqs archive
- **zeus**: point slurm script to external configs directory
- **reliability**: address final review feedback for mix perturbation
- **seqops**: address review feedback for apply_mix
- derive reliability NPZ crop_size from crop_sizes when crop_size is absent
- correct padded_batch shapes and defaults in reliability inference dataset
- **data**: pad runtime crops to max crop size for fixed shapes
- use single-crop validation by default to avoid slow multi-crop eval
- **loaders**: runtime cropper now handles unpadded object arrays
- **train**: build classifier train_args after data is loaded
- **convert**: true sharded streaming to avoid final concat OOM
- **loaders**: correct ragged one-hot conversion and infer codon depth
- **convert**: account for overlap multiplier in streaming batch sizing and expand tests
- **convert**: use finalize helper in fast path and add one-hot multi-crop test
- **convert**: correct one-hot nucleotide padding axis and expand finalize tests
- **slurm**: generate both translated and nucleotide variable-length NPZs
- **train**: skip reliability training when reliability data is absent
- **data**: keep numpy datasets on CPU to avoid GPU OOM
- **data**: support binary reliability labels in numpy loader
- default optimize-data num_workers to all CPUs and add small-dataset fallback
- pass through single NMD unchanged; update spec/plan files
- robust NMDMerge naming, docstring, and tuple check
- avoid mutable default in NMDMerge
- align NMDLayer statistics with MaskedBatchNorm
- **nnlib**: preserve batch dimension in projection head
- **recipe**: add missing runtime deps to bioconda recipe

### Refactor

- **train**: extract _build_numpy_split and guard grouped batching for single features
- **train**: lazy-load classifier/reliability training data
- **ood**: rewrite OOD dataset builder as OODDatasetBuilder class
- remove Identity wrappers and check output dict key

### Perf

- **reliability**: show Keras progress bar during classifier.predict()
- **reliability**: use classifier.predict() instead of manual batch loop
- **reliability**: disk-back synthetic OOD filtering chunks
- **reliability**: stream synthetic sequence generation to reduce memory
- **predict**: reduce tf.function retracing in InferModel
- **reliability**: add configurable inference_batch_size for reliability data generation
- **reliability**: parallelize synthetic sequence generation with multiprocessing
- densify object arrays on load to use fast crop path
- prefetch and parallelize one-hot conversion in NumPy loaders
- **data**: add fast NPZ compression and one-hot memory guard to optimize-data
- **data**: batchwise integer-to-onehot for numpy loader
- **dataops**: use ThreadPool, project logger, and per-crop-size batches in optimize-data

## v1.26.4 (2026-06-13)

### Feat

- major data/seq ops refactor + CLI smoke tests + deps (#33)
- linear genome plots and prophage prediction in new pipeline (#32)
- ONNX inference, INT8 quantization, and performance optimizations (#31)
- add one-liner install script

### Fix

- prevent GPU OOM by moving predictions to CPU immediately (#30)
- resolve inference pipeline bugs in v1.26.3
- resolve smoke test failures and CI lint errors
- add test data files to git tracking
- include SavedModel graph directory in pdm-backend build
- rename jaeger test to jaeger health and add diagnostics (#26)
- add pdm-backend to host requirements for conda build
- use underscore in PyPI source URL (jaeger_bio instead of jaeger-bio)

## v1.26.1 (2026-06-04)

## v1.26.1b1 (2026-01-06)

## v1.26.1b0 (2026-01-06)

## [v1.1.30](https://github.com/Yasas1994/Jaeger/releases/tag/v1.1.30) - 2025-06-05

<small>[Compare with v1.1.30-alpha](https://github.com/Yasas1994/Jaeger/compare/v1.1.30-alpha...v1.1.30)</small>

### Added

- added pydustmasker ([15acef0](https://github.com/Yasas1994/Jaeger/commit/15acef0248bab2588cfdbe00b507949cfc02a3a4) by Yasas1994).
- added separate models ([a6a8a79](https://github.com/Yasas1994/Jaeger/commit/a6a8a79ddf3d7e1b635e2db10efbbb6ab0a9bc4d) by Yasas1994).
- added support for multigpu training ([cbe6e95](https://github.com/Yasas1994/Jaeger/commit/cbe6e95809437c06ff992824757597d47e2b1d68) by Yasas1994).
- added jinja ([cdb2e53](https://github.com/Yasas1994/Jaeger/commit/cdb2e53fbc936867e7ab1aea091b8de790ab5700) by Yasas1994).
- added data dir ([848d95c](https://github.com/Yasas1994/Jaeger/commit/848d95cf60591d531443df78fe618969d2d35a1b) by Yasas1994).
- added DOI ([795700c](https://github.com/Yasas1994/Jaeger/commit/795700cfc50f727c87d109570059e29b6dbd8d8e) by Yasas1994).

## [v1.1.30-alpha](https://github.com/Yasas1994/Jaeger/releases/tag/v1.1.30-alpha) - 2024-08-13

<small>[Compare with v1.1.26](https://github.com/Yasas1994/Jaeger/compare/v1.1.26...v1.1.30-alpha)</small>

### Added

- added tests ([3811cec](https://github.com/Yasas1994/Jaeger/commit/3811cecf60e2501d645521ad40017cb56738da0d) by Yasas1994).
- added progressbar2 ([409bfec](https://github.com/Yasas1994/Jaeger/commit/409bfec4e4f1bbdf1c6e382ee0cdc23a0fc43fb8) by Yasas1994).
- added pyfastx ([34d7265](https://github.com/Yasas1994/Jaeger/commit/34d7265fa65cb98c677469c54084d9f8b0d690a6) by Yasas1994).
- added pycirclize ([c1df16a](https://github.com/Yasas1994/Jaeger/commit/c1df16a4d58a6870516da07030ef1d0e90655658) by Yasas1994).
- added pyproject.toml ([0a75eae](https://github.com/Yasas1994/Jaeger/commit/0a75eae81b424316fc6d90f93f454e33a34d8dd4) by Yasas1994).
- added --version ([7c7562d](https://github.com/Yasas1994/Jaeger/commit/7c7562dd5d96af5dd30548575795b0c581300dcc) by Yasas1994).
- added runtime and memory usage ([325d27c](https://github.com/Yasas1994/Jaeger/commit/325d27cd4e2d7147d2a3142c121acd6a934ba96d) by Yasas1994).
- added pycirclize to requirements ([e0cccc7](https://github.com/Yasas1994/Jaeger/commit/e0cccc7bdce40a9b14d231fd79d5574bb296bc21) by Yasas1994).
- added bioconda setup ([4f150fa](https://github.com/Yasas1994/Jaeger/commit/4f150faf26e09d456330c831e1ea04f02562b643) by Yasas1994).

### Fixed

- fixed a typo ([735f1d6](https://github.com/Yasas1994/Jaeger/commit/735f1d60ee00612a7fc0b895b17de61c1039f436) by Yasas1994).
- fix macos install ([f015c01](https://github.com/Yasas1994/Jaeger/commit/f015c013b64fb55a4652f170dd63c6d97dc54218) by Yasas1994).
- fixing zerodivision error in g + c ([3601428](https://github.com/Yasas1994/Jaeger/commit/3601428fcb574676d0d8b565c496e3c8261d1496) by Yasas1994).
- fixed zerodivision error in g+c ([dea614f](https://github.com/Yasas1994/Jaeger/commit/dea614f1a2640f780ef06a084f251992668f3928) by Yasas1994).
- fixed zero division error due to Ns ([0dcf93e](https://github.com/Yasas1994/Jaeger/commit/0dcf93ec43f2084efbf95dd164afac91733c05e2) by Yasas1994).
- fixed wrong download link ([e05eef9](https://github.com/Yasas1994/Jaeger/commit/e05eef9bacbcd61a62597d9188db155cfa50a7c2) by Yasas1994).

### Changed

- changed to fit unix convention ([aaa6ff5](https://github.com/Yasas1994/Jaeger/commit/aaa6ff5c18a283a84020821244925316fccefbae) by Yasas1994).

### Removed

- removed Jaeger_inference ([6505347](https://github.com/Yasas1994/Jaeger/commit/6505347d8c33d5c5d78b87883ec4d57dc0b37d4d) by Yasas1994).

## [v1.1.26](https://github.com/Yasas1994/Jaeger/releases/tag/v1.1.26) - 2024-04-11

<small>[Compare with v1.1.23](https://github.com/Yasas1994/Jaeger/compare/v1.1.23...v1.1.26)</small>

### Added

- added tags ([0375c48](https://github.com/Yasas1994/Jaeger/commit/0375c4803b0964572fccad8dbcb871f080030b60) by Yasas1994).
- added a citation section ([0b1cd5c](https://github.com/Yasas1994/Jaeger/commit/0b1cd5cf17d5b66975d231562f91be08d476cbfc) by lingyi-owl).
- added changelog ([b11cefb](https://github.com/Yasas1994/Jaeger/commit/b11cefb9e3bd67ded4ad3ca6f29b4876b465be4a) by Yasas1994).
- added acknowlegements ([6a277e9](https://github.com/Yasas1994/Jaeger/commit/6a277e9ba1add77dcc8017d9e16d6e227ef0d746) by Yasas1994).
- added Jaeger_parallel script and updated reqs ([ae95ada](https://github.com/Yasas1994/Jaeger/commit/ae95ada91fb8a7214108b0042a9aa3f24e071400) by Yasas1994).
- add support to setect gpus and use virtualgpus ([ccf28c4](https://github.com/Yasas1994/Jaeger/commit/ccf28c430f24c2d9efa37a0efe33bfb391901e16) by Yasas1994).
- added psutils to setup file ([e7a9fec](https://github.com/Yasas1994/Jaeger/commit/e7a9fec266d918106b7dc058475efe12fb4ed1d8) by Yasas1994).

### Fixed

- fixed the input length issue ([510be59](https://github.com/Yasas1994/Jaeger/commit/510be59920c67835ecd726c3c21b9bde1e535b71) by Yasas1994).
- fixed missing name ([5f81410](https://github.com/Yasas1994/Jaeger/commit/5f81410ff0f9cdd17cef6d1c654abcdba85ba61d) by Yasas1994).

### Changed

- changed to false ([5f89a97](https://github.com/Yasas1994/Jaeger/commit/5f89a97c76befa8c43a2a3112c88a79f24aeafbb) by Yasas1994).
- changed to store_false ([a7d40d0](https://github.com/Yasas1994/Jaeger/commit/a7d40d0f82b3240d2876d30340094ae8113f8b1f) by Yasas1994).

### Removed

- removed misc ([de3610b](https://github.com/Yasas1994/Jaeger/commit/de3610b3cf458e37197f2a3b3db0734d200639e0) by Yasas1994).
- remove an unwanted test file ([542fd7d](https://github.com/Yasas1994/Jaeger/commit/542fd7d7544c679d9d3b944c8efa78ae0dfc4391) by Yasas1994).
- removed gpu mode cmbline option ([9977556](https://github.com/Yasas1994/Jaeger/commit/99775565af516e5cf741ddd3c9a5e9027498bfdd) by Yasas1994).

## [v1.1.23](https://github.com/Yasas1994/Jaeger/releases/tag/v1.1.23) - 2023-06-29

<small>[Compare with v1.1.22](https://github.com/Yasas1994/Jaeger/compare/v1.1.22...v1.1.23)</small>

## [v1.1.22](https://github.com/Yasas1994/Jaeger/releases/tag/v1.1.22) - 2023-01-25

<small>[Compare with v1.1.21](https://github.com/Yasas1994/Jaeger/compare/v1.1.21...v1.1.22)</small>

## [v1.1.21](https://github.com/Yasas1994/Jaeger/releases/tag/v1.1.21) - 2023-01-25

<small>[Compare with v1.1.2](https://github.com/Yasas1994/Jaeger/compare/v1.1.2...v1.1.21)</small>

### Fixed

- fixed: Exception ignored in: <function Pool.__del__ at 0x149864e50> ([ba7bcc9](https://github.com/Yasas1994/Jaeger/commit/ba7bcc9945cddf7b887b8d4794a404e465a60204) by Yasas1994).

## [v1.1.2](https://github.com/Yasas1994/Jaeger/releases/tag/v1.1.2) - 2023-01-25

<small>[Compare with v1.1.1](https://github.com/Yasas1994/Jaeger/compare/v1.1.1...v1.1.2)</small>

### Added

- added python setup file ([66e24ef](https://github.com/Yasas1994/Jaeger/commit/66e24ef1d7ff4b6aad1e45e4371e8e0a50fe362f) by Yasas1994).
- added a test file ([f0ff9f7](https://github.com/Yasas1994/Jaeger/commit/f0ff9f79084992d5a1c0c5894348aabc91e60ea3) by Yasas1994).
- added python api ([04bfd25](https://github.com/Yasas1994/Jaeger/commit/04bfd253755f179bffff4f96eac5dd50b47611ea) by Yasas1994).

## [v1.1.1](https://github.com/Yasas1994/Jaeger/releases/tag/v1.1.1) - 2023-01-24

<small>[Compare with v1.0.0](https://github.com/Yasas1994/Jaeger/compare/v1.0.0...v1.1.1)</small>

### Added

- Add files via upload ([698d5f8](https://github.com/Yasas1994/Jaeger/commit/698d5f8749032e6a8a005414ce08498286063672) by Yasas Wijesekara).
- added .gitignore ([74eabf7](https://github.com/Yasas1994/Jaeger/commit/74eabf7ebbfd519683da738387b61133714894be) by Yasas1994).
- Added weights for the AA model ([e00dbdb](https://github.com/Yasas1994/Jaeger/commit/e00dbdb16317fe8b872ccd523676f85dd016d1d2) by Yasas Wijesekara).

### Fixed

- fixed path to weights dir ([151e33f](https://github.com/Yasas1994/Jaeger/commit/151e33f0fec30f8728a3f45e01ea00d7dc80ccad) by Yasas1994).

### Changed

- changed tf logging level ([b6f99de](https://github.com/Yasas1994/Jaeger/commit/b6f99deaf0079a9253a815aade79f24f97620a34) by Yasas Wijesekara).

## [v1.0.0](https://github.com/Yasas1994/Jaeger/releases/tag/v1.0.0) - 2022-08-28

<small>[Compare with first commit](https://github.com/Yasas1994/Jaeger/compare/a69c9cbe9f9831efed33d03904a3cc02973384cf...v1.0.0)</small>

### Added

- added support to write predicted contigs to a fa ([0630a3a](https://github.com/Yasas1994/Jaeger/commit/0630a3a47375a56d218b4545db1365c2001e9378) by Yasas1994).
- added convlstm model ([d503094](https://github.com/Yasas1994/Jaeger/commit/d5030944c995a6addb035f63e601490c7df280d2) by Yasas Wijesekara).
- added model weights to the repo ([1c5dce9](https://github.com/Yasas1994/Jaeger/commit/1c5dce9b9facee121ddf555651aca28aa7beb660) by Yasas Wijesekara).
- added all jaeger code base to the repo ([73d0b9e](https://github.com/Yasas1994/Jaeger/commit/73d0b9efc81aab2525fee110ad8cad7d8bb78d6d) by Yasas Wijesekara).

