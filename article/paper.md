# Label Granularity, Not Model Capacity, Limits Multi-Source Mammography Classification: An Attention-Based Multiple Instance Learning Approach

**Authors:** *[Author 1]*<sup>1</sup>, *[Author 2]*<sup>1,2</sup>, *[Author 3]*<sup>2</sup>

<sup>1</sup> Artificial Intelligence and Medical Computing Research Laboratory (LIIACOM), Facultad de Medicina y Ciencias Biomédicas, Universidad Autónoma de Chihuahua, Circuito Universitario, Chihuahua, 31125, Chihuahua, México.

<sup>2</sup> Faculty of Engineering, Universidad Autónoma de Chihuahua, Circuito Universitario, Chihuahua, 31125, Chihuahua, México.

*Corresponding author: [email]@uach.mx*

---

## Abstract

Deep learning systems for mammographic benign–malignant discrimination are increasingly trained on aggregations of public datasets, a practice that raises two under-examined validity concerns: the source dataset can act as a shortcut for the label, and the annotation granularity of most public corpora does not match the image-level supervision these systems assume. This work presents a benign–malignant classifier trained across six public mammography datasets and, more importantly, a diagnostic protocol that separates genuine radiological signal from these two confounds. We first show that a classifier with no access to pixel data, predicting solely from the source dataset identity, reaches 70.63% accuracy on a conventionally constructed test set, and that 91.6% of that test set consists of patients also present in training. After enforcing patient-grouped partitioning and reporting a within-source AUC that counts only benign–malignant pairs originating from the same dataset, apparent performance falls from an image-level accuracy of 82.03% to a within-source AUC of 0.6734. We then demonstrate that this residual ceiling is not a modelling limitation. Quadrupling input resolution to 1024 px, which restores microcalcification-scale detail for 100% of the evaluated evidence, and applying strong regularization, which reduces the train–validation gap from 28 to 17 points, both leave the within-source AUC unchanged within 0.017. Instead, we identify annotation granularity as the binding constraint: in three of four datasets, 100% of patients imaged bilaterally carry an identical label on both breasts, so that under the clinical unilaterality of breast cancer approximately half of all images labelled malignant depict a healthy breast. We formalize this as a label-noise model that predicts an AUC gain proportional to the contaminated fraction *f* upon patient-level aggregation, and confirm the prediction across datasets (*r* = 0.93). Reformulating training as attention-based multiple instance learning over patient bags, so that the loss no longer penalizes correct low scores on the contralateral healthy breast, yields a test bag-level AUC of 0.8294 (95% CI [0.797, 0.862]) and a within-source AUC of 0.7671, at an operating point of 86.44% sensitivity and 58.74% specificity. We further report three negative results, on resolution, regularization and model ensembling, that constrain the space of plausible explanations for the remaining gap.

**Keywords:** Mammography, Multiple Instance Learning, Attention Pooling, Shortcut Learning, Label Noise, Multi-Source Datasets, ConvNeXt

---

## 1 Introduction

Breast cancer remains among the most frequently diagnosed malignancies worldwide, and mammographic screening is the principal population-level detection modality. The interpretive burden and inter-reader variability of screening mammography have motivated sustained interest in automated benign–malignant classification, with several systems now reporting reader-level or super-reader-level discrimination on large proprietary cohorts [1, 2].

Reproducing such results in academic settings is constrained by data access. Public mammography corpora are individually small, and a common response is to pool several of them into a single training set. This practice is methodologically appealing, as it increases sample size and nominally broadens the distribution of acquisition hardware and populations. It also introduces two failure modes that conventional evaluation does not surface.

The first is **shortcut learning** [3]. Public mammography datasets differ systematically in detector, dynamic range, compression, burned-in annotation and, critically, in class prevalence, because each was assembled for a different purpose. A screening cohort may contain no malignant cases at all, while a diagnostic cohort may be predominantly malignant. A network that learns to recognize the acquisition signature of each source thereby acquires substantial predictive power over the label without extracting any radiological information. This mechanism is well documented in chest radiography, where models have been shown to key on site-specific markers rather than pathology [4, 5], but is rarely quantified in multi-source mammography work.

The second is **annotation granularity mismatch**. Most public mammography corpora distribute a diagnosis at the level of the patient or the case, not the individual image. When such a corpus is unrolled into an image-level training set, every view of a patient inherits the patient's label. Because breast cancer is unilateral in the large majority of cases, the images of the contralateral, unaffected breast of a malignant patient are labelled malignant while containing no lesion. Image-level supervision therefore asks the network to produce a positive score for radiologically normal tissue.

**Figure 1** *Overview of the proposed pipeline. Raw mammograms undergo breast-region extraction, contrast enhancement and background neutralization; images are grouped into bags whose level is detected per dataset; a ConvNeXt encoder with residually gated CBAM produces per-instance embeddings that gated attention pooling combines into a single bag decision. Evaluation reports the within-source AUC.*

Table 1 situates this work among representative studies. Our emphasis differs from the prevailing direction of the field: rather than proposing a novel architecture, we treat the evaluation protocol and the supervision granularity as the primary objects of study, and adopt a deliberately standard backbone so that reported differences are attributable to protocol rather than to model capacity.

**Table 1** Design choices in representative mammographic classification studies. PGS: patient-grouped splits. SC: explicit source-confound control. LG: label-granularity-aware supervision.

| Study | Backbone / Formulation | Sources | Decision level | PGS | SC | LG |
|---|---|---|---|---|---|---|
| This study | ConvNeXt-S + CBAM / attention MIL | 6 public | Bag (patient/breast) | Yes | Yes | Yes |
| Wu et al. [1] | Multi-view ResNet | 1 institutional | Breast | Yes | n/a<sup>†</sup> | Yes |
| McKinney et al. [2] | Ensemble of CNNs | 2 institutional | Breast / patient | Yes | Partial | Yes |
| Shen et al. [6] | Patch-then-whole-image CNN | 1–2 public | Image | Yes | n/a<sup>†</sup> | No |
| Ilse et al. [7] | Attention MIL (histopathology) | 2 public | Bag | Yes | n/a<sup>†</sup> | Yes |

<sup>†</sup> Single-source studies are not exposed to the source-as-shortcut confound; the column is not applicable rather than unaddressed.

The contributions of this study are as follows:

- **A quantitative shortcut audit for multi-source mammography.** We introduce a source-prior baseline, a classifier that observes only the dataset of origin, and show it attains 70.63% accuracy under a conventional split. We further propose decomposing the pooled AUC into within-source and cross-source components, and report the within-source component as the confound-free primary metric.

- **Identification and formalization of label granularity as the binding constraint.** We measure, per dataset, whether the two breasts of a patient share a label, derive the contaminated fraction *f* of positively labelled images, and formulate a noise model predicting the AUC gain from patient-level aggregation. The prediction is confirmed across datasets (*r* = 0.93 between *f* and the observed gain).

- **An attention-based MIL formulation matched to the measured granularity.** Bag level is detected per dataset from the data rather than assumed, so that corpora annotated per breast are not coarsened. Gated attention pooling allows suppression of uninformative views and exposes which view drove each decision.

- **Three negative results that constrain the explanation space.** We show that input resolution, regularization strength, and ensembling with an image-level model each fail to move the within-source AUC beyond noise, evidence that argues against capacity- and resolution-based accounts of the residual gap.

---

## 2 Methodology

### 2.1 Dataset

We use a harmonized multi-source corpus assembling seven public mammography collections. Restricting to records with a definite benign or malignant label yields 14,294 images from 4,648 patients. Table 2 summarizes composition.

**Table 2** Composition of the source corpus, restricted to definite benign/malignant labels.

| Source | Benign | Malignant | Patients | Malignant (%) |
|---|---|---|---|---|
| cdd-cesm [8] | 331 | 331 | 284 | 50.0 |
| cmmd [9] | 1,108 | 4,094 | 1,856 | 78.7 |
| dmid | 154 | 24 | 178 | 13.5 |
| inbreast [10] | 243 | 57 | 300 | 19.0 |
| kau-bcmd [11] | 265 | 22 | 76 | 7.7 |
| mini-ddsm [12] | 2,684 | 2,716 | 1,350 | 50.3 |
| rsna-screening [13] | 2,265 | 0 | 604 | 0.0 |
| **Total** | **7,050** | **7,244** | **4,648** | **50.7** |

The spread of per-source malignancy prevalence, from 0.0% to 78.7%, is the structural basis of the source shortcut. The rsna-screening subset is an extreme case: it contributes 2,265 images and no malignant instances, so that recognizing this source alone resolves 2,265 labels. We therefore exclude it from all experiments, retaining 12,029 images from 4,044 patients at 60.2% malignant prevalence. Retaining it is available as a configuration flag but is not recommended.

### 2.2 Partitioning protocol

Each patient contributes multiple images through the standard craniocaudal (CC) and mediolateral oblique (MLO) projections of each breast. Partitioning at the image level therefore places different views of the same breast on both sides of the train–test boundary. In the partition originally available for this corpus, **1,310 of 1,430 test rows (91.6%)** and 1,320 of 1,429 validation rows (92.4%) belonged to patients also present in training.

We repartition using `StratifiedGroupKFold` with the patient as the grouping unit and the (source, label) pair as the stratification variable, at a 70/15/15 ratio. Grouping by patient eliminates leakage; stratifying jointly on source and label ensures that the three partitions carry the same mixture of acquisition domains, so that the train-to-test difference is not itself a domain shift. Patient identifiers are prefixed by source, since identifier ranges collide across collections. Verified leakage after repartitioning is 0.00%.

### 2.3 Preprocessing

Each image is converted to grayscale and processed as follows.

**Breast region extraction.** Otsu thresholding followed by connected-component analysis retains the largest component, which is invariably the breast. The component mask is morphologically closed to fill interior holes in dense tissue, and the image is cropped to the component bounding box. This step removes background, burned-in laterality and projection text, and metallic skin markers, all of which differ systematically between collections and thus constitute direct evidence of source identity.

**Contrast enhancement.** Contrast-limited adaptive histogram equalization (CLAHE, clip limit 2.0, 8×8 tiles) is applied **before** masking. The ordering is material: applied after masking, the equalizer encounters the large mass of zero-valued background pixels and displaces them to a small positive value (measured: 0 → 3), which prevents the background from being subsequently zeroed cleanly. Tissue is then floored at intensity 1 so that "pixel = 0" identifies background unambiguously and the mask can be recovered from the cached image.

Notably, we do **not** apply erosion or morphological opening, both of which appear in prior mammography preprocessing pipelines. With a 3×3 structuring element these operations remove structures a few pixels in extent, which is precisely the scale of microcalcifications, the most discriminative radiological sign for malignancy.

**Figure 2** *Effect of preprocessing. Left: raw image with background, burned-in laterality text and skin markers. Centre: largest connected component after Otsu thresholding. Right: cropped, CLAHE-enhanced and masked input, with background at exactly zero. The removed annotations are direct evidence of source identity.*

**Geometric normalization and background neutralization.** Images are resized preserving aspect ratio and zero-padded to a square canvas. Intensities are normalized with ImageNet statistics to match the distribution expected by the pretrained backbone; the breast mask is then applied **after** normalization so that background is exactly 0.0. Without this step a black pixel maps to −2.118, nearly 30 times the magnitude of mean tissue (0.074), so that a region occupying 35–50% of the canvas dominates the input signal. Zero background contributes neither activation nor gradient.

---

## 3 Quantifying the two confounds

### 3.1 The source-prior baseline

Let $D(x)$ denote the source dataset of image $x$ and $\pi_d$ the malignant prevalence of dataset $d$ estimated on the training partition. Define the source-prior classifier

$$\hat{y}_{\text{prior}}(x) = \mathbb{1}\left[\pi_{D(x)} > 0.5\right] \tag{1}$$

which observes no pixel data whatsoever. On the original image-level test partition this classifier attains **70.63% accuracy**, and on the repartitioned test set it attains an AUC of 0.694 when $\pi_{D(x)}$ is used directly as a score. Any reported figure must be interpreted against this floor.

### 3.2 Within-source AUC decomposition

The AUC is the probability that a randomly chosen positive is ranked above a randomly chosen negative. Equivalently, over the set $P$ of all (positive, negative) pairs,

$$\text{AUC} = \frac{1}{|P|}\sum_{(i,j) \in P} \mathbb{1}\left[s_i > s_j\right] \tag{2}$$

In a multi-source corpus $P$ partitions into pairs whose two members share a source, $P_w$, and pairs that cross sources, $P_c$. Cross-source pairs can be resolved by recognizing source identity alone, because sources differ in prevalence; within-source pairs cannot. The pooled AUC is their pair-count-weighted mixture:

$$\text{AUC}_{\text{pooled}} = \frac{|P_w|}{|P|}\,\text{AUC}_w + \frac{|P_c|}{|P|}\,\text{AUC}_c \tag{3}$$

$$\text{AUC}_w = \frac{\sum_{d} |P_d|\,\text{AUC}_d}{\sum_{d} |P_d|} \tag{4}$$

where $P_d$ are the pairs internal to dataset $d$. On our bag-level test set, $|P_w| = 24{,}472$ and $|P_c| = 60{,}801$: **71.3% of the evidence entering the pooled AUC comes from cross-source comparisons.** We therefore report $\text{AUC}_w$ as the primary metric.

**Figure 3** *Share of within-source benign–malignant pairs contributed by each source, with its test AUC. Weight is the product of the class counts, not the number of bags, so cmmd and mini-ddsm carry 96.3% of the evidence while dmid and kau-bcmd contribute essentially none despite AUCs of 1.000.*

Because $|P_d| = n_d^+ n_d^-$, datasets contribute to $\text{AUC}_w$ in proportion to the product of their class counts, not their image counts. In our test set cmmd and mini-ddsm carry 58.7% and 37.6% of the within-source evidence respectively, while dmid and kau-bcmd carry 0.1% and 0.04%. We report 95% confidence intervals by the Hanley–McNeil method [14] and explicitly flag datasets contributing fewer than 500 pairs as uninterpretable.

### 3.3 Label granularity and the contaminated fraction

For each dataset we test whether the two breasts of a patient carry distinct labels. Table 3 reports, among patients imaged bilaterally, the proportion whose left and right breasts share a label.

**Table 3** Label concordance between the two breasts of the same patient.

| Source | Bilaterally imaged patients | Identical label on both breasts |
|---|---|---|
| cmmd | 745 | **100.0%** |
| mini-ddsm | 1,350 | **100.0%** |
| kau-bcmd | 67 | **100.0%** |
| cdd-cesm | 91 | 37.4% |

In three of four datasets the concordance is exactly 100%, which is not a clinical finding but an artifact of annotation: the diagnosis is recorded per patient and propagated to every image. Only cdd-cesm annotates per breast.

Breast cancer is unilateral in approximately 95–98% of cases. For a malignant patient imaged bilaterally, roughly half the images therefore depict a healthy breast while carrying a malignant label. Define the contaminated fraction

$$f_d = \frac{1}{2}\cdot\frac{\left|\{x : y_x = 1,\ \text{patient}(x) \text{ imaged bilaterally}\}\right|}{\left|\{x : y_x = 1\}\right|} \tag{5}$$

giving $f = 0.36$ for cmmd, $f = 0.50$ for mini-ddsm and $f = 0.02$ for cdd-cesm.

**A falsifiable prediction.** Model the contaminated positives as indistinguishable from true negatives. A pair formed with a contaminated positive is then concordant with probability 0.5, and

$$\text{AUC}_{\text{obs}} = (1 - f)\,\text{AUC}_{\text{true}} + f \cdot 0.5 \tag{6}$$

Equation (6) implies that aggregating predictions to the level at which the label is actually correct, the patient, should raise the measured AUC by an amount monotonically increasing in $f$. Table 4 reports the test.

**Table 4** Predicted and observed effect of patient-level aggregation. Model and predictions are identical across the two columns; only the evaluation granularity differs.

| Source | *f* | Image-level AUC | Patient-level AUC | Gain |
|---|---|---|---|---|
| mini-ddsm | 0.50 | 0.647 | 0.761 | **+0.114** |
| cmmd | 0.36 | 0.676 | 0.717 | **+0.041** |
| cdd-cesm | 0.02 | 0.806 | 0.803 | −0.003 |

**Figure 4** *Observed AUC gain from patient-level aggregation against the contaminated fraction $f$, with the least-squares fit through the origin implied by Equation (6). The ordering predicted by the model is exact ($r$ = 0.93), while the fitted slope implies $\text{AUC}_{\text{true}}$ = 0.69 rather than the ~0.80 measured directly, indicating that aggregation removes the contamination from the metric but not from the model, which was trained on contaminated labels.*

The ordering is exact and the correlation between $f$ and the observed gain is *r* = 0.93 across the three interpretable datasets. The dataset already annotated per breast gains nothing, as predicted. Inverting Equation (6) on the observed image-level values yields $\text{AUC}_{\text{true}}$ estimates of 0.782 (mini-ddsm), 0.797 (cmmd) and 0.821 (cdd-cesm), a tight cluster consistent with the directly measured patient-level AUC of 0.806.

We note the limitation that Equation (6) is a first-order model: it assumes contaminated positives are wholly uninformative, whereas contralateral tissue may carry weak population-level risk signal. It also treats *f* as exactly one half of bilaterally imaged positives, neglecting the 2–5% bilateral-disease rate. Both approximations bias the estimated $\text{AUC}_{\text{true}}$ conservatively.

---

## 4 Bag-level classification with attention MIL

### 4.1 Problem formulation

The measurements of Section 3.3 indicate that supervision is available at the level of a group of images, not an individual image. This is precisely the multiple instance learning setting [7]: a **bag** $B = \{x_1, \ldots, x_K\}$ carries a single label, positive if at least one instance is positive.

Under image-level training, the network views the contralateral healthy breast, correctly assigns it a low malignancy score, and receives a gradient penalizing that correct behaviour. Under bag-level training the loss constrains only the aggregated bag score, leaving the network free to score the healthy breast low provided it scores the affected breast high.

**Bag level is detected from the data rather than assumed.** For each dataset, if the two breasts of a patient share a label in at least 90% of bilaterally imaged patients, the annotation is patient-level and the bag is the patient. Otherwise the annotation is already per breast and the bag is the (patient, laterality) pair. On our partitions this resolves cdd-cesm to breast-level bags and all remaining datasets to patient-level bags, so that the one corpus with finer annotation is not coarsened. Training bags number 2,965 (median size 4).

### 4.2 Instance encoder

Each instance is encoded by a ConvNeXt-Small backbone [15] pretrained on ImageNet, augmented with Convolutional Block Attention Modules (CBAM) [16] on the final two stages.

Inserting randomly initialized attention into a pretrained backbone is destructive: a standard CBAM multiplies activations by $\sigma(\cdot) \approx 0.5$ from the first optimization step, degrading the very features transfer learning is meant to exploit. We therefore wrap each module in a residual gate initialized at zero,

$$x' = x + \gamma\left(\text{CBAM}(x) - x\right), \qquad \gamma_{\text{init}} = 0 \tag{7}$$

so that the module begins as the exact identity and the network learns how much attention to apply at each stage. The instance embedding $h_k \in \mathbb{R}^{D}$ is obtained by global average pooling followed by layer normalization.

### 4.3 Gated attention pooling

Bag aggregation uses the gated attention mechanism of Ilse et al. [7]:

$$a_k = \frac{\exp\left\{w^{\top}\left(\tanh(V h_k) \odot \sigma(U h_k)\right)\right\}}{\sum_{k' \in B} \exp\left\{w^{\top}\left(\tanh(V h_{k'}) \odot \sigma(U h_{k'})\right)\right\}} \tag{8}$$

$$z = \sum_{k \in B} a_k h_k \tag{9}$$

The $\tanh$ branch extracts evidence and the $\sigma$ branch acts as a multiplicative gate that can drive a term to zero, allowing outright suppression of an instance rather than mere down-weighting. This distinction is functionally important here: the contralateral healthy breast should not contribute, not merely contribute less. The attention weights $a_k$ additionally identify which view drove each decision, providing coarse localization without lesion masks or ROI annotations.

Bags vary in size from 1 to 14 instances. Rather than padding to a common length, which would introduce phantom instances diluting the attention distribution, all bags in a mini-batch are flattened into a single tensor accompanied by a bag-index vector, and the softmax in Equation (8) is computed segment-wise over that index.

**Figure 5** *Bag-level inference. The four standard views of a patient are encoded independently; gated attention assigns weights summing to one within the bag, permitting suppression of the contralateral healthy breast; the decision is taken on the weighted sum. A single loss is computed per bag.*

### 4.4 Sampling and optimization

Bags are drawn by a weighted sampler equalizing the mass of each (source, label) cell, counted in bags rather than images. Equalizing these cells removes the mutual information between source identity and label within a batch, rendering the shortcut uninformative.

Models are trained at 1024×1024 resolution with one bag per step and gradient accumulation over 16 steps, for 30 epochs. Optimization uses AdamW [17] with a base learning rate of 2×10⁻⁴, layer-wise learning rate decay of 0.75 through the backbone depth, weight decay 0.05 excluded from norms and biases, cosine schedule with one epoch of linear warmup over the full run, gradient norm clipping at 1.0, label smoothing 0.05, and stochastic depth [18] at 0.4. An exponential moving average of the weights is maintained with a warmup ramp $\min(\beta, (1+t)/(10+t))$; without this ramp the average retains a substantial fraction of the random initialization for several epochs, since gradient accumulation reduces the number of weight updates per epoch by a factor of 16. Both the raw and averaged models are evaluated each epoch and the better selected by validation AUC.

At inference, instances are encoded in chunks so that large bags do not require all views in memory simultaneously, and test-time augmentation averages predictions over the horizontal mirror, the only transformation that preserves anatomy.

### 4.5 Operating point selection

The decision threshold is selected **exclusively on the validation partition** and applied unchanged to test. The Youden index [19], $J = \text{sensitivity} + \text{specificity} - 1$, weights the two error types equally, which is inappropriate for cancer screening: a false negative is a missed tumour, a false positive an additional workup. We therefore select the most stringent threshold still attaining a target sensitivity of 85%.

---

## 5 Results and Discussion

### 5.1 Main results

On the held-out test partition of 586 bags, the proposed method attains a bag-level AUC of **0.8294 (95% CI [0.797, 0.862])** and a within-source AUC of **0.7671**. At the validation-selected operating point, sensitivity is 86.44%, specificity 58.74%, accuracy 73.72% and macro F1 0.7265.

**Figure 6** *Receiver operating characteristic curves on the test partition, overall and for the two datasets carrying 96.3% of the within-source evidence, with the three candidate operating points marked and the selected one filled.*

**Figure 7** *Confusion matrix at the validation-selected operating point (85% target sensitivity), with row-normalized percentages.*

**Table 5** Per-source test performance at the bag level. Pairs, not bag counts, determine each source's weight in the within-source AUC.

| Source | Bags | Pairs | AUC | 95% CI |
|---|---|---|---|---|
| cmmd | 265 | 14,364 | 0.803 | [0.750, 0.856] |
| mini-ddsm | 193 | 9,202 | 0.699 | [0.623, 0.774] |
| cdd-cesm | 49 | 558 | 0.875 | [0.778, 0.971] |
| inbreast | 44 | 315 | 0.921 | [0.794, 1.000]<sup>‡</sup> |
| dmid | 25 | 24 | 1.000 | [1.000, 1.000]<sup>‡</sup> |
| kau-bcmd | 10 | 9 | 1.000 | [1.000, 1.000]<sup>‡</sup> |
| **Within-source (weighted)** | | **24,472** | **0.7671** | |

<sup>‡</sup> Fewer than 500 pairs; not interpretable. dmid contributes a single malignant bag against 24 benign, so its AUC of 1.000 summarizes 24 comparisons.

We stress that the near-perfect AUCs of dmid, kau-bcmd and inbreast should not be reported as findings. Together they account for 1.4% of the within-source evidence. The interpretable result rests on cmmd and mini-ddsm, which jointly carry 96.3%.

### 5.2 Effect of each protocol correction

**Table 6** Progression of measured performance. All rows use the same patient-grouped partitions; the first row of the original code additionally suffered 91.6% patient leakage and is not comparable.

| Configuration | Pooled AUC | Within-source AUC | Decision level |
|---|---|---|---|
| Original pipeline (leaked partitions) | — (82.03% acc.) | — | Image |
| 512 px, ConvNeXt-B | 0.7520 | 0.6734 | Image |
| 512 px + background neutralization | 0.7471 | 0.6561 | Image |
| 1024 px, ConvNeXt-S + regularization | 0.7332 | 0.6616 | Image |
| ↳ same model, aggregated to patient | 0.7872 | 0.7378 | Patient |
| **Attention MIL (proposed)** | **0.8294** | **0.7671** | Bag |
| Rank ensemble (MIL + image model) | 0.8204 | 0.7698 | Bag |

Two observations follow. First, the largest single improvement, from 0.6616 to 0.7378 within-source, comes from changing the **evaluation** granularity alone, with the model held fixed. Second, MIL adds a further +0.029 on top of that, indicating that correcting the training signal contributes beyond correcting the measurement.

Per source, the MIL gain is not uniform: cmmd improves from 0.717 to 0.803 while mini-ddsm declines from 0.761 to 0.699. The per-source confidence intervals of Table 5 overlap in both cases, but comparing two AUCs by the overlap of their individual intervals is an incorrect and highly conservative test: those intervals describe the uncertainty of each AUC under independent resampling, not the uncertainty of their difference [21]. Because both models are evaluated on identical cases and share a backbone and training data, their errors are strongly correlated, and the appropriate test is DeLong's paired comparison [22], which estimates and removes that covariance. We use the fast implementation of Sun and Xu [23].

**Table 7** DeLong paired comparison of the MIL model against the image-level model aggregated to the patient, on the 537 test bags for which both are defined. *r* is the DeLong-estimated correlation between the two models.

| Subset | *n* | AUC MIL | AUC image | Difference | 95% CI of difference | *r* | *p* |
|---|---|---|---|---|---|---|---|
| All matched | 537 | 0.824 | 0.787 | +0.037 | [+0.005, +0.069] | 0.60 | **0.025** |
| cmmd | 265 | 0.803 | 0.717 | +0.086 | [+0.026, +0.147] | 0.53 | **0.005** |
| mini-ddsm | 193 | 0.699 | 0.761 | −0.063 | [−0.133, +0.008] | 0.53 | 0.081 |
| inbreast | 44 | 0.921 | 0.927 | −0.006 | [−0.041, +0.028] | 0.98 | 0.720 |

The paired test resolves what the overlapping intervals could not. The overall gain is significant (*p* = 0.025), the cmmd gain is significant at *p* = 0.005, and the mini-ddsm decline does not reach significance (*p* = 0.081, with a difference interval that includes zero). The correct statement is therefore that MIL produces a significant overall improvement driven by the dataset with the highest structural contamination, and that the apparent regression on mini-ddsm is not established by the data. We nonetheless refrain from claiming a uniform improvement across sources.

### 5.3 Negative results

Three interventions failed to move the within-source AUC beyond noise. We report them because they constrain the space of plausible explanations for the residual gap.

**Input resolution.** Microcalcifications measure 0.2–0.5 mm. At a typical full-field digital detector pitch of 70 µm and native long-edge dimensions of 2,294–2,577 px for the two dominant datasets, resizing to 512 px yields 0.31–0.35 mm per pixel, placing microcalcifications below one pixel. Since 44% of malignant cmmd cases involve calcifications, this suggested a physical ceiling. Training at 1024 px, which brings 100% of the evaluated evidence below the 0.2 mm/px threshold, changed the within-source AUC from 0.6734 to 0.6616.

**Regularization.** Stochastic depth 0.4, mixup [20] 0.3 and dropout 0.5 reduced the train–validation accuracy gap from 28 to 17 points, confirming the intervention acted as intended on overfitting. The within-source AUC did not respond.

**Ensembling.** Rank-averaging the MIL and image-level models with the mixing weight selected on validation gave a pooled AUC of 0.8204 against 0.8293 for MIL alone. The diagnosis is more informative than the result: the two models exchange rank between partitions, the image-level model leading by +0.015 on validation and trailing by −0.040 on test. The weight was therefore assigned in good faith to the model that generalized worse.

This last observation exposes a limitation that applies to every validation-selected quantity in this study, including the checkpoint epoch, the threshold and the mixing weight. With 584 validation bags and a confidence interval of width ≈0.03, differences below that magnitude are not resolvable, and several decisions here were taken on differences of that order.

### 5.4 Interpretation

Across five configurations spanning a 4× change in input pixels, a substantial change in regularization and two changes in decision granularity, the within-source AUC of the image-level formulation varied by 0.017, while a change in supervision granularity alone moved it by 0.076. Taken together with the confirmed prediction of Equation (6), the evidence supports annotation granularity rather than model capacity as the binding constraint in this setting.

The practical implication for the field is that pooled multi-source mammography benchmarks require both a within-source metric and an explicit statement of annotation granularity for results to be comparable. An image-level AUC reported on a pooled corpus with patient-level annotations conflates three quantities: radiological discrimination, source recognition, and the contaminated fraction of the positive class.

The principal remaining limitation is clinical rather than methodological. At 86.44% sensitivity the specificity is 58.74%, meaning roughly four in ten benign cases are flagged. This is bounded by the AUC, and mini-ddsm, carrying 37.6% of the within-source evidence at an AUC of 0.699, is the weakest component.

---

## 6 Conclusion

We presented a benign–malignant mammography classifier trained across six public datasets, together with a diagnostic protocol that isolates genuine radiological signal from two confounds endemic to pooled corpora. We showed that a conventionally reported image-level accuracy of 82.03% rested on a test set in which 91.6% of rows shared patients with training, and against a floor of 70.63% attainable from source identity alone.

The central finding is that the residual performance ceiling of the image-level formulation was not attributable to model capacity, input resolution or overfitting, each of which we excluded experimentally, but to a mismatch between the granularity of the available annotations and the granularity of the supervision. Because most public mammography corpora record a diagnosis per patient and propagate it to every view, approximately half of the images labelled malignant depict a healthy contralateral breast. We formalized this as a noise model, derived a falsifiable prediction relating the aggregation gain to the contaminated fraction, and confirmed it across datasets.

Reformulating the task as attention-based multiple instance learning over bags whose level is detected per dataset yielded a bag-level AUC of 0.8294 (95% CI [0.797, 0.862]) and a within-source AUC of 0.7671, at 86.44% sensitivity. Three negative results, on resolution, regularization and ensembling, constrain alternative explanations for the remaining gap.

Future work should pursue two directions. The first is annotation: with lesion-level or at minimum breast-level labels across the full corpus, the formulation admits substantially more supervision than is currently available. The second is the interpretability afforded by attention weights, which identify the contributing view at no annotation cost and which we have not yet validated against radiologist-marked lesion locations. Such validation would establish whether the mechanism suppresses the contralateral breast for the reason hypothesized here.

---

## Declarations

**Funding.** *[To be completed.]*

**Conflict of interest.** The authors declare no competing interests.

**Ethics approval and consent to participate.** This study used exclusively publicly available, de-identified mammography datasets. No new human data were collected.

**Consent for publication.** Not applicable.

**Data availability.** All datasets used are publicly available from their respective providers, as cited.

**Code availability.** *[Repository URL to be completed.]*

**Author contribution.** *[To be completed.]*

---

## References

[1] Wu, N., Phang, J., Park, J., Shen, Y., Huang, Z., Zorin, M., Jastrzębski, S., Févry, T., Katsnelson, J., Kim, E., Wolfson, S., Parikh, U., Gaddam, S., Lin, L.L.Y., Ho, K., Weinstein, J.D., Reig, B., Gao, Y., Toth, H., Pysarenko, K., Lewin, A., Lee, J., Airola, K., Mema, E., Chung, S., Hwang, E., Samreen, N., Kim, S.G., Heacock, L., Moy, L., Cho, K., Geras, K.J.: Deep neural networks improve radiologists' performance in breast cancer screening. IEEE Transactions on Medical Imaging **39**(4), 1184–1194 (2020) https://doi.org/10.1109/TMI.2019.2945514

[2] McKinney, S.M., Sieniek, M., Godbole, V., Godwin, J., Antropova, N., Ashrafian, H., Back, T., Chesus, M., Corrado, G.S., Darzi, A., Etemadi, M., Garcia-Vicente, F., Gilbert, F.J., Halling-Brown, M., Hassabis, D., Jansen, S., Karthikesalingam, A., Kelly, C.J., King, D., Ledsam, J.R., Melnick, D., Mostofi, H., Peng, L., Reicher, J.J., Romera-Paredes, B., Sidebottom, R., Suleyman, M., Tse, D., Young, K.C., De Fauw, J., Shetty, S.: International evaluation of an AI system for breast cancer screening. Nature **577**(7788), 89–94 (2020) https://doi.org/10.1038/s41586-019-1799-6

[3] Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., Wichmann, F.A.: Shortcut learning in deep neural networks. Nature Machine Intelligence **2**(11), 665–673 (2020) https://doi.org/10.1038/s42256-020-00257-z

[4] Zech, J.R., Badgeley, M.A., Liu, M., Costa, A.B., Titano, J.J., Oermann, E.K.: Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study. PLOS Medicine **15**(11), 1002683 (2018) https://doi.org/10.1371/journal.pmed.1002683

[5] DeGrave, A.J., Janizek, J.D., Lee, S.-I.: AI for radiographic COVID-19 detection selects shortcuts over signal. Nature Machine Intelligence **3**(7), 610–619 (2021) https://doi.org/10.1038/s42256-021-00338-7

[6] Shen, L., Margolies, L.R., Rothstein, J.H., Fluder, E., McBride, R., Sieh, W.: Deep learning to improve breast cancer detection on screening mammography. Scientific Reports **9**(1), 12495 (2019) https://doi.org/10.1038/s41598-019-48995-4

[7] Ilse, M., Tomczak, J.M., Welling, M.: Attention-based deep multiple instance learning. In: Proceedings of the 35th International Conference on Machine Learning (ICML), pp. 2127–2136 (2018)

[8] Khaled, R., Helal, M., Alfarghaly, O., Mokhtar, O., Elkorany, A., El Kassas, H., Fahmy, A.: Categorized contrast enhanced mammography dataset for diagnostic and artificial intelligence research. Scientific Data **9**(1), 122 (2022) https://doi.org/10.1038/s41597-022-01238-0

[9] Cui, C., Li, L., Cai, H., Fan, Z., Zhang, L., Dan, T., Li, J., Wang, J.: The Chinese Mammography Database (CMMD): An online mammography database with biopsy confirmed types for machine diagnosis of breast. The Cancer Imaging Archive (2021) https://doi.org/10.7937/tcia.eqde-4b16

[10] Moreira, I.C., Amaral, I., Domingues, I., Cardoso, A., Cardoso, M.J., Cardoso, J.S.: INbreast: Toward a full-field digital mammographic database. Academic Radiology **19**(2), 236–248 (2012) https://doi.org/10.1016/j.acra.2011.09.014

[11] Alsolami, A.S., Shalash, W., Alsaggaf, W., Ashoor, S., Refaat, H., Elmogy, M.: King Abdulaziz University Breast Cancer Mammogram Dataset (KAU-BCMD). Data **6**(11), 111 (2021) https://doi.org/10.3390/data6110111

[12] Lekamlage, C.D., Afzal, F., Westerberg, E., Cheddad, A.: Mini-DDSM: Mammography-based automatic age estimation. In: 2020 3rd International Conference on Digital Medicine and Image Processing, pp. 1–6 (2020) https://doi.org/10.1145/3441369.3441370

[13] Radiological Society of North America: RSNA Screening Mammography Breast Cancer Detection Challenge (2023). https://www.kaggle.com/competitions/rsna-breast-cancer-detection

[14] Hanley, J.A., McNeil, B.J.: The meaning and use of the area under a receiver operating characteristic (ROC) curve. Radiology **143**(1), 29–36 (1982) https://doi.org/10.1148/radiology.143.1.7063747

[15] Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., Xie, S.: A ConvNet for the 2020s. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11976–11986 (2022)

[16] Woo, S., Park, J., Lee, J.-Y., Kweon, I.S.: CBAM: Convolutional block attention module. In: Proceedings of the European Conference on Computer Vision (ECCV), pp. 3–19 (2018)

[17] Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. In: International Conference on Learning Representations (ICLR) (2019)

[18] Huang, G., Sun, Y., Liu, Z., Sedra, D., Weinberger, K.Q.: Deep networks with stochastic depth. In: Proceedings of the European Conference on Computer Vision (ECCV), pp. 646–661 (2016)

[19] Youden, W.J.: Index for rating diagnostic tests. Cancer **3**(1), 32–35 (1950)

[21] Schenker, N., Gentleman, J.F.: On judging the significance of differences by examining the overlap between confidence intervals. The American Statistician **55**(3), 182–186 (2001) https://doi.org/10.1198/000313001317097960

[22] DeLong, E.R., DeLong, D.M., Clarke-Pearson, D.L.: Comparing the areas under two or more correlated receiver operating characteristic curves: A nonparametric approach. Biometrics **44**(3), 837–845 (1988) https://doi.org/10.2307/2531595

[23] Sun, X., Xu, W.: Fast implementation of DeLong's algorithm for comparing the areas under correlated receiver operating characteristic curves. IEEE Signal Processing Letters **21**(11), 1389–1393 (2014) https://doi.org/10.1109/LSP.2014.2337313

[20] Zhang, H., Cissé, M., Dauphin, Y.N., Lopez-Paz, D.: mixup: Beyond empirical risk minimization. In: International Conference on Learning Representations (ICLR) (2018)
