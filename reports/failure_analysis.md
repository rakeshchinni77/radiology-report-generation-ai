# Failure Analysis

## Overview

This project implements a multimodal chest X-ray report generation system built around a frozen vision encoder, a constrained GPT-style language decoder, and cross-attention fusion. In the current lightweight configuration, the model is capable of producing syntactically plausible outputs, but its clinical reliability remains limited. The dominant failure pattern is not complete incoherence; rather, the system often produces partially relevant statements with repetitive phrasing, weak semantic control, and unstable decoding behavior. For example, recent outputs such as `earch Trooper orememorymemorymemory` and `earch activity activity activity` show that the generator can collapse into repeated fragments or token loops instead of producing a stable radiology impression.

From a research perspective, these failures are expected under a setting that emphasizes CPU-safe execution, frozen backbones, limited fine-tuning, and a small effective training subset. The model has learned some radiology vocabulary, but the mapping from visual evidence to clinically grounded language remains underconstrained. Attention heatmaps and cross-attention visualizations are useful for inspection, but they do not by themselves guarantee faithful explanation or diagnosis-level correctness.

## Error Categories

| Category                  | Typical Manifestation                          | Example                                                                             |
| ------------------------- | ---------------------------------------------- | ----------------------------------------------------------------------------------- |
| Anatomical Confusion      | Left/right or spatial localization errors      | “left lower lobe opacity” when the evidence suggests a diffuse bilateral pattern    |
| Severity Mismatch         | Under- or overestimation of disease burden     | describing “mild cardiomegaly” when the intended finding is moderate enlargement    |
| Hallucination Errors      | Fabricated findings not supported by the image | introducing pleural effusion, focal consolidation, or pneumothorax without evidence |
| Repetition / Degeneration | Token loops and unstable beam outputs          | `activity activity activity`, `memorymemorymemory`                                  |

### Anatomical Confusion

Anatomical confusion occurs when the model identifies the correct general pathology but assigns it to the wrong location or laterality. This is especially plausible in chest radiography, where patch-level ambiguity and low-contrast boundaries can obscure fine localization. The current model can attend to coarse structures, but the frozen image backbone and limited supervision reduce its capacity to distinguish subtle left-versus-right asymmetries, apical-versus-basal distribution, or mediastinal versus parenchymal abnormalities.

### Severity Mismatch

Severity mismatch is one of the most clinically important failures. The model may downgrade a more significant abnormality to a mild descriptor or, conversely, overstate the seriousness of a subtle finding. In generated reports this appears as stable vocabulary with incorrect intensity, such as describing “mild cardiomegaly” or “minimal atelectasis” even when the learned target distribution may contain stronger language. This makes the report superficially plausible while remaining clinically inaccurate.

### Hallucination Errors

Hallucinations arise when the decoder produces findings that are not supported by the visual evidence. A common failure mode is the insertion of unsupported phrases such as pleural effusion, focal consolidation, or opacity descriptions that are absent from the image. Because the decoder is language-stronger than the visual grounding module, it can emit high-probability clinical phrases even when the cross-attention signal is weak or ambiguous.

### Repetition and Decoding Degeneration

The lightweight generation setup still exhibits unstable beam behavior and repetition collapse. The observed outputs demonstrate phrase recycling and token duplication, which indicate that language-model priors dominate the decoding process when visual grounding is insufficient. This can also be amplified by limited vocabulary adaptation, where the decoder relies on a narrow subset of common tokens and fails to explore semantically richer alternatives.

## Root Causes

Several architectural and data-related constraints explain the above errors:

- Limited training epochs reduce exposure to rare findings and long-tail phrasing patterns.
- Lightweight fine-tuning restricts the model’s capacity to specialize for radiology language.
- Frozen backbone constraints prevent end-to-end adaptation of visual features to the report-generation objective.
- The small dataset subset limits coverage of clinically diverse image-report pairs.
- Limited medical-domain adaptation means the language model does not fully learn report style, radiographic reasoning, and evidence-to-text alignment.
- Decoding is still vulnerable to repetition loops when the next-token distribution becomes overconfident.
- Attention supervision is indirect, so cross-attention may correlate with plausible regions without guaranteeing causal faithfulness.

## Proposed Improvements

The most important direction is to increase the fidelity of visual grounding and expand the statistical support available to the model.

- Train on a larger radiology corpus to improve rare-finding coverage and clinical diversity.
- Replace generic components with domain-specific pretrained models, especially for the language decoder and tokenizer.
- Train longer with stronger regularization and more stable optimization schedules.
- Use improved decoding strategies such as diverse beam search, constrained decoding, and repetition penalties.
- Incorporate reinforcement learning or preference optimization using clinically preferred report outputs.
- Adapt the tokenizer to medical phrases, abbreviations, and report templates so the model represents domain-specific language more naturally.
- Add better attention supervision, including alignment objectives that encourage evidence-grounded region focus.
- Scale to larger GPU training so that the full model can be fine-tuned more effectively instead of remaining heavily frozen.
- Explore mixed teacher forcing to stabilize early generation while preserving autoregressive realism.
- Use curriculum learning, beginning with easier report templates and gradually introducing more complex abnormal cases.

## Future Work

Future work should evaluate whether attention maps actually improve interpretability and not merely visual appeal. Cross-attention visualization is helpful for debugging, but it should be treated as a diagnostic aid rather than proof of explanation. A stronger study would compare attention heatmaps against radiologist annotations, report-level evidence spans, and downstream clinical metrics. Additional work should also quantify how decoding settings affect hallucination rate, repetition frequency, and severity calibration.

Potential next experiments include:

- Comparing frozen versus partially unfrozen visual encoders.
- Evaluating domain-specific tokenizers and medical-language pretraining.
- Measuring whether stronger attention regularization reduces hallucinations.
- Testing whether larger beam sizes improve report completeness or increase repetition.
- Assessing calibration of severity terms against expert reference reports.

## Conclusion

The current multimodal report generator demonstrates the correct overall system shape, but its outputs remain limited by data scarcity, frozen-feature constraints, and language-model degeneration. The observed failure modes are clinically meaningful: anatomical confusion, severity mismatch, hallucinated findings, and repetitive generation all reduce trustworthiness. Explainable AI tools such as attention heatmaps and cross-attention overlays are useful for analysis, but they do not resolve the underlying faithfulness problem. Substantial gains will likely require larger-scale domain training, stronger grounding objectives, and better decoding and evaluation protocols.
