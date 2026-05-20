# Qualitative Evaluation Rubric

## Evaluation Overview

This rubric provides a structured manual evaluation framework for generated radiology reports produced by the multimodal report generation system. It is intended for expert review of report quality in a research setting, with emphasis on clinical correctness, language quality, and alignment between image evidence and textual findings. The rubric is designed to support consistent human scoring across samples, facilitate error analysis, and complement automated NLP metrics such as BLEU, ROUGE-L, and CIDEr.

Each report should be assessed independently by a qualified reviewer, ideally a radiologist or a clinician with chest imaging expertise. Scores should reflect the overall quality of the generated report as well as the severity of any clinically relevant error.

## Scoring Criteria

| Criterion         | Score Range |
| ----------------- | ----------- |
| Clinical Accuracy | 1–5         |
| Completeness      | 1–5         |
| Fluency           | 1–5         |

Additional evaluation dimensions:

- Anatomical Correctness
- Medical Consistency
- Hallucination Severity
- Report Coherence

### Clinical Accuracy

Clinical Accuracy measures whether the reported findings are supported by the chest X-ray.

- 1 = completely incorrect findings; report is clinically unreliable
- 2 = major errors with multiple unsupported or incorrect findings
- 3 = partially correct, but contains noticeable inaccuracies or omissions
- 4 = mostly correct with only minor clinical imprecision
- 5 = clinically accurate and reliable; findings are well supported by the image

### Completeness

Completeness measures whether the report captures the major expected findings and relevant negative statements.

- 1 = severely incomplete; missing most salient findings
- 2 = incomplete; several important findings omitted
- 3 = moderate coverage; some key findings present, but not comprehensive
- 4 = nearly complete; only minor omissions
- 5 = fully complete and appropriately detailed

### Fluency

Fluency measures grammatical quality, readability, and natural report style.

- 1 = incoherent, repetitive, or ungrammatical text
- 2 = frequent linguistic errors that impede readability
- 3 = understandable but awkward or uneven phrasing
- 4 = fluent and mostly well-formed with minor stylistic issues
- 5 = polished, professional, and radiology-appropriate language

### Anatomical Correctness

This criterion evaluates whether the anatomical site, laterality, and spatial localization are correct.

- 1 = major anatomical errors; incorrect body region or laterality throughout
- 2 = frequent localization mistakes
- 3 = occasional localization ambiguity or patch-level confusion
- 4 = mostly anatomically correct with minor uncertainty
- 5 = precise anatomical localization with consistent laterality and region labeling

### Medical Consistency

This criterion assesses whether the report is internally consistent and clinically coherent across sentences.

- 1 = contradictory statements dominate the report
- 2 = multiple inconsistencies between findings and impression
- 3 = some inconsistencies, but overall interpretation is partially coherent
- 4 = mostly consistent with only minor interpretive mismatch
- 5 = fully consistent and clinically coherent throughout

### Hallucination Severity

This criterion captures the extent of fabricated or unsupported findings.

- 1 = severe hallucination; multiple unsupported abnormalities introduced
- 2 = substantial hallucination with clinically important false findings
- 3 = moderate hallucination; some unsupported details present
- 4 = minimal hallucination; only small unsupported embellishments
- 5 = no hallucination; all findings are image-grounded

### Report Coherence

This criterion measures whether the report follows a logical structure from findings to impression and maintains stable topic flow.

- 1 = highly disorganized or repetitive
- 2 = weak structure and poor progression of ideas
- 3 = partially coherent but uneven in structure
- 4 = coherent and easy to follow
- 5 = highly coherent with clear radiology-style organization

## Score Interpretation

A practical interpretation of the scores is shown below:

| Average Score | Interpretation                                 |
| ------------- | ---------------------------------------------- |
| 1.0–1.9       | Unacceptable; clinically unsafe for use        |
| 2.0–2.9       | Weak; requires substantial revision            |
| 3.0–3.4       | Moderate; partially usable with caution        |
| 3.5–4.4       | Strong; generally acceptable with minor issues |
| 4.5–5.0       | Excellent; high-quality report generation      |

Reviewers should consider both the average score and the distribution of errors. For example, a report with high fluency but low clinical accuracy should not be considered strong overall, because stylistic quality cannot compensate for incorrect medical content.

## Example Evaluation

| Criterion              | Score | Notes                                                                        |
| ---------------------- | ----: | ---------------------------------------------------------------------------- |
| Clinical Accuracy      |     3 | Mild anatomical confusion and incomplete description of the abnormality      |
| Completeness           |     3 | Mentions key findings but omits relevant negative statements                 |
| Fluency                |     4 | Grammatically coherent and easy to read                                      |
| Anatomical Correctness |     2 | Localization is partially ambiguous                                          |
| Medical Consistency    |     3 | Mostly consistent, but the impression is slightly stronger than the findings |
| Hallucination Severity |     3 | One unsupported opacity description appears in the impression                |
| Report Coherence       |     4 | Logical radiology-style flow with minor repetition                           |

**Example qualitative summary:**
The generated report is fluent and structurally plausible, but it demonstrates mild anatomical confusion and a limited ability to align descriptive language with image evidence. The report is acceptable as a draft, yet it requires expert review before clinical interpretation.

## Notes

- Automated metrics are useful for large-scale benchmarking, but they do not fully capture clinical correctness, localization quality, or hallucination severity.
- Radiologist review remains essential for judging whether the generated report is safe, clinically meaningful, and aligned with the image.
- Expert validation is especially important for borderline cases, such as mild cardiomegaly, subtle atelectasis, and low-volume studies where report wording can change the clinical meaning.
- Human raters should be calibrated with shared examples before scoring to improve inter-rater consistency.
- When multiple raters are available, use consensus scoring or adjudication for low-agreement cases.

## Future Improvements for Human Evaluation

Future evaluation protocols should include inter-rater agreement analysis, severity-weighted scoring, and a structured comparison against radiologist-authored reference reports. Additional improvements include:

- defining benchmark cases for common abnormalities and normal studies
- recording rater confidence alongside the score
- separating detection, localization, and severity assessment into distinct sub-scores
- using blinded review to reduce expectation bias
- incorporating error taxonomy labels for anatomical confusion, hallucination, and severity mismatch

A more mature human evaluation framework should combine rubric-based scoring with case-level critique, enabling both aggregate benchmarking and actionable qualitative analysis for model improvement.
