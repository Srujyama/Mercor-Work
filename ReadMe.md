# Mercor Work Summary

This repository documents the work completed across evaluation systems, rubric transformations, model scoring pipelines, analytical workflows, and internal infrastructure updates at Mercor. All work is grouped by domain for clarity.

---

## Lucius Projects

### 1. Criterion Field Transformation: Combine `criterion_type` and `source`

**Objective:** Improve rubric formatting and remove redundant fields.

**Summary:**

* Extracted styling and formatting labels from the `sources` field (e.g., `(answer_first_general)`).
* Appended each label to the corresponding value in the `criterion_type` array.
* Updated each criterion to match the new expected structure:

  * Old:

    ```
    "criterion_type": ["Information Architecture"],
    "sources": "(answer_first_general)"
    ```
  * New:

    ```
    "criterion_type": ["Information Architecture (answer_first_general)"],
    "sources": ""
    ```
* Implemented this across all relevant rubric entries to support next-day rubric revisions.

---

### 2. AI Studio: Gemini 2.5 Pro Scoring and Loss Analysis Update

**Objective:** Replace chat-interface generations with API-driven outputs for improved consistency and reproducibility.

**Summary:**

* Generated six model outputs per prompt using the Gemini 2.5 Pro API:

  * Three generations with Search enabled.
  * Three generations with Search disabled.
* Re-scored all six outputs using the existing Lucius rubric sets.
* Updated the EDU/HLE loss analysis with:

  * New failure patterns
  * Updated misalignment metrics
  * Revised failure contribution distribution per dimension
* Integrated results into the working loss analysis document.

---

### 3. Continued Loss Analysis Development

**Objective:** Expand the coverage, accuracy, and interpretability of the Lucius evaluation pipeline.

**Summary:**

* Calculated misalignment percentages across multiple task categories.
* Built aggregated scoring tables for:

  * Image Input
  * Image Output
  * Text-Only
  * IA Rubrics
* Extracted weighted failures and dimension-level contributions.
* Updated scripts for Airtable ingestion, scoring, and metric computation.

---

### 4. Plato: Backtesting Model Behavior

**Objective:** Identify a lightweight model whose failure pattern approximates GPT-5-DR for early-stage testing.

**Summary:**

* Ran backtests on multiple models using Plato rubrics.
* Compared failure types, frequencies, and alignment to GPT-5-DR failures.
* Identified viable candidates for quick-turnaround testing before DR evaluation.
* Provided recommendations on which model best matches DR behavior with and without search.

---

## 11/11 Analysis and Visualization

### Grouped Bar Graphs for Image Output, Image Input, Text-Only, and IA Rubrics

**Objective:** Produce visualizations consistent with the style of existing loss analysis charts.

**Summary:**

* Generated grouped bar charts comparing Gemini vs GPT performance.
* Calculated means and 95% confidence intervals for each task group.
* Added:

  * Clean numeric labels
  * Consistent visual formatting
  * Legend, axes, and CI annotations
* Ensured the style matched prior loss analysis graphics for seamless integration.

---

## Additional Mercor Work

### 1. APEX: Expose Fields in Finance Team Lead Control Room

**Objective:** Surface previously hidden metadata to enable improved workflow visibility.

**Summary:**

* Added the following fields to the Finance Team Lead Control Room (Cached Task Versions):

  * Tool SoT Outputs
  * Data Explorer Path
* Updated UI and data-layer visibility without disrupting existing processes.

---

### 2. LLM Call Host Documentation

**Objective:** Improve onboarding and maintainability of the LLM Call Host codebase.

**Summary:**

* Authored an architectural document covering:

  * Call routing and execution flow
  * Provider abstraction layers
  * Batching and concurrency logic
  * Error handling and observability integration
* Added inline comments that clarify design decisions and enhance long-term maintainability.


If you want this customized further, such as adding a table of contents, adjusting tone for recruiting, rewriting for performance review materials, or converting into bullet-point resumes, I can generate those formats as well.
