
# Power of Prompt Engineering

## Overview

This repository explores the use of Mistral language models through the Mistral API for multilingual banking intent classification.

The primary objective is to determine whether a banking query requests **general banking information (FAQ)** or **customer-specific banking information (Customer)**. The experiments investigate how prompt design influences classification accuracy across English and Arabic queries.

The project follows a reproducible experimental workflow: prepare a labeled dataset, construct classification prompts, query a Mistral model through the API, record predictions, and evaluate the results against ground-truth labels.

## Research Objectives

- Evaluate Mistral language models on multilingual banking intent classification.
- Investigate the effect of prompt engineering on classification performance.
- Assess model behavior on both English and Arabic banking queries.
- Improve the distinction between general banking questions and questions requiring customer-specific information.
- Record predictions in CSV files to support quantitative analysis and error inspection.
- Establish a reproducible baseline for subsequent prompt and model comparisons.

## Classification Task

The task is formulated as a binary classification problem.

| Label | Definition | Example |
|---|---|---|
| `faq` | General banking information that does not require access to a specific customer's account or transaction history. | What types of accounts do you offer? |
| `customer` | A request concerning the user's own account, balance, card activity, transactions, payments, or other personal banking records. | Was the annual card maintenance fee charged to me? |

### Important Classification Principle

Classification depends on the **intent of the question**, not simply the presence of banking terminology or a personal-sounding phrase.

For example:

- "What are the requirements for opening a savings account?" → `faq`
- "Was money deducted from my account?" → `customer`
- "ما هي خطوات فتح حساب توفير؟" → `faq`
- "هل تم خصم رسوم من حسابي؟" → `customer`

The classifier should not assume that every question containing words such as "account," "card," or "fee" requires customer data. It must determine whether the user is requesting general information or information about their own banking activity.

## Dataset

The supplied experimental results use a labeled dataset containing 100 questions.

| Property | Description |
|---|---|
| Number of questions | 100 |
| Languages | English and Arabic |
| English questions | 50 |
| Arabic questions | 50 |
| Classification labels | `faq`, `customer` |
| Ground-truth distribution | 50 FAQ, 50 Customer |
| Input fields | `id`, `question`, `language`, `route` |

The `route` column contains the reference label used to evaluate each prediction.

### Dataset Format

The dataset is represented as a CSV file with the following columns:

- `id`: Unique identifier for each example.
- `question`: The banking query to classify.
- `language`: The language associated with the query.
- `route`: The ground-truth classification label.

Example:

```csv
id,question,language,route
1,What types of accounts do you offer?,english,faq
2,Was the annual card maintenance fee charged to me?,english,customer
3,ما هي خطوات فتح حساب توفير؟,arabic,faq
```

The examples above illustrate the expected format.

## Experimental Methodology

The experimental workflow consists of the following stages.

### 1. Dataset Preparation

Load the labeled dataset and identify the question, language, and ground-truth label for each example.

### 2. Prompt Construction

Define the classification instructions, label definitions, and any additional prompt-engineering techniques being evaluated.

Prompt versions should be kept separate so that changes in classification performance can be associated with the corresponding prompt configuration.

### 3. API Inference

Send each question to the selected Mistral model using the Mistral API.

The model is instructed to return one of the two permitted labels:

- `faq`
- `customer`

The classification response should contain the label only, without explanations or additional text.

### 4. Prediction Recording

Save the model's predictions alongside the original question and reference label.

This preserves the information needed to investigate individual errors and reproduce the evaluation.

### 5. Evaluation

Compare the predicted labels with the ground-truth labels and calculate the relevant performance metrics.

### 6. Experimental Comparison

Compare results across prompt versions while keeping the dataset and evaluation procedure consistent.

## Experimental Results

The following results were calculated from the supplied `results_p4.csv` and `results_p5.csv` files by comparing the prediction column against the `route` column.

| Metric | P4 | P5 |
|---|---:|---:|
| Number of examples | 100 | 100 |
| Correct predictions | 98 | 99 |
| Incorrect predictions | 2 | 1 |
| Accuracy | 98.00% | 99.00% |
| English examples | 50 | 50 |
| Arabic examples | 50 | 50 |

### Interpretation

P4 achieved 98% accuracy, while P5 achieved 99% accuracy on the supplied dataset.

P5 therefore improved overall accuracy by one percentage point, reducing the number of incorrect predictions from two to one.

These results indicate that the P5 configuration performed better on this particular evaluation set. However, they do not establish statistical significance or guarantee that the same improvement will generalize to unseen banking queries.

A more complete evaluation should also examine precision, recall, F1-score, per-language performance, and the individual misclassified examples.

**Important:** P4 and P5 are treated here as experimental configuration names. Their exact prompt differences should be documented in the experiment scripts or accompanying research notes.

## Evaluation Metrics

### Accuracy

Accuracy measures the proportion of correctly classified questions.

\[
\text{Accuracy} =
\frac{\text{Correct Predictions}}{\text{Total Predictions}}
\]

### Precision

Precision measures how many predictions assigned to a particular class are correct.

\[
\text{Precision} =
\frac{TP}{TP+FP}
\]

### Recall

Recall measures how many examples belonging to a particular class are correctly identified.

\[
\text{Recall} =
\frac{TP}{TP+FN}
\]

### F1-Score

The F1-score is the harmonic mean of precision and recall.

\[
F1 = 2 \times
\frac{\text{Precision}\times\text{Recall}}
{\text{Precision}+\text{Recall}}
\]

For this binary classification task, per-class precision, recall, and F1-score are useful for detecting whether the model favors `faq` over `customer`, or vice versa.

## Technology Stack

- **Python 3.12** — Experiment implementation.
- **Mistral AI API** — Language model inference.
- **Mistral Python SDK** — API integration.
- **Pandas** — Dataset processing and CSV result management.
- **Python standard library** — Environment variable access, timing, and experiment utilities.

## Installation and Setup

### Prerequisites

- Python 3.12 or a compatible Python version.
- A Mistral API account.
- An API key with access to the model selected in the experiment.
- Internet access for API requests.

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_DIRECTORY>
```

Replace the placeholders with the actual repository URL and directory name.

### 2. Create a Virtual Environment

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks environment activation, use the Python executable inside `.venv` directly or follow your local Python environment policy.


## Results and Reproducibility

The repository includes two supplied result files:

- `results_p4.csv` — Predictions from experimental configuration P4.
- `results_p5.csv` — Predictions from experimental configuration P5.

Each result file contains the original example information, its ground-truth route, and the prediction generated by the corresponding experiment.

These files make it possible to compare individual predictions, identify classification errors, and calculate evaluation metrics without repeating API inference.

For reproducible experiments, record:

- The model identifier.
- The exact system and user prompts.
- The dataset version.
- The inference parameters.
- The execution date.
- The evaluation script and metric definitions.
- API errors, retries, and any examples that could not be evaluated.

Because hosted models can change over time, recording the model identifier and experimental configuration is important when interpreting results.

## Limitations

- The current evaluation contains 100 examples and may not represent the full diversity of real-world banking queries.
- English and Arabic performance should be evaluated separately.
- Arabic dialects, spelling variations, transliteration, and code-switching require dedicated test examples if they fall within the intended scope.
- Overall accuracy alone does not explain the cost of individual classification errors.
- Results from repeated experiments may be affected by model updates, API behavior, and inference settings.
- High accuracy on the current dataset does not guarantee performance on unseen examples or production traffic.


## Security and Responsible Use

This repository evaluates intent classification and is not, by itself, a production banking system.

A predicted `customer` label does not authorize access to customer information. Any downstream banking application must independently enforce authentication, authorization, data-access restrictions, and privacy requirements.

Use synthetic or appropriately anonymized data for experimentation, and do not include real customer credentials, account numbers, or confidential banking records in public datasets or repositories.

## Author

**Reem Hatem Zekry**

AI Engineer | Artificial Intelligence Research

GitHub: [eng-reemhatemzekry](https://github.com/eng-reemhatemzekry)

LinkedIn: [linkedin.com/in/reemhatemzekry](https://linkedin.com/in/reemhatemzekry)
