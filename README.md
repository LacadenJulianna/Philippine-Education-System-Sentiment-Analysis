# Philippine Education System Sentiment Analysis

Data Mining midterm group project (SLU, 1st semester 2026–2027).

We collected public posts, comments, and news articles about the Philippine education system from YouTube, Reddit, TikTok, and Philippine news sites. We then:

- built a rule-based sentiment classifier for English, Tagalog, and Taglish text,
- measured how similar or different the records, platforms, and search keywords are,
- found the main discussion topics with topic modeling,
- combined the two to see how people feel about each topic.

## Repository contents

| File | What it is |
|---|---|
| `datasets/` | All CSV data (see [Datasets](#datasets)) |
| `DatasetFixes.ipynb` | One-time fixes to the final 200-record dataset: replaces off-topic records, rebuilds lemmas, verifies the 90/10 split, and lists the records that still need manual labels |
| `lexicon_corrected.ipynb` | Builds the Tagalog/Taglish VADER lexicon and exports the labeling template that became `final_manual_labels.csv`. **Reference only, no need to re-run** |
| `Sentimental_Analysis.ipynb` | Rule-based sentiment analysis: 3 rule versions, 10-fold cross-validation, final test on the unseen 10%, sentiment per topic |
| `SimilaritiesandDissimilarities.ipynb` | TF-IDF vectors with cosine, Jaccard, and Euclidean comparisons across records, platforms, and keywords, plus heatmaps |
| `TopicModeling.ipynb` | LDA (main model), with LSA and BERTopic for comparison; chooses the number of topics using coherence |

## Datasets

All in `datasets/`. Every notebook finds these files by name.

| File | Rows | What it is |
|---|---|---|
| `final_combined_data.csv` | 17,948 | Raw scraped records from all sources, merged and de-duplicated |
| `final_cleaned_combined_data.csv` | 250 | Cleaned and preprocessed records: the 200 used, plus 50 reserve records used as replacements |
| `final_training_data.csv` | 180 | Training set (90%), stratified by record type |
| `final_unseen_data.csv` | 20 | Unseen / holdout set (10%). Used only for the final test |
| `final_manual_labels.csv` | 180 | Manual sentiment labels by `record_id`. `manual_label`: `1` positive, `0` neutral, `-1` negative |

The 200 records are 80 YouTube comments, 55 Reddit comments, 30 articles, 20 TikTok videos, and 15 Reddit posts.

## Pipeline: what to run, in order

```
 (done)  scraping → final_combined_data.csv
 (done)  cleaning + preprocessing → final_cleaned_combined_data.csv
 (done)  90/10 split → final_training_data.csv + final_unseen_data.csv
            │
 Step 1  DatasetFixes.ipynb ─────────► fixed training/unseen files + records_to_label.csv
            │
 Step 2  manual labeling ────────────► add the new labels to final_manual_labels.csv
            │
            ├─► Step 3  TopicModeling.ipynb ──────────────► topic_modeling_outputs/
            ├─► Step 4  SimilaritiesandDissimilarities.ipynb ► similarity CSVs + heatmaps
            │
 Step 5  Sentimental_Analysis.ipynb ─► metrics + sentiment per topic + predictions_all.csv
```

Steps 3 and 4 don't depend on each other. Run Step 5 **after** Step 3 so the sentiment-per-topic results use the group's LDA topics.

### Step 1: `DatasetFixes.ipynb` (run once)
- **Input:** `final_combined_data.csv`, `final_cleaned_combined_data.csv`, `final_training_data.csv`, `final_unseen_data.csv`, `final_manual_labels.csv`
- **What it does:**
  - Replaces 20 off-topic records with on-topic records of the same type, in the same split.
  - Rebuilds the `lemmas` column with WordNet.
  - Counts the raw records and lists the search keywords.
  - Checks that the 90/10 split is stratified.
- **Output** (saved next to the inputs):
  - `final_training_data.csv` and `final_unseen_data.csv`, overwritten with the fixed versions.
  - `records_to_label.csv`: records with no manual label yet.
  - `replaced_records_log.csv`: for the appendix.
  - `raw_record_counts.csv` and `search_keywords_raw.csv`: for the Methods section.
- Re-running it is safe, because records that were already replaced are skipped. **Do not re-split the data.**

### Step 2: Manual labeling
Fill in `manual_label` in `records_to_label.csv` (`1` / `0` / `-1`) and append those rows to `final_manual_labels.csv`. The unseen records must be labeled, or the final test can't be scored.

### Step 3: `TopicModeling.ipynb`
- **Input:** `final_training_data.csv`, plus `final_unseen_data.csv` (optional)
- **Manual step:** after reading the top words, fill in `TOPIC_LABELS` with a short name for each topic.
- **Output:** `topic_modeling_outputs/`, which contains:
  - `topic_assignments_train.csv` and `topic_assignments_holdout.csv`: used by Step 5.
  - `lda_topics.csv`, `lsa_topics.csv`, `bertopic_topics.csv`.
  - `lda_k_selection_summary.csv` and `model_comparison.csv`.

### Step 4: `SimilaritiesandDissimilarities.ipynb`
- **Input:** `final_training_data.csv` (must be exactly 180 records)
- **Output:**
  - Heatmaps.
  - About 25 CSVs: record, platform, and keyword similarity matrices (`*_cosine_*`, `*_jaccard_*`, `*_euclidean_*`), `top20_*_pairs.csv`, and `remarks.csv`.

### Step 5: `Sentimental_Analysis.ipynb`
- **Input:**
  - `final_training_data.csv`, `final_unseen_data.csv`, `final_manual_labels.csv`.
  - `topic_assignments_train.csv` and `topic_assignments_holdout.csv` (optional). It looks for them in `topic_modeling_outputs/` or `datasets/`.
- **What it does:**
  1. Compares three rule versions: v1 small word list → v2 expanded English/Tagalog lexicon, Tagalog affix matching, and text cleaning → v3 adds negation and intensifiers.
  2. Runs 10-fold stratified cross-validation on the labeled training records (accuracy, precision, recall, macro-F1) and picks the best version.
  3. Tests that version once on the unseen 10% (metrics and confusion matrix).
  4. Builds the sentiment-per-topic table and chart.
- **Output:** `predictions_all.csv` (record, dataset, topic, manual label, score, predicted label)
- If the topic files are missing, it falls back to a placeholder NMF topic model. Re-run it after Step 3 for the final results.

## How to run

**Google Colab:** open the notebook, upload the input files from `datasets/` (and `topic_modeling_outputs/` for Step 5) to `/content/`, then run all cells. Outputs are downloaded automatically where a notebook supports it.

**Locally:** run the notebooks from the repository root. They read from `datasets/` and write their outputs to the repository root (or `topic_modeling_outputs/`).

```
pip install pandas numpy scipy scikit-learn matplotlib seaborn nltk "gensim>=4.4.0" bertopic pyLDAvis
```

`DatasetFixes.ipynb` downloads NLTK's WordNet data on first run. If the download fails on your network, run that notebook in Colab.

## Current status (2026-10-10)

- [ ] **Step 1 not yet run on the files in `datasets/`.** The repo copies still contain the off-topic records and the old lemmas.
- [ ] **49 records need manual labels** after Step 1: 29 training records and all 20 unseen records. 27 existing labels belong to records that are no longer in the dataset and are ignored.
- [ ] Topic names (`TOPIC_LABELS`) in `TopicModeling.ipynb` not yet filled in.
- Sentiment results so far (10-fold cross-validation on 163 labeled training records, before Step 1):

  | Version | Accuracy | Macro-F1 |
  |---|---|---|
  | v1_baseline | 0.447 | 0.381 |
  | v2_+lexicon+cleaning | 0.459 | 0.428 |
  | v3_+negation+intensity | 0.478 | 0.453 |

- The paper's raw record count should be **17,948**, the actual number of rows in `final_combined_data.csv`.
