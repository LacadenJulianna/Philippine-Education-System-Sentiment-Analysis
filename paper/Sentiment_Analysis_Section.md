# Sentiment Analysis (Draft Section)

> Draft for the final paper. Numbers come from `Sentimental_Analysis.ipynb` run on the fixed dataset (after `DatasetFixes.ipynb`) with **151 labeled training records**.
> Items marked **[TO UPDATE]** change once the remaining records in `datasets/records_to_label.csv` are labeled and the notebook is re-run.

## 1. Overview

A rule-based (lexicon-based) classifier labeled each record as **positive**, **neutral**, or **negative**. A rule-based approach was chosen because:

- the dataset is small (200 records), which is too little to train a machine-learning classifier;
- the records mix English, Tagalog, and Taglish, which standard English sentiment tools handle poorly;
- every prediction can be explained by the words that triggered it.

The classifier was built in five versions. Each version was scored against the group's manual labels with 10-fold cross-validation on the training set (90%). The best version was then kept for a single final test on the unseen set (10%).

## 2. Data

| Set | Records | With manual label | Used for |
|---|---|---|---|
| Training (90%) | 180 | 151 | 10-fold cross-validation, rule improvement |
| Unseen (10%) | 20 | 0 **[TO UPDATE]** | Final test only, never used for building rules |

The labeled training records are 56 positive, 51 neutral, and 44 negative, so the three classes are fairly balanced. Records with no comment text (articles and Reddit posts that only have a title) were scored using their title, so no records were dropped.

## 3. How the classifier works

Each word in a record is looked up in a positive and a negative word list (English and Tagalog). A positive word adds +1 and a negative word adds −1. The record's **sentiment score** is the sum of these values:

- score > 0 → **positive**
- score < 0 → **negative**
- score = 0 → **neutral**

## 4. Iterative rule improvement

| Version | Rule change | Why |
|---|---|---|
| **v1_baseline** | Small English + Tagalog word list, exact word matching | Starting point |
| **v2_+lexicon+cleaning** | Word list expanded to about 270 words; Tagalog affix matching (*mahirap*, *nahirapan* → *hirap*); removal of emoji-code leftovers; *problem solving* no longer counted as a complaint; ambiguous words *sana* and *mahal* removed | v1 marked most records neutral because it knew too few words, and Tagalog words appear with many prefixes and suffixes |
| **v3_+negation+intensity** | Negation: *not, no, never, hindi, di, wala, walang, ayaw* within 3 words before a sentiment word flips its sign. Intensifiers: *very, sobra, grabe, napaka, talaga* within 3 words multiply it by 1.5 | *"hindi maganda"* and *"not good"* were being counted as positive |
| **v4_+vocabulary** | Added everyday praise words (*gift, award, honor, respect, inspired, dedicated, thankful*), complaint words (*hopeless, sadly, killing, inutil, walanghiya, nakakalungkot*), and Filipino sighs (*haist, hays, hahayy*) | Reviewing v3's errors on the training set showed these common words were missing, so many opinions were scored as neutral |
| **v5_+contrast** | Contrast words (*but, pero, kaso, however, although, subalit, ngunit*) halve the score of everything before them. Hyphenated words are kept whole, so *never-ending* is not read as the negation *never* | In mixed comments, the opinion usually follows the contrast word, e.g. *"maganda yung idea pero walang pondo"* |

v4 and v5 were designed only from errors in the **training** set. The words added are general words, not phrases copied from individual records, and the unseen set was never looked at.

### Results per version (10-fold cross-validation, n = 151)

| Version | Accuracy | Precision (macro) | Recall (macro) | Macro-F1 | Predictions changed vs previous | Agreement with previous (κ) |
|---|---|---|---|---|---|---|
| v1_baseline | 0.490 ± 0.123 | 0.454 ± 0.202 | 0.471 ± 0.123 | 0.426 ± 0.140 | – | – |
| v2_+lexicon+cleaning | 0.503 ± 0.128 | 0.488 ± 0.178 | 0.489 ± 0.125 | 0.468 ± 0.140 | 25.5% | 0.574 |
| v3_+negation+intensity | 0.523 ± 0.140 | 0.521 ± 0.180 | 0.510 ± 0.135 | 0.491 ± 0.151 | 5.0% | 0.921 |
| v4_+vocabulary | 0.570 ± 0.109 | 0.549 ± 0.162 | 0.553 ± 0.112 | 0.537 ± 0.131 | 7.0% | 0.890 |
| **v5_+contrast** | **0.589 ± 0.104** | **0.570 ± 0.162** | **0.576 ± 0.109** | **0.556 ± 0.129** | 3.5% | 0.945 |

Values are mean ± standard deviation across the 10 folds. "Predictions changed" and κ (Cohen's kappa) compare each version with the one before it, over all 200 records.

**Predicted class mix per version (training set, n = 180):**

| Version | Positive | Neutral | Negative |
|---|---|---|---|
| v1_baseline | 34.4% | 57.8% | 7.8% |
| v2_+lexicon+cleaning | 38.3% | 42.8% | 18.9% |
| v3_+negation+intensity | 36.1% | 43.9% | 20.0% |
| v4_+vocabulary | 39.4% | 38.9% | 21.7% |
| v5_+contrast | 40.0% | 38.3% | 21.7% |

**Remarks:**

- Every version scored higher than the one before it. Accuracy rose from 0.490 to 0.589 and macro-F1 from 0.426 to 0.556.
- The biggest change in *behavior* came from v2: it changed 25.5% of predictions (κ = 0.574). v1 knew so few words that it marked 57.8% of records as neutral and only 7.8% as negative, even though 29% of the labeled records are negative.
- The biggest change in *accuracy* came from v4 (+0.047). v1 to v3 had fixed how words are scored, but many common opinion words were still missing from the list.
- v5's contrast rule changed only 3.5% of predictions, but it gave the best and most stable result: the lowest SD across folds (±0.104).
- **v5_+contrast** had the highest mean macro-F1 and was selected for the final test.

## 5. Final model: v5_+contrast

### Per-fold results (10-fold cross-validation)

| Fold | n | Accuracy | Precision | Recall | Macro-F1 |
|---|---|---|---|---|---|
| 1 | 16 | 0.625 | 0.722 | 0.611 | 0.635 |
| 2 | 15 | 0.600 | 0.572 | 0.572 | 0.568 |
| 3 | 15 | 0.533 | 0.375 | 0.478 | 0.420 |
| 4 | 15 | 0.600 | 0.600 | 0.589 | 0.591 |
| 5 | 15 | 0.600 | 0.579 | 0.622 | 0.560 |
| 6 | 15 | 0.600 | 0.550 | 0.550 | 0.530 |
| 7 | 15 | 0.600 | 0.578 | 0.600 | 0.585 |
| 8 | 15 | 0.733 | 0.738 | 0.733 | 0.722 |
| 9 | 15 | 0.333 | 0.236 | 0.333 | 0.275 |
| 10 | 15 | 0.667 | 0.750 | 0.667 | 0.677 |
| **Mean ± SD** | 151 | **0.589 ± 0.104** | **0.570 ± 0.162** | **0.576 ± 0.109** | **0.556 ± 0.129** |

### Per-class results (all folds pooled)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Positive | 0.64 | 0.70 | 0.67 | 56 |
| Neutral | 0.52 | 0.57 | 0.54 | 51 |
| Negative | 0.62 | 0.48 | 0.54 | 44 |
| **Accuracy** | | | **0.59** | 151 |
| Macro avg | 0.59 | 0.58 | 0.58 | 151 |

### Confusion matrix (all folds pooled)

| | Predicted positive | Predicted neutral | Predicted negative |
|---|---|---|---|
| **Actual positive** | **39** | 13 | 4 |
| **Actual neutral** | 13 | **29** | 9 |
| **Actual negative** | 9 | 14 | **21** |

### Accuracy by platform and language (labeled training records)

| Platform | Accuracy | n | | Language | Accuracy | n |
|---|---|---|---|---|---|---|
| YouTube | 0.677 | 62 | | English | 0.633 | 79 |
| News articles | 0.591 | 22 | | Tagalog | 0.588 | 17 |
| Reddit | 0.526 | 57 | | English/Tagalog (Taglish) | 0.527 | 55 |
| TikTok | 0.400 | 10 | | | | |

**Remarks:**

- **Positive** was the easiest class (F1 0.67). Praise is usually direct: *salamat po*, *great teacher*, *galing*.
- **Negative** had the lowest recall (0.48). Of the 44 negative records, 14 were predicted neutral and 9 positive. Many complaints were expressed through sarcasm, rhetorical questions, or describing a situation without opinion words. For example, *"saan napupunta ang pondo ng gobyerno?"* is clearly negative but contains no negative word.
- **Neutral** was confused in both directions. News headlines and questions that mention words like *quality*, *free*, or *support* got a positive score even though they state facts.
- YouTube comments, which are short and direct, scored best. Reddit, with long arguments full of mixed opinions, and TikTok captions, mostly hashtags and promotions, scored worst. Taglish records (0.527) were harder than English-only ones (0.633), because code-switching splits an opinion across two word lists.
- Fold 9 (0.333) shows how much the results depend on which 15 records land in a fold. With only 151 labeled records, the mean ± SD across folds is more meaningful than any single number.

## 6. Final test on the unseen set (10%)  **[TO UPDATE]**

> The 20 unseen records don't have manual labels yet, so the final test can't be scored. Once they are labeled and added to `final_manual_labels.csv`, re-running `Sentimental_Analysis.ipynb` fills in this table and the confusion matrix (Step 6).

| | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| v5_+contrast on unseen data | **[TO UPDATE]** | **[TO UPDATE]** | **[TO UPDATE]** | **[TO UPDATE]** |

Until then, the predicted sentiment mix of the two sets can be compared:

| | Positive | Neutral | Negative |
|---|---|---|---|
| Training (n = 180) | 40.0% | 38.3% | 21.7% |
| Unseen (n = 20) | 45.0% | 45.0% | 10.0% |

A chi-square test found no significant difference between the two mixes (p = 0.470). The model behaves similarly on data it has not seen. With only 20 unseen records, a difference of one or two records shifts the shares by 5–10 points.

## 7. Sentiment per topic

Each record's predicted sentiment was combined with its LDA topic from `TopicModeling.ipynb` (k = 5). Records that the topic model skipped (too few words after preprocessing) are listed as *Unassigned*.

| Topic | n | Positive | Neutral | Negative |
|---|---|---|---|---|
| PISA results & language of instruction | 26 | 42.3% | 30.8% | 26.9% |
| Classroom learning, grades & DepEd programs | 34 | 47.1% | 38.2% | 14.7% |
| Critical thinking & global competitiveness | 37 | 40.5% | 40.5% | 18.9% |
| Public school quality & teaching problems | 46 | 45.7% | 21.7% | **32.6%** |
| Curriculum relevance & work readiness | 31 | 35.5% | 41.9% | 22.6% |
| Unassigned (short comments) | 26 | 26.9% | 73.1% | 0.0% |

**Remarks:**

- **Public school quality & teaching problems** was the most negative topic (32.6% negative) and also the most divided: it had the fewest neutral records (21.7%). Comments here compare public and private schools, criticize mass promotion, and describe a lack of teacher training and classrooms. Both the complaints and the defenses of teachers are strongly worded.
- **PISA results & language of instruction** was the second most negative (26.9%). It reflects concern about the country's international rankings and about English versus Tagalog as the language of instruction.
- **Classroom learning, grades & DepEd programs** was the most positive and least negative topic (47.1% / 14.7%). It includes DepEd announcements and appreciation for teachers and programs.
- **Curriculum relevance & work readiness** was the most neutral (41.9%). Many records here describe experiences or suggest subjects to add, without a clear opinion.
- *Unassigned* records are mostly short replies like *thank you po*. The classifier rarely finds sentiment words in them.
- **Check against the manual labels:** among labeled training records, *Public school quality & teaching problems* (40.6% negative) and *PISA results & language of instruction* (41.2% negative) were also the most negative by human judgment. The ranking of topics holds even though the classifier under-detects negative sentiment, which matches its low negative recall.

## 8. Limitations

- **Accuracy is moderate (0.59).** Lexicon rules can't understand sarcasm, rhetorical questions, or complaints made without opinion words.
- **The cross-validation results for v4 and v5 are optimistic.** Their rules were designed by reading training errors, so they are somewhat fitted to the training records. The unseen-set test **[TO UPDATE]** is the fair measure of how well the final rules work on new data.
- **Small labeled set.** With 151 labeled training records and 20 unseen records, results vary noticeably between folds (SD ±0.10).
- **Words with more than one meaning.** For example, *mahal* means both "expensive" and "dear", and *free* can describe tuition or freedom. Some were removed from the word lists, and others remain a source of error.
- **Manual labels came from the group.** No inter-annotator agreement was measured.

## 9. Script / Output / Remarks table

| Step | Script (`Sentimental_Analysis.ipynb`) | Output | Remarks |
|---|---|---|---|
| 1 | `load()`: reads `final_training_data.csv` and `final_unseen_data.csv`, uses the title when `text` is empty, merges `final_manual_labels.csv` by `record_id` | 180 training records (151 labeled), 20 unseen records | Labels mapped from 1 / 0 / −1 to positive / neutral / negative |
| 2 | `RuleBasedSentiment` class with `VERSIONS` v1–v5 | Five classifier configurations | Each version adds one group of rules (Section 4) |
| 3 | Predict every record with every version | Class mix per version (Section 4 table) | v1 under-predicts negative (7.8%) |
| 4 | `cv_scores()`: `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)` | Accuracy, precision, recall, macro-F1 per version; per-fold table; pooled confusion matrix | v5 selected (macro-F1 0.556) |
| 5 | Compare consecutive versions: % changed, `cohen_kappa_score` | Version change table | v2 changed the most predictions (25.5%) |
| 6 | Run v5 once on the unseen set; `classification_report`, `confusion_matrix`, chi-square test | Unseen metrics **[TO UPDATE]**; class mix comparison (p = 0.470) | Unseen set never used to build rules |
| 7 | Merge `topic_assignments_train.csv` and `topic_assignments_holdout.csv` (`lda_topic_label`); `pd.crosstab(topic, predicted_label)` | Sentiment share per topic and stacked bar chart | Public school quality is the most negative topic |
| 8 | Save `predictions_all.csv` | One row per record: dataset, topic, manual label, score, predicted label | Input for the Discussion of Results |
