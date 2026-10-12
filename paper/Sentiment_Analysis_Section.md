# Sentiment Analysis (Draft Section)

> Draft for the final paper. Numbers come from `Sentimental_Analysis.ipynb` run on the fixed dataset (after `DatasetFixes.ipynb`) with all **200 records manually labeled** (180 training, 20 unseen).

## 1. Overview

A rule-based (lexicon-based) classifier labeled each record as **positive**, **neutral**, or **negative**. A rule-based approach was chosen because:

- the dataset is small (200 records), which is too little to train a machine-learning classifier;
- the records mix English, Tagalog, and Taglish, which standard English sentiment tools handle poorly;
- every prediction can be explained by the words that triggered it.

The classifier was built in five versions. Each version was scored against the group's manual labels with 10-fold cross-validation on the training set (90%). The best version was then kept for a single final test on the unseen set (10%).

## 2. Data

| Set | Records | With manual label | Used for |
|---|---|---|---|
| Training (90%) | 180 | 180 | 10-fold cross-validation, rule improvement |
| Unseen (10%) | 20 | 20 | Final test only, never used for building rules |

The labeled training records are 62 positive, 64 neutral, and 54 negative, so the three classes are fairly balanced. The unseen records are 7 positive, 4 neutral, and 9 negative. Records with no comment text (articles and Reddit posts that only have a title) were scored using their title, so no records were dropped.

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

### Results per version (10-fold cross-validation, n = 180)

| Version | Accuracy | Precision (macro) | Recall (macro) | Macro-F1 | Predictions changed vs previous | Agreement with previous (κ) |
|---|---|---|---|---|---|---|
| v1_baseline | 0.483 ± 0.098 | 0.483 ± 0.182 | 0.469 ± 0.092 | 0.429 ± 0.115 | – | – |
| v2_+lexicon+cleaning | 0.522 ± 0.091 | 0.531 ± 0.118 | 0.512 ± 0.094 | 0.499 ± 0.097 | 25.5% | 0.574 |
| v3_+negation+intensity | 0.539 ± 0.087 | 0.561 ± 0.133 | 0.529 ± 0.090 | 0.517 ± 0.097 | 5.0% | 0.921 |
| v4_+vocabulary | 0.578 ± 0.054 | 0.608 ± 0.076 | 0.570 ± 0.055 | 0.561 ± 0.058 | 7.0% | 0.890 |
| **v5_+contrast** | **0.594 ± 0.074** | **0.631 ± 0.101** | **0.588 ± 0.076** | **0.581 ± 0.080** | 3.5% | 0.945 |

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

- Every version scored higher than the one before it. Accuracy rose from 0.483 to 0.594 and macro-F1 from 0.429 to 0.581.
- The biggest change came from v2. It changed 25.5% of predictions (κ = 0.574) and gave the largest macro-F1 gain (+0.070). v1 knew so few words that it marked 57.8% of records as neutral and only 7.8% as negative, even though 30% of the labeled records are negative.
- The second-largest gain came from v4 (+0.044 macro-F1, +0.039 accuracy). v1 to v3 had fixed how words are scored, but many common opinion words were still missing from the list. v4 was also the most stable version across folds (SD ±0.054).
- v5's contrast rule changed only 3.5% of predictions but still added +0.020 macro-F1, giving the best mean on every metric.
- **v5_+contrast** had the highest mean macro-F1 and was selected for the final test.

## 5. Final model: v5_+contrast

### Per-fold results (10-fold cross-validation)

| Fold | n | Accuracy | Precision | Recall | Macro-F1 |
|---|---|---|---|---|---|
| 1 | 18 | 0.611 | 0.722 | 0.575 | 0.552 |
| 2 | 18 | 0.556 | 0.556 | 0.546 | 0.549 |
| 3 | 18 | 0.667 | 0.719 | 0.671 | 0.686 |
| 4 | 18 | 0.667 | 0.756 | 0.657 | 0.646 |
| 5 | 18 | 0.500 | 0.531 | 0.490 | 0.500 |
| 6 | 18 | 0.611 | 0.605 | 0.605 | 0.605 |
| 7 | 18 | 0.556 | 0.581 | 0.556 | 0.554 |
| 8 | 18 | 0.500 | 0.540 | 0.500 | 0.478 |
| 9 | 18 | 0.722 | 0.780 | 0.722 | 0.717 |
| 10 | 18 | 0.556 | 0.522 | 0.556 | 0.521 |
| **Mean ± SD** | 180 | **0.594 ± 0.074** | **0.631 ± 0.101** | **0.588 ± 0.076** | **0.581 ± 0.080** |

### Per-class results (all folds pooled)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Positive | 0.60 | 0.69 | 0.64 | 62 |
| Neutral | 0.57 | 0.61 | 0.59 | 64 |
| Negative | 0.64 | 0.46 | 0.54 | 54 |
| **Accuracy** | | | **0.59** | 180 |
| Macro avg | 0.60 | 0.59 | 0.59 | 180 |

### Confusion matrix (all folds pooled)

| | Predicted positive | Predicted neutral | Predicted negative |
|---|---|---|---|
| **Actual positive** | **43** | 15 | 4 |
| **Actual neutral** | 15 | **39** | 10 |
| **Actual negative** | 14 | 15 | **25** |

### Accuracy by platform and language (labeled training records)

| Platform | Accuracy | n | | Language | Accuracy | n |
|---|---|---|---|---|---|---|
| YouTube | 0.681 | 72 | | English | 0.663 | 92 |
| Reddit | 0.571 | 63 | | Tagalog | 0.545 | 22 |
| News articles | 0.519 | 27 | | English/Tagalog (Taglish) | 0.515 | 66 |
| TikTok | 0.444 | 18 | | | | |

**Remarks:**

- **Positive** was the easiest class (F1 0.64). Praise is usually direct: *salamat po*, *great teacher*, *galing*.
- **Negative** had the lowest recall (0.46). Of the 54 negative records, 15 were predicted neutral and 14 positive. Many complaints were expressed through sarcasm, rhetorical questions, or describing a situation without opinion words. For example, *"saan napupunta ang pondo ng gobyerno?"* is clearly negative but contains no negative word.
- **Neutral** was confused in both directions. News headlines and questions that mention words like *quality*, *free*, or *support* got a positive score even though they state facts.
- YouTube comments, which are short and direct, scored best. TikTok captions, mostly hashtags and promotions, scored worst, followed by news articles, whose factual wording rarely matches the opinion word lists. Taglish (0.515) and Tagalog (0.545) records were harder than English-only ones (0.663), because the Tagalog word list is smaller and code-switching splits an opinion across two word lists.
- Fold accuracy ranged from 0.500 (folds 5 and 8) to 0.722 (fold 9). With 18 records per fold, one or two records move a fold's accuracy by 5–11 points, so the mean ± SD across folds is more meaningful than any single number.

## 6. Final test on the unseen set (10%)

The selected version, v5_+contrast, was run **once** on the 20 unseen records.

| | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| v5_+contrast, cross-validation (n = 180) | 0.594 | 0.631 | 0.588 | 0.581 |
| **v5_+contrast, unseen data (n = 20)** | **0.450** | **0.463** | **0.561** | **0.432** |

**Per-class results (unseen):**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Positive | 0.44 | 0.57 | 0.50 | 7 |
| Neutral | 0.44 | 1.00 | 0.62 | 4 |
| Negative | 0.50 | 0.11 | 0.18 | 9 |

**Confusion matrix (unseen):**

| | Predicted positive | Predicted neutral | Predicted negative |
|---|---|---|---|
| **Actual positive** | **4** | 2 | 1 |
| **Actual neutral** | 0 | **4** | 0 |
| **Actual negative** | 5 | 3 | **1** |

**Predicted sentiment mix, training vs unseen:**

| | Positive | Neutral | Negative |
|---|---|---|---|
| Training (n = 180) | 40.0% | 38.3% | 21.7% |
| Unseen (n = 20) | 45.0% | 45.0% | 10.0% |

A chi-square test found no significant difference between the two predicted mixes (p = 0.470).

**Remarks:**

- Unseen accuracy (0.450) is **0.144 lower** than the cross-validation accuracy (0.594). Two reasons explain most of the gap:
  1. **The unseen set has more negative records.** 45% of the unseen records are negative (9 of 20) against 30% of the training records, and negative is the class the classifier detects worst (cross-validation recall 0.46).
  2. **v4 and v5 were tuned on training errors**, so the cross-validation result for them is optimistic. The unseen test is the fair estimate.
- The classifier found only **1 of the 9 negative records**. The other 8 show the same weaknesses seen in training:
  - complaints with no negative word: *"kala ko ba sa pilipinas bakit sa luzon lang..."*, *"admu's admissions remain highly competitive..."*;
  - positive words used inside a complaint or a question: *"i can't imagine myself applying scholarships here... **to be honest**"*, *"what do you **recommend**... the **best** possible academic education"*;
  - a negative event described with kind words: a school-shooting post that says *"please stay **safe**"* was scored positive.
- All 4 neutral records were classified correctly (recall 1.00), and positive was 4 of 7.
- With only 20 records, each record is worth 5 accuracy points, so this result has a wide margin of error. It still shows that the rules mostly miss indirect negative opinions, which is the main limitation of a lexicon-based approach.

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
- **Check against the manual labels (all 200 records):** by human judgment, *PISA results & language of instruction* (46.2% negative) and *Public school quality & teaching problems* (45.7% negative) were also the two most negative topics, and *Classroom learning, grades & DepEd programs* was the least negative (17.6%). The classifier puts the top two in the opposite order, but they are almost tied in the manual labels. The overall ranking holds, even though the classifier under-detects negative sentiment in every topic, which matches its low negative recall.

## 8. Limitations

- **Accuracy is moderate**: 0.59 in cross-validation and 0.45 on the unseen set. Lexicon rules can't understand sarcasm, rhetorical questions, or complaints made without opinion words.
- **The cross-validation results for v4 and v5 are optimistic.** Their rules were designed by reading training errors, so they are somewhat fitted to the training records. The lower unseen result (0.450) is the fairer measure of how well the final rules work on new data.
- **Small labeled set.** With 180 labeled training records and 20 unseen records, results vary noticeably between folds (SD ±0.07), and each unseen record is worth 5 accuracy points.
- **Words with more than one meaning.** For example, *mahal* means both "expensive" and "dear", and *free* can describe tuition or freedom. Some were removed from the word lists, and others remain a source of error.
- **Manual labels came from the group.** No inter-annotator agreement was measured.

## 9. Script / Output / Remarks table

| Step | Script (`Sentimental_Analysis.ipynb`) | Output | Remarks |
|---|---|---|---|
| 1 | `load()`: reads `final_training_data.csv` and `final_unseen_data.csv`, uses the title when `text` is empty, merges `final_manual_labels.csv` by `record_id` | 180 training records and 20 unseen records, all labeled | Labels mapped from 1 / 0 / −1 to positive / neutral / negative |
| 2 | `RuleBasedSentiment` class with `VERSIONS` v1–v5 | Five classifier configurations | Each version adds one group of rules (Section 4) |
| 3 | Predict every record with every version | Class mix per version (Section 4 table) | v1 under-predicts negative (7.8%) |
| 4 | `cv_scores()`: `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)` | Accuracy, precision, recall, macro-F1 per version; per-fold table; pooled confusion matrix | v5 selected (macro-F1 0.581) |
| 5 | Compare consecutive versions: % changed, `cohen_kappa_score` | Version change table | v2 changed the most predictions (25.5%) |
| 6 | Run v5 once on the unseen set; `classification_report`, `confusion_matrix`, chi-square test | Unseen accuracy 0.450, macro-F1 0.432; class mix comparison (p = 0.470) | Unseen set never used to build rules; negative recall only 0.11 |
| 7 | Merge `topic_assignments_train.csv` and `topic_assignments_holdout.csv` (`lda_topic_label`); `pd.crosstab(topic, predicted_label)` | Sentiment share per topic and stacked bar chart | Public school quality is the most negative topic |
| 8 | Save `predictions_all.csv` | One row per record: dataset, topic, manual label, score, predicted label | Input for the Discussion of Results |
