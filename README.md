## Jigsaw Rate Severity of Toxic Comments

This repository provides a comprehensive description of the methods used in the *Jigsaw “Rate Severity of Toxic Comments”* competition (<https://www.kaggle.com/competitions/jigsaw-toxic-severity-rating>), enabling achievement of the best private leaderboard score of **0.81404**. The full source code is publicly available at Kaggle: <https://www.kaggle.com/code/alex56mod/nlp-jigsaw-toxicity-debra-v3-large-fin>.

We employed a transformer-based multi-label classification framework for toxic comment prediction, leveraging the **DeBERTa-v3-large (Decoding-enhanced BERT with disentangled attention)** model as a backbone. Our system was designed to address six toxicity-related categories: *toxic*, *severe\_toxic*, *obscene*, *threat*, *insult*, and *identity\_hate*, and to predict toxicity scores to rank the comment pairs. The model was trained and evaluated using a five-fold cross-validation strategy, ensuring robust generalization despite limited training epochs (link to the model configuration dataset below). 

Link to the Kaggle notebook: <https://www.kaggle.com/code/alex56mod/nlp-jigsaw-toxicity-debra-v3-large-fin> 

Link to the model configuration dataset (~28 GB): <https://www.kaggle.com/datasets/alex56mod/mod-deberta-v3-large-jigsaw-tox>

Software requirement: 

torch==2.6.0
pytorch\_lightning==2.5.1
transformers==4.51.3

Hardware Requirement: GPU P100

### Datasets (all files are available at the corresponding competition page)

To train and evaluate our model for toxicity detection and ranking, we utilized several publicly available datasets from previous Jigsaw competitions and other related sources. 

The dataset from the *Jigsaw “Toxic Comment Classification Challenge”* (https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge) was used as the primary training source (based on ensemble model analysis of 1st place solution of this competition). It consists of comments labeled for six binary toxicity categories: *toxic*, *severe\_toxic*, *obscene*, *threat*, *insult*, and *identity\_hate*. The dataset was subjected to class balancing via undersampling: non-toxic examples were randomly sampled to match the number of toxic instances, based on the aggregate toxicity score.

We also processed additional datasets for exploratory purposes (however, models trained on these datasets did not outperform those trained exclusively on the jigsaw\_01 dataset in our experiments):

- *Jigsaw “Unintended Bias in Toxicity Classification”* (https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification): This dataset included individual annotations for toxicity and identity-related subgroups. Mean scores were aggregated per comment and binarized using a threshold of 0.5.
- *Ruddit dataset* (https://www.kaggle.com/datasets/rajkumarl/ruddit-jigsaw-dataset): This dataset provided scalar toxicity scores (offensiveness\_score) for Reddit comments.

Validation was performed on the *Jigsaw “Rate Severity of Toxic Comments”* dataset (https://www.kaggle.com/competitions/jigsaw-toxic-severity-rating), specifically using the validation\_data.csv file, which contains pairwise annotations (less\_toxic, more\_toxic) used for ranking.

### Text Normalization

All datasets were preprocessed in a consistent pipeline involving text normalization and label transformation. Among them, the dataset prepared using the jigsaw\_01\_dataset\_proc function consistently yielded the best results across multiple evaluation metrics, and thus was used as the primary training corpus in our final model.

Raw text inputs from all datasets were passed through a standardized normalization function inspired by best practices in prior Jigsaw competitions. This function performed the following operations:

- Removal of hyperlinks and HTML tags using regular expressions and the BeautifulSoup parser.
- Filtering of emojis, excessive punctuation, and specific non-linguistic artifacts such as IP addresses.
- Replacement of common obfuscations of offensive words (e.g., "f\*\*k", "s$x") with their canonical forms.
- General formatting cleanup, such as extra whitespaces, newline characters, and special symbols.

This normalization ensured uniformity in text representation across all datasets, improving the robustness of the learned representations.

### Metrics

The objective function was the **binary cross-entropy with logits**, computed independently for each label. During validation, we evaluated **pairwise ranking accuracy**, assessing how well the predicted toxicity scores agreed with human annotators on pairs of comments labeled as *more toxic* or *less toxic*.\
This validation setup was designed to mirror the competition’s evaluation metric, which is based on average agreement with annotator preferences across comment pairs.

### Experiment Setup

The training process was managed using **PyTorch Lightning**, facilitating cleaner model encapsulation and reproducibility. The optimization was conducted using the **AdamW** optimizer with a learning rate of 1e-5 and a weight decay of 1e-5. Gradient accumulation over two batches and gradient clipping (clip value of 100) were employed to mitigate instability during backpropagation.

The learning rate schedule followed a **cosine annealing with warm-up** strategy, with 200 warm-up steps and 0.5 cosine cycles. Models were trained for a maximum of 4 epochs per fold (total 5 folds), using batch sizes of 4 and 8 for training and validation respectively.

For both the tokenizer and the backbone model, a pre-trained version was loaded from the Hugging Face Transformers library unless cached weights and configurations were already available locally. Inputs were tokenized up to the maximum sequence length (max\_length = 256).

To ensure model robustness, 5-fold cross-validation was performed. For each fold, a checkpoint of the best model (based on validation loss) was saved (<https://www.kaggle.com/datasets/alex56mod/mod-deberta-v3-large-jigsaw-tox>). The inference was conducted fold-wise using these checkpoints, and out-of-fold (OOF) predictions were collected where required. During inference, the model predictions were averaged across folds to obtain the final output.
