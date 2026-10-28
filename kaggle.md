UNSW-NB15 is a network traffic dataset generated using the IXIA PerfectStorm tool at the Cyber Range Lab of UNSW Canberra. Each row represents a single network flow, and the features describe measurable properties of that flow, such as packet counts, TCP window size, protocol information, and connection state.

Each training sample is labeled as either normal or anomalous. An anomalous flow corresponds to a network attack. The dataset contains multiple attack categories, including denial-of-service traffic and Generic attacks, which are designed to work against block ciphers regardless of the underlying algorithm.

This competition is a data augmentation challenge. You should use the released labeled training data to generate a synthetic dataset that represents the normal data distribution.

Your submission must contain exactly 70,000 synthetic normal samples and follow the same feature schema as the provided submission template. The submitted synthetic data will be used to train fixed downstream anomaly detection models, specifically ECOD and Isolation Forest (IForest), which will then be evaluated on hidden validation and test data. Categorical features are one-hot encoded before evaluation. IForest is evaluated over 10 fixed random seeds and its AUPRC is averaged across the 10 runs. ECOD is deterministic and is evaluated once. The Kaggle leaderboard score is the average of the mean IForest AUPRC and the ECOD AUPRC.

The validation and test sets are not released during the competition.

Files
train.csv: The labeled UNSW-NB15 training set containing both normal and anomalous network flows.
sample_submission.csv: A template showing the required submission columns, column order, and format. Please refer to the Starter Kit on Canvas for details on how to generate it.
There is no public test.csv. The validation and test sets remain hidden during the competition and are used only for evaluation.

Data Format
The UNSW-NB15 dataset contains 42 input features, including 39 numerical features and 3 categorical features. The training set additionally contains an is_anomaly label, where 0 indicates a normal flow and 1 indicates an anomalous flow or attack.

Submission Requirements
You should submit synthetic normal-only data with:

exactly 70,000 rows
exactly the feature columns specified in sample_submission.csv
no anomaly-label or prediction column.
Each submitted row must also include an id column containing a unique row identifier, for example:

id feature1 feature2

0 0.5 18

1 0.7 16

Submissions with an incorrect number of rows or columns will not be accepted for evaluation.
--------------------------------------------------------------------

Dataset Description
NSL-KDD is a refined version of the KDD Cup 1999 network intrusion dataset. Each row represents a single network connection, and the features describe measurable properties of that connection, such as connection duration, byte counts, protocol type, and service information.

Each training sample is labeled as either normal or anomalous. An anomalous connection corresponds to a network attack. The dataset contains multiple types of attacks, including denial-of-service and network probing/surveillance attacks.

This competition is a data augmentation challenge. You should use the released labeled training data to generate a synthetic dataset that represents the normal data distribution.

Your submission must contain exactly 40,000 synthetic normal samples and follow the same feature schema as the provided submission template. The submitted synthetic data will be used to train fixed downstream anomaly detection models, specifically ECOD and Isolation Forest (IForest), which will then be evaluated on hidden validation and test data. Categorical features are one-hot encoded before evaluation. IForest is evaluated over 10 fixed random seeds and its AUPRC is averaged across the 10 runs. ECOD is deterministic and is evaluated once. The Kaggle leaderboard score is the average of the mean IForest AUPRC and the ECOD AUPRC.

The validation and test sets are not released during the competition.

Files
train.csv: The labeled NSL-KDD training set containing both normal and anomalous network connections.
sample_submission.csv: A template showing the required submission columns, column order, and format. Please refer to the Starter Kit on Canvas for details on how to generate it.
There is no public test.csv. The validation and test sets remain hidden during the competition and are used only for evaluation.

Data Format
The NSL-KDD dataset contains 41 input features, including 38 numerical features and 3 categorical features. The training set additionally contains an is_anomaly label, where 0 indicates a normal connection and 1 indicates an anomalous connection or attack.

Submission Requirements
You should submit synthetic normal-only data with:

exactly 40,000 rows
exactly the feature columns specified in sample_submission.csv
no anomaly-label or prediction column.
Each submitted row must also include an id column containing a unique row identifier, for example:

id feature1 feature2

0 0.5 18

1 0.7 16

Submissions with an incorrect number of rows or columns will not be accepted for evaluation.