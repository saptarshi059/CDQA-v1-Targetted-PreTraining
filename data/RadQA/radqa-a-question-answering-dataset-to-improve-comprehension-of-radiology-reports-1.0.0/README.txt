RadQA: A Question Answering Dataset to Improve Comprehension of Radiology Reports

RadQA is an electronic health record (EHR) machine reading comprehension (MRC) dataset that aims to overcome the issues with the existing resources for the MRC task in clinical domain. The following are the main characteristics of RadQA:

• The questions in RadQA reflect true information needs of clinicians ordering radiology reports (as the queries are inspired from the clinical referral section of the radiology reports).
• The corpus contains 3074 unique question-report pairs encompassing 1009 radiology reports from 100 patients.
• Each question has two answers for a radiology report (in its Findings and Impressions sections), resulting in a total of 6148 distinct question-answer evidence pairs (including unanswerable questions, that no available MRC dataset includes).
• The answers are oftentimes present in the form of phrases or span multiple lines (as opposed to only multi-word answer entities in available MRC datasets), fulfilling the clinical information needs.
• The questions require a wide variety of reasoning and domain knowledge to answer, that makes it a challenging dataset for advanced models.
• The distribution of the sampled radiology reports is similar to that in the MIMIC-III database.
• The dataset is publicly available (as the radiology reports come from the publicly available MIMIC-III database).

The following three files contain the different splits used for training, development, and testing.

• train.json - training set
• dev.json - development set
• test.json - testing set