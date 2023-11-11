## Introduction

### Module Overview
- Explore ways to recognize model dependency on data
- Make cost-conscious engineering decisions
- Know when to roll back a model to an earlier version
- Debug observed model behavior
- Implement a pipeline immune to a specific dependency

### Containers and Kubernetes
- Containers: abstraction packaging applications and libraries
- Kubernetes makes hosting large applications better

## Adapting to Data

When it comes to adapting to change in machine learning, several factors may be prone to change. Let's explore these aspects:

### Factors Prone to Change
- An upstream model
- A data source maintained by another team
- The relationship between features and labels
- The distributions of inputs

It's essential to understand that **all of these can and often do change**.

### Example Scenarios

#### Scenario 1: Upstream Model Changes
- Imagine you've built a model predicting umbrella demand based on a weather prediction model.
- Unbeknownst to you, the upstream weather prediction model was trained on incorrect years of data.
- When the model owners fix this without notifying you, your model's performance drops as it expected the old model's data distribution.

#### Scenario 2: Changes in Data Source
- Your data science team ingests traffic logs from the web development team.
- The web development team changes their logging format without notifying your team, causing unexpected null features in your model.
- To address this, it's crucial to stop consuming data from a source that doesn't notify downstream consumers and consider creating a local version of the upstream model.

#### Scenario 3: Poorly Chosen Features
- Sometimes, models are trained on features added without proper scrutiny.
- Adding features indiscriminately may lead to performance issues.
- For instance, a rush decision to include several new features without understanding their relationship to the label can hurt performance.
- It's essential to scrutinize features before adding them and conduct leave-one-out evaluations to assess their importance.

## Changing Distributions

### Model Performance and Distribution Changes
- Ingesting an upstream model with different input expectations can degrade our model's performance.
- Changes in the likelihood of observed values, such as model inputs, are termed as changes in the distribution.

### Reasons for Distribution Changes
- The distribution of data can change for various reasons.
- Examples include changes in the distribution of labels, like in the case of the natality dataset where baby weight has changed over time.

### Impact of Label Distribution Changes
- Changes in label distribution can affect the relationship between features and labels.
- Model predictions, matching the label distribution in the training set, may become less accurate.

### Feature Distribution Changes
- Feature distribution changes can also impact model performance.
- For instance, using postal codes as a feature for predicting population movement patterns can lead to issues due to changes in postal codes over time.

### Handling Feature Distribution Changes
- Representing categorical features, like postal codes, may lead to skewed distributions if not handled properly.
- Monitoring, comparing descriptive summaries, and analyzing model residuals can help identify and address distribution changes.
- Extrapolation, making predictions in regions far from training data, can lead to inaccuracies and should be avoided.

### Protecting Against Distribution Changes
- Vigilance through monitoring is crucial.
- Descriptive summaries of inputs, mean, and variance changes can be indicators of distribution shifts.
- Examining model residuals helps identify changes in prediction accuracy.
- If a relationship is believed to change over time, custom loss functions or retraining on recent data can be implemented.

## Right and Wrong Decisions


### Decisions about Data

- Some decisions involve weighing cost versus benefit, such as short-term performance goals versus long-term maintainability.

### Example: Model Deployment Issue

- Trained a model to predict "probability a patient has cancer" from medical records.
- Selected features: patient age, gender, prior medical conditions, hospital name, vital signs, and test results.
- Model performed well on held-out test data but poorly on new patients.
- Issue: The model was trained using a feature, 'hospital name,' which wasn't legitimately available at decision time.
- Change in the distribution of this feature during production made it an unreliable predictor.
- Data leakage occurred due to the use of a feature (hospital name) not available at decision time.

### Hospital Name as a Feature

- Some hospitals focus on diseases like cancer, leading the model to learn that 'hospital name' was important.
- At decision time, 'hospital name' wasn't available to the model.
- Model interpreted the hospital name as an empty string, causing data leakage.
- This concept is referred to as data leakage, where the label leaks into the training data.


### Example: Naive Testing of Hypothesis

- Professor believed a relationship between how an author thought about the mind and their political affiliation in 18th-century literature.
- Naive machine learning testing: Using mind metaphors as features and political affiliations as labels.
- Data partitioning issue: Sentences from each author were randomly assigned to training, validation, and test sets.
- Suspiciously amazing results due to the implicit inclusion of 'person name' in the feature set.
- When data partitioning changed to be by author instead of by sentence, model accuracy dropped to a more reasonable level.

### Key Takeaways

- Be cautious about features that may not be available at decision time to avoid data leakage.
- Consider proper data partitioning to prevent implicit inclusion of features that are linked to the label.

