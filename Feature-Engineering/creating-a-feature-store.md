# Feature Store and Entity Type Creation

## Overview
- Describes creating a feature store, entity type, adding features, and ingestion process.

## Challenges with ML Features
- Features are hard to share and reuse.
- Serving features in production with low latency is challenging.
- Inadvertent skew in feature values between training and serving is common.

## Preprocessing Source Data
- Ensure clean and tidy features with no missing values.
- Data types must be correct.
- One-hot encoding of categorical values should be done.

## Source Data Requirements
- Vertex AI Feature Store can ingest data from BigQuery or Cloud Storage.
- Files in Cloud Storage must be in Avro or CSV format.
- Must have a column for entity IDs (type: STRING).
- Source data value types must match destination feature value types.

# Feature Store Creation

## Creating a Feature Store
- Can be done in Vertex AI console or Workbench using API.
- Steps to create:
  1. Click "Features" on the dashboard.
  2. Select region.
  3. Click "Create Feature Store."
  4. Name the feature store.
  5. Enter the number of nodes.
  6. Optional: Use a customer-managed encryption key.
  7. Click "Create."

## Note
- Feature store deletion and addition must be done through the API.

# Entity Type Creation

## Creating an Entity Type
- Click "Create Entity Type."
- Name the entity type (e.g., budget_id).
- Optional: Add description.
- Click "Create."

## Entity Type Properties
- Basic information presented includes name, region, feature-store name, creation and update dates, and any description.

# Adding Features

## Overview
- A feature is a measurable attribute of an entity type.
- Features are associated with values stored in BigQuery or Cloud Storage.

## Adding Features
- Use "Add Feature" window.
- Required fields: feature name, value type, and interval.

## Note
- Data type confirmation for features based on XYZ team's dataset.

## Feature Monitoring
- Enable monitoring for feature values to detect data drift over time.

# Ingestion Process

## Overview
- Ingest features data from BigQuery or Cloud Storage.
- Batch ingestion for bulk data import.

## Ingestion Jobs
- Define corresponding entity type and features before importing data.
- Properties include creation time, processing duration, region, workers, and link to data source.

## Note
- Minimum 1,000 rows required for data set to be uploaded into Vertex AI.

## Data Columns for Ingestion
- Entity_id: ID of the ingested entity.
- Timestamp: Timestamp of feature generation.
- Feature columns matching destination feature names.
