# Feature Serving Methods: Batch and Online

## Overview

In this lesson, we describe feature serving using batch and online methods.

## Feature Serving Definition

- Feature serving is the process of exporting stored feature values for training or inference.

## Vertex Feature Store

- Uses a combination of storage systems and back-end components.
- Key APIs: batch ingestion API and online serving API.

## Batch Ingestion API

- Allows ingestion of feature values in bulk from a valid data source.
- Maximum 100 features per entity type per request.
- One batch ingestion job per entity type to avoid collisions.
- Requires specifying the location of the source data and how it maps to features in the feature store.

## Online Serving API

- Used for low latency data retrieval of small batches of data for real-time processing.
- Requires specifying a list of required features.

## Batch Serving

- High throughput, serving large volumes of data for offline processing (e.g., model training, batch predictions).

## Example: Predicting Baby's Weight

- Using historical data from an open dataset of births available in BigQuery.
- Features: date of birth, location of the birth, baby's birth weight, mother's age at birth, and duration of pregnancy.
- Historical features are ingested in batch and served to a mobile app for predicting baby's weight.

## Batch Serving Request Information

- List of existing features to get values for.
- Read-instance list containing information for each training example.
- Destination URI and format for the output (CSV or TFRecord).

## Output Formats

- CSV file in a regional or multi-regional Cloud Storage bucket.
- TFRecord file in a Cloud Storage bucket.

## Batch Serving Jobs

- Must be created in the Feature Store API.

# Resources

## Video Tutorial

- [YouTube - Introduction to Vertex AI Feature Store](https://youtu.be/jXD8Sfx4hvQ)

## Google Cloud Documentation

- [Google Cloud Vertex AI - Feature Store Overview](https://cloud.google.com/vertex-ai/docs/featurestore/overview)
- [Google Cloud Vertex AI - Feature Store Documentation](https://cloud.google.com/vertex-ai/docs/featurestore)

## External Resources

- [Feature Store Community](https://www.featurestore.org/)
