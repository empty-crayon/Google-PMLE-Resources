# MLOps: Continuous Delivery and Automation in Machine Learning

## Overview:
- MLOps combines DevOps principles with ML systems for unified development and operation.
- Businesses invest in data science and ML for predictive models to drive value.

## ML System Complexity:
- ML system elements encompass more than just code.
- Components include configuration, automation, data handling, testing, deployment, monitoring, and more.

## DevOps vs. MLOps:
- DevOps practices like CI and CD are fundamental in ML systems but with unique challenges.
- ML projects involve experimental nature, complex testing, and specialized deployment.

## Steps in ML Model Development:
- Data extraction, analysis, preparation, model training, evaluation, validation, serving, and monitoring.
- The maturity of an ML process is defined by the automation level in these steps.

## MLOps Levels:
### Level 0: Manual Process
- Manual and script-driven process with no automation.
- Disconnection between ML and operations, infrequent iterations, lack of CI/CD, and no active performance monitoring.
- Challenges: Model adaptation, infrequent updates, and lack of monitoring.

### Level 1: ML Pipeline Automation
- Automation of ML pipeline, enabling continuous training in production.
- Rapid experiments, model training, modularized code, and continuous delivery of model prediction service.
- Additional components: Data/model validation, feature store, metadata management, pipeline triggers.
- Challenges: Manual testing, limited pipeline deployments, and new ML idea exploration.

### Level 2: CI/CD Pipeline Automation
- Robust CI/CD for rapid exploration and deployment of new ML components.
- Development, continuous integration, delivery, automated triggering, and monitoring.
- Continuous integration involves testing components, convergence, integrity, and compatibility.
- Continuous delivery involves infrastructure verification, service testing, and performance validation.
- Gradual implementation recommended for improved automation in ML system development and production.