# Building Hybrid Machine Learning Models

## Introduction

### Scenarios for Hybrid Solutions:

1. **On-Premises Constraints:**

   - Inability to be fully cloud-native due to on-premises infrastructure commitments.
   - Challenges in moving training data from on-premise clusters or data centers.

2. **Multi-Cloud Requirements:**

   - Need for a multi-cloud solution due to data production on a different cloud or application deployment on another cloud.

3. **Edge Computing:**
   - Necessity for machine learning on the edge due to connectivity constraints, common in IoT applications.
   - Inference on the edge, requiring predictions to be made on the device itself.

### Kubernetes and Kubeflow:

- Kubernetes is introduced as a container orchestration system, allowing orchestration across on-premises and cloud environments.
- Kubeflow, an open-source machine learning stack built on Kubernetes, facilitates migration between cloud and on-prem environments.
- Kubeflow can be run on GKE on Google Cloud or on various platforms, maintaining consistent code with minor configuration changes.

## Machine Learning on Hybrid Cloud

**Key Concepts in Building Hybrid Machine Learning Systems**

1. **Composability:**

   - Importance: Building effective hybrid ML systems requires support for composability.
   - Definition: Composability involves the ability to integrate various microservices in ML stages (data analysis, training, validation, monitoring) based on what suits the problem.
   - Challenge: Different systems handle these stages differently, emphasizing the need for flexible composition.

2. **Portability**

   - Importance: Essential for moving ML frameworks across environments (on-premises, cloud) in a hybrid setup.
   - Explanation: Once a framework is built, it needs to be portable to different environments without redoing configurations, libraries, and testing.
   - Scenarios: Necessary for adapting to changes in inputs, accommodating scalability needs, and transitioning between on-premises and cloud environments.
   - Significance: Addresses the practical challenges of changing setups due to evolving requirements or diverse environments.

3. **Scalability**
   - Definition: In the context of ML, scalability extends beyond Kubernetes scaling to include accelerators (GPUs, TPUs), disks, diverse skill sets (software engineers, researchers, data scientists), and collaboration across teams.
   - Scope: Encompasses the scaling of experiments, teams, and infrastructure elements required for successful ML operations in a hybrid cloud environment.
