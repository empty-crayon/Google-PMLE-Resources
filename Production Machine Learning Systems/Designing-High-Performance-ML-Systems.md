# Designing High Performance ML Systems

## Introduction

### Performance Considerations
- Performance is a key consideration in machine learning systems.
- The module emphasizes learning how to identify performance considerations specific to machine learning models.


### Focus on Performance Improvements
- Depending on the model, the focus could be on improving Input/Output (IO) performance or enhancing computational speed.

### Infrastructure Requirements
- The choice of machine learning infrastructure depends on the specific performance goals.
- Considerations include whether to scale out with multiple machines or scale up on a single machine using GPUs or TPUs.

### Acceleration Strategies
- In some cases, both scaling out and scaling up might be necessary, achieved by using a machine with multiple accelerators.

### Beyond Hardware
- Hardware selection is not the only factor; it also influences the distribution strategy chosen for the machine learning model.
- The chosen hardware informs decisions about how the model is distributed for optimal performance.

## Training

### Definition of High Performance in Machine Learning
- **High Performance** in machine learning can be defined in various ways:
  - Powerfulness
  - Handling large datasets
  - Speed of training
  - Ability to train for extended periods
  - Achieving the best accuracy

### Key Aspect: Time Taken to Train a Model
- **Critical Factor:** Time taken to train a model.
- Performance is assessed based on efficiency; for example, if one architecture takes less time to achieve the same accuracy as another, it is considered more performant.

### Module Assumption
- **Assumption:** Throughout the module, models with the **same accuracy** are considered.
- Focus on **infrastructure performance** rather than model accuracy.

### Budget Considerations
- **Critical Aspects of Budget:**
  - Time
  - Cost
  - Scale


## Time Considerations
- **Business Use Case Impact:** Time to train might be driven by the business use case.
- **Practical Constraints:** Deployment time, A/B testing, etc.
- **Example:** If a model needs to be trained daily, the training must finish within 24 hours.

### Cost Considerations
- **Business Decision:** Determining how much to spend on computing costs.
- **Incremental Benefit:** Avoid training for extended periods(eg. 20 hours everyday on an expensive machine) if the benefits are not substantial. 

### Scale Considerations
- **Model Computational Expense:** Models differ in computational requirements.
- **Data Size Impact:** More data generally leads to higher accuracy, but there are diminishing returns.
- **Infrastructure Choices:** Single expensive machine vs. multiple cheaper machines.
  
### Factors Limiting Model Training Performance
- **Three Key Factors:**
  - Input/Output
  - CPU
  - Memory

![Graph](./src/training.png)

### IO-Bound Training
- **Indicators:**
  - Large, heterogeneous input data
  - Small models with trivial compute requirements
  - Input data on a low-throughput storage system
- **Improvements:**
  - Efficient data storage
  - Parallelizing reads
  - Reducing batch size

### CPU-Bound Training
- **Indicators:**
  - Simple IO but complex, computationally expensive models
  - Running on underpowered hardware
- **Improvements:**
  - Faster accelerators (GPUs, TPUs)
  - Simpler models or activation functions
  - Training for fewer steps

### Memory-Bound Training
- **Indicators:**
  - Large input size or complex models with many parameters
  - Limited accelerator memory
- **Improvements:**
  - Adding more memory to individual workers
  - Using fewer layers in the model
  - Reducing batch size for memory-bound systems

## Predictions

### Batch Prediction
- **Key Considerations:**
  - Time for predictions (how long does it take to finish predictions)
  - Business-driven time constraints
    - Example: Precomputing top 20% user recommendations in 5 hours for 18 hours of training.
  - Cost considerations
  - Scale considerations
    - Single machine or distributed to multiple workers
    - Hardware availability on workers (e.g., GPUs)

### Online Prediction
- **Differences in Performance Considerations:**
  - Users wait for predictions here
  - Cannot distribute the prediction graph
  - Computation for one end user on one machine
  - Scaling out predictions to multiple workers
    - Each prediction handled by a microservice
    - Replication and scaling predictions using Kubernetes or AI Platform workbook
    - AI Platform workbook predictions as a higher-level abstraction
  
### Performance Metrics for Online Prediction
- **Performance Target:**
  - Queries per second (QPS)
  - Different from training steps per minute
- **Design Considerations:**
  - Separate design for training and performance, especially for online predictions
  - Balancing batch predictions and online predictions
  - Striking the right performance trade-off
  - Measurement necessary to determine the optimal trade-off (after building the system in most cases)

## Distributed Training Architectures


## Scaling in TensorFlow
It's crucial to understand the high-level concepts of distributed training.

- **Single Machine Scaling:**
  - TensorFlow automatically scales on multi-core CPUs.
  - Accelerators like GPUs can be added to speed up training with minimal effort.

- **Distributed Training:**
  - Progresses from one machine with a single device to multiple machines, possibly with multiple devices each.

## Distributed Training Architectures

Distributed training distributes workloads across worker nodes. Two common architectures are data parallelism and model parallelism.

### Data Parallelism

Data parallelism is model-agnostic and widely used for parallelizing neural network training.

- **Working Principle:**
  - Same model and computation run on every device.
  - Each device trains on different data samples.
  - Loss and gradients are computed independently on each device based on the samples that it sees.
  - The updated model is then
used in the next round of computation.

- **Gradient Calculation:**
  - Synchronous: Devices communicate gradients and update model collectively.
  - Asynchronous: Devices can run independently, communicating with peers or through parameter servers.

- **Pros and Cons:**
  - Synchronous: Overhead in gradient calculation due to waiting for all devices.
  - Asynchronous: Potential out-of-sync issues, but scales well with no waiting.

### The Two Data Parallelism approaches
There are currently 2 approaches used to update the model using gradients from
various devices.
- **Async parameter server**
  - Some devices are designated to be parameter servers, and others as workers
  -  Each worker independently fetches the latest parameters from the PS and computes gradients based on a subset training samples.
  - It then sends the gradients back to the PS. Which then updates its copy of the parameters with those gradients.
  - Each worker does this independently. This allows it to scale well to a large number of workers.
  - These don’t hurt the scaling because workers are not waiting for each other.
  - The downside of this approach, however, is that workers get out of sync. They compute parameter updates based on stale values and this can delay convergence.
- **Sync allreduce architecture**
  - Each worker holds a copy of the model’s parameters - there are no special servers holding the parameters.
  - Each worker computes gradients based on the training samples they see and communicate between themselves to propagate the gradients and update their parameters.
  - All workers are synchronized - conceptually the next forward pass doesn’t begin until each worker has received the gradients and updated their parameters
  - With better links and low sync overhead, this may help faster covergence. 

### Model Parallelism

A simple way to describe model parallelism is when your model is so big that it
doesn’t fit on one device’s memory. So you divide it into smaller parts that compute
over the same training samples on multiple devices.

- **Working Principle:**
  - Each processor gets the same data but applies a different parts of the model.
  - Weights of the net are split equally among threads.
  - For example, you could put different layers on different devices.

- **Challenges:**
  - Synchronization needed after each layer for input to the next layer.
  - Assigning layers to GPUs is more complex than data parallelism.

### Hybrid Approach

Sometimes, a hybrid of data and model parallelism is used in the same architecture for optimal results.

## Choosing the Right Approach

The choice between asynchronous parameter server and synchronous Allreduce approaches depends on the characteristics of the model.

- **Asynchronous Parameter Server:**
  - Suitable for sparse models with fewer features.
  - Consumes less memory, ideal for clusters of CPUs.

- **Synchronous Allreduce:**
  - Preferred for dense models with many features.
  - All machines share the load of storing and maintaining global parameters.

In conclusion, there's no one-size-fits-all solution, and the choice depends on the specific requirements and characteristics of the model being trained.
## TF Distributed Training Strategies
There are four TensorFlow distributed training strategies that support data parallelism.
## Distributed Training with Mirrored Strategy

### Introduction
- **Mirrored Strategy:** Simplest approach for distributed training.
- **Use Case:** Single machine with multiple GPU devices.

### Mirrored Strategy Basics
- Replica Creation:
  - Mirrored strategy replicates the model on each GPU.
- Data Distribution:
  - During training, a minibatch is split into "n" parts (number of GPUs).
  - Each part is fed to one GPU device.
- Coordination:
  - Mirrored strategy manages data distribution and gradient updates across GPUs.

### Image Classification Example
- Keras ResNet model with functional API.
- Data Preparation:
  - Download Cassava dataset from TensorFlow datasets.
  - Implement a preprocess_data function to scale images.
  - Map, shuffle, and prefetch data.
- Model Definition:
  - Define the ResNet model using the functional API.

### Implementing Mirrored Strategy
1. Create a MirroredStrategy object using `tf.distribute.MirroredStrategy`.
2. Model Configuration:
   - Define model variables within the strategy scope.
   - Variables include loss, optimizer, and metrics for accuracy.
3. Adjust Batch Size:
   - Batch size in distributed training now refers to the global batch size.
   - Global Batch Size: Total across all GPUs.
   - Per Replica Batch Size: Batch size processed by each GPU.
   - Scale batch size by the number of replicas.
4. Data Processing:
   - Map, shuffle, and prefetch the data.

### Training Process
1. Call `model.fit` on the training data.
2. Scaling Batch Size:
   - Adjust batch size to utilize multiple GPUs effectively.
   - Each machine processes a fraction of the global batch size.
   - Example: If global batch size is 64 with two GPUs, each processes 32 examples per step.
3. Run Training:
   - Run multiple passes (epochs) of the entire training dataset.

### Understanding Model.fit without Strategy
- Example with Simple Linear Model:
  - Computational graph (DAG) with matmul and add operations.
  - Explanation of data parallelism with two GPUs.
  - Each GPU processes different slices of the input batch.

### Conclusion
- Mirrored strategy simplifies distributed training on a single machine with multiple GPUs.
- Scaling batch size ensures optimal GPU usage.

## Multi-worker Mirrored Strategy

## TPU Strategy

## Parameter Server Strategy

## Inference



