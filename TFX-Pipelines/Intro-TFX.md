# Intro to TFX

## TFX Concepts

1. **TFX Components:**
   - Definition: TFX components are implementations of machine learning tasks within a pipeline.
   - Artifacts: Each component produces and consumes structured data representations known as artifacts, which are utilized by subsequent components in the workflow.

2. **Composition of Components:**
   - Components consist of five elements:
     - **Component Specification:** Describes inputs, outputs, and runtime parameters using protocol buffers.
     - **Driver Class:** Coordinates job execution, accessing metadata and artifacts.
     - **Executor Class:** Implements the code for specific ML tasks like ingestion or transformation.
     - **Component Interface:** Packages the specification, driver, executor, and publisher.
     - **Publisher:** Logs component runs and writes output artifacts to the artifact store.

3. **Execution Flow of TFX Components:**
   - Components execute sequentially at runtime:
     - Driver reads the component spec and retrieves required artifacts.
     - Executor performs computation on input artifacts and generates outputs.
     - Publisher logs the run and writes output artifacts to the store.

4. **TFX Pipelines:**
   - Sequence of components linked through a directed acyclic graph based on artifact dependencies.
   - Components communicate through input and output channels.
   - Channels are abstract connections linking data producers and consumers.

5. **Pipeline Parameters:**
   - Allow altering pipeline behavior using Configuration Protocol Buffers without code changes.
   - Enable experimentation by running pipelines with different parameters (e.g., training steps, data splits) for improved performance analysis.

6. **Metadata Store and Artifacts:**
   - TFX uses the ML metadata library to standardize metadata storage.
   - Artifacts are stored on local or remote cloud filesystems, ensuring consistency across ML projects.

7. **Orchestrators in TFX:**
   - Coordinate pipeline runs, ensuring execution order, retries, and parallelization.
   - Task-aware: Can be manually run as tasks, either entire pipelines or individual components and downstream tasks.

8. **Data-Aware Pipelines:**
   - Store all artifacts from every component run to check for changes and determine re-computation necessity, enhancing speed and resource efficiency.

9. **TFX Horizontal Layers:**
   - Shared libraries, utilities, and protobuffs for abstraction across different computing and orchestration environments.
   - Users typically interact with the integrated front end, while deeper layers are for additional customization.

10. **Layers Overview:**
    - Integrated Front End: GUI-based controls for pipeline management, debugging, and visualization.
    - Orchestrators: Run TFX pipelines, schedule components based on dependencies.
    - Shared Libraries: Abstractions controlling pipeline aspects like data representations.
    - Pipelines Storage: ML metadata records and organizes pipeline executions and artifact locations.

These concepts form the foundation of understanding and utilizing the TFX platform for efficient and scalable machine learning pipelines.