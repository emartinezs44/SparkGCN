## Graph Convolutional Networks in Apache Spark

Implementation of the GCN(https://arxiv.org/abs/1609.02907) on top of Spark using [BigDL 2.0](https://bigdl.readthedocs.io/en/latest/). It is inspired on the initial [Keras-based implementation](https://github.com/tkipf/keras-gcn).

It is a pure Scala project that relies on [BigDL DLlib](https://bigdl.readthedocs.io/en/latest/doc/DLlib/Overview/dllib.html) and [Breeze](https://github.com/scalanlp/breeze) libraries and can be used in graph processing pipelines based in Spark like GraphX that can be executed in big data clusters.

### Cora Example
To see how it works it is implemented the [Cora](https://graphsandnetworks.com/the-cora-dataset/) example that consists in a semi-supervised classification problem where only 140 samples from a set of 2708 are used in the training process.  With these labeled nodes the optimization process calculates a set weights which can be considered as filter parameters of convolutional layers that are shared across the graph and encode node features and information from its connections. For more details see: ["Semi-Supervised Classification with Graph Convolutional Networks"](https://arxiv.org/abs/1609.02907) (Thomas N. Kipf, Max Welling).

#### Results

| No propagation Model      | GNC Propagation Model |
| ----------- | ----------- |
| 0.53 Accuracy | 0.78 Accuracy       |

### Execution

You can use SBT version >= 1.0(https://www.scala-sbt.org/download.html) to spawn the training process, indicating the propagation function model:
- **0** to NOT apply convolution
- **1** to apply GCN propagation model(see https://arxiv.org/pdf/1609.02907.pdf).

You must also include the number of epochs you want to train the neural network, and the path of the node and edes data from the Cora dataset. They are included in the resources folder.

Example:
	```sbt run 1 200 cora.content cora.cites```

This sbt command starts the optimization process and executes the inference to the whole graph. As final result, the accuracy metric is calculated.

You can train the neural network in your spark cluster using **spark-submit** indicating the main class and the parameters **ems.gcn.CoraExample [mode] [epochs] [cora.content] [cora.cites]** including the **BigDL 2.0** dependency, and the artifact generated with **sbt package**.

**IMPORTANT NOTE:**

This project applies the convolution in **one Spark Partition**, so in case of submitting the application to a Spark cluster **you must set the number of cores to 1**. This is because in each iteration the convolution is applied to the whole graph and if you use more cores the data will be split across threads and the process will fail.

More work about processing much bigger graphs with graph neural networks using Spark clusters will be added soon in this repository.