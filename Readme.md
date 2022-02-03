## Graph Convolutional Networks in Apache Spark

Implementation of the GCN(https://arxiv.org/abs/1609.02907) on top of Spark using Analytics Zoo. It is based on the Keras repo https://github.com/tkipf/keras-gcn.

It is implemented for the moment only the Cora example. The main program included in the CoraExample module starts a training process using only 140 labeled samples from the Cora dataset.

You can read more info about the experiment in https://emartinezs44.medium.com/graph-convolutions-networks-ad8295b3ce57.

### Execution

You can use SBT version >= 1.0(https://www.scala-sbt.org/download.html) to spawn the training process, indicating the propagation function model:
- **0** to NOT apply convolution
- **1** to apply GCN propagation model(see https://arxiv.org/pdf/1609.02907.pdf).

You must also include the number of epochs you want to train the neural network.

Example:
	```sbt run 1 200```

This sbt command starts the optimization process and executes the inference to the whole graph. As final result the accuracy metric is calculated. You can see the differences between executions with and without graph convolution.

You also can train the neural network in your spark cluster using **spark-submit** indicating the main class and the parameters **ems.gcn.CoraExample [mode] [epochs]** including the Analytics Zoo dependency(see https://analytics-zoo.readthedocs.io/en/latest/doc/UserGuide/scala.html) and the artifact generated with **sbt package**. The sbt package task generates the resulting jar in the directory **target/scala-2.12**.

**IMPORTANT NOTE**

This project applies the convolution in **one Spark Partition**, so in case of submitting the application to a Spark cluster **you must set the number of cores to 1**. This is because in each iteration the convolution is applied to the whole graph and if you use more cores the data will be split across threads and the process will fail. More work about dividing the graph within a Spark cluster for applying the convolution across executors in case of very big graphs will be added in this repo.