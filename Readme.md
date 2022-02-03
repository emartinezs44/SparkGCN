## Graph Convolutional Networks in Apache Spark

Implementation of the GCN(https://arxiv.org/abs/1609.02907) on top of Spark using Analytics Zoo. It is based on the Keras repo https://github.com/tkipf/keras-gcn.

It is implemented for the moment only the Cora example. The main program included in the CoraExample module starts a training process using only 140 labeled samples from the Cora dataset.

You can read more info about the experiment in https://emartinezs44.medium.com/graph-convolutions-networks-ad8295b3ce57.

### Execution

You can use SBT version >= 1.0(https://www.scala-sbt.org/download.html) to spawn the training process, indicating the propagation function model to apply: **0** to NOT apply convolution and **1** to apply GCN propagation model(see https://arxiv.org/pdf/1609.02907.pdf) and the number of epochs.

Example:
	```sbt run 1 200```

You can create a jar using **sbt package**. The data will be included in the jar.

You also can train the neural network in your spark cluster using **spark-submit** indicating the main class and the parameters **ems.gcn.CoraExample [mode] [epochs]** and including the Analytics Zoo dependency. See https://analytics-zoo.readthedocs.io/en/latest/doc/UserGuide/scala.html.

**IMPORTANT NOTE**
This project applies the convolution in **one Spark Partition**, so in case of submit the application to your Spark cluster **you must set the number of cores equal to 1**. This is because in each iteration the convolution is applied to the whole graph and if you select more cores the data will be splitted across threads and the process will fail. More work about splitting the graph across the cluster for applying the convolution in different machines in case of very big graphs has been done and will be released in this repo soon.