## Graph Convolutional Networks in Apache Spark

Implementation of the GCN(https://arxiv.org/abs/1609.02907) on top of Spark using Analytics Zoo. It is based on the Keras repo https://github.com/tkipf/keras-gcn.

It is implemented for the moment only the Cora example. The main program included in the CoraExample module starts a training process using only 140 labeled samples from the Cora dataset.

You can read more info about the experiment in https://emartinezs44.medium.com/graph-convolutions-networks-ad8295b3ce57.

**Execution**

You can use SBT version >= 1.0(https://www.scala-sbt.org/download.html) to execute the example, indicating the propagation function model to apply, **0** to NOT apply convolution and **1** to apply GCN propagation model(see https://arxiv.org/pdf/1609.02907.pdf) and the number of epochs.

Example:
	```sbt run 1 200```

You can also create a jar using **sbt package**. The data is included in the jar sou you can test the model in your spark cluster using **spark-submit** indicating as the main class and the parameters **ems.gcn.CoraExample [mode] [epochs]**.
