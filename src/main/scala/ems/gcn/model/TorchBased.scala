package ems.gcn.model

import com.intel.analytics.bigdl.nn.{Dropout, Linear, LogSoftMax, ReLU, Sequential, SoftMax}
import com.intel.analytics.bigdl.optim.L2Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import ems.gcn.layers.{GraphConvolution, LogToFile}

object TorchBased {
  def getModel(
      dropout: Double,
      matrix: Tensor[Float],
      batchSize: Int,
      inputSize: Int,
      intermediateSize: Int,
      labelsNumber: Int
  ): Sequential[Float] = {

    Sequential[Float]()
      .add(Dropout(dropout))
      .add(GraphConvolution[Float](matrix, batchSize, inputSize))
      .add(Linear[Float](inputSize, intermediateSize, wRegularizer = L2Regularizer(5e-4)).setName("layer-1"))
      .add(ReLU())
      .add(Dropout(dropout))
      .add(GraphConvolution[Float](matrix, batchSize, intermediateSize))
      .add(Linear[Float](intermediateSize, labelsNumber).setName("layer-2"))
      .add(LogSoftMax())
  }
}
