package ems.gcn.model

import com.intel.analytics.bigdl.dllib.keras.Sequential
import com.intel.analytics.bigdl.dllib.keras.layers.{Dense, Dropout, GraphConvolutionK}
import com.intel.analytics.bigdl.dllib.optim.L2Regularizer
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.Shape

object KerasBased {
  def getModel(
    dropout: Double,
    matrix: Tensor[Float],
    batchSize: Int,
    inputSize: Int,
    intermediateSize: Int,
    labelsNumber: Int
  ): Sequential[Float] = {
    val model = Sequential[Float]()
    model.add(Dropout[Float](dropout, inputShape = Shape(1432)))
    model.add(GraphConvolutionK[Float](matrix, batchSize, inputSize))
    model.add(Dense[Float](intermediateSize, activation = "relu", wRegularizer = L2Regularizer(5e-4)).setName("layer-1"))
    model.add(GraphConvolutionK[Float](matrix, batchSize, intermediateSize))
    model.add(Dense[Float](labelsNumber, activation = "log_softmax").setName("layer-2"))

    model
  }

}
