package ems.gcn.layers

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.keras.KerasLayer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils

import scala.reflect.ClassTag

/** Keras Analytics Zoo Layer based layer.
  * The graph convolutional operator from the
  * “Semi-supervised Classification with Graph Convolutional Networks” paper
  * Emiliano Martinez.
  */
class GCNConv[T: ClassTag](adjacencyMatrix: Tensor[T], var inputShape: Shape = null)(implicit ev: TensorNumeric[T])
    extends KerasLayer[Activity, Activity, T](KerasUtils.addBatch(inputShape)) with Net {
  override def doBuild(inputShape: Shape): AbstractModule[Activity, Activity, T] = {
  ???
  }
}
