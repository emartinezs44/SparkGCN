package com.intel.analytics.bigdl.optim

import com.intel.analytics.bigdl.{Criterion, Module}
import com.intel.analytics.bigdl.dataset.{DatasetUtils, _}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

object CustomOptimizer {
  def apply[T: ClassTag](
      model: Module[T],
      sampleRDD: RDD[Sample[T]],
      criterion: Criterion[T],
      batchSize: Int,
      featurePaddingParam: PaddingParam[T] = null,
      labelPaddingParam: PaddingParam[T] = null
  )(implicit ev: TensorNumeric[T]): Optimizer[T, MiniBatch[T]] = {

    val _featurePaddingParam = if (featurePaddingParam != null) Some(featurePaddingParam) else None
    val _labelPaddingParam = if (labelPaddingParam != null) Some(labelPaddingParam) else None

    new DistriOptimizer[T](
      _model = model,
      _dataset = (DatasetUtils.rdd(sampleRDD, batchSize) ->
        SampleToMiniBatch(batchSize, _featurePaddingParam, _labelPaddingParam))
        .asInstanceOf[DistributedDataSet[MiniBatch[T]]],
      _criterion = criterion
    ).asInstanceOf[Optimizer[T, MiniBatch[T]]]
  }
}


