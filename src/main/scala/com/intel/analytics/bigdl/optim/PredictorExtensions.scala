package com.intel.analytics.bigdl.optim

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dataset.{PaddingParam, Sample}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

object PredictorExtensions {
  implicit class AbstractModuleExtendedOps[T: ClassTag](model: Module[T]) extends Serializable {

    def predictWithLabelAndIndex(
        dataSet: RDD[(Int, Sample[T])],
        batchSize: Int = -1,
        shareBuffer: Boolean = false,
        batchPerPartition: Int,
        featurePaddingParam: Option[PaddingParam[T]]
    )(implicit ev: TensorNumeric[T]) = {
      val partitionNum = dataSet.partitions.length

      println("Number of partitions: " + partitionNum)
      val totalBatch = if (batchSize > 0) {
        require(
          batchSize % partitionNum == 0,
          s"Predictor.predict: total batch size $batchSize " +
            s"should be divided by partitionNum ${partitionNum}"
        )
        batchSize
      } else {
        batchPerPartition * partitionNum
      }

      println("Total batch: " + totalBatch)

      val indexArray = dataSet.map { case (i, _) => i }.collect()
      val featuresTensor = dataSet.map { case (_, tensor) => tensor.feature() }.collect()
      val labelsTensor = dataSet.map { case (_, tensor) => tensor.label() }.collect()
      val outputTensor = Tensor[T](Array(featuresTensor.length, featuresTensor.head.toArray().length)).zero()

      (0 to (featuresTensor.length - 1)).map(i => {
        outputTensor.update(i + 1, featuresTensor(i))
      })

      val output = model.forward(outputTensor).toTensor
      val elements = output.toTensor.size(1)
      var mutableArr = new ArrayBuffer[Tensor[T]](elements)

      (1 to elements) map { i =>
        mutableArr = mutableArr.+:(output(i))
      }

      (indexArray, mutableArr.toArray zip labelsTensor)
    }
  }
}
