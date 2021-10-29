package com.intel.analytics.bigdl.dataset

import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.Logger
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

/**
 * Here we force to not to shuffle the data in order to conserve partitions content.
 * This is not available in BigDL.
 */
object DatasetUtils {
  val logger = Logger.getLogger(getClass)
  def rdd[T: ClassTag](data: RDD[T], batchSize: Int): DistributedDataSet[T] = {
    val nodeNumber = Engine.nodeNumber()
    new CachedDistriDataSet[T](
      data.coalesce(nodeNumber, false)
        .mapPartitions(iter => {
          Iterator.single(iter.toArray)
        }).setName("cached dataset")
        .cache()
      ,true, batchSize)
  }
}
