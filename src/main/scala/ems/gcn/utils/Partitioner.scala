package ems.gcn.utils

import org.apache.log4j.Logger
import org.apache.spark.Partitioner

object Partitioner {

  class ExactPartitionerPerKey(override val numPartitions: Int) extends Partitioner {
    def getPartition(key: Any): Int = {
      val k0 = key.asInstanceOf[Int]
      return k0
    }
  }

  class ExactPartitioner(override val numPartitions: Int, elements: Int) extends Partitioner {
    /** This creates partitions depending of a Int column from 0 to n elements **/
    def getPartition(key: Any): Int = {
      val k = key.asInstanceOf[Int]
      // `k` is assumed to go continuously from 0 to elements-1.
      return k * numPartitions / elements
    }
  }
}

object UtilFunctions {
  val logger = Logger.getLogger(getClass)
  def time[R](block: => R)(msg: String): R = {
    val t0 = System.currentTimeMillis()
    val result = block // call-by-name
    val t1 = System.currentTimeMillis()
    logger.info("Elapsed time: " + (t1 - t0) + "ms in " + msg)
    result
  }
}
