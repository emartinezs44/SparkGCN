package ems.gcn.datasets

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.tensor.Tensor
import ems.gcn.CoraExample.spark
import ems.gcn.ElementWithIndexAndNumericLabel
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{asc, col}
import org.apache.spark.sql.types.IntegerType

import scala.collection.mutable

object Operations {

  private[gcn] def splitsRDDs(elements: RDD[ElementWithIndexAndNumericLabel], interval: Array[Int]) = {
    elements.map { case ElementWithIndexAndNumericLabel(element, index, label) =>
      if (index >= interval(0) && index <= interval(1)) {
        ElementWithIndexAndNumericLabel(element, index, label)
      } else {
        ElementWithIndexAndNumericLabel(element, index, -1)
      }
    }
  }

  private[gcn] def createInputDataset(rdd: RDD[ElementWithIndexAndNumericLabel]) : RDD[Sample[Float]] = {
    rdd.map { case ElementWithIndexAndNumericLabel(element, _, label) =>
      val elementsSum = element.words.map(_.toFloat).sum
      Sample(
        Tensor(element.words.map(el => el.toFloat / elementsSum), Array(1432)),
        Tensor(Array(label), Array(1))
      )
    }
  }

  private[gcn] def createInputDatasetWithNumeric(rdd: RDD[ElementWithIndexAndNumericLabel]): RDD[(Int, Sample[Float])] = {
    rdd.map { case ElementWithIndexAndNumericLabel(element, index, label) =>
      val elementsSum = element.words.map(_.toFloat).sum
      (index, Sample(
        Tensor(element.words.map(el => el.toFloat / elementsSum), Array(1432)),
        Tensor(Array(label), Array(1))
      ))
    }
  }

  private[gcn] def splitDatasets(inputDataframe: DataFrame, interval: (Int, Int))(implicit sparkSession: SparkSession) = {
    import spark.implicits._
    inputDataframe
      .map(row => {
        val index = row.getAs[Int]("numeric")
        val label = {
          if (index >= interval._1 && index <= interval._2)
            row.getAs[Float]("labelNumeric")
          else
            -1.0F
        }

        (
          row.getAs[mutable.WrappedArray[String]]("words"),
          row.getAs[String]("label"),
          row.getAs[Int]("node"),
          label,
          row.getAs[Float]("originalLabel"),
          row.getAs[Int]("numeric")
        )
      })
      .toDF("words", "label", "node", "labelNumeric", "originalLabel", "numeric")
  }

  private[gcn] def sortDataframe(in: DataFrame)(implicit spark: SparkSession): DataFrame = {
    in.withColumn(
      "node",
      col("_c0").cast(IntegerType)
    )
      .sort(asc("node"))
  }

}
