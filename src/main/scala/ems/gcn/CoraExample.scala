package ems.gcn

import breeze.linalg.CSCMatrix
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, Sequential}
import com.intel.analytics.bigdl.optim.{Adam, CustomOptimizer, Top1Accuracy, Trigger, ValidationMethod, ValidationResult}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Engine
import ems.gcn.model.TorchBased
import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}


case class Element(_c0: String, words: Array[String], label: String)
case class ElementWithIndexAndNumericLabel(element: Element, index: Int, label: Float)
case class Edge(orig: Int, dest: Int)

import ems.gcn.matrix.Adjacency._
import ems.gcn.datasets.Operations._

import com.intel.analytics.bigdl.tensor.SparseTensorUtils._

object CoraExample {

  def buildAdjacencyMatrixFromCoordinates(edges: Array[Edge], nElements: Int) = {
    val builder = new CSCMatrix.Builder[Float](nElements, nElements)
    edges.foreach { case Edge(r, c) =>
      builder.add(r, c, 1.0F)
    }
    builder.result
  }

  val logger = Logger.getLogger(getClass)

  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)

  def getOptimizer(
      modelIn: Sequential[Float],
      train: RDD[Sample[Float]],
      batchSize: Int,
      maxEpochs: Int,
      checkpointPath: String,
      weights: String,
      optimizer: String,
      learningRate: Float
  ) =
    CustomOptimizer(
      model = modelIn,
      sampleRDD = train,
      criterion = {
        if (weights.isEmpty) {
          new ClassNLLCriterion[Float]()
        } else {
          val classesArray = weights.split(",")
          // Weights added.
          val classesTensor = Tensor(classesArray.map(_.toFloat), Array(classesArray.size))
          new ClassNLLCriterion[Float](classesTensor)
        }
      },
      batchSize = batchSize
    ).setOptimMethod(new Adam(learningRate = learningRate))
      .setEndWhen(Trigger.maxEpoch(maxEpochs)) //.setCheckpoint(checkpointPath, Trigger.everyEpoch)

  private def extractMetricValue(in: Map[(ValidationResult, ValidationMethod[Float]), Int]): Float = {
    in.map {
      case ((valResult, valMethod), _) =>
        valResult.result() match {
          case (v, i) => v
        }
      case _ => 0F
    }.head
  }

  val conf = Engine.createSparkConf()

  implicit val spark: SparkSession =
    SparkSession
      .builder()
      .config(conf)
      .master("local[1]")
      .appName("GCN")
      .getOrCreate()

  import spark.implicits._

  val dataset: String = getClass.getResource("/data/cora.content").getPath
  val edges: String = getClass.getResource("/data/cora.cites").getPath

  val contentDF: DataFrame = spark.read.csv(dataset)
  val contentRDD: RDD[Element] = contentDF.as[String].rdd.map { str =>
    val el = str.split("\t")
    Element(el(0), el.slice(1, 1433), el(1434))
  }

  val edgesDF: DataFrame =
    spark.read
      .option("delimiter", "\t")
      .csv(edges)

  val nodesNumber = contentDF.count().toInt

  val useIdentityAsPropFunction = false

  def main(args: Array[String]): Unit = {

    /** It uses only one partition */
    val content = contentRDD.coalesce(1)

    val edgesUnordered = edgesDF.rdd
      .coalesce(1)
      .map(row => Edge(row.getAs[String](0).toInt, row.getAs[String](1).toInt))

    val contentWithNumeric = content.zipWithIndex()

    /** Extract labels set **/
    val labelsSet = contentWithNumeric.map { case (element, _) => element.label }.collect().distinct

    val contentWithIndexAndLabel = contentWithNumeric.map {
      case (element, index) => ElementWithIndexAndNumericLabel(element, index.toInt, labelsSet.indexOf(element.label) + 1)
    }

    val idx = contentWithIndexAndLabel.map(el => (el.element._c0.toInt -> el.index)).collect().toMap
    val edgesMap = edgesUnordered.map(edge => Edge(idx(edge.orig), idx(edge.dest))).collect()

    val sparseAdj = buildAdjacencyMatrixFromCoordinates(edgesMap, nodesNumber)

    val tensor = if (!useIdentityAsPropFunction) {
      val symAdj = transformToSymmetrical(sparseAdj)
      /* Change the way of creating the adjacency */
      logger.info("Normalized matrix")
      val normalAdj = normalizationSparseFast(symAdj, nodesNumber)
      createSparseTensorFromBreeze[Float](normalAdj, nodesNumber, nodesNumber)
    } else {
      getIdentityMatrix(nodesNumber)
    }

    /** Take 5% of the training samples. We put -1 label to pad this training samples */
    val completeDataset = splitsRDDs(contentWithIndexAndLabel, Array(0, 2708))
    val trainingDatasetWithIndex = splitsRDDs(contentWithIndexAndLabel, Array(0, 140))
    val evaluationDatasetWithIndex = splitsRDDs(contentWithIndexAndLabel, Array(150, 300))
    val testDatasetWithIndex = splitsRDDs(contentWithIndexAndLabel, Array(500, 1500))

    val trainingRDD = createInputDataset(trainingDatasetWithIndex)
    val evaluationRDD = createInputDataset(evaluationDatasetWithIndex)
    val testDatasetRDD = createInputDatasetWithNumeric(testDatasetWithIndex)

    val completeDatasetRDD = createInputDataset(completeDataset)

    val batchSize = trainingDatasetWithIndex.count().toInt
    val model = TorchBased.getModel(0.5, tensor, batchSize, 1432, 16, 7)

    Engine.init

    val optimizer = CustomOptimizer[Float](
      model = model,
      sampleRDD = trainingRDD,
      criterion = {
        new ClassNLLCriterion[Float]()
      },
      batchSize = batchSize
    ).setOptimMethod(new Adam(learningRate = 0.01)).setEndWhen(Trigger.maxEpoch(1000))

    val modelTrained = optimizer.optimize()

    /** At the moment we calculate the metrics doing inference to the whole dataset **/
    println(modelTrained.evaluate(completeDatasetRDD, Array(new Top1Accuracy[Float]()), Some(batchSize)).toList)

    spark.close()
  }

}

