package ems.gcn

import breeze.linalg.CSCMatrix
import com.intel.analytics.bigdl.dllib.NNContext
import com.intel.analytics.bigdl.dllib.nn.ClassNLLCriterion
import com.intel.analytics.bigdl.dllib.optim.{Adam, Top1Accuracy}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}

case class Element(_c0: String, words: Array[String], label: String)
case class ElementWithIndexAndNumericLabel(element: Element, index: Int, label: Float)
case class Edge(orig: Int, dest: Int)

object CoraExample {
  def buildAdjacencyMatrixFromCoordinates(edges: Array[Edge], nElements: Int) = {
    val builder = new CSCMatrix.Builder[Float](nElements, nElements)
    edges.foreach {
      case Edge(r, c) =>
        builder.add(r, c, 1.0F)
    }
    builder.result
  }

  val logger = Logger.getLogger(getClass)

  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)

  val conf = new SparkConf()
    .setMaster("local[1]")
  val sc = NNContext.initNNContext(conf, appName = "CoraExample")

  implicit val spark: SparkSession =
    SparkSession
      .builder()
      .master("local[1]")
      .appName("GCN")
      .getOrCreate()

  import spark.implicits._

  def main(args: Array[String]): Unit = {

    require(
      args.length == 4,
      "Include propagation mode." +
        "\n 0: to NOT apply GCN" +
        "\n 1: to APPLY GNC" +
        "And the number of epochs" +
        "\n 2: dataset path" +
        "\n 3: edges path"
    )

    val useIdentityAsPropFunction = args(0).toInt == 0

    if (!useIdentityAsPropFunction) logger.info("Training with GCN")

    val maxEpochs = args(1).toInt

    /** Input node and edges files **/
//    val dataset: String = getClass.getClassLoader.getResource("data/cora.content").getFile
//    val edges: String = getClass.getClassLoader.getResource("data/cora.cites").getFile
    val dataset: String = args(2)
    val edges: String = args(3)

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

    import ems.gcn.matrix.Adjacency._
    import com.intel.analytics.bigdl.dllib.tensor.SparseTensorUtils._
    val tensor = if (!useIdentityAsPropFunction) {
      val symAdj = transformToSymmetrical(sparseAdj)
      /* Change the way of creating the adjacency */
      val normalAdj = normalizationSparseFast(symAdj, nodesNumber)
      createSparseTensorFromBreeze[Float](normalAdj, nodesNumber, nodesNumber)
    } else {
      getIdentityMatrix(nodesNumber)
    }

    /** Take 5% of the training samples. We put -1 label to pad this training samples */
    import ems.gcn.datasets.Operations._
    val completeDataset = splitsRDDs(contentWithIndexAndLabel, Array(0, 2708))
    val trainingDatasetWithIndex = splitsRDDs(contentWithIndexAndLabel, Array(0, 140))
    val evaluationDatasetWithIndex = splitsRDDs(contentWithIndexAndLabel, Array(150, 300))
    val testDatasetWithIndex = splitsRDDs(contentWithIndexAndLabel, Array(500, 1500))

    val trainingRDD = createInputDataset(trainingDatasetWithIndex)
    val evaluationRDD = createInputDataset(evaluationDatasetWithIndex)
    val testDatasetRDD = createInputDatasetWithNumeric(testDatasetWithIndex)

    val completeDatasetRDD = createInputDataset(completeDataset)

    val batchSize = trainingDatasetWithIndex.count().toInt

    val model = ems.gcn.model.KerasBased.getModel(0.5, tensor, batchSize, 1432, 16, 7)
    model.compile(new Adam[Float](learningRate = 0.01), new ClassNLLCriterion[Float]())

    model.fit(trainingRDD, batchSize, 1000, shuffleData = false, groupSize = batchSize)

    val res = model.evaluate(completeDatasetRDD, Array(new Top1Accuracy[Float]()), Some(batchSize)).toList
    println("accuracy:", res(0))
    spark.close()
  }

}
