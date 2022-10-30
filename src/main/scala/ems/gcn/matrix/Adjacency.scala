package ems.gcn.matrix

import breeze.linalg
import breeze.linalg.{CSCMatrix, SparseVector, kron}
import breeze.numerics.pow
import com.intel.analytics.bigdl.dllib.tensor.SparseTensorUtils
import ems.gcn.utils.UtilFunctions.time
import org.apache.log4j.Logger

object Adjacency {

  val logger = Logger.getLogger(getClass)

  private[gcn] def spdiagFast(a: SparseVector[Float]): CSCMatrix[Float] = {
    val size = a.size
    val result = new linalg.CSCMatrix.Builder[Float](size, size)

    var i = 0
    while (i < size) {
      result.add(i, i, a(i))
      i += 1
    }

    result.result
  }

  private[gcn] def getIdentityMatrix(nodesNumber: Int) = {
    logger.info("Using identity matrix as adjacency")
    val builder_sp = new CSCMatrix.Builder[Float](nodesNumber, nodesNumber)
    for (i <- 0 to (nodesNumber - 1)) {
      builder_sp.add(i, i, 1)
    }

    val eye = builder_sp.result
    SparseTensorUtils.createSparseTensorFromBreeze(
      eye,
      nodesNumber,
      nodesNumber
    )
  }

  private[gcn] def normalizationSparseFast(
      adj: CSCMatrix[Float],
      nElements: Int
  ): CSCMatrix[Float] = {

    val builder_sp = new CSCMatrix.Builder[Float](nElements, nElements)

    for (i <- 0 to (nElements - 1)) {
      builder_sp.add(i, i, 1)
    }

    val eye = builder_sp.result
    val T = adj + eye

    val sumVector = SparseVector.zeros[Float](nElements)

    time {
      logger.info("Creating degree vector ")
      T.activeIterator.foreach { case ((row, col), valu) =>
        sumVector(row) += valu
      }
    }("Creating degree vector")

    val T2 = time {
      logger.info("Pow vector operation")
      pow(sumVector, -0.5f)
    }("Pow vector operation")

    val T3 =
      time {
        logger.info("Creating diag.matrix")
        /* Changed spdiaf for Builder */
        spdiagFast(T2)
      }("Creating diag.matrix Fast")

    val T4 = {
      time {
        logger.info("Doing Normalization")
        (T * T3).t * T3
      }("Doing Normalization")
    }

    T4
  }

  private[gcn] def transformToSymmetrical(
      sparseAdj: CSCMatrix[Float]
  ): CSCMatrix[Float] = {
    val r = time {
      sparseAdj +:+ (sparseAdj.t *:* (sparseAdj.t >:> sparseAdj)
        .map(el =>
          if (el) 1.0f else 0.0f
        )) - (sparseAdj *:* (sparseAdj.t >:> sparseAdj)
        .map(el => if (el) 1.0f else 0.0f))
    }("Transform to symmetric")

    r
  }

  private[gcn] def buildAdjacencySparseFast(
      m: Map[Int, Array[(Int, Int)]],
      nElements: Int
  ): CSCMatrix[Float] = {
    val index = 0 to nElements
    val result_map = time {
      logger.info("Searching in Map")
      index.map(el => (el, m.getOrElse(el, Array())))
    }("Searching in Map")

    val builder = new CSCMatrix.Builder[Float](nElements, nElements)

    result_map.foreach {
      case (_, list_positions) => {
        list_positions.foreach {
          // add a factor x10
          case (r, c) => {
            builder.add(r, c, 1.0f)
          }
        }
      }
    }

    builder.result
  }

}
