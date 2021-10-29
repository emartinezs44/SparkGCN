package com.intel.analytics.bigdl.tensor

import breeze.linalg.CSCMatrix
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

object SparseTensorUtils {
  def createSparseTensorFromBreeze[T: ClassTag](matrix: CSCMatrix[T], rows: Int, cols: Int)(
      implicit ev: TensorNumeric[T]
  ): SparseTensor[T] = {

    var arrRows = ArrayBuffer.empty[Int]
    val colRows = ArrayBuffer.empty[Int]
    val vBuffer = ArrayBuffer.empty[T]

    // Storage in memory for sorting
    val arr = matrix.activeIterator.toArray.sortBy(_._1._1)

    /*matrix.activeIterator*/
    arr.foreach {
      case ((r, c), v) =>
        arrRows.append(r)
        colRows.append(c)
        vBuffer.append(v)
    }

    val indices = Array(arrRows.toArray, colRows.toArray)
    val sparseTensor = SparseTensor(indices, Storage(vBuffer.toArray), Array(rows, cols), 2)

    sparseTensor
  }
}
