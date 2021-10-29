package ems.gcn.layers

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import org.apache.log4j.Logger

import java.io.PrintWriter
import scala.reflect.ClassTag

class LogToFile[T: ClassTag](
  implicit ev: TensorNumeric[T]
) extends AbstractModule[Tensor[T], Tensor[T], T] {

  @transient lazy val logger = Logger.getLogger(getClass)
  import java.io._

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    val inputDimSize = input.size(1)
    (1 to inputDimSize).foreach { el =>
      val cad = input(el).toArray().toList.mkString(",")
      /*val pw = new PrintWriter(new FileOutputStream(
        new File("ouput-layer.csv"),
        true /* append = true */))
      pw.append(cad)
      pw.close()*/
    }
    output.resizeAs(input)
    output = input
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(gradOutput).copy(gradOutput)
    gradInput
  }
}


object LogToFile {
  def apply[@specialized(Float, Double) T: ClassTag](
    implicit ev: TensorNumeric[T]
  ): LogToFile[T] = {
    new LogToFile[T]
  }
}