package ems.gcn.layers

import com.intel.analytics.bigdl.dllib.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import org.apache.log4j.Logger

import java.io.PrintWriter
import scala.reflect.ClassTag

class LogToFile[T: ClassTag](
  implicit ev: TensorNumeric[T]
) extends AbstractModule[Tensor[T], Tensor[T], T] {

  @transient lazy val logger = Logger.getLogger(getClass)
  import java.io._

  var epoch =  0

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    val inputDimSize = input.size(1)
    epoch = epoch + 1
    logger.info("---> Epoch: " + epoch)
    if(epoch % 200 == 0)
    (1 to inputDimSize).foreach { el =>
      val cad = input(el).toArray().toList.mkString(",")
      val pw = new PrintWriter(new FileOutputStream(
        new File(s"output-layer-${epoch}-identity.csv"),
        true /* append = true */))
      pw.append(cad)
      pw.append("\n")
      pw.close()
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