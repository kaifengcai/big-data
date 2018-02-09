import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.rdd.RDD
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by Mayanka on 14-Jul-15.
  */
object jinm {

  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setAppName("SparkWordCount").setMaster("local[*]")
    val sc=new SparkContext(sparkConf)
    val data = sc.textFile("data.txt")
    val parsedData = data.map(line => {line.split(',')}).map(line=>(line(0).toDouble,line(1).toDouble))
    //val parseLineToTuple: String => Array[(Double,Double)] = s => s=> s.split(',').map(fields => (fields(2).toDouble,fields(3).toDouble))

    val metrics = new MulticlassMetrics(parsedData)
    val cfMatrix = metrics.confusionMatrix
    println(" |=================== Confusion matrix ==========================")
    println(cfMatrix)
    println(metrics.fMeasure)
    println(parsedData)
    /*val sparkConf = new SparkConf().setAppName("SparkWordCount").setMaster("local[*]")
    val sc=new SparkContext(sparkConf)
    val data = sc.textFile("data.txt")*/


    printf(
      s"""

         |=================== Confusion matrix ==========================
         |          | %-15s                     %-15s
         |----------+----------------------------------------------------
         |Actual = 0| %-15f                     %-15f
         |Actual = 1| %-15f                     %-15f
         |===============================================================
         """.stripMargin, "Predicted = 0", "Predicted = 1",
      cfMatrix.apply(0, 0), cfMatrix.apply(0, 1), cfMatrix.apply(1, 0), cfMatrix.apply(1, 1))

    println("\nACCURACY " + ((cfMatrix(0,0) + cfMatrix(1,1))/(cfMatrix(0,0) + cfMatrix(0,1) + cfMatrix(1,0) + cfMatrix(1,1))))
    println("\nMisclassification_Rate " + ((cfMatrix(0,1) + cfMatrix(1,0))/(cfMatrix(0,0) + cfMatrix(0,1) + cfMatrix(1,0) + cfMatrix(1,1))))
    println("\nTrue_Positive_Rate " + ((cfMatrix(0,0))/(cfMatrix(0,0) + cfMatrix(0,1))))
    println("\nFalse_Positive_Rate " + ((cfMatrix(0,1))/(cfMatrix(1,0) + cfMatrix(1,1))))
    println("\nSpecificity" + ((cfMatrix(1,1))/(cfMatrix(1,0) + cfMatrix(1,1))))
    println("\nPrecision" +(cfMatrix(0,0))/(cfMatrix(0,0)+(cfMatrix(0,1))))
    println("\nPrevalence" + (cfMatrix(0,0) + cfMatrix(0,1))/(cfMatrix(0,0) + cfMatrix(0,1) + cfMatrix(1,0) + cfMatrix(1,1)))




  }
}
