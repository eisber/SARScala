package gramhagen.sarscala

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.param.shared.HasPredictionCol
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{functions => f}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, Encoder, Encoders, KeyValueGroupedDataset, Row}
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.api.java.function.MapFunction
import org.apache.spark.api.java.function.FlatMapGroupsFunction
import scala.reflect.ClassTag
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable.{ArrayBuilder, HashSet, ListBuffer, PriorityQueue, HashMap}
import util.control.Breaks._
import gramhagen.sarscala.SARExtensions._

case class ItemScore(idx:Int, score: Float) extends Ordered[ItemScore] {
  // Required as of Scala 2.11 for reasons unknown - the companion to Ordered
  // should already be in implicit scope
  import scala.math.Ordered.orderingToOrdered

  def compare(that: ItemScore): Int = -(this.score compare that.score)
}

/**
  * Common params for SARScala Model.
  */
trait SARScalaModelParams extends Params with HasPredictionCol {

  /** @group param */
  val userCol = new Param[String](this, "userCol", "column name for user ids, all ids must be integers")

  /** @group param */
  def getUserCol: String = $(userCol)

  /** @group param */
  val itemCol = new Param[String](this, "itemCol", "column name for item ids, all ids must be integers")

  /** @group param */
  def getItemCol: String = $(itemCol)

  /** @group param */
  val ratingCol = new Param[String](this, "ratingCol", "column name for ratings")

  /** @group param */
  def getRatingCol: String = $(ratingCol)

  /** @group param */
  val timeCol = new Param[String](this, "timeCol", "column name for timestamps, all timestamps must be longs")

  /** @group param */
  def getTimeCol: String = $(timeCol)

  /** @group param */
  val topK = new Param[Int](this, "topK", "return top-k recommendations per user")

  /** @group param */
  def getTopK: Int = $(topK)

}

/**
  * Common parameters for SARScala algorithm
  */
trait SARScalaParams extends SARScalaModelParams {

  /** @group param */
  val timeDecay = new Param[Boolean](this, "timeDecay", "flag to enable time decay on ratings")

  /** @group param */
  def getTimeDecay: Boolean = $(timeDecay)

  /** @group param */
  val decayCoefficient = new Param[Double](this, "decayCoefficient", "time decay coefficient, number of days for rating to decay by half")

  /** @group param */
  def getDecayCoefficient: Double = $(decayCoefficient)

  /** @group param */
  val similarityMetric = new Param[String](this, "similarityMetric", "metric to use for item-item similarity, must one of: [ cooccur | jaccard | lift ]]")

  /** @group param */
  def getSimilarityMetric: String = $(similarityMetric)

  /** @group param */
  val countThreshold = new Param[Int](this, "countThreshold", "ignore item co-occurrences that are less than the threshold")

  /** @group param */
  def getCountThreshold: Int = $(countThreshold)
}

trait SARTransformer {
  def transform(dataset:DataFrame, topK:Int): DataFrame
}
@SerialVersionUID(1L)
class SARScalaModelInternal[T:ClassTag](
    itemOffsets:Array[Int],
    itemIds:Array[Int], itemValues:Array[Float], itemMapping:Map[T, Int]) 
    extends SARTransformer with Serializable {

  def transform(df: DataFrame, topK: Int): DataFrame = {
    def getUserRatings(userItems:Iterator[Row]): Array[ItemScore] = 
      userItems.map(r => {
          val itemId = r.getAs[T](1)
          val idx = itemMapping.apply(itemId)

          ItemScore(idx, r.getFloat(2))
        })
        .toArray
        .sortWith(_.idx < _.idx)
 
    // it is slower for single partitions, but should provide speed improvement with increasing number of partitions
    // https://umbertogriffo.gitbooks.io/apache-spark-best-practices-and-tuning/content/when_to_use_broadcast_variable.html
    val itemOffsetsBc = df.sqlContext.sparkContext.broadcast(itemOffsets)
    val itemIdsBc = df.sqlContext.sparkContext.broadcast(itemIds)
    val itemValuesBc = df.sqlContext.sparkContext.broadcast(itemValues)

    import df.sqlContext.implicits._

    val schema = StructType(
           StructField("u1", StringType, nullable = false) ::
           StructField("i1", StringType, nullable = false) :: 
           StructField("score", FloatType, nullable = false) :: Nil)

    implicit val encoder = RowEncoder(schema)

    def flatMapGroups[K](gdf:KeyValueGroupedDataset[K, Row]):DataFrame =
      gdf.flatMapGroups((u1, rowsForEach:Iterator[Row]) =>  {
           new SARScalaPredictor(itemMapping, itemOffsetsBc.value, itemIdsBc.value, itemValuesBc.value)
             .predict(u1, getUserRatings(rowsForEach), topK)
            })
        .toDF

    df.schema(0).dataType match {
      case DataTypes.LongType => flatMapGroups(df.groupByKey(r => r.getAs[Long](0)))
      case DataTypes.IntegerType => flatMapGroups(df.groupByKey(r => r.getAs[Int](0)))
      case DataTypes.StringType => flatMapGroups(df.groupByKey(r => r.getAs[String](0)))
      case _ => throw new IllegalArgumentException("DataType not supported");
    }
  }
}

/**
  * Model fitted by SAR
  *
  * @param itemSimilarity item similarity matrix
  * @param processedRatings user affinity matrix
  */
class SARScalaModel (
  override val uid: String,
  @transient val processedRatings: DataFrame,
  @transient val itemSimilarity: DataFrame,
  val transformer:SARTransformer)
  extends Model[SARScalaModel] with SARScalaModelParams with MLWritable {

  /** @group setParam */
  def setUserCol(value: String): this.type = set(userCol, value)

  /** @group setParam */
  def setItemCol(value: String): this.type = set(itemCol, value)

  /** @group setParam */
  def setRatingCol(value: String): this.type = set(ratingCol, value)

  /** @group setParam */
  def setTimeCol(value: String): this.type = set(timeCol, value)

  /** @group setParam */
  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  /** @group setParam */
  def setTopK(value: Int): this.type = set(topK, value)

  def getUserAffinity(test_df: Dataset[_]): Dataset[_] = 
    test_df.filter(col($(ratingCol)) > 0)
      .select(col($(userCol)))
      .distinct()
      .join(processedRatings, Seq($(userCol)))
      .select(col($(userCol)), col($(itemCol)), col($(ratingCol)).as("value"))
      .repartition(col($(userCol)), col($(itemCol)))
      .sortWithinPartitions()

  override def transform(dataset: Dataset[_]): DataFrame = {

    transformSchema(dataset.schema)

    // filter processed ratings down to selected users
    val df = processedRatings
      .join(dataset.select($(userCol)).distinct(), $(userCol))
      // .join(itemMapping, processedRatings.col($(itemCol)) <=> itemMapping.col("i2"))
      .select(
        col($(userCol)),
        col($(itemCol)),
        col($(ratingCol)).cast(FloatType))

    transformer.transform(df, $(topK))
  }

  override def transformSchema(schema: StructType): StructType = {
    // append prediction column
    StructType(schema.fields :+ StructField($(predictionCol), IntegerType, nullable = false))
  }

  override def copy(extra: ParamMap): SARScalaModel = {
    new SARScalaModel(uid, processedRatings, itemSimilarity, transformer)
      .setUserCol(getUserCol)
      .setItemCol(getItemCol)
      .setRatingCol(getRatingCol)
      .setPredictionCol(getPredictionCol)
  }

  override def write: MLWriter = new SARScalaModel.SARScalaModelWriter(this)
}

object SARScalaModel extends MLReadable[SARScalaModel] {

  override def read: MLReader[SARScalaModel] = new SARScalaModelReader

  override def load(path: String): SARScalaModel = super.load(path)

  private[SARScalaModel] class SARScalaModelWriter(instance: SARScalaModel) extends MLWriter {

     override protected def saveImpl(path: String): Unit = {

       val metadataPath = new Path(path, "metadata").toString

       val metadata = Seq(Row(
         instance.uid,
         instance.userCol,
         instance.itemCol,
         instance.ratingCol,
         instance.timeCol,
         instance.predictionCol))

       val schema = Seq(
           StructField("uid", StringType, nullable = false),
           StructField("userCol", StringType, nullable = false),
           StructField("itemCol", StringType, nullable = false),
           StructField("ratingCol", StringType, nullable = false),
           StructField("timeCol", StringType, nullable = true),
           StructField("predictionCol", StringType, nullable = true))

       sparkSession.createDataFrame(sparkSession.sparkContext.parallelize(metadata), StructType(schema))
         .write.format("parquet").save(metadataPath)

       // TODO: write arrays 
       // val itemSimilarityPath = new Path(path, "itemSimilarity").toString
       // instance.itemSimilarity.write.format("parquet").save(itemSimilarityPath)

       val processedRatingsPath = new Path(path, "processedRatings").toString
       instance.processedRatings.write.format("parquet").save(processedRatingsPath)
    }
  }

  private class SARScalaModelReader extends MLReader[SARScalaModel] {

    override def load(path: String): SARScalaModel = {
      val metadataPath = new Path(path, "metadata").toString
      val metadata = sparkSession.read.format("parquet").load(metadataPath).first()

      // val itemSimilarityPath = new Path(path, "itemSimilarity").toString
      // val itemSimilarity = sparkSession.read.format("parquet").load(itemSimilarityPath)

      val processedRatingsPath = new Path(path, "processedRatings").toString
      val processedRatings = sparkSession.read.format("parquet").load(processedRatingsPath)

      /*
      new SARScalaModel(metadata.getAs[String]("uid"), itemSimilarity, processedRatings)
        .setUserCol(metadata.getAs[String]("userCol"))
        .setItemCol(metadata.getAs[String]("itemCol"))
        .setRatingCol(metadata.getAs[String]("ratingCol"))
        .setTimeCol(metadata.getAs[String]("timeCol"))
        .setPredictionCol(metadata.getAs[String]("predictionCol"))*/
        throw new IllegalArgumentException

      val x:SARScalaModel = null
      x
    }
  }
}

class SARScala (override val uid: String) extends Estimator[SARScalaModel] with SARScalaParams {

  // set default values
  this.setTimeDecay(false)
  this.setDecayCoefficient(30)
  this.setSimilarityMetric("jaccard")
  this.setCountThreshold(0)

  def this() = this(Identifiable.randomUID("SARScala"))

  /** @group setParam */
  def setUserCol(value: String): this.type = set(userCol, value)

  /** @group setParam */
  def setItemCol(value: String): this.type = set(itemCol, value)

  /** @group setParam */
  def setRatingCol(value: String): this.type = set(ratingCol, value)

  /** @group setParam */
  def setTimeCol(value: String): this.type = set(timeCol, value)

  /** @group setParam */
  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  /** @group setParam */
  def setTimeDecay(value: Boolean): this.type = set(timeDecay, value)

  /** @group setParam */
  def setDecayCoefficient(value: Double): this.type = set(decayCoefficient, value)

  /** @group setParam */
  def setSimilarityMetric(value: String): this.type = set(similarityMetric, value.toLowerCase)

  /** @group setParam */
  def setCountThreshold(value: Int): this.type = set(countThreshold, value)

  def getItemCoOccurrence[U,I](df: Dataset[_]): Dataset[_] = {
    df.select(col($(userCol)).as("u1"), col($(itemCol)).as("i1"))
      .join(df.select(col($(userCol)).as("u2"), col($(itemCol)).as("i2")),
        col("u1") <=> col("u2") && // remove nulls with <=>
        col("i1") <= col("i2"))
  /* // this code should avoid the join, but it seems that shuffle to the next stage hurts much more
    import df.sqlContext.implicits._

    df.groupByKey(r => r.getAs[U](0))
      .flatMapGroups((u, rows) => {
        rows.flatMap(y => { 
          val i1 = y.getAs[I](1)
              
          case class OutputTuple[U,V](i1:U, i2:V)
          rows
            .filter(x => i1 <= x.getAs[I](1))
            .map(x => OutputTuple(i1,x.getAs[I](1))) 
        })
      })        
      */
      .groupBy(col("i1"), col("i2"))
      .count()
      .filter(col("count") >= $(countThreshold))
      .repartition(col("i1"), col("i2"))
      .sortWithinPartitions()
      .persist(StorageLevel.MEMORY_ONLY)
  }

  def getItemSimilarity(df: Dataset[_]): DataFrame = {

    // count each item occurrence
    val itemCount = df.filter(col("i1") === col("i2"))
    // println("itemCount")
    // itemCount.show()

    // append marginal occurrence counts for each item
    val itemMarginal = df
      .join(itemCount.select(col("i1"), itemCount.col("count").as("i1_count")), "i1")
      .join(itemCount.select(col("i2"), itemCount.col("count").as("i2_count")), "i2")

    // itemMarginal.show()

    // compute upper triangular of the item-item similarity matrix using desired metric between items
    val upperTriangular = $(similarityMetric) match {
      case "cooccurrence" =>
          df.select(col("i1"), col("i2"), col("count").as("value"))
      case "jaccard" => 
        itemMarginal.select(col("i1"), col("i2"),
          (col("count") / (col("i1_count") + col("i2_count") - col("count"))).as("value"))
      case "lift" =>
        itemMarginal.select(col("i1"), col("i2"),
          (col("count") / (col("i1_count") * col("i2_count"))).as("value"))
      case _ =>
        throw new IllegalArgumentException("unsupported similarity metric")
    }
    
    upperTriangular.persist(StorageLevel.MEMORY_ONLY)

    // upperTriangular

    // fill in the lower triangular
    // TODO: just return triangular
    upperTriangular.union(
      upperTriangular.filter(col("i1") =!= col("i2"))
        .select(col("i2"), col("i1"), col("value")))
      .select(col("i1"), col("i2"), col("value").cast(FloatType))
      // .repartition(col("i1"))
      // .sortWithinPartitions()
  }

  def getProcessedRatings(df: Dataset[_]): DataFrame = {

    var dfResult:DataFrame = null

    if ($(timeDecay)) {
      val latest = df.select(f.max($(timeCol))).first().get(0)
      val decay = -math.log(2) / ($(decayCoefficient) * 60 * 60 * 24)

      dfResult = df.groupBy($(userCol), $(itemCol))
        .agg(f.sum(col($(ratingCol)) * f.exp(f.lit(decay) * (f.lit(latest) - col($(timeCol))))).as($(ratingCol)))
    } else {
      dfResult = df.select(col($(userCol)), col($(itemCol)), col($(ratingCol)).cast(DoubleType),
        f.row_number().over(Window.partitionBy($(userCol), $(itemCol)).orderBy(f.desc($(timeCol)))).as("latest"))
        .filter(col("latest") === 1)
        .drop("latest")
    }

    // important to sort as the scoring depends on it
    dfResult
        .repartition(col($(userCol)))
        .sortWithinPartitions()
        .persist(StorageLevel.DISK_ONLY)
  }

  def getMappedArrays[T](itemSimilarity:DataFrame, nullElement:T): (Array[Int], Array[Int], Array[Float], Map[T, Int]) = {
    val itemSimilarityLocal = itemSimilarity.orderBy(col("i1"), col("i2")).collect()
    // i1,i2,similarity

    val itemOffsetsBuffer = new ArrayBuilder.ofInt
    val itemIdsBuffer = new ArrayBuilder.ofInt
    val itemValuesBuffer = new ArrayBuilder.ofFloat
    var itemMappingMap = new HashMap[T, Int]()

    var rowNumber = 0
    var offsetCount = 0
    var lastId:T = nullElement

    // build item id to index mapping
    for (row <- itemSimilarityLocal) {
      val i1 = row.getAs[T](0)
      if (lastId != i1)
      {
        itemMappingMap += (i1 -> itemMappingMap.size)
        lastId = i1
      }
    }

    // build the arrays prepared for prediciton
    lastId = nullElement
    for (row <- itemSimilarityLocal) {
      val i1 = row.getAs[T](0)

      // data set is sorted
      if (lastId != i1) {
        itemOffsetsBuffer += rowNumber
        offsetCount += 1

        lastId = i1
      }    

      // store item id indicies and values in parallel array
      itemIdsBuffer += itemMappingMap.apply(row.getAs[T](1))
      itemValuesBuffer += row.getFloat(2)

      rowNumber += 1
    }

    itemOffsetsBuffer += rowNumber

    (itemOffsetsBuffer.result, itemIdsBuffer.result, itemValuesBuffer.result, itemMappingMap.toMap)
  }

/*
  def getMappedArraysSpark: (DataFrame, Array[Int], Array[Int], Array[Float]) = {

    val itemOffsets = 
      itemSimilarity.groupBy("i1")
      .count()
      .orderBy("i1")
      .select(col("count").cast(IntegerType)) // reduce amount transferred
      .collect()
      .map((r: Row) => r.getAs[Int]("count"))
      .scanLeft(0)(_+_) // cumulative sum to get to offsets

    val itemMapping =
      itemSimilarity.select(col("i2"))
      .distinct()
      .select(col("i2"), (f.row_number().over(Window.orderBy("i2")) - 1).as("idx"))
      .repartition(col("i2"))
      .sortWithinPartitions()

    val itemIdsBuffer = new ArrayBuilder.ofInt
    val itemValuesBuffer = new ArrayBuilder.ofFloat
    itemSimilarity.join(itemMapping, "i2")
      .select(col("i1"), col("idx").as("i2"), col("value").cast(FloatType))
      .orderBy("i1")
      .collect()
      .foreach((r: Row) => {
        itemIdsBuffer += r.getAs[Int]("i2")
        itemValuesBuffer += r.getAs[Float]("value")
    })

    (itemMapping,
     itemOffsets,
     itemIdsBuffer.result,
     itemValuesBuffer.result)
  }
*/
  override def fit(dataset: Dataset[_]): SARScalaModel = {

    // apply time decay to ratings if necessary otherwise remove duplicates
    val processedRatings = getProcessedRatings(dataset)
    // count item-item co-occurrence
    val itemCoOccurrence = getItemCoOccurrence(processedRatings)

    // calculate item-item similarity
    val itemSimilarity = getItemSimilarity(itemCoOccurrence)

    val internal = itemSimilarity.schema(0).dataType match {
      case DataTypes.LongType => {
        val (itemOffsets, itemIds, itemValues, itemMapping) = getMappedArrays(itemSimilarity, Long.MaxValue)
        new SARScalaModelInternal[Long](itemOffsets, itemIds, itemValues, itemMapping)
      }
      case DataTypes.IntegerType => {
        val (itemOffsets, itemIds, itemValues, itemMapping) = getMappedArrays(itemSimilarity, Int.MaxValue)
        new SARScalaModelInternal[Int](itemOffsets, itemIds, itemValues, itemMapping)
      }
      case DataTypes.StringType => {
        val (itemOffsets, itemIds, itemValues, itemMapping) = getMappedArrays[String](itemSimilarity, null)
        new SARScalaModelInternal[String](itemOffsets, itemIds, itemValues, itemMapping)
      }
      case _ => throw new IllegalArgumentException("DataType for item column not supported");
    }
 
    new SARScalaModel(uid, processedRatings, itemSimilarity, internal)
      .setUserCol($(userCol))
      .setItemCol($(itemCol))
      .setRatingCol($(ratingCol))
      .setTimeCol($(timeCol))
      .setTopK(10)
  }

  override def transformSchema(schema: StructType): StructType = {
    transformSchema(schema)
  }

  override def copy(extra: ParamMap): SARScala = {
    copy(extra)
      .setTimeDecay(getTimeDecay)
      .setDecayCoefficient(getDecayCoefficient)
      .setSimilarityMetric(getSimilarityMetric)
      .setCountThreshold(getCountThreshold)
  }
}