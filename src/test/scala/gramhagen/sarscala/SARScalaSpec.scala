package gramhagen.sarscala

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.scalatest.{Outcome, fixture}
import scala.collection.immutable.Range


class SARScalaSpec extends fixture.FlatSpec {

  val ATOL = 1e-7
  val TEST_USER = "0003000098E85347"

  case class FixtureParam(data: Map[String, DataFrame])

  def withFixture(test: OneArgTest): Outcome = {
    var spark = SparkSession.builder()
      .master("local[*]")
      .appName("SARScalaSpec")
      .config("spark.sql.shuffle.partitions", value = 4)
      .config("spark.default.parallelism", value = 4)
      .config("spark.ui.enabled", value = false)
      .config("spark.sql.crossJoin.enabled", value = true)
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    // load dataframes from csv files in resources directory
    val dataFiles = Seq(
      "getProcessedRatingsInput",
      "getItemCoOccurrenceInput",
      "getItemSimilarityInput",
      "getUserAffinityInput",
      "itemSimilarity",
      "processedRatings",
      "demoUsage",
      "sim_count1",
      "sim_count3",
      "sim_jac1",
      "sim_jac3",
      "sim_lift1",
      "sim_lift3",
      "user_aff",
      "userpred_count3_userid_only",
      "userpred_jac3_userid_only",
      "userpred_lift3_userid_only",
      "dummy",
      "dummy-3-pred",
      "dummy-3-input"
      )

    var data = dataFiles.map((name: String) => {
      (name, spark.sqlContext
        .read
        .format("csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .load(getClass.getResource(f"/$name%s.csv").getPath))
    }).toMap

    data += ("demoUsageWithRating" -> data.apply("demoUsage")
        .select(
            col("userId"),
            col("productId"), 
            to_timestamp(col("timestamp"), "yyyy/MM/dd'T'HH:mm:ss")
              .cast(DataTypes.LongType)
              .as("timestamp"),
            lit(1).as("rating")))

    val theFixture = FixtureParam(data)

    try {
      withFixture(test.toNoArgTest(theFixture))
    }
    finally {
      spark.stop
      spark = null
      SparkSession.clearDefaultSession()
      SparkSession.clearActiveSession()
    }
  }

  it should "build and fit" in { f =>

    val sar = new SARScala()
      .setUserCol("user")
      .setItemCol("item")
      .setRatingCol("rating")
      .setTimeCol("time")
    assert(sar.isInstanceOf[SARScala])

    val model = sar.fit(f.data.apply("getProcessedRatingsInput"))
    assert(model.isInstanceOf[SARScalaModel])
  }

  it should "calculate item co-occurrence" in { f =>

    val expected = Seq(
      (1, 1, 3), (1, 2, 3), (1, 3, 2), (1, 4, 1), (1, 7, 1), (1, 8, 1), (1, 9, 1), (1, 10, 1),
      (2, 2, 3), (2, 3, 2), (2, 4, 1), (2, 7, 1), (2, 8, 1), (2, 9, 1), (2, 10, 1),
      (3, 3, 2), (3, 4, 1),
      (4, 4, 1),
      (7, 7, 1), (7, 8, 1), (7, 9, 1), (7, 10, 1),
      (8, 8, 1), (8, 9, 1), (8, 10, 1),
      (9, 9, 1), (9, 10, 1),
      (10, 10, 1)
    )

    new SARScala()
      .setUserCol("user")
      .setItemCol("item")
      .getItemCoOccurrence(f.data.apply("getItemCoOccurrenceInput"))
      .orderBy("i1", "i2")
      .toDF()
      .collect()
      .zip(expected)
      .foreach({ case (row, testValue) =>
        assert(row.getAs[Int]("i1") === testValue._1)
        assert(row.getAs[Int]("i2") === testValue._2)
        assert(row.getAs[Int]("count") === testValue._3)
      })
  }

  it should "calculate item-item similarity - jaccard" in { f =>
    val expected = Seq(
      (1, 1, 1.0), (1, 2, 1.0), (1, 3, 0.6), (1, 4, 0.3), (1, 7, 0.3), (1, 8, 0.3), (1, 9, 0.3), (1, 10, 0.3),
      (2, 1, 1.0), (2, 2, 1.0), (2, 3, 0.6), (2, 4, 0.3), (2, 7, 0.3), (2, 8, 0.3), (2, 9, 0.3), (2, 10, 0.3),
      (3, 1, 0.6), (3, 2, 0.6), (3, 3, 1.0), (3, 4, 0.5),
      (4, 1, 0.3), (4, 2, 0.3), (4, 3, 0.5), (4, 4, 1.0),
      (7, 1, 0.3), (7, 2, 0.3), (7, 7, 1.0), (7, 8, 1.0), (7, 9, 1.0), (7, 10, 1.0),
      (8, 1, 0.3), (8, 2, 0.3), (8, 7, 1.0), (8, 8, 1.0), (8, 9, 1.0), (8, 10, 1.0),
      (9, 1, 0.3), (9, 2, 0.3), (9, 7, 1.0), (9, 8, 1.0), (9, 9, 1.0), (9, 10, 1.0),
      (10, 1, 0.3), (10, 2, 0.3), (10, 7, 1.0), (10, 8, 1.0), (10, 9, 1.0), (10, 10, 1.0)
    )

    new SARScala()
      .getItemSimilarity(f.data.apply("getItemSimilarityInput"))
      .orderBy("i1", "i2")
      .toDF()
      .collect()
      .zip(expected)
      .foreach({case(row, testValue) =>
        assert(row.getAs[Int]("i1") === testValue._1)
        assert(row.getAs[Int]("i2") === testValue._2)
        assert(math.abs(row.getAs[Float]("value") - testValue._3) < 0.1)
      })
  }

  it should "calculate item-item similarity - cooccur" in { f =>
    val expected = Seq(
      (1, 1, 3.0), (1, 2, 3.0), (1, 3, 2.0), (1, 4, 1.0), (1, 7, 1.0), (1, 8, 1.0), (1, 9, 1.0), (1, 10, 1.0),
      (2, 1, 3.0), (2, 2, 3.0), (2, 3, 2.0), (2, 4, 1.0), (2, 7, 1.0), (2, 8, 1.0), (2, 9, 1.0), (2, 10, 1.0),
      (3, 1, 2.0), (3, 2, 2.0), (3, 3, 2.0), (3, 4, 1.0),
      (4, 1, 1.0), (4, 2, 1.0), (4, 3, 1.0), (4, 4, 1.0),
      (7, 1, 1.0), (7, 2, 1.0), (7, 7, 1.0), (7, 8, 1.0), (7, 9, 1.0), (7, 10, 1.0),
      (8, 1, 1.0), (8, 2, 1.0), (8, 7, 1.0), (8, 8, 1.0), (8, 9, 1.0), (8, 10, 1.0),
      (9, 1, 1.0), (9, 2, 1.0), (9, 7, 1.0), (9, 8, 1.0), (9, 9, 1.0), (9, 10, 1.0),
      (10, 1, 1.0), (10, 2, 1.0), (10, 7, 1.0), (10, 8, 1.0), (10, 9, 1.0), (10, 10, 1.0)
    )

    new SARScala()
      .setSimilarityMetric("cooccurrence")
      .getItemSimilarity(f.data.apply("getItemSimilarityInput"))
      .orderBy("i1", "i2")
      .toDF()
      .collect()
      .zip(expected)
      .foreach({case(row, testValue) =>
        assert(row.getAs[Int]("i1") === testValue._1)
        assert(row.getAs[Int]("i2") === testValue._2)
        assert(math.abs(row.getAs[Float]("value") - testValue._3) < ATOL)
      })
  }

  it should "calculate item-item similarity - lift" in { f =>
    val expected = Seq(
      (1, 1, 0.3), (1, 2, 0.3), (1, 3, 0.3), (1, 4, 0.3), (1, 7, 0.3), (1, 8, 0.3), (1, 9, 0.3), (1, 10, 0.3),
      (2, 1, 0.3), (2, 2, 0.3), (2, 3, 0.3), (2, 4, 0.3), (2, 7, 0.3), (2, 8, 0.3), (2, 9, 0.3), (2, 10, 0.3),
      (3, 1, 0.3), (3, 2, 0.3), (3, 3, 0.5), (3, 4, 0.5),
      (4, 1, 0.3), (4, 2, 0.3), (4, 3, 0.5), (4, 4, 1.0),
      (7, 1, 0.3), (7, 2, 0.3), (7, 7, 1.0), (7, 8, 1.0), (7, 9, 1.0), (7, 10, 1.0),
      (8, 1, 0.3), (8, 2, 0.3), (8, 7, 1.0), (8, 8, 1.0), (8, 9, 1.0), (8, 10, 1.0),
      (9, 1, 0.3), (9, 2, 0.3), (9, 7, 1.0), (9, 8, 1.0), (9, 9, 1.0), (9, 10, 1.0),
      (10, 1, 0.3), (10, 2, 0.3), (10, 7, 1.0), (10, 8, 1.0), (10, 9, 1.0), (10, 10, 1.0)
    )

    new SARScala()
      .setSimilarityMetric("lift")
      .getItemSimilarity(f.data.apply("getItemSimilarityInput"))
      .orderBy("i1", "i2")
      .toDF()
      .collect()
      .zip(expected)
      .foreach({case(row, testValue) =>
        assert(row.getAs[Int]("i1") === testValue._1)
        assert(row.getAs[Int]("i2") === testValue._2)
        assert(math.abs(row.getAs[Float]("value") - testValue._3) < 0.05)
      })
  }

  it should "get mapped arrays" in { f =>

    val expectedCounts = Array(8, 8, 4, 4, 6, 6, 6, 6)
    val expectedIndices = Array(
      0, 1, 2, 3, 4, 5, 6, 7,
      0, 1, 2, 3, 4, 5, 6, 7,
      0, 1, 2, 3,
      0, 1, 2, 3,
      0, 1, 4, 5, 6, 7,
      0, 1, 4, 5, 6, 7,
      0, 1, 4, 5, 6, 7,
      0, 1, 4, 5, 6, 7
    )
    val expectedValues = Array(
      1.0, 1.0, 0.333, 0.333, 0.333, 0.333, 0.333, 0.333,
      1.0, 1.0, 0.333, 0.333, 0.333, 0.333, 0.333, 0.333,
      0.333, 0.333, 1.0, 1.0,
      0.333, 0.333, 1.0, 1.0,
      0.333, 0.333, 1.0, 1.0, 1.0, 1.0,
      0.333, 0.333, 1.0, 1.0, 1.0, 1.0,
      0.333, 0.333, 1.0, 1.0, 1.0, 1.0,
      0.333, 0.333, 1.0, 1.0, 1.0, 1.0)

    val itemSimilarity = f.data.apply("itemSimilarity")
    /*
    val (counts, indices, values) = new SARScalaModel("uid_1", itemSimilarity, itemSimilarity).getMappedArrays

    assert(counts === expectedCounts)
    assert(indices === expectedIndices)
    assert(values === expectedValues)
    */
  }

  it should "get processed ratings - no decay" in { f =>
    
    val expected = Seq(
      (1,1,1.0),
      (1,2,1.0),
      (1,3,1.0),
      (1,4,1.0),
      (2,1,1.0),
      (2,2,1.0),
      (2,7,1.0),
      (2,8,1.0),
      (2,9,1.0),
      (2,10,1.0),
      (3,1,1.0),
      (3,2,1.0),
      (3,3,1.0))
    
    new SARScala()
      .setUserCol("user")
      .setItemCol("item")
      .setRatingCol("rating")
      .setTimeCol("time")
      .getProcessedRatings(f.data.apply("getProcessedRatingsInput"))
      .orderBy("user", "item")
      .toDF()
      .collect()
      .zip(expected)
      .foreach({case(row, testValue) =>
        assert(row.getAs[Int]("user") === testValue._1)
        assert(row.getAs[Int]("item") === testValue._2)
        assert(math.abs(row.getAs[Double]("rating") - testValue._3) < ATOL)
      })
  }

  it should "get processed ratings - time decay" in { f =>

    // TODO: double check these values
    val expected = Seq(
      (1,1,0.007),
      (1,2,0.004),
      (1,3,0.008),
      (1,4,0.018),
      (2,1,0.004),
      (2,2,0.008),
      (2,7,0.002),
      (2,8,0.018),
      (2,9,0.040),
      (2,10,0.090),
      (3,1,0.201),
      (3,2,0.448),
      (3,3,1.0))

    new SARScala()
      .setUserCol("user")
      .setItemCol("item")
      .setRatingCol("rating")
      .setTimeCol("time")
      .setTimeDecay(true)
      .setDecayCoefficient(0.00001)
      .getProcessedRatings(f.data.apply("getProcessedRatingsInput"))
      .orderBy("user", "item")
      .toDF()
      .collect()
      .zip(expected)
      .foreach({case(row, testValue) =>
        println(row)
        assert(row.getAs[Int]("user") === testValue._1)
        assert(row.getAs[Int]("item") === testValue._2)
        assert(math.abs(row.getAs[Double]("rating") - testValue._3) < 0.0005)
      })
  }

  it should "get user affinity" in { f =>

    val expected = Seq(
      (1,1,0.007),
      (1,2,0.004),
      (1,3,0.008),
      (1,4,0.018),
      (2,1,0.004),
      (2,2,0.008),
      (2,7,0.002),
      (2,8,0.018),
      (2,9,0.040),
      (2,10,0.090))
  /*
    new SARScalaModel("uid", f.data.apply("itemSimilarity"), f.data.apply("processedRatings"))
      .setUserCol("user")
      .setItemCol("item")
      .setRatingCol("rating")
      .getUserAffinity(f.data.apply("getUserAffinityInput"))
      .orderBy("user", "item")
      .toDF()
      .collect()
      .zip(expected)
      .foreach({case(row, testValue) =>
        assert(row.getAs[Int]("user") === testValue._1)
        assert(row.getAs[Int]("item") === testValue._2)
        assert(math.abs(row.getAs[Double]("value") - testValue._3) < ATOL)
      })
      */
  }

  // Tests 1-6
  it should "have same item similarity 1-cooccurrence" in { f => testSarItemSimiliarity(f, 1, "cooccurrence", "count") }
  it should "have same item similarity 3-cooccurrence" in { f => testSarItemSimiliarity(f, 3, "cooccurrence", "count") }
  it should "have same item similarity 1-jaccard" in { f => testSarItemSimiliarity(f, 1, "jaccard", "jac") }
  it should "have same item similarity 3-jaccard" in { f => testSarItemSimiliarity(f, 3, "jaccard", "jac") }
  it should "have same item similarity 1-lift" in { f => testSarItemSimiliarity(f, 1, "lift", "lift") }
  it should "have same item similarity 3-lift" in { f => testSarItemSimiliarity(f, 3, "lift", "lift") }

  def testSarItemSimiliarity(f:FixtureParam, threshold: Int, similarityType: String, file: String) {
    val itemSimilarityRef = f.data.apply(s"sim_$file$threshold")

    // melt the dataframe
    val colLength = itemSimilarityRef.columns.length-1
    val colList = itemSimilarityRef.columns.drop(1).map(s => s"'$s', `$s`").mkString(",")

    val itemSimilarityRefLong = itemSimilarityRef
      .selectExpr("_c0 as i1", s"stack($colLength, $colList) as (i2, value)")
      .filter(col("value") =!= 0)
      .orderBy("i1", "i2")

    val itemSimilarity = new SARScala()
      .setUserCol("userId")
      .setItemCol("productId")
      .setRatingCol("rating")
      .setTimeCol("timestamp")
      .setCountThreshold(threshold)
      .setSimilarityMetric(similarityType)
      .fit(f.data.apply("demoUsageWithRating"))
      .itemSimilarity
      .orderBy("i1", "i2")

    assert(itemSimilarityRefLong.count() == itemSimilarity.count())

    val differences = itemSimilarityRefLong.join(itemSimilarity, Seq("i1", "i2"))
      .select(col("i1"), col("i2"), abs(itemSimilarityRefLong.col("value") - itemSimilarity.col("value")).as("diff"))
      .filter(col("diff") > ATOL)

    // differences.show() // uncomment if there are differences
    assert(differences.count() == 0)
  }

  it should "have same user affinity" in { f =>
    val testUsers = f.data.apply("demoUsageWithRating").filter(col("userId") === TEST_USER)
    var userAffinityRef = f.data.apply("user_aff")

    // melt the dataframe
    val colLength = userAffinityRef.columns.length-1
    val colList = userAffinityRef.columns.drop(1).map(s => s"'$s', `$s`").mkString(",")

    // ensure all values are converted to double
    for (column <- userAffinityRef.columns) {
      userAffinityRef = userAffinityRef.withColumn(column, col(column).cast(DoubleType))
    }
    val userAffinityRefLong = userAffinityRef
      .selectExpr("_c0 as userId", s"stack($colLength, $colList) as (productId, value)")
      .filter(col("value") =!= 0)

    val userAffinity = new SARScala()
      .setUserCol("userId")
      .setItemCol("productId")
      .setRatingCol("rating")
      .setTimeCol("timestamp")
      .fit(f.data.apply("demoUsageWithRating"))
      .getUserAffinity(testUsers)

    assert(userAffinityRefLong.count() == userAffinity.count())

    val differences = userAffinityRefLong.join(userAffinity, Seq("userId", "productId"))
      .select(col("userId"), col("productId"), abs(userAffinityRefLong.col("value") - userAffinity.col("value")).as("diff"))
      .filter(col("diff") > ATOL)

    // differences.show() // uncomment if there are differences
    assert(differences.count() == 0)
  }


  // TODO: test 7

  // Tests 8-10
  it should "have same prediction 3-cooccurence" in { f => testSarUserPred(f, 3, "cooccurrence", "count") }
  it should "have same prediction 3-jaccard" in     { f => testSarUserPred(f, 3, "jaccard", "jac") }
  it should "have same prediction 3-lift" in        { f => testSarUserPred(f, 3, "lift", "lift") }

  def testSarUserPred(f:FixtureParam, threshold: Int, similarityType: String, file: String) {
    val demoUsage = f.data.apply("demoUsageWithRating")

    val timeNow = demoUsage.agg(max(col("timestamp"))).head

    // TODO: support timeNow

    val model = new SARScala()
      .setUserCol("userId")
      .setItemCol("productId")
      .setRatingCol("rating")
      .setTimeCol("timestamp")
      .setTimeDecay(true)
      .setDecayCoefficient(30)
      .setCountThreshold(threshold)
      .setSimilarityMetric(similarityType)
      .fit(demoUsage)

    val reference = f.data.apply(s"userpred_${file}${threshold}_userid_only")
    val expected = Range(1, 11)
      .map(i => (reference.head.getString(i), reference.head.getDouble(i + 10)))
      .toMap

    /*
    for ((k,v) <- expected)
      println(s"$k: $v")
    */
    val refTuples = Range(1, 11).map(i => (reference.head.getString(i), reference.head.getDouble(i + 10)))

    val preds = model.transform(demoUsage.filter(demoUsage.col("userID") <=> "0003000098E85347"))
    // preds.orderBy("score").show()

    import preds.sqlContext.implicits._

    val comp = preds.map({ r => (r.getString(1), r.getFloat(2))}).collect.toMap

    assert(comp.size == 10)

    for ((item, score) <- comp) {
      expected get item match {
        case Some(scoreExpected) => {
          val diff = Math.abs(scoreExpected - score)
          if (diff >= 1e-5)
            println(s"$item: $scoreExpected != $score")

          assert(diff <= 1e-5)
         }
        case None => {
          println(s"\tnot found: $item")
          assert(false)
        }
      }
     }

    val missing = expected.keySet.filterNot(comp.keySet).mkString(",")
    if (missing.length > 0)
      println(s"missing: $missing")
    assert(missing.length == 0)
  }

  it should "return correct preds for dummy dataset" in { f =>
    // TODO: remove timestamp
    val df = f.data.apply("dummy").withColumn("timestamp", lit(1))
    // df.show()

    val expected = f.data.apply("dummy-3-pred")
    // expected.show()

    val model = new SARScala()
      .setUserCol("user")
      .setItemCol("item")
      .setRatingCol("rating")
      .setTimeCol("timestamp") // TODO: ignore timeCol
      .setTimeDecay(false)
      .setCountThreshold(1)
      .setSimilarityMetric("jaccard")
      .fit(df)

    val predInput = f.data.apply("dummy-3-input").withColumn("timestamp", lit(1))
    // predInput.show()

    // TODO: better way of doing this?
    model.setTopK(3)

    var actual = model.transform(predInput)
    // actual.show()

    // TODO: compare dataset
  }
  // def comparePredictions(actual:DataFrame, expected:DataFrame) = {
     // val actualSorted = actual.orderBy("")
  // }
}