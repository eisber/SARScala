name := "sarscala"

version := "0.2"

scalaVersion := "2.11.8"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-sql" % "2.3.1",
  "org.apache.spark" %% "spark-mllib" % "2.3.1",
  "org.scalatest" %% "scalatest" % "3.0.5" % "test"
)
