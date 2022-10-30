val ScalaVersion = "2.12.10"

lazy val spark_mlib = "org.apache.spark" %% "spark-mllib" % "3.1.2"
lazy val spark_sql = "org.apache.spark" %% "spark-sql" % "3.1.2"

lazy val zoo =
  "com.intel.analytics.bigdl" % "bigdl-dllib-spark_3.1.2" % "2.1.0"

lazy val gcn_with_zoo = project
  .in(file("."))
  .settings(
    name := "graph_conv_network",
    organization := "ems.gcn",
    scalaVersion := ScalaVersion,
    version := "0.0.1",
    libraryDependencies ++= Seq(
      spark_mlib,
      spark_sql,
      zoo
    )
  )

resolvers +=
  "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/groups/public/"
