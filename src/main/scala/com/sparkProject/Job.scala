package com.sparkProject

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._


object Job {


  

  def main(args: Array[String]): Unit = {

    // SparkSession configuration
    val spark = SparkSession
      .builder
      .appName("spark session TP_parisTech")
      .getOrCreate()

    val sc = spark.sparkContext

    sc.setLogLevel("WARN")

    import spark.implicits._


    /********************************************************************************
      *
      *        TP 1
      *
      *        - Set environment, InteliJ, submit jobs to Spark
      *        - Load local unstructured data
      *        - Word count , Map Reduce
      ********************************************************************************/



    // ----------------- word count ------------------------

    /*val df_wordCount = sc.textFile("/users/maxime/spark-1.6.2-bin-hadoop2.6/README.md")
      .flatMap{case (line: String) => line.split(" ")}
      .map{case (word: String) => (word, 1)}
      .reduceByKey{case (i: Int, j: Int) => i + j}
      .toDF("word", "count")

    df_wordCount.orderBy($"count".desc).show() */


    /********************************************************************************
      *
      *        TP 2 : début du projet
      *
      ********************************************************************************/

    val dfexop = spark.read.format("csv") // returns a DataFrameReader, giving access to methods “options” and “csv”
      .option("header", "true") // Use first line of all files as header
      .option("inferSchema", "true") // Automatically infer data types
      .option("comment", "#") // All lines starting with # are ignored
      .load("/cal/homes/ebenoit/spark-2.0.0-bin-hadoop2.6/cumulative.csv")


    //println("number of columns", dfexop.columns.length)
    //println("number of rows", dfexop.count)



    val columns = dfexop.columns.slice(10, 20) // df.columns returns an Array. In scala arrays have a method “slice” returning a slice of the array
    dfexop.select(columns.map(col): _*) // .show(20) //

    //dfexop.select("koi_period").show(10)

    //dfexop.printSchema()

    //dfexop.groupBy($"koi_disposition").count().show()

    val dfc = dfexop.filter($"koi_disposition" === "CONFIRMED" || $"koi_disposition" === "FALSE POSITIVE")
    dfc.select(columns.map(col): _*) //.show(10)

    dfc.groupBy($"koi_eccen_err1").count() //.show()
    dfc.select("koi_eccen_err1") //.show(20)

    val dfc2 = dfc.select(columns.map(col): _*).drop("koi_eccen_err1") //.show(10)

    //dfc2.printSchema()

    val notNecessaryColumns = List("koi_eccen_err1", "index", "kepid", "koi_fpflag_nt", "koi_fpflag_ss",
      "koi_fpflag_co",
      "koi_fpflag_ec", "koi_sparprov", "koi_trans_mod", "koi_datalink_dvr", "koi_datalink_dvs", "koi_tce_delivname",
      "koi_parm_prov", "koi_limbdark_mod",  "koi_fittype", "koi_disp_prov", "koi_comment", "kepoi_name",
      "kepler_name",  "koi_vet_date", "koi_pdisposition") // can be a Seq

    val dfc3 = dfc.select(columns.map(col): _*)
      .drop(notNecessaryColumns:_*)

    //dfc3.printSchema()

    // q4e) val dfc4 = dfc3.columns.filter( for(val <- ) Essai perso





    //////////// Tentative resolution 1
    /*val uselessColumn = dfc3.columns.filter{ case (column:String) =>
      dfc3.agg(countDistinct(column)).first().getLong(0) <= 1 }

    //////////// Tentative resolution 2
    val useless = for(col <- dfc3.columns if dfc3.select(col).distinct().count() <= 1 ) yield col
    val dfc3b = dfc3.drop(useless: _*)
    */
    //////////// Tentative resolution 3
    val good_columns= dfc3.columns
      .map(x => dfc3.select(x).distinct().count()<=1)
      .filter(_==true)


    val dfc4 = dfc3.na.fill(0.0)


    dfc4.printSchema()





    // Q6 Joindre deux df

    val df_labels = dfc4.select("rowid", "koi_disposition") //"rowid",
    val df_features = dfc4.drop("koi_disposition")

    val df_joined = df_features.join(df_labels, usingColumn = "rowid")

    // Q8 Ajouter et manipuler des colonnes


    def udf_sum = udf((col1: Double, col2: Double) => col1 + col2)



    val df_newFeatures = df_joined
      .withColumn("koi_ror_min", udf_sum($"koi_ror", $"koi_ror_err2"))
      .withColumn("koi_ror_max", $"koi_ror" + $"koi_ror_err1")

  // Q9 Sauvegarder un df

    df_newFeatures
      .coalesce(1) // optional : regroup all data in ONE partition, so that results are printed in ONE file
      .write
      .mode("overwrite")
      .option("header", "true")
      .csv("/cal/homes/ebenoit/spark-2.0.0-bin-hadoop2.6/cleanedDataFrame.parquet")






  }


}
