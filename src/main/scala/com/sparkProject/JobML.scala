package com.sparkProject


import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.tuning.{TrainValidationSplit, ParamGridBuilder}
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression}
import org.apache.spark.ml.param.ParamMap

/**
  * Created by ebenoit on 27/10/16.
  */
object JobML {

  def main(args: Array[String]): Unit = {

    // SparkSession configuration
    val spark = SparkSession
      .builder
      .appName("spark session TP_parisTech")
      .getOrCreate()

    val sc = spark.sparkContext

    sc.setLogLevel("WARN")

    /**********************************************************************************************
      *                                                                                           *
      *        TP 4/5 - Machine Learning                                                          *
      *                                                                                           *
      *********************************************************************************************/


    val path = "/cal/homes/ebenoit/spark-2.0.0-bin-hadoop2.6/"
    val df = spark.read.parquet(path + "cleanedDataFrame.parquet")

    // df.printSchema()
    // df.show(10)
    // df.columns


    /**********************************************************************************************
      *                                                                                           *
      *       Question 1 - Prepare data                                                           *
      *                                                                                           *
      *********************************************************************************************/

    print("#########################################################################")

    // Q1.a - Assembling features in one single vector
    val assembler = new VectorAssembler()
      .setInputCols(df.drop("rowid", "koi_disposition").columns) // .array (facultative)
      .setOutputCol("features")

    val featVectored = assembler.transform(df)
    // featVectored.show(5)
    // featVectored.select("koi_disposition").show(5)


    // Q1.b - Indexation of text labels in numerical data
    val indexerLab = new StringIndexer()
      .setInputCol("koi_disposition")
      .setOutputCol("label")

    val dfForML = indexerLab.fit(featVectored).transform(featVectored)

    print("\nOur initial dataframe has as label values:\n")
    dfForML.select("features", "label").groupBy("label").count().show(5)


    /**********************************************************************************************
      *                                                                                           *
      *       Question 2 - Logistic regression to predict label values and tuning of regression   *
      *       hyper-parameter                                                                     *
      *                                                                                           *
      *********************************************************************************************/


    // Q2.a - Prepare data into a training and a test set, resp. 90% / 10%
    val seed = 4001
    val Array(training, test) = dfForML.randomSplit(Array(0.9,0.1), seed)

    // trainingSet.select("label", "features").show(5)
    // testSet.select("label", "features").show(5)



    // Q2.b - Training classifier

    // Model
    val lr = new LogisticRegression()
      .setLabelCol("label")
      .setStandardization(true)  // to scale each feature of the model
      .setFitIntercept(true)  // we want an affine regression (with false, it is a linear regression)
      .setTol(1.0e-5)  // stop criterion of the algorithm based on its convergence
      .setElasticNetParam(1.0) // L1-norm regularization : LASSO
      .setMaxIter(2000) // a security stop criterion to avoid infinite loops


    print("\nNow we will tune the regParam to find out the best one. We will use a grid " +
      "search which contains following values:\n")

    // Step is 0.5 in log from 1.0e-6 to 1.0
    val gridSearch = (-6.0 to 0.0 by 0.5 toArray).map(x => math.pow(10, x))
    print(gridSearch.deep.mkString(" | ") + "\n\n")


    // We use a ParamGridBuilder to construct a grid of parameters to search over.
    // TrainValidationSplit will try all combinations of values and determine best model using
    // the evaluator.
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, gridSearch)
      .build()

    print("\nAs we don't want iteration number to be reached too quickly, we set it as 2000\n")


    // In this case the estimator is the logistic regression.
    // A TrainValidationSplit requires an Estimator, a set of Estimator ParamMaps (grid search), and
    // an Evaluator.
    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(lr)
      .setEvaluator(new BinaryClassificationEvaluator())
      .setEstimatorParamMaps(paramGrid)
      // 70% of the data will be used for training and the remaining 30% for validation.
      .setTrainRatio(0.7)

    // Run train validation split, and choose the best set of parameters.
    val model = trainValidationSplit.fit(training)


    // Make predictions on test data. model is the model with combination of parameters
    // that performed best.
    val df_WithPredictions = model.transform(test)


    print("\nWe tuned hyper-parameter with choosing the best on predictions. Here the first 20 "
      + "predictions compared to label \n")
    df_WithPredictions.select("features", "label", "prediction").show()



    // create an Evaluator for binary classification, which expects two input columns: prediction
    // and label.
    val evaluator = new BinaryClassificationEvaluator().setLabelCol("label") //
    // .setRawPredictionCol("prediction")

    // Evaluation of the accuracy - with hyper parameter tuning
    // Evaluates predictions and returns a scalar metric areaUnderROC(larger is better).
    val accuracy = evaluator.evaluate(df_WithPredictions)

    print("\nAccuracy with tuning of regression hyper-parameter is:"
      + "\n %.2f ".format(accuracy*100) + " %\n\n")


    print("Confusion matrix:\n")
    df_WithPredictions.select("features", "label", "prediction")
      .groupBy("label", "prediction")
      .count.show()


    // Saving model
    val trainedModel = model.write.overwrite.save(path + "model")

  }
}