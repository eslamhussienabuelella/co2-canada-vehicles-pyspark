#!/usr/bin/env python
# coding: utf-8

# ## install libraries
"""
#libraries installation

!pip install pyspark
!pip install findspark
!pip install pandas numpy

"""


# # import necessary libraries

import findspark # this will be used to find the and use Apache Spark
findspark.init() # Initiate the findspark to locate and utilise the Spark
from pyspark.sql import SparkSession 

from pyspark.sql.functions import lower, regexp_replace, col, trim
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
#from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
#from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
#from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
#from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
#from pyspark.ml import Pipeline

import pandas as pd 


#start spark session & check it
spark = SparkSession \
    .builder \
    .appName("Eslam Canada Vehicles CO2 emissions Spark Session") \
    .getOrCreate()

# Check if my spark session started
if "spark" in locals() and isinstance(spark, SparkSession):
    print("The spark session", spark.sparkContext.appName, "is active and ready to use")
else:
    print("No sparksessions is active, please create a one to proceed")


# ## Loading the data

# Use Spark to read the my csv file.
raw_co2_df = spark.read.csv(r"/home/eslamabuelella/Desktop/MY1995_2023_Fuel_Consumption_Ratings.csv",inferSchema=True,header=True)


# # 3. EDA & Data wrangling

# Print the Schema of the Spark DataFrame
raw_co2_df.printSchema()


# to standerdise the column names 
co2_df=raw_co2_df
rename_dict = {
    "ModelYear": "model_year",
    "Make":"make",
    "Model":"model",
    "VehicleClass":"vehicle_class",
    "EngineSize_L":"engine_size_l",
    "Cylinders":"cylinders",
    "Transmission":"transmission",
    "FuelType":"fuel_type",
    "FuelConsCity_L100km":"fuel_cons_city_L100km",
    "FuelConsHwy_L100km":"fuel_cons_hwy_L100km",
    "Comb_L100km":"fuel_cons_comb_L100km",
    "Comb_mpg":"fuel_cons_comb_MPG",
    "CO2Emission_g_km":"CO2_emission_gkm",
    "CO2Rating":"CO2_rating",
    "SmogRating":"smog_rating",
       }
for old_col, new_col in rename_dict.items():
    co2_df = co2_df.withColumnRenamed(old_col,new_col)

co2_df.printSchema()


# Inspecting the categorical variables: vehicle classes
for column_name in ["make","model","vehicle_class","transmission","fuel_type"]:
    print("column:", column_name)
    print("number of unqie categories:", co2_df.select(column_name).distinct().count())

print(co2_df.select("fuel_type").distinct().orderBy(column_name).show())


print(co2_df.select("transmission").distinct().orderBy(column_name).show(30))


print(co2_df.select("vehicle_class").distinct().orderBy(column_name).show(33))

# we can see clear redundancy in categories due to none standardised data entry
# so we need to clean this mess

# from pyspark.sql.functions import lower, regexp_replace, col, trim

# lowercase and trim white spaces
co2_df = co2_df.withColumn("vehicle_class", lower(trim(col("vehicle_class"))))

# replace non-alphanumeric character with "-"
co2_df = co2_df.withColumn("vehicle_class", regexp_replace(col("vehicle_class"), "[^a-zA-Z0-9]+", "-"))

# remove "-" from begings or endings
co2_df = co2_df.withColumn("vehicle_class", regexp_replace(col("vehicle_class"), "^-+|-+$", ""))

# 4. merge multiple "-" into one
co2_df = co2_df.withColumn("vehicle_class", regexp_replace(col("vehicle_class"), "-+", "-"))

print("Clean classes counts:", co2_df.select("vehicle_class").distinct().count())
co2_df.select("vehicle_class").distinct().orderBy("vehicle_class").show(truncate=False)

# checking the count of nulls in my dataset
co2_df.toPandas().info()

# Since CO2_rating & smog_rating are +70% nulls we shouldn't rely on them for finding or predicting CO2 in this data set
co2_df=co2_df.drop("co2_rating", "smog_rating")
co2_df.printSchema()

co2_df.toPandas().info()  

# creating lists of predictors names according to their nature

numeric_predictors = ["engine_size_l", "cylinders", "model_year", 
                      "fuel_cons_city_L100km", "fuel_cons_hwy_L100km", 
                      "fuel_cons_comb_L100km", "fuel_cons_comb_MPG"]

categorical_predictors = ["vehicle_class", "make", "transmission", "fuel_type"]

target = "CO2_emission_gkm"


## Building base model using all numeric predictors 


## Building base model using all numeric predictors 
# from pyspark.ml.regression import LinearRegression
# from pyspark.ml.evaluation import RegressionEvaluator

# defining linear reg modeel
lr1 = LinearRegression(featuresCol="features", labelCol="CO2_emission_gkm")

# building the linear regresion evaluator
evaluator = RegressionEvaluator(labelCol="CO2_emission_gkm", predictionCol="prediction", metricName="r2")


# from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# building grid of regularisation parameters
paramGrid = (ParamGridBuilder() .addGrid(lr1.regParam, [0.0, 0.01,0.05, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5]).build())


# building vector assembler to be used in pyspark model
# from pyspark.ml.feature import VectorAssembler

assembler_lr1_all_num = VectorAssembler(inputCols= numeric_predictors, outputCol="features")
output_lr1_all_num = assembler_lr1_all_num.transform(co2_df)

# prepare training and testing data stes
lr1_train_full, lr1_test_full = output_lr1_all_num.randomSplit([0.8, 0.2], seed=42)
lr1_train_full.select("features").show(3)

# building  regression cross validator with 5 folds to validate mmmodels against  regularisation paramters
cv_lr1 = CrossValidator(estimator=lr1, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)

cvModel_lr1 = cv_lr1.fit(lr1_train_full)

# selecting best model
bestModel = cvModel_lr1.bestModel

# show R2 per regulrisation parameter
print("Cross-validated R2 for each regParam:")
for params, metric in zip(paramGrid, cvModel_lr1.avgMetrics):
    print(f"regParam={params[lr1.regParam]} -> avg R2={metric}")

# Extract the best  regulrisation parameter from the bestModel
#best_regParam = bestModel._java_obj.parent().getRegParam()
#print("\nBest regParam selected by CV:", best_regParam)


best_predictions = bestModel.transform(lr1_test_full)

test_rmse = evaluator.evaluate(best_predictions, {evaluator.metricName: "rmse"})
test_r2   = evaluator.evaluate(best_predictions, {evaluator.metricName: "r2"})
test_mae  = evaluator.evaluate(best_predictions, {evaluator.metricName: "mae"})

print("Test RMSE:", test_rmse)
print("Test MAE :", test_mae)
print("Test R²  :", test_r2)


#print("Best regParam:", best_regParam)
print("Intercept:", bestModel.intercept)

for name, coef in zip(numeric_predictors, bestModel.coefficients):
    print(f"{name:25s}: {coef:.4f}")


# # Build Models using all variables (numeric + categorical)

# defining the common processes to deal with any categorical var
# building indexer to convert to categories into numbers

all_cat_indexers = [StringIndexer(inputCol=cat_col, outputCol=f"{cat_col}_idx", handleInvalid="keep")
                    for cat_col in categorical_predictors]

# building encoder to convert to categories indexes into dummy variables

all_cat_encoders = [OneHotEncoder(inputCol=f"{cat_col}_idx", outputCol=f"{cat_col}_vec")
                    for cat_col in categorical_predictors]

# building assembler for all predictors (cat + numbers)
assembler_all = VectorAssembler(inputCols=[f"{cat_col}_vec" for cat_col in categorical_predictors] + numeric_predictors, outputCol="features")


# ## Build random forest non linear model using all variables (categorical + numrical)


# Build random forest non linear model using all variables (categorical + numrical)

#feature_pipeline = Pipeline(stages=indexers + [encoder, assembler])
rf1_pipeline = Pipeline(stages=all_cat_indexers + all_cat_encoders + [assembler_all])
rf = RandomForestRegressor(
    labelCol=target,
    featuresCol="features",
    predictionCol="prediction",
    seed=42)

# apply random forestt piplibe to engineer features & 
rf_pipeline = Pipeline(stages=rf1_pipeline.getStages() + [rf])

# definition of random forest grid
rf_paramGrid = (ParamGridBuilder()
    .addGrid(rf.numTrees, [25, 50])#, 75])
    .addGrid(rf.maxDepth, [ 9, 12])#, 15])
    .addGrid(rf.minInfoGain, [0.01])#, 0.1, 0.5])
    .build())

# random forest evaluator
rf_evaluator = RegressionEvaluator(labelCol=target, predictionCol="prediction", metricName="r2")

# cross validation step for randmon forest
rf_cv = CrossValidator(estimator=rf_pipeline, estimatorParamMaps=rf_paramGrid, evaluator=rf_evaluator, numFolds=5)


rf_train_full, rf_test_full = co2_df.randomSplit([0.8, 0.2], seed=42)


# random forest model fitting to training
rf_cv_model = rf_cv.fit(rf_train_full)

# identifcation of best rf model
rf_best_model = rf_cv_model.bestModel
# defining the best parameters
best_rf_model = rf_best_model.stages[-1]


# Extract key hyperparameters
best_num_trees = best_rf_model.getNumTrees
best_max_depth = best_rf_model.getMaxDepth()
best_min_gain = best_rf_model.getMinInfoGain()

print("\nBest Random Forest hyperparameters found by CrossValidator:")
print(f"number of trees: {best_num_trees}")
print(f"maximum depth: {best_max_depth}")
print(f"minimum gain: {best_min_gain}")

# predicting using training data
rf_train_predictions = rf_best_model.transform(rf_train_full)

# predict using random forest model best model
rf_predictions = rf_best_model.transform(rf_test_full)
# build model evaluator
rf_evaluator_rmse = RegressionEvaluator(labelCol=target, predictionCol="prediction", metricName="rmse")
rf_evaluator_mae  = RegressionEvaluator(labelCol=target, predictionCol="prediction", metricName="mae")
rf_evaluator_r2   = RegressionEvaluator(labelCol=target, predictionCol="prediction", metricName="r2")
# evaluating model's performance on training data & test data to ensure that the model is not overfitting 
rf_best_model_rmse = rf_evaluator_rmse.evaluate(rf_train_predictions)
rf_best_model_mae  = rf_evaluator_mae.evaluate(rf_train_predictions)
rf_best_model_r2   = rf_evaluator_r2.evaluate(rf_train_predictions)

rf_test_rmse = rf_evaluator_rmse.evaluate(rf_predictions)
rf_test_mae  = rf_evaluator_mae.evaluate(rf_predictions)
rf_test_r2   = rf_evaluator_r2.evaluate(rf_predictions)

print("\n\n Random Forest Regression metric scores on trainindata\n\n")
print(f"Random Forest on training dataset RMSE: {rf_best_model_rmse}")
print(f"Random Forest on training dataset MAE : {rf_best_model_mae}")
print(f"Random Forest on training dataset R2  : {rf_best_model_r2}")


print("\n\n Random Forest Regression metric scores on test set\n\n")
print(f"Random Forest on test set RMSE: {rf_test_rmse}")
print(f"Random Forest on test set MAE : {rf_test_mae}")
print(f"Random Forest on test set R2  : {rf_test_r2}")

# feature importance 
predictors_importances = best_rf_model.featureImportances.toArray()
# converting to numpy to python floats
predictors_importances = [float(importance_value) for importance_value in predictors_importances]  

predictors_names = assembler_all.getInputCols()
# fill feature importance in spark df
predictors_importance_df = (spark.createDataFrame(list(zip(predictors_names, predictors_importances)),["feature", "importance"]).orderBy("importance", ascending=False))
# export to feature importance to csv
predictors_importance_df.toPandas().to_csv(r"/home/eslamabuelella/Desktop/RF_predictors_importance_df.csv", index=True)

predictors_importance_df.show()


# from pyspark.ml import Pipeline

# building linear regression object

lr2_cat_num = LinearRegression(labelCol=target, featuresCol="features")

# buiilding indexer, encoder, assembeler pipe line
lr2_pipeline_full = Pipeline(stages=all_cat_indexers + all_cat_encoders + [assembler_all, lr2_cat_num])

# train test splitting
all_cat_train_full, all_cat_test_full = co2_df.randomSplit([0.8, 0.2], seed=42)

lr2_evaluator = RegressionEvaluator(labelCol=target, predictionCol="prediction", metricName="r2")

lr2_paramGrid = (ParamGridBuilder().addGrid(lr2_cat_num.regParam, [0.0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5]).build())

lr2_cv_full = CrossValidator(estimator=lr2_pipeline_full, estimatorParamMaps=lr2_paramGrid, evaluator=lr2_evaluator, numFolds=5)

lr2_cvModel_full = lr2_cv_full.fit(all_cat_train_full)


# best linear model selection 
bestModel_lr2_full = lr2_cvModel_full.bestModel

best_lr2_stage = bestModel_lr2_full.stages[-1]

print("Cross-validated RMSE for each regParam:")
for params, metric in zip(lr2_paramGrid, lr2_cvModel_full.avgMetrics):
    print(f"for regularisation parameter = {params[lr2_cat_num.regParam]} has avg R2={metric}")

best_regParam_full_lr2 = best_lr2_stage._java_obj.parent().getRegParam()
print("\nBest regularisation Parameter selected by CV:", best_regParam_full_lr2)


lr2_pred_full = bestModel_lr2_full.transform(all_cat_test_full)

lr2_test_rmse_full = lr2_evaluator.evaluate(lr2_pred_full, {lr2_evaluator.metricName: "rmse"})
lr2_test_mae_full  = lr2_evaluator.evaluate(lr2_pred_full, {lr2_evaluator.metricName: "mae"})
lr2_test_r2_full   = lr2_evaluator.evaluate(lr2_pred_full, {lr2_evaluator.metricName: "r2"})

print("Test RMSE:", lr2_test_rmse_full)
print("Test MAE :", lr2_test_mae_full)
print("Test R²  :", lr2_test_r2_full)


lr2_coeffs = best_lr2_stage.coefficients
lr2_intercept = best_lr2_stage.intercept
print("Intercept:", lr2_intercept)
print("Number of coefficients:", len(lr2_coeffs))


print("coefficients of best linear regression 'lr2' model:", lr2_coeffs)



sample_lr2_df = bestModel_lr2_full.transform(all_cat_train_full.limit(1))
sample_lr2_df

# Extract metadata from the 'features' column
lr2_feature_metadata = sample_lr2_df.schema["features"].metadata["ml_attr"]["attrs"]
lr2_feature_metadata


# collect attributes from ohe variables and numeric values


lr2_attrs = []


for lr2_attr_type in ["binary", "numeric"]:
    if lr2_attr_type in lr2_feature_metadata:
        lr2_attrs.extend(lr2_feature_metadata[lr2_attr_type])

# sort index to be in correct order
lr2_attrs = sorted(lr2_attrs, key=lambda variable: variable["idx"])

# Extract feature names in order
lr2_feature_names = [variable_name["name"] for variable_name in lr2_attrs]

#print("Number of feature names:", len(lr2_feature_names))
#print("Number of coefficients:", len(best_lr2_stage.coefficients))


# import pandas as pd 
coef_values = best_lr2_stage.coefficients.toArray()

coef_df = pd.DataFrame({"feature": lr2_feature_names, "coefficient": coef_values})

coef_df["abs_coefficient"] = coef_df["coefficient"].abs()

# Sort by absolute size (strongest effects first)
coef_df_sorted = coef_df.sort_values("abs_coefficient", ascending=False)

print("Intercept:", best_lr2_stage.intercept)
coef_df_sorted.head(50)


coef_df_sorted.to_csv(r"/home/eslamabuelella/Desktop/full_linear_model_coefficients.csv", index=False)


coef_df = pd.DataFrame({"feature": lr2_feature_names, "coefficient": coef_values})
coef_df["abs_coefficient"] = coef_df["coefficient"].abs()


# the idea behind creating a function to use it in any model later
def get_field_name(vectorised_feature: str):
    # to handle one hot encoded categorical similar to "fuel_type_vec_E"
    if "_vec_" in vectorised_feature:
        return vectorised_feature.split("_vec_")[0]
    # to handle encoded but without category (rare)
    elif vectorised_feature.endswith("_vec"):
        return vectorised_feature[:-4]
    # numeric feature
    else:
        return vectorised_feature

coef_df["field"] = coef_df["feature"].apply(get_field_name)


coef_df[["feature", "field"]].head(20)


avg_abs_by_field = coef_df.groupby("field")["abs_coefficient"].agg(avg_abs_coefficient_LR = "mean", 
                                                                    sum_abs_coefficient_LR = "sum", 
                                                                    median_abs_coefficient_LR = "median").reset_index()

avg_abs_by_field = avg_abs_by_field.rename(columns={"abs_coefficient": "avg_abs_coefficient_LR"}).sort_values("avg_abs_coefficient_LR", ascending=False)

avg_abs_by_field


# ## inspect the colinearity between the numeric variables


#co2_df[numeric_predictors+[target]].toPandas.corr()

numeric_co2_df_corr = co2_df.select(numeric_predictors + [target]).toPandas().corr()
# export correlation matrix to be visualised in Tabluea
numeric_co2_df_corr.to_csv(r"/home/eslamabuelella/Desktop/numeric_co2_df_corr.csv", index=True)
numeric_co2_df_corr


# * Since our Target variable is CO2_emission_gkm we found that the stronest correlated predictor:
#     1. fuel_cons_comb_L100km : 0.933518
#     2. fuel_cons_city_L100km : 0.930649
#     3. fuel_cons_hwy_L100km : 0.908668
#     4. fuel_cons_comb_MPG : -0.905320
#     5. engine_size_l : 0.824808
#     6. cylinders : 0.791384
# * We found that the stronget predictor **fuel_cons_comb_L100km** is strongly correlated with all the above predictors which indicate clear colinearity between all the strongly correlated predictor.
#     1. fuel_cons_city_L100km : 0.993385
#     2. fuel_cons_hwy_L100km : 0.979384
#     3. fuel_cons_comb_MPG : -0.923746
#     4. engine_size_l : 0.806844
#     5. cylinders : 0.763336

# ## Build Model using strong predictors (feult_comb_cons_L100km + feul_type)

numeric_predictors_lr3 = ["fuel_cons_comb_L100km"]

categorical_predictors_lr3 = ["fuel_type"]

target_lr3 = "CO2_emission_gkm"



lr3_indexers = [StringIndexer(inputCol=cat_col, outputCol=f"{cat_col}_idx", handleInvalid="keep")
                for cat_col in categorical_predictors_lr3]

lr3_encoders = [OneHotEncoder(inputCol=f"{cat_col}_idx", outputCol=f"{cat_col}_vec")
                for cat_col in categorical_predictors_lr3]


assembler_all = VectorAssembler(inputCols=[f"{cat_col}_vec" for cat_col in categorical_predictors_lr3] + numeric_predictors_lr3, outputCol="features")


lr3_cat_num = LinearRegression( labelCol=target, featuresCol="features")

lr3_pipeline_full = Pipeline(stages=lr3_indexers + lr3_encoders + [assembler_all, lr3_cat_num])

lr3_train_full, lr3_test_full = co2_df.randomSplit([0.8, 0.2], seed=42)


lr3_evaluator = RegressionEvaluator(labelCol=target, predictionCol="prediction", metricName="r2")

lr3_paramGrid = (ParamGridBuilder().addGrid(lr3_cat_num.regParam, [0.0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5]).build())

lr3_cv_full = CrossValidator(estimator=lr3_pipeline_full, estimatorParamMaps=lr3_paramGrid, evaluator=lr3_evaluator, numFolds=5)

lr3_cvModel_full = lr3_cv_full.fit(lr3_train_full)



bestModel_lr3_full = lr3_cvModel_full.bestModel
best_lr3_stage = bestModel_lr3_full.stages[-1]

print("Cross-validated RMSE for each regParam:")
for params, metric in zip(lr3_paramGrid, lr3_cvModel_full.avgMetrics):
    print(f"Reglusation parameter = {params[lr3_cat_num.regParam]} has avg R2 = {metric}")

best_regParam_full_lr3 = best_lr3_stage._java_obj.parent().getRegParam()
print("\nBest regParam selected by CV:", best_regParam_full_lr3)



lr3_pred_full = bestModel_lr3_full.transform(lr3_test_full)

lr3_test_rmse_full = lr3_evaluator.evaluate(lr3_pred_full, {lr3_evaluator.metricName: "rmse"})
lr3_test_mae_full  = lr3_evaluator.evaluate(lr3_pred_full, {lr3_evaluator.metricName: "mae"})
lr3_test_r2_full   = lr3_evaluator.evaluate(lr3_pred_full, {lr3_evaluator.metricName: "r2"})

print("Test RMSE:", lr3_test_rmse_full)
print("Test MAE :", lr3_test_mae_full)
print("Test R²  :", lr3_test_r2_full)


lr3_coeffs = best_lr3_stage.coefficients
lr3_intercept = best_lr3_stage.intercept
print("Intercept:", lr3_intercept)
print("Number of coefficients:", len(lr3_coeffs))

# use the pipeline to transform a small sample to infer schema

sample_lr3_df = bestModel_lr3_full.transform(lr3_train_full.limit(1))
# Extract metadata from the 'features' column
lr3_feature_metadata = sample_lr3_df.schema["features"].metadata["ml_attr"]["attrs"]

# collect attributes one hot encodes and numeric value

lr3_attrs = []

for lr3_attr_type in ["binary", "numeric"]:
    if lr3_attr_type in lr3_feature_metadata:
        lr3_attrs.extend(lr3_feature_metadata[lr3_attr_type])

# sort index to be in correct order
lr3_attrs = sorted(lr3_attrs, key=lambda lr3_variable: lr3_variable["idx"])

# Extract feature names in order
lr3_feature_names = [variable_name["name"] for variable_name in lr3_attrs]

lr3_coef_values = best_lr3_stage.coefficients.toArray()

lr3_coef_df = pd.DataFrame({"feature": lr3_feature_names, "coefficient": lr3_coef_values})

lr3_coef_df["abs_coefficient"] = lr3_coef_df["coefficient"].abs()

# Sort by absolute size (strongest effects first)
coef_df_sorted = lr3_coef_df.sort_values("abs_coefficient", ascending=False)

print("Intercept:", best_lr3_stage.intercept)
coef_df_sorted.head(len(coef_df_sorted))
