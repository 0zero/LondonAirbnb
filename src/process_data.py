import sys
from typing import Tuple, List

import pandas as pd
from sklearn.model_selection import train_test_split

from data_processor import DataProcessor, get_key_feature_columns
from fit_regressor import build_linear_model, build_randforest_model, evaluate_model, save_model

# TODO: add logging


def main():
    if len(sys.argv) == 5:

        listings_path, listings_summary_path, neighbourhoods_path, database_filename = sys.argv[1:]
        data_processor = DataProcessor(
            listings_path=listings_path,
            listings_summary_path=listings_summary_path,
            neighbourhoods_path=neighbourhoods_path,
            database_filename=database_filename
        )

        print("Loading and cleaning data...\n    Listings: {}\n    Listings Summary: {}\n    Neighbourhood: {}"
              .format(listings_path, listings_summary_path, neighbourhoods_path))
        data_processor.load_and_clean_data()
        data_processor.create_neighbourhood_features()

        # print("Saving data...\n    DATABASE: {}".format(database_filename))
        # data_processor.save_data()

        print("Cleaned data saved to database!")

        print("Building model...")
        target_variable = "price_numeric"

        modelling_df = data_processor.create_listing_features(remove_all_nans=True)
        modelling_df = modelling_df[modelling_df[target_variable] < 1000]
        numeric_cols, cat_cols = get_key_feature_columns(modelling_df)

        X = modelling_df[numeric_cols+cat_cols]
        y = modelling_df[target_variable]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=42)

        model_linear = build_linear_model(
            numeric_cols=numeric_cols,
            categorical_cols=cat_cols,
            model_name="ols",
        )
        model_randfo = build_randforest_model(
            numeric_cols=numeric_cols,
            categorical_cols=cat_cols,
        )

        print("Training linear model...")
        model_linear.fit(X_train, y_train)
        print("Training RandomForest model...")
        model_randfo.fit(X_train, y_train)

        print("Evaluating model...")
        linear_r2, linear_rmse = evaluate_model(model_linear, X_test, y_test)
        linear_train_r2, linear_train_rmse = evaluate_model(model_linear, X_train, y_train)
        randfo_r2, randfo_rmse = evaluate_model(model_randfo, X_test, y_test)
        randfo_train_r2, randfo_train_rmse = evaluate_model(model_randfo, X_train, y_train)

        print(f"The r-squared and rmse scores for your Linear model was {linear_r2}, {linear_rmse} on {len(y_test)} values.")
        print(f"The r-squared and rmse scores on the training data for your Linear model was {linear_train_r2}, {linear_train_rmse}.")

        print(f"The r-squared and rmse scores for your RandomForest model was {randfo_r2}, {randfo_rmse} on {len(y_test)} values.")
        print(f"The r-squared and rmse scores on the training data for your RandomForest model was {randfo_train_r2}, {randfo_train_rmse}.")
    else:
        print("Please provide the filepaths of the listings, listings summary, and neighbourhoods "
              "datasets as the first, second, and third argument respectively, as "
              "well as the filename of the database to save the cleaned data "
              "to as the fourth argument. \n\nExample: python process_data.py "
              "listings.csv listings_summary.csv neighbourhoods.geojson"
              "LondonAirbnbDatabase.db")


if __name__ == "__main__":
    main()
