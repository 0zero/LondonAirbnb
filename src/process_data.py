import sys

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.model_selection import train_test_split

from data_processor import DataProcessor, get_key_feature_columns
from fit_regressor import build_linear_model, build_randforest_model, evaluate_model, save_model


# TODO: add logging
from plotting import plot_multiple_column_maps, plot_ridge_plot, plot_pred_vs_test


def print_top_n_price(
        df: gpd.GeoDataFrame,
        column_name: str,
        top_n: int = 3
):
    """
        Returns top n values of a column_name from a DataFrame
        Input:
            df: pandas dataframe
            column_name: name of column that we want to use in function
            top_n: number of top results we want from column_name
        Output:
            printed statements
    """
    print(f"\nLondon's most expensive Airbnb boroughs, using {column_name}, are:")
    for i in range(top_n):
        print(
            f"\t{df[column_name].nlargest(top_n).index.to_numpy()[i]}: £ {np.round(df[column_name].nlargest(top_n).values[i], 0)} per night.")


# TODO: refactor main to be more concise and grouping themes
def main():
    # TODO: add option to take 2/3 input variables if we don't want to re-process the data
    if len(sys.argv) == 5:

        # TODO: user parser to get inputs
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

        print("Saving data...\n    DATABASE: {}".format(database_filename))
        data_processor.save_data()

        data_processor.create_neighbourhood_features()

        print_top_n_price(data_processor.neighbourhoods, "price_mean")
        print_top_n_price(data_processor.neighbourhoods, "price_median")

        print("Cleaned data saved to database!")

        print("Building model...")
        target_variable = "price_numeric"
        target_price_limit = 1000

        modelling_df = data_processor.create_listing_features(remove_all_nans=True)
        modelling_df = modelling_df[modelling_df[target_variable] < target_price_limit]
        numeric_cols, cat_cols = get_key_feature_columns(modelling_df)

        X = modelling_df[numeric_cols + cat_cols]
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

        print(
            f"The r-squared and rmse scores for your Linear model was {linear_r2}, {linear_rmse} on {len(y_test)} values.")
        print(
            f"The r-squared and rmse scores on the training data for your Linear model was {linear_train_r2}, {linear_train_rmse}.")

        print(
            f"The r-squared and rmse scores for your RandomForest model was {randfo_r2}, {randfo_rmse} on {len(y_test)} values.")
        print(
            f"The r-squared and rmse scores on the training data for your RandomForest model was {randfo_train_r2}, {randfo_train_rmse}.")

        print("Creating figures...")

        plot_multiple_column_maps(
            data_processor.neighbourhoods,
            ["price_mean", "price_median"],
            ["Mean Price / £", "Median Price / £"],
            savefig=True,
        )

        plot_ridge_plot(data_processor.listings, "price_numeric", "Log10 Price / £", True, savefig=True)

        plot_multiple_column_maps(
            data_processor.neighbourhoods,
            ["number_of_reviews", "number_of_reviews_per_listings_zero_rev"],
            ["Total Reviews", "Normalised Reviews (reviews>0)"],
            figname="num_reviews",
            savefig=True
        )

        plot_pred_vs_test(
            y_test,
            model_linear.predict(X_test),
            [f"price < {target_price_limit}: $r^2$={linear_r2}"],
            savefig=True,
        )

    else:
        print("Please provide the filepaths of the listings, listings summary, and neighbourhoods "
              "datasets as the first, second, and third argument respectively, as "
              "well as the filename of the database to save the cleaned data "
              "to as the fourth argument. \n\nExample: python process_data.py "
              "listings.csv listings_summary.csv neighbourhoods.geojson"
              "LondonAirbnbDatabase.db")


if __name__ == "__main__":
    main()
