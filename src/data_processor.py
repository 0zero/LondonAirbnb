import numpy as np
import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine
from pathlib import Path
from typing import Tuple, List, Union, Optional


# TODO: add logging


class DataProcessor:
    def __init__(
            self,
            listings_path: str,
            listings_summary_path: str,
            neighbourhoods_path: str,
            database_filename: str,
            use_clean_data: bool = False,
    ):
        self.listings_path = Path(listings_path)
        self.listings_summary_path = Path(listings_summary_path)
        self.neighbourhoods_path = Path(neighbourhoods_path)
        self.database_filename = database_filename
        self.use_clean_data = use_clean_data
        self.listings: Optional[pd.DataFrame] = None
        self.neighbourhoods: Optional[gpd.GeoDataFrame] = None

    def load_and_clean_data(self) -> None:
        """
        Load in data from file paths, remove duplicates and some NaNs (if all rows in column are NaN
        or target variable row is NaN)
        :return: None but assigns cleaned DataFrame and GeoDataFrame to class attributes
        """
        if self.use_clean_data:
            # Load clean data
            engine = create_engine(f"sqlite:///{self.database_filename}")
            df_clean_listings = pd.read_sql_table("Listings", engine)

            geojson_path = self.neighbourhoods_path.parent / "neighbourhoods_cleaned.geojson"
            gdf_clean_neighbourhoods = load_geojson_data(geojson_path)
        else:
            # Load raw data
            df_raw_listings = load_csv_data(self.listings_path)
            df_raw_listings_summary = load_csv_data(self.listings_summary_path)
            gdf_raw_neighbourhoods = load_geojson_data(self.neighbourhoods_path)

            # Remove duplicates and columns with only NaNs
            df_clean_listings = clean_data(df_raw_listings)
            df_clean_listings_summary = clean_data(df_raw_listings_summary)
            gdf_clean_neighbourhoods = clean_data(gdf_raw_neighbourhoods)

            # Combine listings data and remove rows where "price" (our target variable) has
            # zero or NaN values.
            df_clean_listings = cast_and_clean_price(df_clean_listings, df_clean_listings_summary)

        self.listings = df_clean_listings
        self.neighbourhoods = gdf_clean_neighbourhoods

    def save_data(self) -> None:
        """
        Export DataFrames to sqlLite database
        :return: None
        """
        # TODO: add try-except statement here
        engine = create_engine(f"sqlite:///{self.database_filename}")
        self.listings.to_sql('Listings', engine, index=False, if_exists="replace")

        # TODO: Add GeoDataFrame to sqlite DB like
        #  https://www.giacomodebidda.com/posts/export-a-geodataframe-to-spatialite/
        # self.neighbourhoods.to_postgis('Neighbourhoods', engine, index=False, if_exists="replace")
        # For now, just export cleaned GeoDataFrame as a new geojson file
        geojson_path = self.neighbourhoods_path.parent / "neighbourhoods_cleaned.geojson"
        self.neighbourhoods.to_file(str(geojson_path), driver="GeoJSON")

    def create_neighbourhood_features(self) -> None:
        """
        Add some summary comparison metrics to neighbourhoods GeoDataFrame. These include
        summary statistics for price, review scores, and number of reviews per Borough.
        :return: Original Neighbours GeoDF
        """
        # TODO: refactor this - separate feature groups and clean up code

        listings = self.listings.copy()
        self.neighbourhoods = self.neighbourhoods.set_index("neighbourhood")

        # Features for Price across London Boroughs
        self.neighbourhoods["price_mean"] = listings.groupby("neighbourhood_cleansed")["price_numeric"].mean()
        self.neighbourhoods["price_std"] = listings.groupby("neighbourhood_cleansed")["price_numeric"].std()
        self.neighbourhoods["price_median"] = listings.groupby("neighbourhood_cleansed")["price_numeric"].median()
        self.neighbourhoods["price_mode"] = listings.groupby("neighbourhood_cleansed")["price_numeric"].agg(
            pd.Series.mode)

        # Features for Reviews across London Boroughs
        listings_review_columns = [
            "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness",
            "review_scores_checkin", "review_scores_communication", "review_scores_location",
            "review_scores_value",
        ]
        # calculate average review ratings
        for col in listings_review_columns:
            self.neighbourhoods[f"{col}_mean"] = self.listings.groupby("neighbourhood_cleansed")[col].mean()
            self.neighbourhoods[f"{col}_std"] = self.listings.groupby("neighbourhood_cleansed")[col].std()

        self.neighbourhoods["reviews_mean"] = self.neighbourhoods.iloc[:, 6::2].mean(axis=1)

        # Features for Popularity across London Boroughs. We're using the number_of_reviews as a
        # proxy for popularity here. We're assuming that, for the most part, if someone rents a listing,
        # they'll leave a review and therefore listings without reviews have never been rented.
        # This assumption is fairly strong but in the absence of data related to how many times
        # each listing has been rented this is a good substitute.

        self.neighbourhoods["number_of_reviews"] = listings.groupby(
            "neighbourhood_cleansed")["number_of_reviews"].sum()
        self.neighbourhoods["number_of_listings"] = listings.groupby(
            "neighbourhood_cleansed")["id"].count()
        self.neighbourhoods["number_of_reviews_per_listings"] = listings.groupby(
            "neighbourhood_cleansed")["number_of_reviews"].sum() / listings.groupby(
            "neighbourhood_cleansed")["id"].count()

        # same popularity features but only for listings with reviews
        listings_zero_revs = listings[listings["number_of_reviews"] != 0].copy()
        self.neighbourhoods["number_of_reviews_zero_rev"] = listings_zero_revs.groupby(
            "neighbourhood_cleansed")["number_of_reviews"].sum()
        self.neighbourhoods["number_of_listings_zero_rev"] = listings_zero_revs.groupby(
            "neighbourhood_cleansed")["id"].count()
        self.neighbourhoods["number_of_reviews_per_listings_zero_rev"] = listings_zero_revs.groupby(
            "neighbourhood_cleansed")["number_of_reviews"].sum() / listings_zero_revs.groupby(
            "neighbourhood_cleansed")["id"].count()

        # utility columns
        self.neighbourhoods["centre"] = self.neighbourhoods["geometry"].centroid

    def create_listing_features(self, remove_all_nans: bool = False) -> pd.DataFrame:
        """
        Create reduced listings dataframe which will be used as basis of modelling operations.
        Option to either remove all NaN data or fill in the NaN values with the mode of the feature.
        :param remove_all_nans: Whether to remove all rows with any NaNs
        :return: A tuple with:
            i) a reduced DF containing all data ready for to go into our training pipeline
            ii) a list of categorical data columns
        """
        raw_features_cols = [
            "neighbourhood_cleansed", "latitude", "longitude", "room_type", "bathrooms_text",
            "accommodates", "bedrooms", "beds", "price_numeric",
        ]

        listings_modelling = self.listings[raw_features_cols].copy()
        # Create bathrooms features
        # bathrooms_text I think we can convert these to a numerical value along with whether the
        # bathroom is private or not. I'm going to assume that if the room_type is the Entire home/apt then the
        # bathroom is also private.
        bath_num, bath_private = get_bathroom_number(self.listings[raw_features_cols], "bathrooms_text", use_bool=True)
        listings_modelling["bathrooms_number"] = bath_num
        listings_modelling["bathrooms_private"] = bath_private

        # There are a few categorical features that have some missing data. In the notebook, I fill the
        # values with the mode (which has the same value as the median in our cases). But I also want to
        # have the option of removing all the rows with missing data, it's only about 6% of rows have
        # missing data and while not ideal, this is still very much a work in progress.
        if remove_all_nans:
            listings_modelling.dropna(axis=0, how="any", inplace=True)
        else:
            values = {
                "bedrooms": listings_modelling.bedrooms.mode()[0],
                "beds": listings_modelling.beds.mode()[0],
                "bathrooms_number": listings_modelling.bathrooms_number.mode()[0],
                "bathrooms_private": listings_modelling.bathrooms_private.mode()[0]
            }
            listings_modelling = listings_modelling.fillna(value=values).copy()
        listings_modelling.drop(columns="bathrooms_text", inplace=True)
        return listings_modelling


def get_nan_columns(df: pd.DataFrame) -> List[str]:
    """
     Return the column names where all their data is NaN from a dataframe
    :param df:
    :return: list of column names where all their data is NaN
    """
    nan_cols = []
    for col in df.columns:
        if df[col].isna().all():
            nan_cols.append(col)
    return nan_cols


def load_csv_data(file_path: Path) -> pd.DataFrame:
    """
    Use Pandas to load a CSV file
    :param file_path: Path to csv file
    :return: Pandas DataFrame with csv file data
    """
    if file_path.exists():
        df = pd.read_csv(file_path)
        return df
    else:
        msg = f"The file at path {str(file_path)} does not exist. Have you got the correct path?"
        raise FileNotFoundError(msg)


def load_geojson_data(file_path: Path) -> gpd.GeoDataFrame:
    """
    Use GeoPandas to load a geojson file
    :param file_path: Path to geojson file
    :return: GeoPandas DataFrame with geojson file data
    """
    if file_path.exists():
        gdf = gpd.read_file(file_path)
        return gdf
    else:
        msg = f"The file at path {str(file_path)} does not exist. Have you got the correct path?"
        raise FileNotFoundError(msg)


def clean_data(df: Union[pd.DataFrame, gpd.GeoDataFrame]) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """
    Remove duplicates and columns where all rows are NaN
    :param df: Geo/DataFrame
    :return: same Geo/DataFrame but with duplicates and NaNs(columns) removed
    """
    nan_columns = get_nan_columns(df)
    if len(nan_columns) != 0:
        df.drop(nan_columns, axis=1, inplace=True)

    if df.duplicated(keep="first").sum() > 0:
        df.drop_duplicates(inplace=True)

    return df


def cast_and_clean_price(df: pd.DataFrame, df_summary: pd.DataFrame) -> pd.DataFrame:
    """
    The price in df_clean_listings is a string with the currency but what we want is
    the actual price as a float. So let's copy the price from df_clean_listings_summary
    over to listings_df and ensure that the unique id's match.
    Also, remove rows where "price" (our target variable) has zero or NaN values.
    :param df: df_clean_listings
    :param df_summary: df_clean_listings_summary
    :return: df_clean_listings with numerical price
    """

    temp_price = []
    for i, row in df_summary.iterrows():
        temp_price.append(float(row["price"]) if row["id"] == df["id"].iloc[i] else None)
    df["price_numeric"] = temp_price

    # Drop rows where price_numeric == 0
    zero_price_df = df[df["price_numeric"] == 0]
    if zero_price_df.shape[0] > 0:
        df.drop(axis=0, index=zero_price_df.index, inplace=True)

    # Drop rows where price_numeric == NaN
    if np.sum(df["price_numeric"].isna()) > 0:
        df.dropna(axis=0, subset=["price_numeric"], inplace=True)

    return df


def get_bathroom_number(
        df: pd.DataFrame,
        column_name: str,
        use_bool: bool = False,
) -> Tuple[List, List]:
    """
        Separate bathroom number and whether it's private or not from a single column
        Input:
            df: pandas dataframe
            column_name: name of column that we want to use in function
            use_bool: private bathroom returns a boolean or an integer representation of a boolean
        Output:
            bathroom_number: number corresponding to how many bathrooms are in property
            bathroom_private: bool/int corresponding to whether the bathroom is private or shared
    """
    bathroom_number = []
    bathroom_private = []

    if use_bool:
        true_value = True
        false_value = False
    else:
        true_value = int(True)
        false_value = int(False)

    for _, row in df.iterrows():
        if isinstance(row[column_name], float):
            if np.isnan(row[column_name]):
                bathroom_number.append(np.nan)
                bathroom_private.append(np.nan)
                continue

        if "private" in row[column_name].lower() or row["room_type"] == "Entire home/apt":
            bathroom_private.append(true_value)
        else:
            bathroom_private.append(false_value)

        if "half-bath" in row[column_name].lower():
            bathroom_number.append(0.5)
        else:
            bathroom_number.append(float(row[column_name].split(" ")[0]))

    return bathroom_number, bathroom_private


def create_dummy_df(df: pd.DataFrame, cat_cols: List[str], dummy_na: bool) -> pd.DataFrame:
    """
    INPUT:
    df - pandas dataframe with categorical variables you want to dummy
    cat_cols - list of strings that are associated with names of the categorical columns
    dummy_na - Bool holding whether you want to dummy NA vals of categorical columns or not

    OUTPUT:
    df - a new dataframe that has the following characteristics:
            1. contains all columns that were not specified as categorical
            2. removes all the original columns in cat_cols
            3. dummy columns for each of the categorical columns in cat_cols
            4. if dummy_na is True - it also contains dummy columns for the NaN values
            5. Use a prefix of the column name with an underscore (_) for separating
    """
    for col in cat_cols:
        try:
            # for each cat add dummy var, drop original column
            df = pd.concat(
                [
                    df.drop(col, axis=1),
                    pd.get_dummies(df[col], prefix=col, prefix_sep='_', drop_first=True, dummy_na=dummy_na),
                ],
                axis=1,
            )
        # TODO: fix this and deal with exceptions properly
        except:
            continue
    return df


def get_key_feature_columns(
        df: pd.DataFrame,
        target_feature: str = "price_numeric",
        limit: float = 0.05,
) -> Tuple[List, List]:
    """
    Use correlation matrix to get features that have more than abs(0.05) linear
    correlation coefficient with our target feature (numeric price)
    :param: df: Dataframe used in modelling
    :return: tuple with list of numeric and categorical feature names
    """

    # Get room_type and neighbourhood_cleansed features
    # df = create_dummy_df(df, ["room_type"], dummy_na=False)
    # df = create_dummy_df(df, ["neighbourhood_cleansed"], dummy_na=False)
    # df.columns = df.columns.str.strip().str.lower().str.replace(
    #     " ", "_").str.replace("(", "").str.replace(")", "")

    categorical = list(df.select_dtypes(include=["object"]).columns)

    corr_mat = df.corr()
    key_features = corr_mat[abs(corr_mat[target_feature]) > limit][target_feature]
    df = df[list(key_features.index)]
    numeric = list(df.select_dtypes(include=["float64", "int64"]).columns)
    numeric.remove(target_feature)

    # print(f"Key features are: {numeric} and {categorical}.")
    return numeric, categorical
