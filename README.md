# London's Airbnb scene

This project sets out to analyse Airbnb data for London, UK compiled in 2021.

## Brief
Under the guise of working for a rental property investor, I will aim to answer, or at the very least provide insight into, the following questions.

1. On average, where in London are Airbnb's most expensive listings?
2. Which of London's boroughs are the most popular for Airbnb listings?
3. Can we predict the price for a London Airbnb listing?

I also look at the following questions but there isn't much to say about them (see the [notebook](airbnb.ipynb) as to why):
- Do visitors prefer to hire the entire house/apartment or a single room?
- Which area in London has the best AirBnB ratings?

## Summary of Results

For more detailed discussions of the findings please refer to the [notebook](airbnb.ipynb) or its corresponding [medium article](https://medium.com/@japheth.yates/scoping-out-londons-airbnb-market-5bea3c1948d1). Below are a summary of the answers to the questions posed above.

1. On average, where in London are Airbnb's most expensive listings?
    * Westminster: £ 258.0 per night.
	* City of London: £ 237.0 per night.
	* Kensington and Chelsea: £ 222.0 per night.

2. Which of London's boroughs are the most popular for Airbnb listings?
    * Westminister or City of London depending on feature used.

3. Can we predict the price for a London Airbnb listing?
    * Not very well, r^2 score ~ 0.4 or an RMSE ~ £50 when filtering data to include only listings with a price lower than £1000 per night.
     

## The Data

I'll be using Airbnb listing data for London which was compiled in December 2021. You can find links to the [full dataset](http://insideairbnb.com/get-the-data/) and some [AirBnB visualisations](http://insideairbnb.com/london/). Airbnb also provide a [spreadsheet with explanations for each data column](https://docs.google.com/spreadsheets/d/1iWCNJcSutYqpULSQHlNyGInUvHg2BoUGoNRIGa6Szc4/edit#gid=982310896) and some [assumptions](http://insideairbnb.com/data-assumptions/) in the data.

For this particular analysis I've only used the following three data files (some of which are stored with [git lfs](https://git-lfs.github.com/)):

* `listings.csv` : Detailed Listings data. Size: 150 Mb.
* `listings_summary.csv` : Summary information and metrics for listings in London (good for visualisations). Size: 8.9 Mb.
* `neighbourhoods.geojson` : GeoJSON file of neighbourhoods of the city. Size: 1 Mb.


## Environment setup

I've used [Anaconda](https://www.anaconda.com/) with Python 3.9.2 to create the environment for this work. You can use the `requirement.yml` file to create the environment locally using:

```
conda env create -f requirement.yml
```

You can then activate it with

```
conda activate airbnb_london
```
This will install `numpy`, `pandas`, `geopandas`, `matplotlib`, `sklearn`, `seaborn`, `plotly` and their dependencies. 

## Other files description
* `airbnb.ipynb` : Jupyter notebook containing all the analysis within this project.
* `.gitattributes` : containing details of which file types are tracked by `git lfs`.


---

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.