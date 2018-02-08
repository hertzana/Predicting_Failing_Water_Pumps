# Water Pumps
In this project, we use various classification methods to predict which water pumps in Tanzania are operational and which ones do not work at all.

The original data can be found here: https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/25/ 

Currently, there are 4 files:
- Water-Exploratory Data Analysis.ipynb: Data Exploration 
- data_cleanup1.py: Consolidates the features 'installer' and 'funder' into larger categories, as each variable has large numbers of installers/funders
- data_cleanup2.py: Data cleaning, normalization & standardization, filling in of missing values
- Modeling 1.ipynb: 1st cut of modeling (more to follow) 

The water pump labels to predict are:
- functional - the waterpoint is operational and there are no repairs needed
-	functional needs repair - the waterpoint is operational, but needs repairs
-	non functional - the waterpoint is not operational


