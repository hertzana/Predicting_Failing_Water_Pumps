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

The list of variables in the data include:
-	amount_tsh - Total static head (amount water available to waterpoint)
-	date_recorded - The date the row was entered
-	funder - Who funded the well
-	gps_height - Altitude of the well
-	installer - Organization that installed the well
-	longitude - GPS coordinate
-	latitude - GPS coordinate
-	wpt_name - Name of the waterpoint if there is one
-	num_private -
-	basin - Geographic water basin
-	subvillage - Geographic location
-	region - Geographic location
-	region_code - Geographic location (coded)
-	district_code - Geographic location (coded)
-	lga - Geographic location
-	ward - Geographic location
-	population - Population around the well
-	public_meeting - True/False
-	recorded_by - Group entering this row of data
-	scheme_management - Who operates the waterpoint
-	scheme_name - Who operates the waterpoint
-	permit - If the waterpoint is permitted
-	construction_year - Year the waterpoint was constructed
-	extraction_type - The kind of extraction the waterpoint uses
-	extraction_type_group - The kind of extraction the waterpoint uses
-	extraction_type_class - The kind of extraction the waterpoint uses
-	management - How the waterpoint is managed
-	management_group - How the waterpoint is managed
-	payment - What the water costs
-	payment_type - What the water costs
-	water_quality - The quality of the water
-	quality_group - The quality of the water
-	quantity - The quantity of water
-	quantity_group - The quantity of water
-	source - The source of the water
-	source_type - The source of the water
-	source_class - The source of the water
-	waterpoint_type - The kind of waterpoint
-	waterpoint_type_group - The kind of waterpoint
