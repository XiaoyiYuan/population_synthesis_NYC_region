This project is supported by the Center for Social Complexity at George Mason University and the Defense Threat Reduction Agency (DTRA) under Grant number HDTRA1-16-0043. The opinions, findings, conclusions or recommendations expressed in this work are those of the researchers and do not necessarily reflect the views of the sponsors.

The Python code include population sysnthesis based on 2010 U.S. Census data:

- Roads: 2010 Census TIGER shapefiles (all roads separately downloaded and combined) https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2010&layergroup=Roads
- Demographics: Census-tract level Demographic Profile (DP) TIGER shapefiles (particular shapefile ZIP) https://www.census.gov/geo/maps-data/data/tiger-data.html
- School info: The Educational Institution dataset  https://geodata.epa.gov/arcgis/rest/services/OEI/ORNL_Education/MapServer
- Establishment numbers: Census Bureau’s County Business Patterns (CBP) https://www.census.gov/data/datasets/2010/econ/cbp/2010-cbp.html
- Workflow: Census Bureau’s Longitudinal Employer- Household Dynamics (LEHD) Origin-Destination Employment Statistics (LODES) https://lehd.ces.census.gov/data/
	    
Since Census data is aggregated data, in order to create realistic population, household, location of the housholds, workplaces, and schools, we came up with several heuristics (see Heuristics.pdf). The data folder includes our cleaned road map and demographic data (note: one data that's not added is road shapefile that's too big to add). The output of the population sysntheis result is organized in a SQL database. This code is a modified version of the original one by Talha Oz who is also a member of our team (https://github.com/oztalha/Population-Synthesis).

