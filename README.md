## Fishing for Fishermen 2: fishing vessel classification
This code submitted for Topcoder competition 'Fishing for Fishermen 2' aims to identify the type of fishing being performed by a vessel, based on automatic identification system (AIS) broadcast reports and contextual data. 

https://www.topcoder.com/challenges/#&query=fishing%20&tracks=datasci&tracks=design&tracks=develop

## Objective
Create an algorithm to effectively identify if a vessel is fishing--and if so, the type of fishing taking place--based on observable behavior and any additional data such as weather, known fishing grounds, etc. regardless of vessel type or declared purpose.

The algorithm uses AIS positional data, and combined oceanographic data as provided in the downloadable data set.

The algorithm should then detect vessels that match the profile of behaviours of vessels engaged in fishing, and identifies what type of fishing each vessel is involved in during each of the tracks in the data set.

## Data description
The data is provided as a set of CSV files, one for each vessel track, as well as a ground truth file (for the training data) indicating the type of fishing being performed on each track.

The vessel track data contains the following fields:

- Track Number
- Relative Time (seconds from the track start)
- Latitude (degrees to the north)
- Longitude (degrees to the east)
- SOG (Speed Over Ground, knots)
- Oceanic Depth (meters)
- Chlorophyll Concentration (milligrams per cubic meter)
- Salinity (Practical Salinity Units)
- Water Surface Elevation (meters)
- Sea Temperature (degrees)
- Thermocline Depth (meters)
- Eastward Water Velocity (meters per second)
- Northward Water Velocity (meters per second)

The ground truth file for training data will contain the following fields:

- Track Number
- Fishing Type
- The fishing type will not be included for the testing data.

## Scoring

Predictions will be scored against the ground truth using the area under the receiver operating characteristic (ROC). Some of the records in the test set will be used for provisional scoring, and others for system test scoring. (You will not know which records belong to which set.)

The ROC curve will be determined and the score will be determined from the area under the ROC curve using the following method:

The contestant's submission will score each track with a probability that the vessel was engaged in each type of fishing.
Each fishing type will be treated as a binary classifier, and it’s AuC will be calculated as in steps 3-5.
The true positive rates and the false positive rates are determined as follows:

TPR_i = Accumulate[s_i] / N_TPR

FPR_i = Accumulate[1 - s_i] / N_FPR;

with the addition: FPR_0 = 0;
where N_TRP is the total number of fishing records of the given type, N_FPR is the total number of records not of that fishing type, and N_TRP + N_FPR = N (total number of records with known status in the test)
Then the AuC is determined as a numerical integral of TRP over FRP:

AuC = Sum [TPR_i * (FPR_i - FPR_i-1)]
Then the four AuC values are weighted and averaged, then scaled to determine the final score:

Score = max(1,000,000 * (2 * WeightedAverage - 1), 0)
The weights are as follows:

- trawler: 40%
- seiner: 30%
- longliner: 20%
- support: 10%

## Useful resources
The following links are included, as they were all listed with the previous "Fishing for Fishermen" contest. Though they are not all strictly necessary or directly useful in this iteration, we have left them included for anyone with general interest in subject.

Useful site for the entire AIS message: (http://catb.org/gpsd/AIVDM.html#_aivdm_aivdo_sentence_layer)

ITU document describing “payload” field (field 6 of each message): (https://www.itu.int/rec/R-REC-M.1371/en)

# How to run the pipeline
This pipeline requires Anaconda Python version 3.4 or above. 

1. `source activate envs/fishing_for_fishermen_2.yml` activate conda environment for the project
2. `source environment_variables` create variables for project folders and add project to python path
3. `python src/model/generate_features.py` this performs data cleaning and generate extra features on the input data including the time difference between each point, the course, and change in course between points, the speed between points, distance to shore, distance to port, and whether the vessel is an an exclusive economic zone (EEZ).
4. `python trajectories/plot_trajectories.py` plots the trajectories of each vessel to visualise the difference between fishing behaviours.
5. `python src/model/aggregate_features.py` aggregates each vessel track into an aggregate feature set. Where aggregate features are described in `docs/features.txt` file. This creates aggregate `training.csv` and `testing.csv` files.
6. `python src/model/magicloops/simpleloop.py` this runs Rayid Ghani's model selection and optimisation magicloops routine (https://github.com/rayidghani/magicloops). A customised weighted ROC scoring system was implemented for this problem according to the competition scoring. The models to run and search grid size can be adjusted. ROC score was calculated for each class seperately. Models that are not multiclass could be run in future using a OneVsRestClassifier, but this has not yet been implemented.
7. Choose the optimum model based on the results of the magicloops and set in the `generate_model` function of `train_model.py`.
8. Run `python train_model.py`
9. Run `python predict.py` specifying the path to the pickled model to use, this will output a final `scoring.csv` file for submission., this will output a final `scoring.csv` file for submission.
10. submit the function `generateAnswerURL.py` for evaluation by TopCoder.

# Authors 
William Grimes
