# Dataset README
## 1. Introductory information
Eyetracking dataset: EyeTracking-data
The folder is organized with subfolders for each participant, which contains the raw eyetracking data for each condition (explained in the next section) in csv format. Additionaly, for each condition there is data ending with _blinks. These contain the raw eyetracking data, as well as annotations at blink events.
For questions about the dataset, contact Simon Gebraad at j.s.gebraad@student.tudelft.nl.

## 2. Methodological information
The raw data comes from a BEP project. During the study, the participants were placed in a VR environment and asked to locate and gather 10 products from shelves in a specified order and place them in a cart. These products were shown on a shopping list. This experiment was done in 5 conditions:

- 0: Control condition: no external influence
- 1: Non-playable characters (NPCs): multiple NPCs were walking through the aisles, creating visual distractions.
- 2: Background noise: audio fragments of a busy supermarket were played, creating auditory distractions.
- 3: Arithmetic task: the participants had to perform an arithmetic task simultaneously with the primary task. 
- 4: Combined conditions: all previous conditions were combined.

Additionaly, during this project, annotated data was created which contains the raw data alongside marks when a blink was recorded. The code for this is included in this repository and explained in more detail under 'Code information'.

## 3. Data specific information
The raw dataset includes:

- Frame
- Timestamp in 'YYYY-MM-DD HH-MM-SEC' format
- Openness_L (dimensionless), with 1.0 indicating a fully open eye
- Openness_R (dimensionless), with 1.0 indicating a fully open eye
- Pupil_diamter_L (mm), with -1 indicating a closed eye
- Pupil_diamter_R (mm), with -1 indicating a closed eye

The annotated dataset includes all of the above and the following:
- Blink (s), marked at the middle of the blink event

# Code README
## 1. Project goal
The goal of this project was to extract blink rate and blink rate variability from raw eye-tracking data in real time. These two features are then used to classify different types of cognitive load.

## 2. Installation instructions
These instructions will help you to get the code running.

### Prerequisites
- Installation of Python with matplotlib, numpy, seaborn, pandas, sklearn and scipy
- A package manager like Anaconda can be helpful for this (Anaconda is freely available)

### Installing
- With the prerequisites, nothing needs to be installed. You can just open analyse_blinking.py

### Running
This section assumes you have met the prerequisites and have analyse_blinking.py open.
#### Reproducing results
- To reproduce the results from the paper, you can run the entire script which will automatically read the data, analyse it and save the plots.

#### Analyse blinking
- If you only want to analyse the blink rate and blink rate variability, run everything up untill 'Analyse blinking'
- To plot the results, also run 'Plot blinkrate' and 'Plot blinkrate variability'
- To perform statistical tests also run 'Compare blinkrate' and 'Compare blinkrate variability'

#### Classify conditions
- If you only want to classify the different cognitive conditions, run the imports and 'Classifying data'
- To plot the results, also run 'Plot classifiers'

#### Create plots
- It is currently not possible to create the plots without running the other sections


## 3. License information
This project is licensed under the [MIT](LICENSE.md) License.

## 4. Citation information
To cite this work, refer to the [citation](CITATION.cff) or click 'Cite this repository' in the 'About' section of this repository.

## 5. Contributing information
If you want to contribute to this project, send a push request with your contribution.
