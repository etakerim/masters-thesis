# Datasets
https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset
https://www.kaggle.com/datasets/jishnukoliyadan/vibration-analysis-on-rotating-shaft
https://www.kaggle.com/datasets/tyhuffman/vibration-analysis-bearing-belt-and-edm-faults


## NASA Bearing Dataset
Prognostic Dataset for Predictive/Preventive Maintenance

### About Dataset
Dataset Description

Four bearings were installed on a shaft. The rotation speed was kept constant at 2000 RPM by an AC motor coupled to the shaft via rub belts. A radial load of 6000 lbs is applied onto the shaft and bearing by a spring mechanism. All bearings are force lubricated.

Rexnord ZA-2115 double row bearings were installed on the shaft as shown in Figure 1. PCB 353B33 High Sensitivity Quartz ICP accelerometers were installed on the bearing housing (two accelerometers for each bearing [x- and y-axes] for data set 1, one accelerometer for each bearing for data sets 2 and 3). Sensor placement is also shown in Figure 1. All failures occurred after exceeding designed life time of the bearing which is more than 100 million revolutions.
Dataset Structure

Three (3) data sets are included in the data packet (IMS-Rexnord Bearing Data.zip). Each data set describes a test-to-failure experiment. Each data set consists of individual files that are 1-second vibration signal snapshots recorded at specific intervals. Each file consists of 20,480 points with the sampling rate set at 20 kHz. The file name indicates when the data was collected. Each record (row) in the data file is a data point. Data collection was facilitated by NI DAQ Card 6062E. Larger intervals of time stamps (showed in file names) indicate resumption of the experiment in the next working day.

#### Set No. 1:
- Recording Duration: October 22, 2003 12:06:24 to November 25, 2003 23:39:56
- No. of Files: 2,156
- No. of Channels: 8
- Channel Arrangement: Bearing 1 – Ch 1&2; Bearing 2 – Ch 3&4; Bearing 3 – Ch 5&6; Bearing 4 – Ch 7&8.
- File Recording Interval: Every 10 minutes (except the first 43 files were taken every 5 minutes)
- File Format: ASCII
- Description: At the end of the test-to-failure experiment, inner race defect occurred in bearing 3 and roller element defect in bearing 4.

#### Set No. 2:
- Recording Duration: February 12, 2004 10:32:39 to February 19, 2004 06:22:39
- No. of Files: 984
- No. of Channels: 4
- Channel Arrangement: Bearing 1 – Ch 1; Bearing2 – Ch 2; Bearing3 – Ch3; Bearing 4 – Ch 4.
- File Recording Interval: Every 10 minutes
- File Format: ASCII
- Description: At the end of the test-to-failure experiment, outer race failure occurred in bearing 1.

#### Set No. 3
- Recording Duration: March 4, 2004 09:27:46 to April 4, 2004 19:01:57
- No. of Files: 4,448
- No. of Channels: 4
- Channel Arrangement: Bearing1 – Ch 1; Bearing2 – Ch 2; Bearing3 – Ch3; Bearing4 – Ch4;
- File Recording Interval: Every 10 minutes
- File Format: ASCII
- Description: At the end of the test-to-failure experiment, outer race failure occurred in bearing 3.

Accessing the Dataset

We have made this dataset available on Kaggle. Watch out for Offical NASA Website.

The dataset is in text format and has been rared, then zipped and also contain breif documentation (README) by the authors itself.

The data set was provided by the Center for Intelligent Maintenance Systems (IMS), University of Cincinnati.
Acknowledgements

J. Lee, H. Qiu, G. Yu, J. Lin, and Rexnord Technical Services (2007). IMS, University of Cincinnati. "Bearing Data Set", NASA Ames Prognostics Data Repository (http://ti.arc.nasa.gov/project/prognostic-data-repository), NASA Ames Research Center, Moffett Field, CA



## Vibration Analysis on Rotating Shaft
Unbalance Detection of a Rotating Shaft Using Vibration Data


### About Dataset
Introduction

The vibration is defined as cyclic or oscillating motion of a machine or machine component from its position of rest.

The use of machinery vibration and the technological advances that have been developed over the years, that make it possible to not only detect when a machine is developing a problem, but to identify the specific nature of the problem for scheduled correction.

Fault detection at rotating machinery with the help of vibration sensors offers the possibility to detect damage to machines at an early stage and to prevent production down-times by taking appropriate measures.

### How Data is collected ?

The setup for the simulation of defined unbalances and the measurement of the resulting vibrations is powered by an electronically commutated DC motor (WEG GmbH, type UE 511 T), which is controlled by a motor controller (WEG GmbH, type W2300) and is fixed to the aluminum base plate by means of a galvanized steel bracket.

 Vibration sensors (PCB Synotech GmbH, type PCB-M607A11 / M001AC) are attached to both the bearing block and the motor mounting and are read out using a 4-channel data acquisition system (PCB Synotech GmbH, type FRE-DT9837).

### The DataSet

Using the setup described in above section, vibration data for unbalances of different sizes was recorded. The vibration data was recorded at a sampling rate of 4096 values per second. By varying the level of unbalance, different levels of difficulty can be achieved, since smaller unbalances obviously influence the signals at the vibration sensors to a lesser extent.

In total, datasets for 4 different unbalance strengths were recorded as well as one dataset with the unbalance holder without additional weight (i.e. without unbalance). The rotation speed was varied between approx. 630 and 2330 RPM in the development datasets and between approx. 1060 and 1900 RPM in the evaluation datasets. Each dataset is provided as a csv-file with five columns:

1. V_in         : The input voltage to the motor controller V_in (in V)
2. Measured_RPM   : The rotation speed of the motor (in RPM; computed from speed measurements using the DT9837)
3. Vibration_1      : The signal from the first vibration sensor
4. Vibration_2     : The signal from the second vibration sensor
5. Vibration_3     : The signal from the third vibration sensor

Overview of the dataset components:

```
ID 	Radius [mm] 	Mass [g]
0D/ 0E 	- 	-
1D/ 1E 	14 ± 0.1 	3.281 ± 0.003
2D/ 2E 	18.5 ± 0.1 	3.281 ± 0.003
3D/ 3E 	23 ± 0.1 	3.281 ± 0.003
4D/ 4E 	23 ± 0.1 	6.614 ± 0.007
```


In order to enable a comparable division into a development dataset and an evaluation dataset, separate measurements were taken for each unbalance strength, respectively.

This separation can be recognized in the names of the csv-files, which are of the form “1D.csv”: The digit describes the unbalance strength (“0” = no unbalance, “4” = strong unbalance), and the letter describes the intended use of the dataset (“D” = development or training, “E” = evaluation).

### Acknowledgements

Special thanks to Oliver Mey, Willi Neudeck, André Schneider and Olaf Enge-Rosenblatt for their effort. Please do citation

@inproceedings{inproceedings,
author = {Mey, Oliver and Neudeck, Willi and Schneider, André and Enge-Rosenblatt, Olaf},
year = {2020},
month = {09},
pages = {1610-1617},
title = {Machine Learning-Based Unbalance Detection of a Rotating Shaft Using Vibration Data},
doi = {10.1109/ETFA46521.2020.9212000}
}

Research paper : https://arxiv.org/abs/2005.12742
Image Credit : Jonathan Borba
Inspiration

    Found out that the data related to vibration analysis is fewer and fewer.
    Bring a non-popular dateset to main stream.
    An appreciation to above mentioned team and their kindness for making this data publicly available.

Keywords : Vibration, Motor, Condition Monitoring, Preventive Maintenance, Mass Unbalance, Rotating Machinery, Deep learning, Machine Learning, Real world, Spectrum Analysis, Signal Processing, Mechanical Data.
NOTE :

The analysis of vibrations on rotating shafts to detect unbalances or to detect damage to roller bearings has proven to be very promising.
    
Unbalances on rotating shafts can cause decreased lifetimes of bearings or other parts of the machinery and, therefore, lead to additional costs. Hence, early detection of unbalances helps to minimize maintenance expenses, to avoid unnecessary production stops and to increase the service life of machines.


## Vibration analysis - bearing, belt, and EDM faults
These sound files were taken from a database that spans over 2,500 machines.


### About Dataset

These are very noisy sound files were taken from a database, where I have labeled the files by the type of fault, a building number, and the point on the equipment, as well as date and time of the recording are in the file title. The way we use these files is by reviewing the Waveform as well as the FFT of the file and the trend value in the velocity spectrum of an overall value. When speech recognition was using Waveform and FFT the end result was poor at best, when ML was used the results got much better. I intend on adding more files as time permits. The files that I have uploaded are all 2 seconds long which represents many revolutions of the motor so each file could be split up into approximately 30 to 60 files to be used for ML.
Context

There's a story behind every dataset and here's your opportunity to share yours.
Content

What's inside is more than just rows and columns. Make it easy for others to get started by describing how you acquired the data and what time period it represents, too.

### Acknowledgements

We wouldn't be here without the help of others. If you owe any attributions or thanks, include them here along with any citations of past research.
Inspiration

Your data will be in front of the world's largest data science community. What questions do you want to see answered?
