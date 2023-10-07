# Master's thesis
- **Title:** Machinery vibrodiagnostics with the industrial internet of things
- **Author:** Bc. Miroslav Hájek
- **Supervisor:** Ing. Marcel Baláž, PhD.
- **Deadline:** June 2024


___
## Research questions
Instead of recording complete signals as is, a process should be devised that records
only key descriptions based on vibrational signal specifics, namely different machine
parts:

1. Which time-frequency features can be extracted from vibrational signals to provide an accurate record of machinery faults?
2. What are the savings in transmission bandwidth when chosen signal features
are used in comparison to raw sampled measurement or lossless compression
techniques?
3. How can the machinery faults be continuously identified based on collected
events?

___
## Analysis outline
- Condition monitoring
	+ Maintenance strategies
	+ Typical machinery faults
	+ Technical standards
- Feature engineering
	+ Signal preprocessing
	+ Feature extraction - Statistical measures, Signal decompositions
	+ Feature transformations
	+ Filter methods for feature selection
- Diagnosis techniques
	+ Novelty detection
	+ Online classification
- Evaluation datasets
	+ MAFAULDA
	+ CWRU
	+ Rotating shaft
- Sensor unit

___
## Goal
Alert mechanism in the factory for machinery faults (asset evaluation according to PdM objectives) that can notify operator about abnormal behavior (later with continuous learning to point to probable source of error) and provide expert means to track down specific fault.

### Data flow preliminary design
1. MEMS accelerometers are placed on at least two distinict measurement points in two perpendicular axis and one sensor in base for denoising. Rotational speed is also captured.
2.  Sensors are triggered in regular intervals (every 15 minutes) to collect sample recording from the *band saw* (machine of interest).
3. Features are computed and compared to recent measurements. If there is an statistically significant change the whole summary (feature vector) is send, otherwise keepalive notification is send.
4. *Possible local communication between sensors to pre-compute clustering / classification information*
5. Database stores history of measurements
6. Diagnosis panel notifies the operator about observed abnormal behavior or possible fault.

### Research goal
- Divise domain specific feature vector (custom metric/method) that captures faults
in way similiar to how human expert of vibrodiagnostics would analyze the frequency spectrum
- propose some kind spectral segmentation based of peak/energy/envelope extraction.
- Evaluate the invented method is optimal in comparision to regular statistics

---
## Feature extraction datasets
- [**Features**](https://drive.google.com/drive/folders/19PlPF5jPp-z7-0l0es14bSgUsELUx3BR?usp=sharing)

____
## Evaluation datasets
- [**MAFAULDA**](https://www02.smt.ufrj.br/~offshore/mfs/page_01.html) - Machinery Fault Database (12.0 GiB)
	+ **Download (in STU domain):** https://drive.google.com/file/d/1AEWEamDs8i9qq3KVYVLhS4hmQwINwXjP/view?usp=share_link
- [**CWRU**](https://engineering.case.edu/bearingdatacenter) - Ball bearing test data for normal and faulty bearings (368.7 MiB)
	- **Download (in STU domain):** https://drive.google.com/file/d/10aCHnOETzRCFiV9HsRROziHCthxW35zf/view?usp=share_link
- [**Rotating shaft**](https://www.kaggle.com/datasets/jishnukoliyadan/vibration-analysis-on-rotating-shaft) - Unbalance Detection of a Rotating Shaft Using Vibration Data (2.6 GiB)
	+ **Download (in STU domain):** https://drive.google.com/file/d/1vgK-Fi8g4GntY1CN3wfHf3fONxbZr6pa/view?usp=share_link

