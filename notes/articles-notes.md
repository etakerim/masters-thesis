## Condition monitoring

https://scikit-multiflow.github.io/
https://riverml.xyz/

### thomson_theory_1993

- All bodies possessing mass and elasticity are capable of vibration. Thus, most engineering machines and structures experience vibration to some degree, and their design generally requires consideration of their oscillatory behavior. 
- s.58 - rotor unbalance 
- Chapter 12 (Classical methods) - The exact analysis for the vibration of systems of many degrees of freedom is generally difficult and its associated calculations are laborious. Many DOF - the results beyond the first few normal modes are often unreliable and meaningless.
- For this purpose, Rayleigh's method (fundamental frequency of multi-DOF systems) and Dunkerley's equation are of great value and importance. In many vibrational systems, we can consider the mass to be lumped. A shaft transmitting torque with several pulleys along its length is an example. 
- Holzer devised a simple procedure for the calculation of the natural frequencies of such a system. Holzer's method was extended to beam vibration by Myklestad and both methods have been matricized into a transfer matrix procedure by Pestel.

### ziaran_technicka_2013
- p.36 - Regular measurement of mechanical oscilatory motion we can find out beginning of the fault (or future failure) and monitor its progress. Method of trend tracking - extrapolation.
- p.40 - normal level, level indicating significant change, level coresponding to fix
- p.50 - the total magnitude of the oscillation can be compared with reference values set by norms or defined for each type of the machine.
- p.54 - oscillation measurement allows us to quantify dynamic load capacity of mechanical systems a its analysis.
- p.100 - principles of oscillation measurement - choice of sensor type, exclusion of errors caused by resonance, placement and direction of the sensor, mounting of acceleration sensor, the influence of the environment
- p.110 - types of signals - periodic, quasiperiodic, nonstationary
- p.114 - transmission of the oscillation signal through the environment - through discontinuity and construction
- p.139 - signal averaging - linear, exponencial, peak based
- p.148 - kurtosis - high kurtosis means more extreme values - peaks 
- p. 153 - Vibroacustic diagnosis is one of the most important methods of early identification of component faults
- p.165 - severity levels for vibrartion amplitudes (same as in ISO standard). 
- p.178 - reference mask
- p.185 - overview of root causes of machine faults - imbalance, misalignment,  resonance, excentricity, loosnessnes, 
- p.243 - displacement, speed, acceleration
- p.253 - Quanitative evaluation criteria - band A, B, C, D
- p.256 - warning, alarm
- p.261 - characteristics of typical faults
- The descriptor variable can be any meaningful statistical quantity, e.g. peak-to-peak, RMS, crest factor, kurtosis, which can be applied to recorded samples or frequency bands.


### davies_techniques_2012
- p.36 - Condition monitoring is much more than a maintenance scheduling tool. Condition monitoring is thus a management technique that uses the regular evaluation of the actual operating condition of plant equipment, production systems and plant management functions, to optimize total plant operation. The output of a condition monitoring programme is data. Until action is taken to resolve the deviations or problems revealed by the programme, plant performance cannot be improved
- p.40 - This data is compared to either a baseline reading taken from a new machine or to vibration severity charts to determine the relative condition of the machine. Normally an unfiltered broadband measurement that provides the total vibration energy between 10 and 10000 Hertz (Hz) is used
- p.38 - Condition monitoring utilizing vibration signature analysis for example is predicated on two basic facts: 
- 'All common failure modes have distinct vibration frequency components that can be isolated and identified.'
-  'The amplitude of each distinct vibration component will remain constant unless there is a change in the operating dynamics of the machine train.'
- p. 269 - Whenever vibration occurs, there are actually four forces involved that determine the characteristics of the vibration. These forces are: 1. the exciting force, such as unbalance or misalignment; 2. the mass of the vibrating system; 3. the stiffness of the vibrating system; 4. the damping characteristics of the vibrating system.
- p. 294 - Even machines in the best of condition will have some vibration and noise associated with their operation due to minor defects. It is therefore important to remember that:
	 1. every machine will have a level of vibration and noise which is regarded as inherent normal; 
	 2. when machinery noise and vibration increase or become excessive, some mechanical defect or operating problem is usually the reason; 
	 3. each mechanical defect generates vibration and noise in its own unique way.
- p. 300 -machinery vibration signatures taken in the horizontal, vertical and axial directions at each bearing point on the machine; imbalance will show up predominantly in the horizontal or vertical directions, whereas misalignment will reveal relatively high amplitudes in the axial direction.
- p. 301 
- When selecting the measurement parameter for analysis, there are two factors to consider. First, why is the reading being taken? Secondly, what is the frequency of the vibration?
- For general machinery analysis, use velocity whenever possible. Amplitude readings in velocity are directly comparable to severity without the need to cross refer amplitude with frequency. Where vibration frequencies are very low (approximately below 600 CPM
- p. 308 - The crest factor, defined as the ratio of the peak to RMS levels, has been proposed as a trending parameter as it includes both parameters (Braun, 1980). However, investigations by the author have shown that this parameter usually increases marginally with incipient failure, and subsequently decreases due to the gradually increasing RMS value typical of progressive failure. Quite often, the trend recorded by this parameter has been found to be similar to another time domain parameter, the Kurtosis factor
- p.312 Kurtosis - sensitive to failure in rolling element bearings (Dyer and Stewart, 1978). However, an independent evaluation of this technique (Mathew and Alfredson, 1984) has shown that high Kurtosis values would only be obtained if the original time waveform was of an impulsive nature
- p.326 - evaluate changes occurring in the signature spectrum is to form a spectral mask. This is derived from the baseline signature, plus an allowable tolerance limit (Randall, 1981a)

### What are predicted variables - result of diagnoses
- Presence of the Fault
- Type of fault present (different characteristics - e.g. frequency content)
- Remaining Useful Life (time until failure) - machines of the same type and different degradation curves

### Remainig useful life models} (RUL) 
- is the expected life or usage time remaining before the machine requires repair or replacement.
- https://www.mathworks.com/help/predmaint/ug/rul-estimation-using-rul-estimator-models.html

- Similarity - run to failure history of similiar machines in database
- Degradation - known failure threshold (warning, alert threshold)
- Survival  - life-span of components and correlated variables


### jung_vibration_2017
- Indirect Measurement: indirect and approximate measurement over the vibration phenomenon of the target equipment.
- Noisy and Unaligned Observations: well aligned / may contain huge amount of noise.
- Variance on Initial Status: initial status of the target equipment different from each other.
- Diversity on Lifetime model: the usage and lifetime model -  number of unknown and external factors.


### mohanty_machinery_2015
- Difference between fault (degrating performace of the machine - higher friction and power consumption) and failure (machine is unusable). (picture)
reaction-based, time-based, or condition-based maintenance

- **Reactive** - run equipment until failure occurs - low stakes operation. Failure can have negative economic impact or can damage adjacent parts
- **Preventive** - predetermined schedule when assets are diagnosed and repairs are made. Crutial to set appropiate maintanance interval. Good parts are replaced before they are completely worn out, preventing critical failure, but creating unneccesary waste. Sometimes faults are not detected soon enough. industrial average life statistics, such as Mean Time To Failure (MTTF),
- **Predictive** - model of expected lifetime, warns about unexpected faults before they become too serious and before affecting the machine.


- During operation, machines give out information or signals in the form of noise, vibration, temperature, lubricating oil condition, quality and quantity of the motor current drawn, and the like
- Once the faults have been detected and diagnosed, the next question is how long the machine will last in the present condition or what is the remaining useful life (RUL) of the machine under observation.

Wear process curve Bath tub curve (page 10)
- **Initial** - large roughness
- **Normal** - contact area fomred
- **Severe** - high friction


- p.11 - Failure Modes Effects and Criticality Analysis (FMECA) FMECA is a methodology widely used in the industry to identify and analyze all potential failure modes of the various parts of a system
- p.30 - Simple Rigid Rotor-Disc System (Jeffcott rotor) - shaft that rotates and is supported at two ends by bearings with disc
- p.26 - machines are designed so that the system’s natural or resonance frequencies are not close to the machine’s operating speed.
- p.93 - In a majority of the rotating machinery, vibration monitoring is preferred. This is due to the fact that every dynamic machine component will manifest itself in the measured vibration response of the machine at its characteristic frequencies. Bearing Coupling, Motor, Mechanical drive
- p.93 - vibrations at any location should be measured in three mutually perpendicular directions. Simultaneous measurement of the vibration in three directions can be done using a triaxial accelerometer. It  is always desirable to record the rotating speed of the  shafts at the instant that vibration measurements are  done, because all the predominant frequencies in the vibration spectra are related to the rotating speeds of the shaft
- p.97 - Eccentric rotors occur when the geometric center of the rotor does not coincide with the center line through the support bearings. crack growth rate is determined by the famous Paris equation, which depends on the material type and type of loading
Unbalance is due to a net uneven distribution of the mass of a rotating component about its axi
- p. 101 - looseness manifests as high levels of impulsive vibrations in the time domain. The typical frequency domain characteristics of looseness in rotating machines is the presence of vibration peaks at fractional harmonics of the rotational speed and their harmonics.

- p. 216 Appendix A1: Vibration-Based Machinery Fault Identification
Shaft/Rotor Unbalance , Misalignment, Looseness, Rub, Crack, Bearings Journal, Rolling Element, Fan/Blowers, Pulley/Belt, Electrical Motor
Appendix A2: Vibration Severity Levels

- Rotordynamics (chapter 4) (p. 29)  - p.97 - Fault types, p.127 - faults in electric motors

- Campbell diagram, Bode plot, forced spring-mass-damper system

- In order to understand, and correctly diagnose the vibratory characteristics of rotating machinery, it is essential for the machinery diagnostician to understand the physics of dynamic motion. This includes the influence of stiffness and damping on the frequency of an oscillating mass — as well as the interrelationship between frequency, displacement, velocity, and acceleration of a body in motion.
- p.395 - Common malfunction   F = ma, torque, $$F_centrifugal = m * r(deviation) * RPM^2$$, $$v = v_0 + \int a$$
- The synchronous, or running speed, or fundamental, or 1X motion of a rotating element is an inherent characteristic of every machine. It should be recognized that all machines function with some level of residual unbalance. The radial forces from an eccentric element will vary with the speed squared as described by expression

- **SYNCHRONOUS RESPONSE (p.395)** - physically impossible to produce a perfectly straight and concentric rotor.vibration response is inversely proportional to the restraint or stiffness when the applied force is held constant
- **MASS UNBALANCE (p.398)** - Mass unbalance represents the most common type of synchronous excitation on rotating machinery. Every rotor consists of a shaft plus a series of integral disks. high dimensional tolerances, a residual unbalance is present in each element. Centrifugal force
- **BENT OR BOWED SHAFT (p.400)** - Bent rotors and shaft bows represent another major class of synchronous 1X motion. It was previously mentioned that all machine parts contain some finite amount of residual unbalance. In a similar manner, all assembled horizontal rotors (and some vertical rotors), will exhibit varying degrees of rotor bows.
- **ECCENTRICITY  (p.406)** - Large machine elements or high rotational speeds are the most susceptible to high forces due to an eccentric element. many motor problems only appear under load.
- **SHAFT PRELOADS (p.410)** - The presence of various types of unidirectional forces acting upon the rotating mechanical system is a normal and expected characteristic of machinery. Just as residual unbalance, rotor bows, and component eccentricity are inherent with the assembly of rotating elements, the presence of shaft preloads are an unavoidable part of assembled mechanical equipment. Gravitational preloads
- **RESONANT RESPONSE (p.416)** - Machines and structures all contain natural frequencies that are essentially a function of stiffness and mass. f = sqrt(k/m) lowest order resonant frequency. For more complex mechanical systems an entire family of resonant responses must be addressed. 

The range of natural frequencies may vary from 60 CPM (1 Hz) for the foundation and support systems, to 1,800,000 CPM (30,000 Hz) for turbine blade The actual number of system natural frequencies may vary from 20 to 50 or more. Campbell diagram  -  natural frequencies (or eigenfrequencies)

#### Rotodynamics
$$ \mathbf{M} \frac{\partial^2 \mathbf{q}(t)}{\partial t^2} + (\mathbf{C} + \mathbf{G})\frac{\partial\mathbf{q}(t)}{\partial t} + (\mathbf{K} + \mathbf{N})\mathbf{q}(t) = \mathbf{f}(t) $$

 - M is the symmetric Mass matrix
 - C is the symmetric damping matrix
 - G is the skew-symmetric gyroscopic matrix
 - K is the symmetric bearing or seal stiffness matrix
 - N is the gyroscopic matrix of deflection for inclusion of e.g., centrifugal elements.
- in which q is the generalized coordinates of the rotor in inertial coordinates and f is a forcing function, usually including the unbalance. axially symmetric rotor rotating at a constant spin speed $\Omega$. The gyroscopic matrix G is proportional to spin speed $\Omega$. The general solution to the above equation involves complex eigenvectors which are spin speed dependent.
- **ROTOR RUBS (p.440)**- The physical contact between rotating elements and stationary machine parts can generate a variety of rub conditions In the frequency domain, the intermittent rub looks like a loose bearing housing with integer fractions of rotative speed (e.g., X/2, X/3, or X/ 4) plus a string of fractional frequencies (e.g. 3X/2, 5X/2, 7X/2, etc.)
- **CRACKED SHAFT** -Machines that are subjected to frequent startups and shutdowns appear to be more susceptible to shaft cracks due to the increased number of cycles through the rotor resonance(s), plus the process heating and cooling. Cracks may originate at high stress points such as the square corners
- p.704 - maintenance activities may be categorized as either reaction-based, time-based, or condition-based maintenance

Machinery Diagnostic Methodology (s.747)
	- Diagnostic objectives - solving the problem
	- Mechanical inspection - hands on examination of the machinery
	- DATA ACQUISITION AND PROCESSING - requires the assembly of the proper transducers and test instrumentation. ata acquired is really a function of the specific machine, the associated problem, and the individual test plan
	- Data interpretation - summary and correlation of all pertinent data acquired during the project. This includes the mechanical configuration, process and maintenance history, field testing data, plus any supportive calculations or analytical models
	
The 30,000 horsepower gas turbine behaves quite differently from a similarly rated steam turbine. Hence, the data must be interpreted in accordance with the physical characteristics of the particular machine type and the operating environment. One approach is to view the data in terms of normal behavior for a particular machine type, and then look for the abnormalities in response characteristics.



### eisenmann_machinery_1997
- Forced Vibration Mechanism
	- Mass Unbalance
	- Misalignment
	- Shaft Bow
	- Gyroscopic
	- Gear Contact
	- Rotor Rubs
	- Electrical Excitations
	- External Excitations
- Free Vibration Mechanism
	- Oil Whirl
	- Oil or Steam Whip
	- Internal Friction
	- Rotor Resonance
	- Structural Resonances
	- Acoustic Resonances
	- Aerodynamic Excitations
	- Hydrodynamic Excitations


### popescu_blind_2010
- The analysis of the behavior of such signals reveals that the most of the changes that occur are either changes in the mean level, or changes in spectral characteristics. In this framework, the problem of segmentation between ‘‘homogenous” part of the signals (or detection of changes in signals) arises more or less explicitly
- the key for their common success resides in the proper design of the statistical criteria according to which separation is forced
- It can be noted a moving of the spectra to the low frequency area, with increased values of the PSD, after the fault produced


- Vibration fault types
There are a few methods of machinery fault identification in vibrational signals based on domain expertise. Data points can be viewed in the time domain and frequency domain. Either as individual stationary profiles obtained during the short duration in the time of measurement, or multiple spaced-out observations with the intent to highlight the long-term trend, e.g. shown in a waterfall plot 

Mechanical faults manifest themselves in the vibration signal at various frequencies. In the low-frequency range (up to 1 kHz) shaft's unbalance, misalignment, bend, crack, and mechanical looseness is present. High frequencies (up to 16 kHz or more) contain bearings faults and gear faults.

Under fault-free circumstances, shaft speed appears as the strongest frequency component. In case of shaft and gear imbalance or damage, synchronous multiples of shaft frequency (harmonics) are amplified. When rub, bad drive belts and chains, or looseness is occurring in the machine then sub-synchronous harmonics or even non-synchronous frequencies appear \cite{mohanty_machinery_2015}.  Therefore it is useful to rescale the horizontal axis to RPM or orders of rotational speed. Complementary methods of fault symptom identification are phase and orbital analysis \cite{scheffer_practical_2004}.


- Bearing faults - vibration on each rotation of rolling elements, CFC (characteric fault frequencies with impulse
\item Rotor bar faults - current will not flow - forces diffrent on both sides of rotor
- Eccentricity Faults - uneven air gap between stator rotor
- Misalignment - parralel / angular
- Cavitation - pumps
- Gearbox fault -broken teeth

measuring vibration with current, thermal, flux is improvement, +acoustic elminited (detect similiar faults)
vibration is better alone, then other methods alone (80 vs. <60%)
 
### goel_methodology_2022

\subsection{Technical standards}
The maintenance procedure usually involves data acquisition cards inside handheld devices with accelerometer sensor probes then mounted firmly to the machine frame by either screwing in, magnets or wax \cite{ziaran_technicka_2013}. The probe placement in axial and perpendicular radial directions is standardized in ISO 20816. The severity of vibrations is mostly assessed in units of velocity ($mm/s$), but acceleration ($m/s^2$) and displacement ($\mu m$) are also used. Based on the observed vibration intensity and one of the four classes of machines (I, II, III, IV) by output power and size, zones (A, B, C, D) for accepted levels are proposed. It is customary to establish operational limits in the form of alarms and trips \cite{iso_20816}.

Standard ISO 13373 categorizes three types of vibration monitoring systems: permanent, semi-permanent, and mobile. More importantly, a structured diagnostic approach is developed here complete with recommendations for formalizing diagnostic techniques \cite{iso_13373}. The next step is the signal analysis with the use of proper units and transformations is the subject of the ISO 18431 \cite{iso_18431}.

ISO-10816 Vibration Severity Chart (include table)


Typical faults produce unusual low-frequency vibrations (10 to 1000 Hz).
Imbalances, misalignments and looseness are recorded at frequencies up to 300 Hz.

### Sensor placement  
- Frequency Limitations Resulting from Mounted Resonance of an Accelerometer, Jack D. Peters
- Axial, Radial (Standard)
- Mounting resonance is a direct result of lowering the accelerometers natural frequency and occurs as the result of reduced stiffness or increased mass. $f_n = \frac{1}{2\pi}\sqrt{\frac{k}{m}}$
- Under ideal circumstances, the accelerometer mounting should provide total use of the transmission region
- p.4 
	- Probe tip (500 Hz)
	- Curved surface magnet (2 kHz)
	- Quick disconnect (6.5 kHz)
	- Flat magnet (10 kHz)
	- Adhesive mount (10 - 15 kHz)
	- Stud mount (15 kHz)









### goumas_classification_2002

- Mapping measurement sapce into feature space
- Pattern classification - partitioning feature space into decision subspaces
- Feature vector - point in N-dimensional feature space - assign to pattern class
- Classification of N-dimensional feature space with M classes may be viewed as a problem of defining hyperplanes to divide N-dimensional Euclidian space into M regions.
- Pattern recognition stages: measurements, feature extraction, classification
- Washing machines 500 Hz, 20 signals
- 8 measurement poinst - Vibrartion on lower part of machine is atenuated because of contant effect with ground - p.7
	Types of machines in population:		
	- z machines - no fault, 
	- b machines - electric motor clamping screws problems
	- p machines - counter weight distorted (loose, broken)

- Daubechies Wavelet function 4 (D4) with fifth-level decomposition FWT  - Detail coeficient (abrupt changes as local variations in coef.) and last-level approx. coef.
	- Autocorrelation function from coeficients - p.8 -> Moving average filter
	- S1, S2 result of moving average filtering on cD1, cD2 (DWT detail coeficients) 
	- Karhunen–Loève transform - PCA transformsoriginal variables into new set of uncorrelated variables called Principal components (PCs) - p.9
	- Bayesian classification, 87\% - Naive Bayes


Aggarwal - Data clustering
In feature selection, original subsets of the features are selected. In dimensionality reduction, linear combinations of features may be used in techniques such as principal component analysis in order to further enhance the feature selection effect. The advantage of the former is greater interpretability, whereas the advantage of the latter is that a lesser number of transformed directions is required for the representation process.

Feature extraction - PCA on features to find most separation 
	- Reduce dimensionality 
	- Singular Value Decomposition (QR algorithm) - eigenvalue algorithm , 


p.232 - Streams typically have massive volume, and it is often not possible to store the data explicitly on disk. Therefore, the data needs to be processed in a single pass, in which all the summary information required for the clustering process needs to be stored and maintained.

### zhuo_research_2022 
### zheng_feature_2018
Statistical features in Time-domain (and correlation to blade wear) 

p. 254 furthest point algorithm - local clustering at each node and merges these different clusters into a single global clustering at low communication cost.

- Root mean square (0.98)
- Mean (0.17)
- Amplitude (0.81)
- Kurtosis (0.042)
- Peak to peak (0.463)
- Signal strength (0.119)
- Standard deviation (0.908)
- Peak value (0.488)
- Shape factor (0.007)
- Skewness (0.118)
- Avearge signal level (0.46)
- Crest factor (0.056, spikeness of the signal - rms/amplitude)

Selection according to high correlation (graph: sawn-trough section vs feature)
Features in time domain with high correlaction: RMS, Standard deviation, Amplitude

Statistical features in Frequency domain (PSD analysis) r >= 0.8 db3 analysis
- Root mean square (0.402)
- Mean (0.497)
- Peak frequency (0.670)
- Kurtosis (0.852)
- Peak to peak (0.076)
- Standard deviation (0.799)
- Peak value (0.787)
- Shape factor (0.851)
- Skewness (0.819)
- Frequency centroid (0.775)

skewness (PSD_S), kurtosis (PSD_k), shape factor (PSD_Sf),
centroid frequency (FFT_fc), wavelet packet energy entropy (WPD_EP) = 0.85
- The WPD energy E8, E10, and E12
- Energy ratios P8 and P13 of frequency bands 8 and 13


### peeters_large_2004
Spectral features  1. Spectral shape description

- Coherence function - correlation between two signals PSD
- Spectral centroid - barycenter of the spectrum (weighted mean of the frequencies present in the signal, with their magnitudes as the weights)
- Spectral spread
- Spectral skewness
- Spectral kurtosis
- Spectral slope - comupted with linear regression - amount of decresing of the spectral amplitude
- Spectral roll-off - 95\% of the signal energy is contained below this frequency
- 2. Temporal variation of spectrum - spectral flux - correlation of normalized cross-correlation between two succesive amplitude spectra


**Harmonic features**
- Fundamental frequency  - Maximu likelihood algorithm
- Noisiness - ratio - energy of noise to the total energy
- Inharmonicity - energy weighted difference of the spectral components from the multiple of fundamental frequency
- Harmonic Spectral Deviation - deviation of amplite harmonics peaks from global spectral envelope



### lagrange_robust_2010
measure the \textbf{spectral flatness} of the ratio between the observations (the peaks) with respect to the model (the spectral envelope)


### jung_vibration_2017
\textbf{Harmonic peak feature} - 
- RUL estimation - Harmonic peak distance is score from baseline - create probability density functions (with Recursive RANSAC regression algorithm - because of noisy distribution) of zones A,  BC, D
and cut on transition to zone D (around >0.21 Peak distance)
- group of pairs of significant peaks’ value and frequency in PSD (p, f)
- Distance is euclidian based - closest peak frequency and value
- Harmonic peak distance - compare baseline harmonic peak feature (20 peaks - points where its first order differential changes from positive to negative-  from smooth spectrum by 16 point Hann window) 


### li_fault_2019
p.8 - rolling element bearing equation mechanics
Detect bearing faults - impulses (transients)
Maximum Correlated Kurtosis Deconvolution (MCKD)

- FIR filter maximizing the CK (Kurtosis) of the impulses - Result coeficients of the filter
- Empirical Wavelet Transform 
	- address the mode mixing or over-estimation phenomenon of the EMD
	- EWT divides the spectrum into several portions, and each portion corresponds to a mode centered at the specific frequency and compact support, such as AM-FM signal
- Algorithm MKCD-EWT
- De-noise the signal by MCKD.
- Spectrum segmentation. Calculate the envelope curve of the amplitude spectrum of the de-noising signal.
- Signal decomposition. Design the wavelet filter banks
- IMF (Intrinsic Mode Function) selection. Calculate the kurtosis of each sub-signal
- Feature extraction. Calculate the squared envelope spectrum and teager energy operator spectrum of the chosen mode
- Highest kurtosis values of these modes IMF1 - 4 in the largest IMF

- Teager-Kaiser operator (TKEO)
- Teager Energy Operator (TEO)
- $x(t)  = (dx/dt)^2+ x(t)(d^2x/dt^2) $
- $[x[n]] = x^2[n] + x[n - 1]x[n + 1]$ TKEO
- When Ψc is applied to signals produced by a simple harmonic oscillator, e.g. a mass spring oscillator who’s equation of motion can be derived from the Newton’s Law - It can track the oscillator’s energy

### A Novel Online Machine Learning Approach for Real-Time Condition Monitoring of Rotating Machines
- These algorithms call **novelty detectors**: When we want to identify a machine's abnormal behavior in the real world, we do not have access to all its possible faults characteristics. So we need an algorithm that can**learn a machine's healthy behavior and discriminate any faults.**
- Each machine has unique characteristics, and so this process should be done repeatedly for each machine which requires monitoring
- These models can be trained exclusively by normal data of machines and identify any abnormalities in working conditions.
- an autoencoder combined with a Long short-term memory (LSTM) neural network was used to extract features from raw vibration signals and detect anomalies in rotating machines
- the network is first trained on highperformance and powerful computers and then flashed to MCUs as just matrices of numbers, which means that these libraries do not support on-device training
- layer called TinyOL attached to the final layer of the autoencoder.
- The MCU runs at 480 MHz and has 1 MB RAM and 2 MB Flash memory used to store the code.
- (STM32H743ZI2) with ARM Cortex-M7 core, a digital 3axis accelerometer (LIS3DSH)
- 1600-Hz sampling frequency and 16-bit resolution, and it has an onboard anti-aliasing filter with 800 Hz cut-off frequency. The Xbee module can send and receive data at 250,000 bps rate
- Feature engineering is extracting useful information from raw data, and it is considered the cornerstone of successful anomaly detection. It is used to reduce data dimensionality and remove nullities in a data set.
- The most prominent features are designed to reduce the dimension of the sample data and extract the most effective information from the raw vibration data of rotating machines - manual selection of features
- time-frequency domain features are considered the best features when it comes to non-stationary signals 
	- Short-Time Fourier Transform (STFT)
	- Continuous Wavelet Transform (CWT)
	- Discrete Wavelet Transform (DWC)
	- **Wavelet Packet Decomposition (WPD)**
- **Processing:** 
	1. from each axis has 2048 sample points. 
	2. the DC gain of each axes has removed by subtracting the mean of the array from each sample point. 
	3. Subsequently, the total 29 features are extracted from each axis and stored in an array with a length of 87
- **Time domain features**: (p.3)
	+ Root Mean squared of signal (RMS)
	+ Square root of the amplitude
	+ Kurtosis value
	+ Skewness Value
	+ Peak-peak value
	+ Crest factor
	+ Impulse Factor
	+ Margin Factor
	+ Shape Factor
	+ Kurtosis Factor
- **Frequency domain features**:
	+ Frequency center
	+ RMS frequency
	+ Root variance frequency
- Common classification methods can not be used because their training requires labeled data from both faulty and healthy states of the machine, which is not available in most cases. In this situation, novelty detection algorithms are the best choice

### yu_concentrated_2020
- spectral kurtosis (SK) method, synchrosqueezing transform (SST)
- Transient-extracting operator (TEO) - from Dirac delta
- transient-extracting transform (TET)
- $ TEO(t, \omega) = \delta(t − t_0(t, \omega))$
- $ Te(t, \omega) = G(t, \omega) \cdot TEO(t, \omega) $ 

\subsection{Power spectral density segmentation}

#### Denoising
- SNR - The traditional approach to calculating SNR is to measure the average Active level, subtract the average Inactive
level, and divide that result by the peak level of noise witnessed on the Inactive level. 

### bechhoefer_review_2009
- Shift mean - remove by averaging filter
- Standardization - Min-max scaler, Standard scaler (clustering - feature have different scales)
- Transformation - Log transformation, Box-Cox
- Adaptive Noise Canceling -  least mean squares (LMS) filter - stocahstic signals
- ICA (independent component analysis) - FastICA, JADE - denoising from enviroment - Same as Adaptive Noise Cancelation, 		- exploitation of the spatial diversity provided by many sensors and is the fundamental basis of BSS. finding an estimate S (n) of the sources X (n) by adapting an unknown separating function which leads to independence of S (n). 
	- instantaneous model can hold when the structure under investigation has a high rigidity and a small size. aim of the problem; i.e., to "nd a linear transformation, which relies on independent components (sources) contributing to the observations of a mixture of them
	- In short, the BSS problem cannot be solved by using only second order statistics because independence is a stronger condition than uncorrelation.
	- real recordings, it is very diffcult to measure the separation quality. Here, prior knowledge about the sources was used; that is, harmonic frequencies in relation to the mechanical components as well as the signals recorded on each source separately in the real environment (the reference).
	- 20000 samples were considered for each frequency channel to perform separation (fs = 500 Hz) -> 40 s
\item TSA - 
	- TSA and show that noise reduction is 1/sqrt(number of revolutions). 
	- The third topic examines TSA techniques when no tachometer signal is available. 
	- it allows the vibration signature of the gear under analysis to be separated from other gears and noise sources in the gearbox
	- phase information can be provided through a n per revolution tachometer signal (such as a Hall sensor or optical encoder) or though demodulation of gear mesh signatures	



- Averaging - Time Synchronous Averaging, Welch method
- TFA (time-frequency analysis) - Real FFT
- Empirical Mode Decomposition na Intrinsic Modes - EMD/EEMD - Each IMF represents a narrow band frequency - amplitude modulation that is often related to a specific physical process (mode mixing phenomenon),\cite{wang_computational_2014}

- SST - originally introduced in the context of audio signal analysis and is shown to be an alternative to EMD -  \cite{herrera_applications_2014}

- Fast CWT
- Synchrosqueezing - SST - extension of the wavelet transform incorporating elements of empirical mode decomposition and frequency reassignment techniques
- EWT 
- Peak identification, Spectral Envelope


major drawbacks of PSD
- PSD is a highdimensional feature (i.e., 1024 dimensions in our case) that often generates singular matrix = regression algorithms.
- PSD feature is unreliable due to a large random fluctuation in their amplitudes over frequency due to measurement noise inherent in MEMS sensor.


### wang_computational_2014
On the computational complexity of the empirical mode decomposition algorithm

- EMD is a nonlinear and nonstationary time domain decomposition method.  adaptive, data-driven algorithm that decomposes a time series into multiple empirical modes - intrinsic mode functions (IMFs). 
- EMD behaves as a dyadic filter bank
- Each IMF represents a narrow band frequency–amplitude modulation
- During the last decade, the EMD/EEMD was shown to be more effective than the traditional Fourier method in many problems
- Intrinsic mode functions (IMFs) which are extracted via an iterative sifting process.
	1. **local maxima and minima** of the signal  - extrema identification procedure The definition of a local maximum in the strict sense (highest between two points - brute force)
	2. **extremes connected by cubic splines** to form the upper/lower envelopes. - for each point between two consecutive maxima upper envelope is constructed using a third order polynomial. - piecewise curve of third degree
$$\tau_j = t_j - t_i$$
$$x(t) = A_i \tau_j^3 + B_i \tau_j^2 + C_i \tau_j + D_i$$
Find coeficients by system of equations - tridiagonal matrix solver
	3. **average of the two envelopes** is then subtracted from the original signal.
	4. This sifting process is then repeated several time

- The result of the EMD is a decomposition of the signal y0(t) into the sum of the IMFs and a residue r(t).
	$$ y_0(t) = \sum_{m=1}^{n_m}{c_m(t) + r(t)}$$
- **EEMD***
	1. generates an ensemble of data sets by adding different realizations of a white noise with finite amplitude ε to the original data.
	2. EMD analysis is then applied to each data series of the ensemble. 
	3. Finally, the IMFs are obtained by averaging the respective components in each realization over the ensemble.
	

### adikaram_non-parametric_2016

- Non-parametric methods, also known as distribution-free methods, depend on fewer number of underlying assumptions - more robust methods
- proposed technique determines **maxima and minima based on the relation of sum of terms in an arithmetic series**. 
- The same relation was used as a non-parametric method (MMS: a method based on maximum, minimum, and sum) for finding outliers in linear relation and non-parametric linear fit identification method
- This work focuses on modifying the methods of MMS for locating extrema in non-linear data series.


### altaf_new_2022

- The features obtained are later integrated with the different machine learning techniques to classify the faults into different categories.
- healthy, outer race fault, inner race fault, and ball fault classes
- **Features**: skewness, kurtosis, average, root mean square
- The same features were then extracted from the second derivative of the time domain vibration signals
- These feature vectors are finally fed into the **K- nearest neighbour**, **support vector machine** and **kernel linear discriminant analysis** for the detection and classification of bearing faults.
- **reduction percentage** of more than **95% percent**
- **average accuracy** of 99.13% using KLDA and 96.64% using KNN classifiers
- Both the AE and vibration signals can effectively be used for the detection and localization of defects in rotating machinery. However, the A**E signal outperforms the vibration signal** in case early and preemptive detection is required and also in fault detection in low speed rotating machines due to the limited efficiency of vibration signals as compared to AE signals.
- signals non-stationary in nature and are complicated to analyse due to the heavy background noise of industrial set up
- **Kurtosis and its different variations**, such as kurtogram, spectral kurtosis, adaptive spectral kurtosis, and Short Term Fourier Transform (STFT) based kurtosis, have been used extensively by the research community for the analysis of vibration signals from rotating machinery;
- **Convolutional Neural Networks (CNN)** with time domain vibration signals for fault diagnosis, with 96% accuracy on Case Western Reserve University (CWRU)
- accuracy of 92% if the model trained on one machine is used for testing another machine
- **Features:**
	- statistical features of time domain signal, 
	- statistical features of signal in Fourier domain 
	- statistical features of signal’s Power Spectral Density
	- statistical features:
		- maximum value
		- minimum value
		- standard deviation
		- mean
		- median
		- variance
		- skewness
		- kurtosis
		- range
		- Fisher Information Ratio
		- Petrosian Fractal Dimension
		- entropy.
+ The Average, Kurtosis, Skewness and Standard Deviation vectors of each domain were concatenated before giving to SVM, KNN and KLDA
- Concealed component decomposition (CCD)

- In oscillation detection methods with a supervised moving window, e.g., EMD, EEMD, and LMD, the longer window is mandatory to determine the reasonable vacillating component

### Time and frequency domain scanning fault diagnosis method based on spectral negentropy and its application
- https://www.mathworks.com/help/signal/ref/pkurtosis.html#mw_95d59e55-8d7b-4145-9009-6f9384f3fd9e
- https://www.mathworks.com/help/signal/ref/kurtogram.html
- Time-frequency domain scanning empirical spectral negentropy method (T-FSESNE)
	+ The signal is filtered twice by EWT filter: 
		1. the central frequencies of all resonance side bands are determined by using frequency-domain spectral negentropy
		2. optimal bandwidth of the resonance side bands is determined by using time-domain spectral negentropy
- Dyer and Stewart introduced **kurtosis** = sensitivity to instantaneous pulse
- Spectral kurtosis (SK)
- Fast kurtogram (FK) - extract the transient characteristics of vibration signals with STFT.
- Because of sensitivity to instantaneous pulse:
	- kurtosis is vulnerable to interference from single impulse signal 
	- and irrelevant signal in low signal-to-noise ratio (SNR) background.
- The accurate determination of the central frequency and bandwidth of the resonance frequency band is very important for a further envelope analysis.
- Spectral kurtosis shortcomings:
	- The decomposition results are influenced by monopulse signals. 
	- Due to the unreasonable division of frequency band, the extracted sideband contains insufficient fault information.
- New method:
	+ the signal is filtered by the EWT scan filter to obtain all components 
	- the average of their spectral negentropy is calculated, retaining components with negative entropy values greater than the mean, 
	- taking the center frequency of these components as the center frequency of the resonant band
	- The envelope spectrum of each component is calculated by Hilbert transform
	- fault diagnosis is carried out according to the envelope spectrum of each component.
- p.4 - kurtosis used in FK is susceptible to accidental shocks, which are very common in engineering signals
- frequency-domain spectrum negentropy (FSNE)
- time-domain spectrum negentropy (TSNE)
- **The center frequency is f_ci with the bandwidth B_w.**

- Empirical Wavelet Decomposition EWT is a new signal-processing algorithm to detect the different vibration modes based on the EMD method and wavelet analysis theory. 
-It can effectively extract the different modes from a mixed vibration signal, by adaptively establishing an appropriate filter bank based on the Fourier spectrum.


### Nearest Neighbor Classification for High-Speed Big Data Streams Using Spark
- There are several possible approaches to learning from data streams.
	1. Rebuilding the classifier whenever new data becomes available. 
	2. Using a sliding window approach. 
	3. Using an incremental or online learner.
- Data streams are often characterized by a phenomenon called concept drift
- Sliding window-based classifiers were designed primarily for drifting data streams, as they incorporate the forgetting mechanism in order to discard irrelevant samples and adapt to appearing changes 
- Recent works in this area incorporate dynamic window size adjustment or usage of multiple windows.
- It is worth noticing that some of popular classifiers can work in incremental or online modes, e.g., Naïve Bayes, neural networks, or nearest neighbor methods.
- alleviate the k-NN search complexity. They range from metric trees (M-trees), which index data through a metric-space ordering; to locally sensitive hashing
- M-tree [41] can be considered as one of the most important and simplest data structure in the spaceindexing domain.
- http://mlwiki.org/index.php/Metric_Trees


### Semi-supervised Learning of Naive Bayes Classifier with feature constraints
- Semi-supervised learning methods (O. Chapelle and Zien, 2006) address the difficulties of integration of information contained in labeled data and unlabeled data. Though labeled data is scarce, unlabeled data is abundantly available
- Kullback–Leibler divergence: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence


Wavelet decompostion (half tree - High pass (Detail coef), Low pass (Approx coef.)
Wavelet packet transform (all subbands recursively) -> Calcultate Wavelet Energy / Entropy
- Wavelets-based Feature Extraction: https://www.youtube.com/watch?v=fxfS0vSAsTA
- https://en.wikipedia.org/wiki/Energy_(signal_processing)
- https://math.stackexchange.com/questions/1086522/how-to-calculate-wavelet-energy
- Entropy - every "suprise" we can expect

### An Improved Empirical Wavelet Transform for Noisy and Non-Stationary Signal Processing
- The EWT method includes three important steps: 
	- getting the local maximum of the spectrum; 
	- segmenting the spectrum by classifying boundaries; 
	- establishing a wavelet filter group. Gilles utilizes the Littlewood-Paley and Meyer wavelets to construct the filter group
- **PCHIP-EWT**
	1. Obtain the noisy and non-stationary signal y(t), and acquire Fourier spectrum Y (f ) by the fast Fourier transform (FFT) algorithm.
	2. Calculate the spectrum envelope of the K (f ) from the spectrum Y (f ) based on the PCHIP. In this method, the PCHIP =PIECEWISE CUBIC HERMITE INTERPOLATING POLYNOMIAL is able to make the Fourier spectrum Y (f ) more smooth
	3. calculate spectrum envelope K (f )based on the PCHIP. Process the signal through the EWT.select the helpful sub-bands based on the LP and threshold λ.
	4. Process the signal through the EWT.
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html

### Fault Detection of Bearing: An Unsupervised Machine Learning Approach Exploiting Feature Extraction

- allowing to work with a reduced number of features, which results in: 
	1. possibility of follow-up of features by specialists—assists in data visualization (given that some assets can present more than 100 features acquired in real time, which makes detailed monitoring of all impracticable); 
	2. avoid introducing irrelevant or correlated features in machine learning models, which would result in a loss of learning quality and, consequently, a reduction in the success rate; 
	3. reduced data storage space; and 
	4. less computational time for training the models.
- **Monitoring rotating machines has a great advantage over other research fields, which is prior knowledge of the behavior** and characteristics of the vast majority of machine failures.
- Monitoring through the vibration signal is one of the non-invasive techniques that provide the greatest amount of information about the dynamic behavior of the asset
- Stands out:
	- it does not need to stop the asset to perform the measurement
	- easy placement of the sensor for data acquisition (in the most common case of accelerometers);
	- widespread knowledge about the characteristics of faults; 
	- fast acquisition time (in most cases), enabling the monitoring of a greater amount of assets; and 
	- it provides information about mechanical, electrical, and even structural conditions.
- Features to detect faults in rotating machinery using vibration signals are commonly extracted from: 
	- **Time domain:** mean, standard deviation, rms (root mean square), peak value, peak-topeak value, shape indicator, skewness, kurtosis, crest factor, clearance indicator, etc. 
	- **Frequency domain:** mean frequency, central frequency, energy in frequency bands, etc. 
	- **Time-frequency domain**: entropy are usually extracted by Wavelet Transform, Wavelet Packet Transform, and empirical model decomposition.
	- https://www.mathworks.com/help/wavelet/ref/wentropy.html
- The higher the number of features, the harder it gets to visualize the training set and then work on it. Sometimes, many of these features are correlated or redundant.
- Techniques to **reduce the dimensionality of the data**: 
	- Principal Component Analysis (PCA)
	- t-distributed Stochastic Neighbor Embedding (t-SNE)
	- Isometric Feature Mapping (ISOMAP)
	- Independent Component Analysis (ICA)
	- Neural Network Autoencoder (AE)
- objective is to reduce the number of features by creating new representative ones and thus discarding the originals. The new set, therefore, should be able to summarize most of the information contained in the original set of features. 
- The advantages are: 
	- reduced data storage space; 
	- less computational time for training the models; 
	- better performance in some algorithms that do not work well in high dimensions; 
	- reduction of correlated variables; 
	- assistance with data visualization. 
- Disadvantages: 
	- loss of explainability of the features (when space transformation occurs)
	- lack of representativeness of the problem under analysis.
- An important parameter of anomaly detection approaches is the ability to summarize a multivariate system in just one indicator, called Anomaly Score (AS)
- Isolation Forest (iForest or IF) is probably the most popular AD approach. It works well in high-dimensional problems that have a large number of irrelevant attributes, and in situations where a training set does not contain any anomalies.
-**Methodology:**
	1. Data Acquisition; 
	2. Feature Extraction; 
	3. Dimensionality Reduction; 
	4. Fault detection: Anomaly Detection; 
	5. Feature Trend Analysis
- In case of bearing analysis, specific features are those that indicate the type of fault (BPFI, BPFO, and BSF) and the remaining features are those that indicate the presence of a defect.
- evolution of the fault, new frequencies tend to appear in other bands, which can be noticed in the energy per sub band and in the wavelet frequency sub bands.


### wagner_semi-supervised_2018
Semi-Supervised Learning on Data Streams via Temporal Label Propagation
- The labels are spread in the graph by a random walk process that moves through the unlabeled nodes until reaching a labeled node. 
- The labeling computed by this process is known as the harmonic solution
- Temporal Label Propagation (TLP), a streaming SSL algorithm
- The short-circuit operator is a way to compress a large graph G into a much smaller graph H
- nodes of interest called terminals, while preserving some global properties of G.
- terminals as the most recent points on the stream,
- Online SSL is a relatively new field that has generated considerable interest
- **transduction vs. induction**
	+ Most graphbased SSL algorithms are **transductive**, which means the unlabeled data is fully given to them in advance
	+ Inductive algorithms can also label new test points. do not use the new points to learn how to label future points (goal of online SSL)
- Graphs - weighted undirected
- https://en.wikipedia.org/wiki/Laplacian_matrix
- Offline - The input to the label propagation algorithm is a weighted undirected graph G = (V, E, w), in which a small subset of nodes Vl ⊂ V are labeled and the rest Vu ⊂ V are unlabeled
- The weight of an edge (x, y) represents some measure of similarity between its endpoints.
- The algorithm computes f_u (unlabeled fractional lables) by **minimizing the energy function of the graph** - is called the harmonic solution
- Electrical network solution:
	- View the similarity graph G as an electric network where every edge (x, y) is a resistor with conductance wx,y.
	- Connect a +1V voltage source to all nodes in Vl labeled with 1
	- a ground source (0V) to all nodes in Vl labeled with 0. 
	- The potentials induced at the unlabeled nodes are equal to the harmonic solution.
	- The short-circuit operator allows us to encode G into a smaller network G〈Vt〉 whose only nodes are the terminals.
	- However, G〈Vt〉 can also be computed by a sequence of local operations, known as **star-mesh transforms**. This will be useful for the streaming setting. (offline: inverting a large Laplacian submatrix with Shur complement)
		1. **Star**: Remove xo from G with its incident edges. 
		2. **Mesh**: Every pair of points where x_o has junction replace with direct edges between neighbours - weight (w_(x-x0) w_(x′_x0)) / deg(xo). If (x, x′) is already in E then add the new weight to its current weight
- The essence of a streaming algorithm is in maintaining a compressed representation of the stream, from which the desired output can still be computed
- The challenge here is two-fold since the algorithm needs to not only compress the data, but also update the compressed representation as new points arrive.
- https://en.wikipedia.org/wiki/Similarity_measure (Cosine similarity)
- we should favor smoothness across temporally adjacent points.
- Experimental Setting. **We use the standard RBF similarity, Sim(x, y) = exp(−‖x − y‖2/σ2).** We set σ = 0.1 for Incart-ECG, Daphnet-Gait, and CamVid and σ = 10 for Caltech10-101.
- However, when there is no natural temporal ordering (such as with Caltech10-101 data), we did not observe an advantage over the other methods.
- For example, on the Incart-ECG dataset, TLP can get to a 95% classification accuracy given only two labeled examples
- **Shingling**. A useful technique when dealing with timeseries data is to group consecutive sequences (N -grams) of points into shingles. This lifts the data into a higher dimension N and allows for a richer representation of inputs.


### zheng_feature_2018 - solve by PCA
Feature importance ranking of Numeric features - Filtering
- High correlation with predictor - band saw blade width of flank face --- to signal statistics
- Low correlation (Decorrelation) among predictors themselves - if they are correlated they produce same response
- ANOVA with F-Test - Variance of the feature - high variance - is high response
- Linearly dependent features are a waste of space and computation power because the information could have been encoded in much fewer features. 


##### Feature selection (for classification) - SelectKBest
- Variance threshold
- Pearson correlation coeficient (Filter methods:)
- ANOVA F-value
- Mutual information (MI) - Information Gain
- Fisher's score
- Spectral Feature Selection (SPEC) algorithm

-  selects subset of highly discriminant features. In other words, it selects features that are capable of discriminating samples that belong to different classes.
- Unlabeled data poses yet another challenge in feature selection. In such cases, defining relevancy becomes unclear. However, we still believe that selecting subset(s) of features may help improve unsupervised learning, 
- With the existence of a large number of features, learning models tend to overfit and their learning performance degenerates
p.30 - However, feature selection is superior in terms of better readability and interpretability since it maintains the original feature values in the reduced space, 
while feature extraction transforms the data from the original space into a new space with lower dimension, which cannot be linked to the features in the original space.
Feature selection is: filter model, wrapper model, embedded model, hybrid model

Feature selection algorithms fall into one of the three categories: 
- subset selection -  returns a subset of selected features identified by the index of the feature 
- feature weighting- returns weight corresponding to each feature (generalization of subset selection  0 - 1) ;
- hybrid of subset selection and feature weighting - returns a ranked subset of features.

Vibration levels are dependent on the type of work (load) of the machine (cite)
- **Sawing process database:** it contains basic information such as sawing machine tools model, band saw blade model, sawing parameters, and the material and size of the workpiece to match the relevant online monitoring model.
- **Online monitoring model database**: it stores online monitoring models of band saw blade wear based on different sawing processes.











% -------------------------------------------------------------------------------
- Inbalanced data, Feature space
- Scale  -  Normalize and log transform
- multi class classification - dynamic 


- Fault or no fault - anomaly detection solutions - unsupervised \cite{torres_automatic_2022}. mean shift clustering algorithm
- Types of faults - clustering, classification} - Multimodal non-Gaussian multivariate probability distribution


- **DenStream** - (based on DBSCAN)  density based params: - continous regions of high density
	- minPts (the minimum number of data points that need to be clustered together for an area to be considered high-density)
	- eps (the distance used to determine if a data point is in the same area as other data points). - compare to MAD. OPTICS for not same density.
- **Half-space Tree (IForestASD)** - isolation teqnique (map feature space to anomaly score) - how many uniform splits does it take before point is isolated (alone in group)
- **Label propagation algorithm** - Temporal Label Propagation - Graph-Based Methods in semi-supervised learning (p.9)
\end{itemize}
- **k-NN** with Mahalanobis distance to class label with M tree (guity by association)
- **Naive Bayes** (low-variance, high-bias) - classifiy based on probabilities (Multinomial Naive Bayes) - Features have to be indepent (they are not) - normal distribution
- https://www.mathworks.com/campaigns/offers/next/choosing-the-best-machine-learning-classification-model-and-avoiding-overfitting.html
- https://www.analyticsvidhya.com/blog/2017/09/naive-bayes-explained/
- https://stats.stackexchange.com/questions/21822/understanding-naive-bayes?noredirect=1&lq=1

Multi-class classification in semi-supervised manner (machinery faults in real-time)
**Algorithm have to adapt to machine** - Transfer learning or learn on the way - may underfit
Types of classification algorithms:     
- Logistic Regression - needs training to fit S-curve
- Naive Bayes
- K-Nearest Neighbors
- Decision Tree - needs training
- Support Vector Machines - needs training
- https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff
**The streaming setting in machine learning is called "online learning".**

https://www.cms.waikato.ac.nz/~abifet/book/chapter_6.html
Naive Bayes is a classification algorithm known for its low computational cost and simplicity. As an incremental algorithm, it is well suited for the data stream setting. However, it assumes independence of the attributes, and that might not be the case in many real data streams.




### chapelle_semi-supervised_2006
p.5
- Semi-supervised smoothness assumption: If two points x1,x2 in a high-density region are close, then so should be the corresponding outputs y1,y2. 
- Cluster assumption: If points are in the same cluster, they are likely to be of the same class.
- The cluster assumption can be formulated in an equivalent way: Low density separation: The decision boundary should lie in a low-density region.
- Manifold assumption: The (high-dimensional) data lie (roughly) on a low-dimensional manifold.
- Consider as an example supervised learning, where predictions of labels y corresponding to some objects x are desired. Generative models estimate the density of x as an intermediate step, while discriminative methods directly estimate the labels.

- Classification can be treated as a special case of estimating the joint density P (x,y), but this is wasteful since x will always be given at prediction time, so there is no need to estimate the input distribution. The terminology “unsupervised learning” is a bit unfortunate: the term density 1. 
- We restrict ourselves to classification scenarios in this chapter estimation should probably be preferred. Traditionally, many techniques for density estimation propose a latent (unobserved) class variable y and estimate P (x) as mixture distribution

- The semi-supervised learning problem belongs to the supervised category, since the goal is to minimize the classification error, and an estimate of P(x) is not sought after


### torres_automatic_2022
Devices and sensors + Wireless protocols limitation: IEEE 802.15.4e, OpenThread
	
https://riverml.xyz/0.15.0/
https://scikit-multiflow.github.io/



#### Kernel Discriminant Analysis
https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/LI1/kda/index.html
- pattern recognition problems, selecting an appropriate representation to extract the most significant features is crucially important. Principal Component Analysis (PCA) has been widely adopted to extract abstract features and to reduce dimensionality in many pattern recognition problems.
- But the features extracted by PCA are actually "global" features for all pattern classes, thus they are not necessarily much representative for discriminating one class from others. Linear Discriminant Analysis (LDA) [2,3,9], which seeks to find a linear transformation by maximising the between-class variance and minimising the within-class variance



## Common bandsaw problems

### Guides and guide arms 
You have to check and adjust the saw guides regularly. Check if they are worn out, and replace if necessary. Position guide arms as close to the work piece as possible.

**Problems:**
- **Band breakage** – guides are worn out or guide settings are too wide.
- **Crooked sawing/cutting out of line/square** – guides are too far apart, worn out or poorly adjusted, guide arm loose.
- **Vibration** – guides need adjusting.


### Bandwheels 
The band wheels have to be kept in good condition and properly aligned.

**Problems:**
- **Band breakage** – Band wheels worn or too small – try thinner bands
- **Band slips on wheel** – Driving wheel is worn out


### Chip Brush
Check that the swarf brush is properly adjusted and change it regularly.

**Problems:**
- **Tooth breakage** – Swarf brush does not work; gullets filled 
- **Rapid tooth wear** – Swarf brush does not work 


### Band Tension
The correct band tension is needed to get a straight cut. Measure with tensionmeter.

**Problems:**
- **Band breakage** – Band tension too high
- **Crooked Sawing** – Band tension too low
- **Vibration** – Band tension too low
- **Band slips on wheel** – Band tension too low


### Coolant / Cutting Fluid. Needed to lubricate and to cool.

Check concentration with a Bahco refractometer. Use good coolant. It should reach the cut with low pressure and with generous flow

**Problems:**
- **Rapid tooth wear** – Too little coolant or incorrect concentration


### Band Speed
The band speed has to be chosen correctly. Check the band speed with a tachometer.

**Problems:**
- **Crooked sawing/cutting out of line/square** – Band speed too low 
- **Rough surface finish** – Band speed too low 
- **Rapid tooth wear** – Band speed too high
- **Vibration** – Natural vibration, band speed is slightly high or low


### Feed Rate
The feed rate has to be chosen so that the teeth of the bandsaw blade can work properly

**Problems:**
- **Band breakage** – Feed rate too high
- **Crooked sawing** – Feed rate too high
- **Tooth breakage** – Feed rate too high
- **Rough surface** – Feed rate too high
- **Rapid tooth wear** – Feed rate too high
- **Vibration** – Feed rate too high
- **Band slips on wheel** – Feed rate too high


### Tooth Pitch
The selection of the right tooth pitch is just as important as choosing the right feed and speed

**Problems:**
- **Crooked sawing** – Tooth pitch too fine
- **Tooth breakage** – Tooth pitch too fine Gullets filled 
- **Rough surface** – Tooth pitch too coarse
- **Rapid tooth wear** –  Tooth pitch too fine


### Tooth Shape
Every tooth shape has its ideal application

**Problems:**
- **Tooth breakage** – Tooth shape too weak
- **Rapid tooth wear** – Wrong tooth shape selection
- **Vibration** – Use combo

 
### Running In
A new bandsaw blade should be run in to obtain maximum bandsaw lifetime. Never saw in old kerf.

**Problems:**
- **Rough surface** – Band not properly run in
- **Rapid tooth wear** – Band not properly run in
- **Vibration** – Band not properly run in

 

### Blade Life
All blades wear out eventually. Look for signs of wear.

**Problems:**
- **Crooked sawing** – Blade worn out
- **Rough surface** – Blade worn out
- **Band slips on wheel** – Blade worn out


### Surface
A bad surface (scale) of the work piece will shorten the life of the blade. Lower the band speed.

**Problems:**
- **Rapid tooth wear** – Surface defects, i.e. scale, rust, sand


### Clamping
Securely clamp work pieces, especially when bundle cutting. Do not use bent or damaged work pieces.

**Problems:**
- **Tooth breakage** – Work piece moves
- **Vibration** – Work piece not properly clamped










