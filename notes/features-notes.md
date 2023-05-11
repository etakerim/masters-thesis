 
## Feature Engineering for Machine Learning
- zheng_feature_2018
- p.16 - After log transformation, the histogram is less concentrated in the low end and more spread out over the x-axis.
- p.23 - The log transform is a specific example of a family of transformations known as power transforms. In statistical terms, these are variance-stabilizing transformations.
- p.23 - This is a heavy-tailed distribution with a variance that is equal to its mean: hence, the larger its center of mass, the larger its variance, and the heavier the tail. Power transforms change the distribution of the variable so that the variance is no longer dependent on the mean.
- p.24 - A simple generalization of both the square root transform and the log transform is known as the **Box-Cox transform**
- \widetilde{x} =
\begin{cases}
\frac{x^\lambda - 1}{\lambda} \quad \mathrm{if} \;\lambda \neq 0, \\\\ 
ln(x)\quad \mathrm{if}\;\lambda = 0
\end{cases}

- p.25 - we have to determine a value for the parameter λ. This may be done via maximum likelihood (finding the λ that maximizes the Gaussian likelihood of the resulting transformed signal) or Bayesian methods. The optimal Box-Cox transform deflates the tail more than the log transform, as is evident from the fact that the tail flattens out under the red diagonal equivalence line.
- p.29 - **Feature Scaling or Normalization** Some features, such as latitude or longitude, are bounded in value. Other numeric features, such as counts, may increase without bound. Models that are smooth functions of the input, such as linear regression, logistic regression, or anything that involves a matrix, are affected by the scale of the input.
- p.30 - **Min-max scaling** squeezes (or stretches) all feature values to be within the range of [0, 1].
- 	\widetilde{x} = \frac{x - \min(x)}{\max(x) - \min(x)}
- **Feature standardization** - It subtracts off the mean of the feature (over all data points) and divides by the variance. Hence, it can also be called variance scaling. The resulting scaled feature has a mean of 0 and a variance of 1.
- 	\widetilde{x} = \frac{x - \bar{x}}{\sigma_x}
- Use caution when performing min-max scaling and standardization on sparse features.

#### Feature selection
- p.37 - **Feature selection** techniques prune away nonuseful features in order to reduce the complexity of the resulting model. The end goal is a parsimonious model that is quicker to compute, with little or no degradation in predictive accuracy.
- In other words, feature selection is not about reducing training time—in fact, some techniques increase overall training time—but about reducing model scoring time.
- **Filtering techniques** preprocess features to remove ones that are unlikely to be useful for the model. For example, one could compute the correlation or mutual information between each feature and the response variable, and filter out the features that fall below a threshold
- **Wrapper methods** These techniques are expensive, but they allow you to try out subsets of features, which means you won’t accidentally prune away features that are uninformative by themselves but useful when taken in combination

#### PCA
- p.99 - With automatic data collection and feature generation techniques, one can quickly obtain a large number of features. But not all of them are useful. In Chapters 3 and 4, we discussed frequency-based filtering and feature scaling as ways of pruning away uninformative features. Now we will take a close look at the topic of feature dimensionality reduction using **principal component analysis (PCA).**
- If the column space is small compared to the total number of features, then most of the features are linear combinations of a few key features. Linearly dependent features are a waste of space and computation power because the information could have been encoded in much fewer features. To avoid this situation, principal component analysis tries to reduce such “fluff” by squashing the data into a much lower-dimensional linear subspace.
- The key idea here is to replace redundant features with a few new features that adequately summarize information contained in the original feature space.
- Any rectangular matrix can be decomposed into three matrices of particular shapes and characteristics: 
- X = U\SigmaV^T
- p.104 - Objective function for principal components: max_{w}{}w^Tw, where w^Tw = 1. The optimal w, as it turns out, is the principal left singular vector of X, which is also the principal eigenvector of XTX. The projected data is called a principal component of the original data.
- p.106 - easiest way to **implement PCA is by taking the singular value decomposition of the centered data matrix.**
- p.108 - the transformed features are no longer correlated. In other words, the inner products between pairs of feature vectors are zero. 
- Sometimes, it is useful to also normalize the scale of the features to 1. In signal processing terms, this is known as whitening.
- When using PCA for dimensionality reduction, one must address the question of **how many principal components** (k) to use.
- One possibility is to pick k to account for a desired proportion of total variance.
- It is difficult to perform PCA in a streaming fashion, in batch updates, or from a sample of the full data. Streaming computation of the SVD, updating the SVD, and
computing the SVD from a subsample are all difficult research problems. Algorithms exist, but at the cost of reduced accuracy.





## Feature Engineering and Selection: A Practical Approach for Predictive Models
- This example demonstrates how an alteration of the predictors, in this case a simple transformation, can lead to improvements to the effectiveness of the model. When comparing the data in Figures 1.2a and Figure 1.3a, it is easier to visually discriminate the two groups of data
- However, different models have different requirements of the data. If the skewness of the original predictors was the issue affecting the logistic regression model, other models exist that do not have the same sensitivity to this characteristic. For example, a neural network can also be used to fit these data without using the inverse transformation of the predictors
- the neural network model has its own drawbacks: it is completely uninterpretable and requires extensive parameter tuning to achieve good results
- Without any specific knowledge of the problem or data at hand, no one predictive model can be said to be the best. In practice, it is wise to try a number of disparate types of models to probe which ones will work well with your particular data set.
- Consequently, another round of feature engineering (g) might be used to compensate for these obstacles. By this point, it may be apparent which models tend to work best for the problem at hand and another
- Chap. 10.1. : Even when a predictive model is insensitive to extra predictors, it makes good scientific sense to include the minimum possible set that provides acceptable results. In some cases, removing predictors can reduce the cost of acquiring data or improve the throughput of the software used to make predictions.
- Chap. 11.2: The most basic approach to feature selection is to screen the predictors to see if any have a relationship with the outcome prior to including them in a model. To do this, a numeric scoring technique is required to quantify the strength of the relationship. Using the scores, the predictors are ranked and filtered with either a threshold or by taking the top p predictors. Scoring the predictors can be done separately for each predictor, or can be done simultaneously across all predictors (depending on the technique that is used).
- 6.3 Many:Many Transformations
When creating new features from multiple predictors, there is a possibility of correcting a variety of issues such as outliers or collinearity. It can also help reduce the dimensionality of the predictor space in ways that might improve performance and reduce computational time for models.
6.3.1 Linear Projection Methods
- An important side benefit of this technique is that the resulting PCA scores are uncorrelated.
-  1.2.5 - sensitive to outliers, it is much less sensitive than a nearest neighbor model.
-  1.2.5 -In a similar manner, models can have reduced performance due to irrelevant predictors causing excess model variation. Feature selection techniques improve models by reducing the unwanted noise of extra variables.
\cite{johnson_feature_2019}
- Chap. 6. how to handle continuous predictors with commonly occurring issues. The predictors may:
    be on vastly different scales.
    follow a skewed distribution where a small proportion of samples are orders of magnitude larger than the majority of the data (i.e., skewness).
    contain a small number of extreme values.
    be censored on the low and/or high end of the range.
    have a complex relationship
    contain relevant and overly redundant information
- 10.2. : Feature selection mythologies fall into three general classes: intrinsic (or implicit) methods, filter methods, and wrapper methods. **Intrinsic methods** have feature selection naturally incorporated with the modeling process. 





## A Novel Online Machine Learning Approach for Real-Time Condition Monitoring of Rotating Machines
- mostafavi_novel_2021
- However, handling this vast amount of data could be a daunting task requiring extensive hardware for data transfer, storage and processing.
- Transferring a large amount of data to a cloud server and processing leads to a bottleneck problem
- Selecting a suitable algorithm plays a key role in designing a real-time and standalone device for anomaly detection of industrial machines. When we want to identify a machine's abnormal behavior in the real world, we do not have access to all its possible faults characteristics. So we need an algorithm that can learn a machine's healthy behavior and discriminate any faults. These algorithms call novelty detectors.
- Feature engineering is extracting useful information from raw data, and it is considered the cornerstone of successful anomaly detection. It is used to reduce data dimensionality and remove nullities in a data set.
- **convolutional neural networks (CNNs), auto-encoders, and recurrent neural networks (RNNs)** have been used for feature extraction through the learning process
- some authors preferred to select the most useful features manually
- Usually, the features for anomaly detection have been classified into three main groups, **time domain, frequency domain, and time-frequency domain**. Using signal rootmean-square (RMS) level, crest factor, and the maximum signal value ratio to its RMS are the most common ways to extract the time domain features. - 
- Kurtosis and peak value are other famous features in the time domain which has been used in [21] for diagnosing faults in roller and ball bearings.
- Alongside these features, some dimensionless factors such as shape factor, impulse factor, kurtosis factor, and margin factor have been utilized in for bearing fault prognosis.
- n non-stationary signals, **time-frequency domain features** are considered the best features when it comes to non-stationary signals 
	- Short-Time Fourier Transform (STFT), 
	- Continuous Wavelet Transform (CWT), 
	- Discrete Wavelet Transform (DWC), 
	- Wavelet Packet Decomposition (WPD)
- **Using WPD**, we decompose the signal into low-frequency and high-frequency parts, level by level until the fourth level. The energies of the last layer nodes, which are **16 nodes, are selected as the time-frequency domain features**.
- vibration sample acquired via the accelerometer from each axis has 2048 sample points. After storing three axes raw vibration data into three different arrays, the DC gain of each axes has removed by subtracting the mean of the array from each sample point. 
- Subsequently, the total **29 features are extracted from each axis** and stored in an array with a length of 87.
- In this study, we utilized one of the most commonly used novelty detector algorithms, autoencoders.**Autoencoders** are feed-forward fully connected artificial neural networks that can learn to reconstruct their unlabeled inputs. If an autoencoder is trained by healthy machine data, it can precisely reproduce every healthy data.
- **mean squared error (MSE) between input and output** of the autoencoder can be used as an indicator for discriminating the healthy and defective state of the machine.
- Offline: 1000 tests under the healthy condition of the pump and 2800 tests under faulty conditions were conducted. All faulty samples were acquired at five different pump operating points, 120, 130, 140, 150, and 160 lit/min for each type of fault, and tests under healthy pump conditions were conducted at 12 operating points, ranging from 90 lit/min to 210 lit/min.
- The total training procedure is done in 300 epochs. In addition, since validation cost is decreasing while training cost is descending, the network has not been overfitted. 99.9% Accuracy and presicion
- when a sample is categorized as healthy by the network, it is healthy with a probability of 99.7%.
- Training on edge Fig. 7 describes the data pipeline for edge computing.
- Equations for features:
	+ rms:   X_{rms} = \left(\frac{1}{N}\sum_{i = 1}^{N}{x_i^2}\right)^\frac{1}{2}
	+ kurtosis value:  X_{kv} = \frac{1}{N}\sum_{i = 1}^{N}{\left(\frac{x_i - \bar{x}}{\sigma}\right)^4}
	+ skewness value: X_{kv} = \frac{1}{N}\sum_{i = 1}^{N}{\left(\frac{x_i - \bar{x}}{\sigma}\right)^3}
	+ peak-peak value: X_{ppv} = \max(x_i) - \min(x_i)
	+ crest factor: X_{cf} = \frac{max(|x_i|)}{\left( \frac{1}{N} \sum_{i=1}^{N}{x_i^2} \right)^\frac{1}{2}}
	+ impulse factor: X_{if} = \frac{max(|x_i|)}{\frac{1}{N} \sum_{i=1}^{N}{|x_i|}}
	+ margin factor: X_{mf} = \frac{max(|x_i|)}{\left( \frac{1}{N} \sum_{i=1}^{N}{\sqrt{|x_i|}} \right)^2}
	+ shape factor: X_{sf} = \frac{
\left( \frac{1}{N} \sum_{i=1}^{N}{x_i^2} \right)^\frac{1}{2}
}{
\frac{1}{N} \sum_{i=1}^{N}{|x_i|} 
}
	+ kurtosis factor: X_{kf} = \frac{X_{kv}}{X_{rms}^2}
	+ Frequency center: X_{fc} = \frac{\sum_{i = 0}^{N - 1}{f_i \cdot s(f_i)}}{\sum_{i = 0}^{N - 1}{s(f_i)}}
	+ RMS frequency: X_{rmsf} = \left(\frac{\sum_{i = 0}^{N - 1}{f_i^2 \cdot s(f_i)}}{\sum_{i = 0}^{N - 1}{s(f_i)}}\right)^\frac{1}{2}
	+ Root variance frequency: X_{rmsf} = \left(\frac{\sum_{i = 0}^{N - 1}{(f_i - X_{fc})^2 \cdot s(f_i)}}{\sum_{i = 0}^{N - 1}{s(f_i)}}\right)^\frac{1}{2}




## A New Statistical Features Based Approach for Bearing Fault Diagnosis Using Vibration Signals
- The statistical features including **skewness, kurtosis, average and root mean square values** of time domain vibration signals are considered. These features are extracted from the second derivative of the time domain vibration signals and power spectral density of vibration signals. 
- The vibration signal is also converted to the frequency domain and the same features are extracted
- 95% percent is achieved, which not only reduces the computational burden but also the classification time. Simulation results show that the signals are classified to achieve an average accuracy of **99.13% using KLDA and 96.64% using KNN classifiers.**
- The vibration signal of healthy and faulty bearings were recorded, using vibration sensors, shaft rotating at a rate of 800 revolutions per minute and a sampling rate of 40,000 samples per second. 
- The output vector of the second difference is then used to calculate the statistical measures as given above.

- **KNN is a popular algorithm** based on distance measure between two feature vectors si and sj, using the Mahalanobis distance:
- d = (s_i − s_j)^T C^{−1} (s_i − s_j),
- where C ∈ R^{p×p} represents the **covariance matrix obtained from training feature vectors**. The C can be computed as a diagonal matrix in case of small training sample sizes, with the feature variances as the diagonal elements.
- The highest accuracy is given by the PSD_P for KLDA, which is 99.13%, 
	- followed by Statistical_P, which is 98.257% using KLDA. 
	- The Fourier is third in the row with 98.0% using KLDA 
	- followed by EMD with 97.01% using KNN.

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
- altaf_new_2022
 

## Fault Detection of Bearing: An Unsupervised Machine Learning Approach Exploiting Feature Extraction and Dimensionality Reduction
- The proposed features were used as input to an unsupervised anomaly detection model (**Isolation Forest**) to identify bearing fault
- **Time domain:** mean, standard deviation, rms (root mean square), peak value, peak-topeak value, shape indicator, skewness, kurtosis, crest factor, clearance indicator, etc. 
- **Frequency domain:** mean frequency, central frequency, energy in frequency bands, etc. 
- **Time-frequency domain:** entropy are usually extracted by Wavelet Transform, Wavelet Packet Transform, and empirical model decomposition.
- \cite{brito_fault_2021}
- Dimensionality reduction:
	- **Curse of dimensionality**: due to the high number of features in relation to the sample, the algorithm tends to suffer overfitting, fitting very well to the training data, but showing a high error rate in the test group. 
	- **Occam’s Razor**: to be used in real applications, the models are intended to be simple and explainable. The greater the number of features present, the greater the difficulty in explaining the model under development.
	- **Garbage In Garbage Out**: when using features that do not present significant information for the model, the final result obtained will be lower than desired.
- allowing to work with a reduced number of features
	- possibility of follow-up of features by specialists—assists in data visualization 
	- avoid introducing irrelevant or correlated features 
	- reduced data storage space; 
	- less computational time for training the models.
- An important parameter of anomaly detection approaches is the ability to summarize a multivariate system in just one indicator, called **Anomaly Score (AS)** (**Health Factor or Deviance Index**).
- The selected features are shown in Table 1. Three **energy features** were calculated in the frequency bands of 10–1000 Hz, 1000–4000 Hz, and 4000–10,000 Hz, and nine wavelet sub bands were extracted for entropy calculation.
- - The **principal frequency** can vary with the appearance of the defect, stabilizing and suffering changing with the fault evolution due to the random behavior caused by the excessive wear. 
- **Crest factor** tends to increase as the amplitude of high frequency impacts in the bearing increase compared to the amplitude of overall broadband vibration. 
- **Skewness** will provide information on how the signal is symmetrical with respect to its mean value. 
- **rms value**, global value from envelope analysis, and absolute energy represent the global behavior of the system, indicating a general degradation and accentuation of the 
- a **greater number of features tends to introduce bias and variance in the system**, increasing the standard deviation of the results, and consequently reducing their robustness
- The dimensionality reduction methods (test iii) showed good results, close to the set of manually selected features **93.11 (All), 98.02 (selected), 97.45 (PCA)**
- A framework composed of five main steps is proposed, namely: (1) Data Acquisition; (2) Feature Extraction; (3) Dimensionality Reduction; (4) Fault detection: Anomaly Detection; and (5) Feature Trend Analysis.
- Features such as absolute energy, kurtosis, skewness, rms, and global value from envelope analysis pk-pk can be characterized as relevant for identifying the fault under study,
- Features
	- Absolute Energy: E_s = \sum_{i = 0}^{N - 1}|x(t)|^2
	- RMS
	- Skewness
	- Crest factor
	- Global value from envelope analysis peak-to-peak
	- Principal frequency \max(x_{fi})
	- Wavelet sub band entropy − \sum p_i \cdot \log(p_i)


## A large set of audio features for sound description
\cite{peeters_large_2004}
Spectral features  1. Spectral shape description

- Coherence function - correlation between two signals PSD
- Spectral centroid (Frequency center) - barycenter of the spectrum (weighted mean of the frequencies present in the signal, with their magnitudes as the weights)
- 
- Spectral spread - variance of the spectral distribution
- Spectral skewness
- Spectral kurtosis
- **Spectral slope** - comupted with linear regression of the spectral amplitude - amount of decresing of the spectral amplitude
- \hat{a}(f) = \mathrm{slope} \cdot f + \mathrm{const} \\
- \mathrm{slope} = \frac{1}{\sum_k a(k)} \cdot \frac{N\sum_k f(k) \cdot a(k) - \sum_k f(k) \cdot \sum_k a(k)}{N\sum_k f^2(k) - \left(\sum_k f(k)\right)^2}

- **Spectral roll-off** - 95\% of the signal energy is contained below this frequency (sr/2 - nyquist frequecy, fc - spectral frequency roll-off)
- \sum_{0}^{f_c}a^2(f) = 0.95 \sum_{0}^{\mathrm{sr} / 2}a^2(f)
- **Temporal variation of spectrum** - spectral flux - correlation of normalized cross-correlation between two succesive amplitude spectra - Close to 0 = similiar, 1 = dissimiliar
- \mathrm{flux} = 1 - \frac{\sum_k a(t-1, k) \cdot a(t,k)}{\sqrt{\sum_k a(t-1, k)^2} \sqrt{\sum_k a(t, k)^2}}


**Harmonic features** -p.17
- **Fundamental frequency**  - Maximum likelihood algorithm - best explain content of signal spectrum
- **Noisiness** - ratio - energy of noise to the total energy
- **Inharmonicity** - energy weighted difference of the spectral components from the multiple of fundamental frequency - 0=purely harmonic, 1=inharmonic
- 
- **Harmonic Spectral Deviation** - deviation of amplite harmonics peaks from global spectral envelope. H - total number of harmonics. a(h) - amplitude of h-th hatmonic, SE(h) - spectral envelope at frequency f(h)
- \mathrm{HDEV} = \frac{1}{H}\sum_h(a(h) - SE(h))
- 


## Research on online intelligent monitoring system of band saw blade wear status based on multi‑feature fusion of acoustic emission signals
\cite{zhuo_research_2022}
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

- **Wavelet packet decomposition (WPD)** has a good resolution to the high-frequency part of signal. good monitoring effect on tool wear. Therefore, the WPD analysis method is used to process the AE signal - wavelet db3 analysis
- **Energy from WPD:** E_{j,n} = \sum_{k=1}^{N}|x_{i,j,k}|^2
- **Energy ratio from WPD:** P_i = \frac{E_i}{\sum_{i = 1}^{N}E_i}


## Early Detection of Imbalance in Load and Machine In Front Load Washing Machines by Monitoring Drum Movement
\cite{mohammadi_early_2020}
- The centrifugal force produced by this laundry can be calculated using the equation ~ F = mω2r, in which m is the mass of the load, ω is its angular velocity, and r is the radius of the drum. 
- Assuming the radius to be 25 centimeters, the force exerted by this unbalanced load on the drum is 3948 N, or 402.6 kilograms perpendicular to the drum
-  load imbalance classification task **XGBoost** (Gradient Boosted Decision Trees) with discrete Fourier transform features has the best performance score with the shortest input. 
- However, the highest performance is achieved using a 10-bin histogram of the input data and **SVM classifier**. 
- In both imbalance detection tasks, the selected models can detect the problem with just **500 ms** of sensor data, and a **performance F_1 score of 95%**. 
- XGBoost with DFT feature extraction performs the best with the F1 score of 0.948 for the load imbalance classification problem.
- We also observe that the performance can be enhanced even more by providing the models with longer sequences of input data.
- **minimum imbalance detection length** (MIDL) as the shortest length of the multivariate time series sensor data for which the model performance is at maximum
- The problem is to make a decision with partial temporal input. Xing et al. (2009) uses a **1-nearest neighbor classifier** to achieve reliable predictions with minimal temporal input lengths. They do not make any assumptions about the form of the underlying distributions on the input.
- p.19 - The kernel is given using Equation 3.12, in which W1 is the 1st Wasserstein distance between the two time series Ti and Tj.

## A Data Mining based Approach for Electric Motor Anomaly Detection Applied on Vibration Data
\cite{egaji_data_2020}
- Often, there is a shortage of faulty machine data samples, whereas healthy example data are readily available
- Hence, it becomes beneficial to adopt an appropriate learning approach such as anomaly detection, that can be modelled using only healthy machine data samples.
- The objective of this process is to find patterns in data that do not conform to the healthy (normal) machine condition. The commonly used anomaly detection techniques include the one-class support vector machine, K-means, Self-organizing Maps (SOM)
- Hilbert-Huang Transform (HHT). The HHT time-frequency was the most effective tool in diagnosing faults of rotational machines (non-stationary and non-linear) as it can detect components of low energy
- this paper proposes an anomaly detection methodology which relies on optimised features extracted from the bearing vibration data
- It utilises statistical features extracted from the vibration data as an input to a **one-dimensional PCA**. The output of the PCA is used to train the machine learning model.
- The performance of three machine learning models (**KNN, SVR (Support Vector Regression) and RF (Random Forest Tree)**) were presented. The RF showed the best performance as compared to the SVR and KNN as it has the lowest false positive and a better detection time than KNN.
- The threshold-based approach has been commonly used for condition monitoring in some industry. This approach requires setting a minimum or maximum threshold for individual sensors reading, and when the reading exceeds that threshold, a fault instance is considered to be present. 
- However, this has shown to cause **false alarms**, which can result from the sensors inaccuracies, human error. -> More robust approach

## Vibration Analysis for IoT Enabled Predictive Maintenance
- The new generation of vibration sensors, based on **MEMS**, greatly reduces the price, power consumption and size. MEMS also enables a wide operation range, due to its high resonance frequency and larger detectable acceleration range.
- For each single measurement, the **sensor collects** 1024 samples, each of which consists of 2 byte readings on three dimension, with 6 Kbyte data in total. Due to the limitation of maximum packet size of low-power radio per transmission, the 6Kbyte data is partitioned and transmitted to the gateway in 120 data packets
- Let us assume that the sampling rate of sensors is set to 150Hz in Fig. 5. Then one option of the system is to collect 2,576 vibration measurements in three years of target lifetime (= 3 year x 365 days x 24 hours / 10.2hours ), - - while another option is to collect 3,650 measurements for 2 years of target lifetime (= 2 year x 365 days x 24 hours / 5.2hours ). 
- In our IoT-based data analytic settings, we must acknowledge that **data is expensive and valuable resource**. This is a significant difference to other data analytical systems, in which it is always cheap to collect more data.
- Assumptions:
	+ data only provides indirect information over the vibrations
	+ samples from different sensors may cover various intervals on the time domain
	+ here is always significant noise in the data.
	
- RUL estimation - Harmonic peak distance is score from baseline - create probability density functions (with Recursive RANSAC regression algorithm - because of noisy distribution) of zones A,  BC, D
and cut on transition to zone D (around >0.21 Peak distance)
- group of pairs of significant peaks’ value and frequency in PSD (p, f)
- Distance is euclidian based - closest peak frequency and value
- Harmonic peak distance - compare baseline harmonic peak feature (20 peaks - points where its first order differential changes from positive to negative-  from smooth spectrum by 16 point Hann window) 
\cite{jung_vibration_2017}


## Condition Monitoring with Vibration Signals
\cite{nandi_condition_2019}
p.36 - Time-domain features
	- Peak Amplitude
	- Mean Amplitude
	- Root Mean Square Amplitude
	- Crest Factor (CF) - measure of the impulsive nature of a vibration signal that will give basic information about how much change is occurring in a normal-condition vibration waveform.
	- Margin Factor - changes significantly with changes in the peak value, which makes it very sensitive to impulse faults in particular.
	- Shape factor - the change resulting in the vibration signal due to unbalance and misalignment defects.
	- A histogram can be assumed to be a discrete PDF of the vibration signal. Two types of features can be obtained from the histogram: the lower bound (LB) and upper bound (UB) p.42
	- A **considerable amount of literature** has been published on vibration monitoring using statistical time domain techniques to preprocess vibration signals as input features, individually or in combination with other techniques. These studies are summarised in Table 3.1.
	- skewness, shape factor, impulse factor, variance, CF, peak-to-peak, RMS, and mean are among the most-used techniques in these studies.
	
- p.63 - Frequency analysis, also called spectral analysis, is one of the most commonly used vibration analysis techniques for monitoring the condition of machines. In fact, **frequency domain analysis** techniques have the ability to divulge information based on frequency characteristics that **are not easily observed in the time domain.**
- p.67 - The fast Fourier transform (FFT) is an efficient algorithm that computes the DFT and its inverse for a stationary time series signal with a significant reduction in complexity.

- p.71 - Envelope analysis, also called high-frequency resonance analysis or resonance demodulation, is a signal-processing technique that is considered a powerful and dependable method for fault detection in rolling element bearings. 
	- the raw vibration signal is band-pass filtered; 
	- it is then rectified or enveloped by folding the bottom part of the time waveform over the top part, usually using the Hilbert-Huang transform (HHT
	- it is transformed utilising FFT
- they showed that envelope analysis is a very useful method to detect incipient failures of rolling element bearings

p.79 
- Thus, the Fourier transform in the frequency domain does not have the ability to provide a time distribution information of the spectral components. Rotating machines, in general, **generate stationary vibration signals**. -
- Nevertheless, Brandt stated that most analysis of rotating machines is based on examining the vibrations during a speed sweep, where machines are either speeded up from low to high revolutions per minute (RPM) or slowed down from high to low RPM.	
- time-frequency analysis techniques have been developed and applied to machinery fault diagnosis: e.g. **short-time Fourier transform (STFT), wavelet transform (WT), Hilbert-Huang transform (HHT), empirical mode decomposition (EMD), local mean decomposition (LMD)**

p.82 
- **Wavelet analysis** is another time-frequency domain analysis approach that decomposes the signal based on a family of ‘wavelets’.
- . Unlike the window used with the STFT, the wavelet families have fixed shapes – e.g. **Haar, Daubechies, symlets, Morlets, coiflets**, etc. 
– but the wavelet function is scalable, which means the wavelet transformation is adaptable to a wide range of frequency- and time-based resolutions.
- **The mother wavelet:** \psi_{s, \tau} = \frac{1}{\sqrt{s}}\psi\left(\frac{t - \tau}{s}\right)
- The three main transforms in wavelets analysis are the **CWT, DWT, and WPT**.
- The **CWT** of the time domain vibration signal (x):
-  W_{x(t)}(s, \tau) = \frac{1}{\sqrt{s}}\int x(t) \cdot \psi^*\left(\frac{t - \tau}{s}\right) dt
- where 𝜓* represents the complex conjugate of 𝜓(t) that is scaled and shifted using the s and 𝜏 parameters, respectively. The translation parameter 𝜏 is linked to the location of the wavelet window, as the window moved through the signal and scale parameter s is linked to the zooming action of the wavelets
- p.84 Zheng et al. showed that the time average wavelet spectrum (TAWS) based on the **CWT using a Morlet wavelet** function can select features for effective **gear fault identification.**
- fast CWT for condition monitoring of bearings and demonstrated its suitability for real-time applications
- Kankar et al. introduced a method for rolling bearing fault diagnosis using the CWT. In this method, two wavelet selection criteria – the **maximum energy to Shannon entropy ratio and the maximum relative wavelet energy** – are utilised and compared to choose a suitable feature extraction.

p.85
- Discrete methods are often required for the computerised implementation and analysis process of wavelet transforms (DWT)
-  W_{x(t)}(s, \tau) = \frac{1}{\sqrt{2^j}}\int x(t) \cdot \psi^*\left(\frac{t - k2^j}{2^j}\right) dt
- s discretised using dyadic scales, i.e. s = 2j
- **DWT is often implemented** with a low-pass scaling filter h(k) and a high-pass wavelet filter g (k) = (−1)kh(1 − k). These filters are created from scaling functions 𝜙(t)and𝜓(t)
- The signal is decomposed into a set of low- and high-frequency signals using the wavelet filters such that
- a_{j,k} = \sum_m h(2k - m) \cdot a_{j-1,m} \\ 
b_{j,k} = \sum_m g(2k - m) \cdot a_{j-1,m}
- To perform the DWT, also called multiresolution analysis, of a given discrete signal x, (i) the signal is filtered using special low-pass filter (L) and high-pass filter (H), e.g. **Daubechies, coiflets, and symlets,** producing two vectors of low and high sub-bands at the first level. 
- The first vector is the **approximation coefficient (A1), and the second is a detailed coefficient (D1)** (see Figure 5.3)
- For the next level of the decomposition, (ii) the approximation coefficient, i.e. the low-pass sub-band, is further filtered using L and H filters, which produce a **further approximation coefficient (A2) and detailed coefficient (D2)** at the second level
- DWT in machine fault diagnosis is denoising the vibration signal in the time domain as well as the frequency domain.

p.89 
- The **WPT is an improvement of the DWT** in which every signal detail obtained by the DWT is further decomposed into an approximation signal and a detail signal
- the time domain vibration signal x(t) **can be decomposed** using the following equations:
- d_{j+1,2n} = \sum_m h(m - 2k) \cdot d_{j,n} \\
d_{j+1,2n+1} = \sum_m g(m - 2k) \cdot d_{j,n}
- idea that by using the WPT, a rich collection of time-frequency characteristics in a signal can be obtained.
- their experimental results, this method showed high classification accuracy for vibration monitoring
- The fault detection is performed by calculating the **Renyi entropy** and Jensen-Renyi divergence

p.91
- The **EMD method** was developed based on the assumption that any signal comprises different simple intrinsic modes of oscillations.
- extract the instantaneous frequencies and amplitudes of the multicomponent amplitude-modulated and frequency-modulated (AM-FM) signals.
- To stop the shifting process, a stoppage criterion must be selected. In (Huang et al. 1998), the stoppage criterion is determined by using a **Cauchy type of convergence test.** In this test, the normalised squared difference between two successive shifting operations
- The EEMD process can be described in the following steps: 
	- add a white noise series to the signal of interest, 
	- decompose the signal with the added white noise in the IMFs,
	- repeat with a different realisation of the white noise series each time, and finally 
	- obtain the (ensemble) means of the corresponding IMFs of the decomposition.

The **kurtogram algorithm** (KUR) was first introduced by Antoni and Randall, and comes from the SK (Antoni and Randall 2006). The KUR computes the SK for several window sizes using a bandpass filter. In this bandpass filter, the central frequency f c and the bandwidth can be determined with which to jointly maximise the kurtogram. Here, for all possible central frequencies and bandwidths, all possible window sizes should be considered. 

p.125 - Semi-supervised learning addresses this limitation by means of utilising a large amount of unlabelled data together with the labelled data, to build better classifiers
p.127 - reparing Vibration Data for Analysis 6.3.3.1 Normalisation As just described, various environmental and operating conditions affect the accuracy of measured vibration signals.

p.127 - **Feature selection**, also called subset selection, techniques aim to select a subset of features that can sufficiently represent the characteristic of the original features. In view of that, this will reduce the computational cost and may remove irrelevant and redundant features and consequently improve learning performance.
- **Filter models**, e.g. Fisher score (FS), Laplacian score (LS), relief algorithms, Pearson correlation coefficient (PCC), information gain (IG), gain ratio (GR), mutual information (MI), Chi-squared (Chi-2), etc
- **Wrapper models**, which can be categorised into sequential selection algorithms and heuristic search algorithms.

- p.132 - **PCA is an orthogonal linear feature projection algorithm** that aims to find all the components (eigenvectors) in descending order of significance where the first few principal components (PCs) possess most of the variance of the original data.

p.173 - **Feature Selection**
- high-dimensional data requires large storage and time for signal processing, and this also may limit the number of machines that can be monitored remotely across **wireless sensor networks (WSNs)** due to bandwidth and power constraints. 
- eature-extraction techniques project the high-dimensional data of n instances {xi}n i=1 with D feature space, i.e. xi ∈ RD,intoanew low-dimensional representation {yi}n i=1 with d feature space, i.e. yi ∈ Rd where d ≪ D. 
- The new **low-dimensional feature space** is often a linear or nonlinear combination of the original features.
- **Feature-selection, also called subset selection**, techniques aim to select a subset of features that can sufficiently represent the characteristic of the original features.
- showed that all strongly relevant and some of the weakly relevant features are required in terms of an optimal **Bayes classifier**.
- SVM can really suffer in high-dimensional spaces where many features are irrelevant.
- **feature relevance alone is not enough** for effective feature selection of high-dimensional data, and they developed a correlation-based method for relevance and redundancy analysis
- finding a **minimal feature set** optimal for classification (MINIMAL-OPTIMAL) and finding **all features relevant to the target variable** (ALL-RELEVANT). They proved that ALL-RELEVANT is much harder than MINIMAL-OPTIMAL.
- **supervised feature-selection methods, semi-supervised methods, and unsupervised methods**
- **Supervised:** relevance is often evaluated by computing the correlation between the feature and class labels
- The main task of the feature-selection technique is to select a subset (Z)ofm features from F, where Z ⊂ F.TheselectedsubsetZ is expected to build a better classification model.

p. 175 - typical feature-selection procedure:
	+ **Subset generation** - adding, removing featues from the set (search algorithms)
	+ **Subset evaluation** - evaluation measure, e.g. independent criteria, distance measures, information measures, dependency measures, and consistency measures.
	+ **Stopping criteria** - search complete, iterations reached, acceptable rate for classification task
	+ **Validation** - compare to previously known results - artificial and or real datatsets
	
- Filter method: ranks features based on certain measures. with the highest rankings are selected to train classification algorithms
	+ **Fisher score (FS)** - compute a subset of features with a large distance between data points in different classes and a small distance between data points in the same class
		- the FS of each feature is computed independently.
		- **Score:** \mathrm{FS}(X^j) = \frac{\sum_{i=1}^{C} l_i(\mu_i^j - \mu^i)^2}{(\sigma^j)^2} \\
(\sigma^i)^2 = \sum_{i=1}^{c} l_i (\sigma_i^i)^2
		- li is the size of the ith class in the reduced data space
		- Let 𝜇j i and 𝜎j i be the mean and the standard deviation of the ith class, corresponding to the jth feature; and let 𝜇j and 𝜎j be the mean and standard deviation of the entire data corresponding to the jth feature. Formally, the FS feature of the jth
	+ **Laplacian score (LS)** (p.177) - is an unsupervised filter-based technique that ranks features depending on their locality-preserving power
		- f_r - rth feature f_ri - ith sample rth feature
		- LS algorithm constructs the nearest neighbour graph G with M nodes, where the ith node corresponds to xi. Next, an edge between nodes i and j is placed; **if xi is among the k nearest neighbours of xj or vice versa, then i and j are connected.** The elements of the weight matrix of graph G is Sij:
		- S_{ij} = e^{-\frac{\lVert x_i - y_j \rVert^2}{t}} (ak nie sú connected tak 0)
		- Vzorec na Laplacian score: https://www.mathworks.com/help/stats/fsulaplacian.html
	+ The **Pearson correlation coefficient (PCC)**
		- is a supervised filter-based ranking technique that examines the relationship between two variables according to their correlation coefficient
		- Next: not capturing correlations that are not linear in nature. correlation measure based on the informationtheoretical idea of entropy.
		- r(i) = \frac{cov(x_i, y)}{\sqrt{var(x_i) \cdot var(y)}}
	- **Information Gain (IG)**
		- https://machinelearningmastery.com/information-gain-and-mutual-information/
		- correlation measure based on the information-theoretical concept of entropy to overcome the limitation of PCC as a linear correlation measure.
		- IG(X \;|\; Y) = H(X) - H(X \;|\; Y)
		- H(X) = - \sum_i P(x_i) \log_2 P(x_i)
		- H(X \;|\; Y) = - \sum_j P(y_i) \sum_i P(x_i \;|\; y_i) \log_2 P(x_i \;|\; y_i)
		- IG = Entropy(Dataset) – (Count(Group1) / Count(Dataset) * Entropy(Group1) + Count(Group2) / Count(Dataset) * Entropy(Group2))
	- **Mutal Information (MI)**
		- is a measure of dependence between two variables, i.e. how much information is shared between two variables. 
		- **Mutual Information and Information Gain are the same thing**, although the context or usage of the measure often gives rise to the different names
		- MI(X, Y) will be vey high if X and Y are closely related to each other; otherwise,
		- MI(X, Y) = \sum_{y \in Y} \sum_{x \in X} p(x, y) \log(\frac{p(x,y)}{p(x)p(y)})
		- Features F = {f 1, f 2, ..., f m}and class labels C = {c1, c2, ..., ck}

	- wavelet packet is selected by PCC-based correlation analysis, and the fault feature of the bearing is extracted from the selected wavelet packet using envelope analysis.
	- Vibration signals acquired from a roller bearing with healthy bearing (HB), IRD, ORD, and ball defect (BD) are used to validate the proposed method. The results demonstrated that 93.54% ten-fold cross-validation accuracy is obtained when the Chi-squared feature ranking method is used along with the RF classifier.		
	- Embedded model–based feature-selection methods are built in the classification algorithm to accomplish the feature selection. LASSO, elastic net, the classification and regression tree (CART), C4.5, and SVM–recursive feature elimination (SVM-RFE)
	- The results showed that the combination of the wrapper feature-selection method and an SVM classifier with 14 selected features gives the maximum accuracy of 96%.


## Non-Parametric Local Maxima and Minima Finder with Filtering Techniques for Bioprocess
\cite{adikaram_non-parametric_2016}
- **magnitude based methods**, the nth term of a series is xn; xnis considered as a peak (maximum) when xn−1 < xn > xn+1. In the same time, xnis considered as a valley (minimum) when xn−1 > xn< xn+1. 
- **gradient-based methods**, extremum can be located by considering slope (gradient) of a certain point and acts as the most popular method. When the slope is zero (first derivative is zero) at a certain point, the point can be described as a peak, valley or a saddle point.
- **Magnitude of prominences and the widths at half prominence** are two properties of signals that are commonly used to **filter extrema**
- Non-parametric methods, also known as distribution-free methods, depend on fewer number of underlying assumptions - more robust methods
- proposed technique determines **maxima and minima based on the relation of sum of terms in an arithmetic series**.
- \mathrm{MMS}_{max} = \frac{a_{max} - a_{min}}{S_n - a_{min} \cdot n} > 2/n
- \mathrm{MMS}_{min} = \frac{a_{max} - a_{min}}{a_{max} \cdot n - S_n} > 2/n
- Therefore, when a window satisfies Equation (7) it implies that the **middle point is a maximum** and once a window satisfies Equation (9) it alternatively implies that the **middle point is a minimum**. Conditions:
    - \mathrm{MMS}_{\mathrm{max}} = \mathrm{MMS}_{\mathrm{max|mid}} \\
      \mathrm{MMS}_{\mathrm{min}} = \mathrm{MMS}_{\mathrm{min|mid}}

    - \frac{a_{max} - a_{min}}{S_3 - a_{min} \cdot 3} = \frac{a_{mid} - a_{min}}{S_3 - a_{min} \cdot 3} \\
      \frac{a_{max} - a_{min}}{a_{max} \cdot 3 - S_3} = \frac{a_{max} - a_{mid}}{a_{max} \cdot 3 - S_3}

- **MMS-Window based filter” or (MMS-WBF)**
- **MMS-SG** Perfect maxima: \mathrm{MMS}_{max} / \mathrm{MMS}_{min} = (n -1)
- **MMS-LH** 
    - R_{LH\_min} = \frac{a_{min} \cdot n}{S_n},\quad R_{LH\_min} \in (0, 1] (17, p.11)
    - R_{LH\_max} = \frac{n}{(a_{max} + 1)\cdot n - S_n},\quad R_{LH\_max} \in (0, 1]  (24, p.13)
- The same relation was used as a non-parametric method (MMS: a method based on maximum, minimum, and sum) for finding outliers in linear relation and non-parametric linear fit identification method
- This work focuses on modifying the methods of MMS for locating extrema in non-linear data series.
- MMS max-min finder

## Identification of harmonics and sidebands in a finite set of spectral components
\cite{gerber_identification_2013}
- automatically **identify the harmonic series and sidebands** taking the uncertainty in frequency estimation into account and without introducing any a priori on the signal
- S of spectral components - C_i(v_i, \delta v_i, A_i):
    - central frequency νi
    - the uncertainty ∆νi 
    - amplitude Ai
- Two series of fundamental frequencies νi and νj **belong to the same family** if their **ratio is a rational number**
- signal containing more than one harmonic family cannot be periodic
- **Harmonic series identification** from estimated components is a nontrivial problem because of **estimation errors**.
- In the proposed algorithm, the search of harmonic series is exhaustive
    - We propose to use a criterion of minimum distance to select νi(r) the harmonic of order r of the fundamental frequency νi.
    - v_i^{(r)} = \frac{v_j}{\min{|v_j - r \cdot v_i|}}
    - To prevent the search interval growing, each time a component is identified as a harmonic of Ci
- each component of the spectrum is not considered as a potential carrier frequency. 
- **The search for sidebands** is only made around the components belonging to the harmonic series Hi previously identified
    - compare the fundamental frequencies from the MkCj+ series to the fundamental frequencies from MkCj-.
    - **If two series have the same fundamental frequency** (with a possible error of maximum ∆νi), both series are merged and are now considered as a modulation series.


## Spectral negentropy and kurtogram performance comparison for bearing fault diagnosis
\cite{avoci_spectral_2020}
- paper is motivated by ideas borrowed from thermodynamics, where transients are seen as departures from a state of equilibrium; 
- it is proposed to measure the negentropy of the **squared envelope (SE)** and the **squared envelope spectrum (SES)** of the signal
- most mechanical degradation presents a series of repetitive transients, the **Kurtogram or SK** often hardly or almost cannot recognize them;
- **spectral negentropy** was demonstrated to be one of the alternatives to overcome this.
- spectral negentropy, a contraction for "negative entropy"
- entropy evaluates the level of the disorder in a system from the state of equilibrium; 
- negentropy measures the inclination of a system to **increase its level of organization**. 
- In the science of information, entropy describes how much information contains a signal
- **complex envelope:** y(k, f, \Delta f)
- **squared envelope (SE)**: e(k, f, \Delta f) = |y(k, f, \Delta f)|^2
- In Fourier domain, the squared envelope spectrum (SES).
- The more impulses the fault induces, the larger is the spectral negentropy.
- \Delta I_E(f, \Delta f) = \sum_{k = 0}^{N - 1}{\frac{E(k, f, \Delta f)^2}{\frac{1}{N} \sum_{k=0}^{N-1} E(k, f, \Delta f)^2}} \cdot \ln\left(\frac{E(k, f, \Delta f)^2}{\frac{1}{N} \sum_{k=0}^{N-1} E(k, f, \Delta f)^2}\right)
- SK for transients detection in a signal, focuses on measurement of distance between random and Gaussian process.
- All those bands reveal the signature of an inner race fault. The SES infograms on Fig. 7 (c) show that cyclostationary events are below 4 kHz, which do not necessarily point out the equivalence in the kurtogram and SES infogram. they are complementary

## Application of Teager–Kaiser Energy Operator in the Early Fault Diagnosis of Rolling Bearings
\cite{shi_application_2022}
Gaussian white noise in bearing vibration signals seriously masks the weak fault characteristics in the diagnosis based on the **Teager–Kaiser energy operator envelope**,
- Improved TKEO can attenuate noise in consideration of computational efficiency while preserving information about the possible fault.
- weak impact caused by the defect is hardly reflected on time-domain statistical parameters
- for AM signals, including bearing vibration signals, the TKEO energy is approximate to the squared envelope.
- statistical parameters to select the optimal frequency band for wavelet packet transform and use TKEO to detect the hidden impact
- TKEO energy only requires three adjacent samples, which is simple and computationally efficient; TKEO has high time resolution
- assumption that the bearing vibration signal is an AM signal
- modulated fault features or information can be extracted from bearing vibration signals by **amplitude envelope analysis**.
    - Discrete verison of TKEO: \psi[x(n)] = [x(n)]^2 - x(n - 1)x(n + 1)
- **Gaussian white noise** easily interferes with the bearing fault detection by TKEO energy
- Removal of white noise before TKEO can greatly improve the accuracy and sensitivity of early fault diagnosis.
- **Improved TKEO** is the method combining TKEO analysis with this denoising method (p.9)



## The fast continuous wavelet transformation (fCWT) for real-time, high-quality, noise-resistant time–frequency analysis
\cite{arts_fast_2022}
- The drawback of the **STFT is its use of a fixed-width window function**, as a result of which frequency analysis is restricted to frequencies with a wavelength close to the window width
- hopping up the signal in short, **fixed-width windows scrambles the signal’s properties**
- reduce the computational burden of the WT, the discrete wavelet transform (DWT) has been proposed, which applies a coarse, logarithmic discretization
- benchmark the performance of fCWT we **compared fCWT to the six widely used CWT** implementations shown in Fig. 3.
- complex Morlet wavelet (σ = 6) was used to calculate the CWT of three signals, all containing N = 100,000 samples.
- The Morlet wavelet is defined as a plane wave modulated by a Gaussian envelope
- the signal content and wavelet choice are irrelevant to the performance of fCWT

- fCWT being, respectively, **122 times and 34 times faster than the reference Wavelib** and the fastest available algorithm, PyWavelet
- fCWT was also compared to two other fast time–frequency estimation algorithms: the STFT and DWT. 
    - STFT uses a Blackman window of 500 ms with 400-ms overlap, 
    - DWT uses 20 dyadic (that is, aj = 2j) scales of Debauchie decomposition
- real-time analysis ratio (RAR) Δtcomputation / Δtsignal
- even the fastest CWT implementation available tends to be extremely slow compared to STFT and DWT
- DWT is powerful in denoising, but not suitable for time–frequency analysis.

- fCWT’s scale-dependent part by exploiting its repeated nature and high parallelizability. 
The scale-independent operations are performed first as their result forms the input for the scale-dependent steps. We pre-calculate two functions: 
    - the input signal’s FFT and 
    - FFT of the mother wavelet function at scale a0 = 2. 
- Both functions are independent of the scale factor a, so they can be pre-calculated and used as look-up tables in the processing pipeline.

- Fourier-based wavelet transform. Applying **Parseval’s theorem to equation**, a **reduction in CWT’s complexity** can be achieved:
- W_{\psi}f(a,b) = \frac{1}{2 \pi} \int \hat{f}(\xi)\bar{\psi_{a,b}(\xi)} \;d\xi
- W_{\psi}f[a,b] = \frac{1}{N} \sum_{k = 0}^{K - 1} \hat{f}[k]\bar{\hat{\psi}[a,k]} e^{i2\pi bk /K}


## A Concentrated Time–Frequency Analysis Tool for Bearing Fault Diagnosis
\cite{yu_concentrated_2020}
 novel time–frequency analysis method termed the transient-extracting transform
- based on the short-time Fourier transform and does not require extended parameters or a priori information.
- different fault signals occupy distinct frequency bands, 
    - the joined time–frequency (TF) analysis (TFA) is an effective tool for characterizing transient faults that have nonstationary TF features
- no TF basis functions that can be compactly supported in the TF domain simultaneously, the linear TFA methods show a poor ability to characterize the precise TF features.
- Methods in realted work:
    - empirical mode decomposition (EMD) 
    - the spectral kurtosis (SK) method 
    - synchrosqueezing transform (SST) 
        - SST is intended to obtain a sharper TF representation, which can characterize the faults in high TF resolution
        - challenging to estimate the IF of the transient component precisely, because the fault signals usually do not meet the weakly time-varying requirement of the SST
        - background noises will introduce serious interference into the SST result
- TFA(t,ω) = 〈s(u), ψt,ω(u)〉,where〈, 〉 denotes the inner product operator.
- STFT, we select the **Gaussian function** as the moved window and the window length is 100 samples. 
- For the WT, we employ the **Morlet function** to address this transient signal.

- Ideal TF representation of the signal (2), the energy should only appear at the time t0 instead of being spread over a large region
- postprocessing procedure called the **transient-extracting operator (TEO)** is proposed:
    - \mathrm{TEO}(t,\omega) = \delta(t − t_0(t, \omega))
- the new TFA method employing the TEO is termed the **transient-extracting transform (TET)** and formulated as 
    - \mathrm{Te}(t,\omega) = G(t,\omega) \cdot TEO(t,\omega)

- First, the TET algorithm needs to calculate two STFTs (G[n, k] and Gtg[n, k]) with respect to the windows g[n] and n · g[n]. 
- Equations on p.4
- Uses CWRU
- he MATLAB code of the TET can be found on: https://www.mathworks.com/matlabcentral/fileexchange/70319-transient-extracting-transform
- TET provides a significantly concentrated result than the STFT.
- uniform discretization of s(t) taken at the time tn = t0 + nT,where T is the sampling interval. The Fourier transform of data s[n]
    - \mathrm{Te}[n, k]= 
\begin{cases}
G[n, k],\quad ∣Re \left[ \frac{G^{tg}[n,k]}{G[n , k]} \right]∣ < \frac{T}{2} \\
0, \quad \mathrm{otherwise}
\end{cases} 
- It can be seen that the proposed TET method provides a **decomposed result with a significantly larger kurtosis than the other methods**. It can be concluded that our proposed method is more suitable for extracting the transient components **(p.7) Table III - V** 38.85 for TET vs. around 4 for SK and EEMD IMF1-4

## Applications of the synchrosqueezing transform in seismic time-frequency analysis
\cite{herrera_applications_2014}
- SST aims to decompose a signal sðtÞ into constituent components with time-varying harmonic behavior. 
- These signals are assumed to be the addition of individual time-varying harmonic components yielding **(Equation)**
- where AkðtÞ is the instantaneous amplitude, ηðtÞ represents additive noise, K stands for the maximum number of components in one signal, and θkðtÞ is the instantaneous phase of the kth component
- The **CWT is the crosscorrelation of the signal sðtÞ with several wavelets** that are scaled and translated versions of the original mother wavelet.
- The **SST reallocates the coefficients of the CWT** to get a concentrated image over the time-frequency plane, from which the instantaneous frequencies are then extracted
- **Conditions:**
- the wavelet choice is a key issue in synchrosqueezing-based methods (Meignen et al., 2012). 
- we need a mother wavelet that satisfies the admissibility condition (i.e., finite energy, zero mean, and bandlimited). 
- At the same time, the wavelet must be a good match for the target signal

## Wavelet Packet Feature Extraction for Vibration Monitoring
\cite{yen_wavelet_2000}
- **faults develop, some of the system dynamics vary**, resulting in significant deviations in the vibration patterns. 
- By employing appropriate data analysis algorithms, it is feasible to detect changes in vibration signatures caused by faulty component
- With the aid of statistical-based feature selection criteria, many of the **feature components containing little discriminant information** could be discarded, - resulting in a feature subset having a reduced number of parameters without compromising the classification performance.
- simple condition monitoring system is approached from a pattern classification perspective. 
- It can be decomposed into **three general tasks**: 
    - data acquisition
    - feature extraction
    - condition classification

- if a multilayer neural network is used to classify unprocessed data, the input layer, 
- which learns from examples, will essentially serve as a **feature extractor**.

- **Short Time Fourier Transform (STFT)** can be employed to detect the localized transient. Unfortunately, the fixed windowing used in the STFT implies fixed time-frequency resolution in the time-frequency plane
- The collection of all wavelet packet coefficients contains far too many elements to efficiently represent a signal.
- Care must be taken in **choosing a subset**of this collection in order to manage the computational complexity in practical situations
- we formulate a systematic method of determining **wavelet-packet-based features** that exploit class-specific differences among interesting signals

- if we are analyzing the **low-frequency** content of a signal, we might desire a wide window function in time. On the contrary, 
- if we were interested in **high-frequency** phenomena, a short-duration window function would be preferred.

- Specifically, the STFT of x(t) and g(t) is a window function
- G(f, \tau) = \int x(t) \cdot g^*(t - \tau) \cdot e^{-i2\pi f t} dt

- To compute the wavelet transform, all we need are filters. 
- Rather than taking the scalar product of the scaling function or the wavelet with the signal, we **convolve the signal with filters**
- **number of points is gradually decreased** through successive decimation. 
    - signal of points 2^j, then, in the following level 2^{j-1}, we have wavelet coefficients. 
    - maximum decomposition level is equal to j

- filtering algorithm is, in fact, a classical scheme known as a two-channel subband coding using **quadrature mirror filters (QMF’s)**
- **Fig. 3. Implementation of discrete WPD.**

- method of decomposition described above does not result in a WPT tree displayed in increasing frequency order. 
- This is because aliasing occurs, which exchanges the frequency ordering of some nodes of the tree. 
- A simple swapping of the appropriate nodes results in the increasing frequency ordering referred to as the **Paley ordering**

- Whereas the **FWT decomposes only the low-frequency components**, 
- **WPT decomposes** the signal utilizing both the low-frequency components and the high-frequency components
- One deficiency that wavelet bases inherently possess is the **lack of a translation-invariant property**
- **node energy representation** provides us with a more robust signal feature for classification than using coefficients directly
    - wavelet packet coefficient: w_{j,n,k} = \langle f, W_{j,k}^n\rangle = \langle f, 2^{j/2} W^n (2^jt-k) \rangle
    - wavelet packet node energy: c_{j,n} = \sum_k w_{j,n,k}^2
    - where is j a scaling parameter, n  is a translation parameter, k is an oscillation parameter.

- each **wavelet packet node energy** value was defined as an **individual feature component** 2^{r+1} - 2 components
- direct manipulation of a whole set of node energies is prohibitive because the space normally has **very high dimensionality**, 
- and the existence of undesired components makes the classification unnecessarily difficult
- One popular technique in reducing the feature dimensionality is the **Karhumen–Loéve (K–L) transform**
- LDA - idea is to find a linear transformation that projects the samples onto a lower dimensional space in 
    - he variability of samples within each class is as close as possible, and the dispersion of the class mean vectors about the mean vector is as separated as possible
    - ability to classify patterns relies on the implied assumption that **different classes occupy distinct regions in the pattern space**.

- criterion function for **evaluating the discriminant power of a feature** could be assessed by measuring the **overlap** between and probs p(x|c1) p(x|c2). A high overlap corresponds to a low discriminant power and vice versa.
    - efficient criterion function known as **Fisher’s criterion**
    - \mu are the mean values of the k-th feature, f_k, for class i, j; and are the variance \sigma
    - When there are more than two classes of data, the general approach is to take the summation of the pairwise combinations of i,j
    - Eq: J(f_1) \geq J(f_2) \geq \dots J(F_n)
    - LDA: \Sigma_B = \frac{1}{C} \sum_{i = 1}^{C} (\mu_i - \mu)(\mu_i - \mu)^T
    - Fisher: J_{f_k}(i,j)= \frac{|\mu_{i,f_k} - \mu_{i,f_k}|^2}{\sigma^2_{i,f_k} + \sigma^2_{i,f_k}}
- select a feature subset based on (3.14) for each possible pair of classes. 
    - Then, we take the **union of feature components selected from each pair of classes** to form the final feature vector. (p.11)
- The Westland data set [24] transmission vibartions
- **mean estimate is subtracted from the random signal** before computing the power spectrum estimate
- some sensors are not sensitive to the detection of specific fault symptoms. This suggests the need to use multiple-sensor data to search for class-specific features.

- there is a slight **amount of frequency overlap among the wavelet basis** functions, 
    - particular frequency may be sensed by two different basis functions. 
    - This frequency leakage may lead to worse performance using wavelet-packet-based features.
- **noise** 
    - from machine components (other than the faulty response), 
    - neighboring machinery, 
    - measurement noise
    - better results are obtained via the wavelet-packet-based approach.

## A wavelet approach to dimension reduction and classiﬁcation of hyperspectral data
\cite{wickmann_wavelet_2007}
- p.29 - i) The Continuous Wavelet Transform (CWT) 
- p.39 - (ii) The Discrete Wavelet Transform (DWT) 
- (iii) The Wavelet Packet Decomposition (WPD)
- p.76 - In figure 4.8 the “reduce” step is replaced by the PCA. The experiment is conducted as described in section 3.4.
	


## On the computational complexity of the empirical mode decomposition algorithm
\cite{wang_computational_2014}
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




## Fault Feature Extraction and Enhancement of Rolling Element Bearings Based on Maximum Correlated Kurtosis Deconvolution and Improved Empirical Wavelet Transform
	\cite{li_fault_2019}
- The key step of **EWT is segmenting the Fourier spectrum** of the collected signal. 
- EWT divides the spectrum into several portions, 
    - each portion corresponds to a **mode centered at the specific frequency and compact support**, such as AM-FM signal.

- Assuming that we want to **segment the Fourier spectrum into N continuous segments**
    - it needs to find **(N − 1) the largest local maxima** in the Fourier spectrum. In most cases, 0 and π are always taken as two boundaries
    - For this set of maxima, we define that ωn is the boundary between each segment, where ω0 = 0 and ωN = π. Each segment is shown as An = [ωn−1, ωn], then ∪N n=1An = [0, π]. Centered on each ωn, we define a transition phase Tn, with width 2τn and τn = γωn; γ is properly chosen to get a tight frame and is given in Equation
    - p. 5 - According to the boundaries defined by the **segmentation of the Fourier spectrum**, 
    - a **series of empirical wavelets** can be constructed as **bandpass filters** based on the idea of the construction 
    of both **Littlewood-Paley and Meyer’s wavelets**
    - \gamma = \min_n\left(\frac{\omega_{n+1} - \omega_n}{\omega_{n+1} + \omega_n}\right)
- following **threshold**according to the method of literature: 
    - λ denotes the pre-set threshold, Ah and Al are, respectively, 
    - the maximum and minimum magnitudes of the spectrum and 
    - r is inversely proportional to the SNR
    - \lambda = A_l + \frac{C}{\mathrm{SNR}}(A_h - A_l)
- If the number of the local maxima points above the threshold is larger than the pre-defined number N of the components, 
    - then keep on calculating the **envelope curve of the spectrum** until the number of the local maxima points is equal or less 
    - than the pre-defined number N of the components
- **Calculate the kurtosis value** of each component and choose the component with the maximum kurtosis value to further detect the fault feature.

- more reliable segmentation methodology for the signal spectrum is introduced by calculating the envelope curve of the amplitude spectrum of the signal.
-  We calculate the envelope curve at first, 
    - and then modify the envelope curve with a **pre-set threshold \lambda** according to the SNR value. 
    - Finally, we segment the signal spectrum based on the local extrema of the modified envelope curve
    - Calculate the envelope curve of the amplitude spectrum based on the **local maxima and minima by linear interpolation method**, then modify the envelope curve according to the threshold

- the **improved EWT can adaptively segment the spectrum** of the rolling element bearing in strong noise conditions.
- Secondly, compared with the FK and OPWT methods, the MCKD-EWT method is more effective in weak fault feature extraction and enhancement.
- When the fault is large enough, FK, OPWT (Optimal Wavelet Packet Transform), and MCKD-EWT methods can find the resonant frequency band and realize fault diagnosis.

- there is a **large rotational speed fluctuation**, the proposed method may be **useless** because the MCKD method needs to ensure the periodicity of the spacing of the pulses. 
- The cost time of building the filters banks is too long, so it **cannot be used in on-line fault diagnosis**
- **Further investigation** should be performed
    - choosing the optimal parameters of MCKD-EWT
    - finding an enhanced method for rotational speed fluctuations condition 
    - finding a faster method.

p.8 - rolling element bearing equation mechanics
Detect bearing faults - impulses (transients)
Maximum Correlated Kurtosis Deconvolution (MCKD)

- FIR filter maximizing the CK (Kurtosis) of the impulses - Result coeficients of the filter
- Empirical Wavelet Transform 
	- address the mode mixing or over-estimation phenomenon of the EMD
	- EWT divides the spectrum into several portions, and each portion corresponds to a mode centered at the specific frequency and compact support, such as AM-FM signal
- Algorithm MKCD-EWT
- De-noise the signal by MCKD 
    - MCKD encourages the **periodicity of the periodic impacts or impulse-like vibration** behaviors by selecting a finite impulse response (FIR) filter to maximize the CK of the filtered signal
    - p.3 - The MCKD is used to search a reverse FIR filter f by maximizing the CK of the impulses y(n). The CK is defined as:
- **Steps** p.7
    - Spectrum segmentation. Calculate the envelope curve of the amplitude spectrum of the de-noising signal.
     Signal decomposition. Design the wavelet filter banks
    - IMF (Intrinsic Mode Function) selection. Calculate the kurtosis of each sub-signal
    - Feature extraction. Calculate the squared envelope spectrum and teager energy operator spectrum of the chosen mode


- Highest kurtosis values of these modes IMF1 - 4 in the largest IMF
- Genetic algorithm in OWPT

- Teager-Kaiser operator (TKEO)
- Teager Energy Operator (TEO)
- $x(t)  = (dx/dt)^2+ x(t)(d^2x/dt^2) $
- $[x[n]] = x^2[n] + x[n - 1]x[n + 1]$ TKEO
- When Ψc is applied to signals produced by a simple harmonic oscillator, e.g. a mass spring oscillator who’s equation of motion can be derived from the Newton’s Law - It can track the oscillator’s energy
- In the squared envelope spectrum and teager energy operator spectrum, it can be seen that both methods successfully extract the outer race fault feature.


## Time and frequency domain scanning fault diagnosis method based on spectral negentropy and its application
\cite{yonggang_time_2020}
p.6 - **The empirical scaling function b ∅n ω ðÞ[25] and the empirical wavelet function b Ψn ω ðÞ [25] are defined as follows:**

- EWT and spectral negentropy are combined to adopt the filtering method by changing the bandwidth and central frequency. 
    - EWT is used to adaptively divide the signal boundary and reconstruct the filtered signal. 
    - Spectral negentropy is used to detect non-equilibrium disturbances in the system, which contains fault information.

1. signal is filtered by the EWT scan filter to obtain all components and the average of their spectral negentropy is calculated,
2. retaining components with negative entropy values greater than the mean, 
3. taking the center frequency of these components as the center frequency of the resonant band. 
4. The **central frequency fcj of all resonance bands** is determined by using the **index of spectral negentropy in frequency domain**. 
5. the scanning filter with a fixed central frequency and a changing bandwidth is used to process signal, 
6. the **bandwidth Bwk of resonance band** is determined by using **spectral negentropy in time domain**.

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
- **New method:**
	+ the signal is filtered by the EWT scan filter to obtain all components 
	- the average of their spectral negentropy is calculated, retaining components with negative entropy values greater than the mean, 
	- taking the center frequency of these components as the center frequency of the resonant band
	- The envelope spectrum of each component is calculated by Hilbert transform
	- fault diagnosis is carried out according to the envelope spectrum of each component.
- p.4 - kurtosis used in FK is susceptible to accidental shocks, which are very common in engineering signals
- frequency-domain spectrum negentropy (FSNE)
- time-domain spectrum negentropy (TSNE)
- **The center frequency is f_ci with the bandwidth B_w.**

- whole normalized Fourier spectrum is divided on [0, π]by [ω f ci −ωBw=2, ω f ci þ ωBw=2 ]. 
- EWT filters are constructed with the boundaries ½ω f ci−ωBw=2, ω f ci þ ωBw=2 ]. 
- the frequency bands ω f ci −ωBw=2; ω f ci þ ωBw=2 are extracted to obtain the components
- center frequency is fci with the bandwidth Bw.

- Empirical Wavelet Decomposition EWT is a new signal-processing algorithm to detect the different vibration modes based on the EMD method and wavelet analysis theory. 
-It can effectively extract the different modes from a mixed vibration signal, by adaptively establishing an appropriate filter bank based on the Fourier spectrum.
- Determining more **reasonable center frequency and bandwidth** is the precondition of accurate envelope analysis and bearing fault diagnosis

## The MFBD: a novel weak features extraction method for rotating machinery
\cite{song_mfbd_2021}
- **multiple frequency bands demodulation (MFBD)** method has better demodulation performance than Fast Kurtogram, Autogram and Fast-SC for weak modulation features

- Compared with the **cyclostationary analysis** method, the **envelope analysis** has lower computational complexity
- Antoni further presented an efficient algorithm and optimized the filter, called **Fast Kurtogram**
- demodulation performance of **Fast Kurtogram would decrease sharply under low signal to noise ratio** (SNR)

- Aodulation frequency band by **Gini index**
- In previous studies, the periodic modulation wave signal is extracted by **combining time frequency analysis and PCA**. 
- This method has been well used for fault feature extraction [20]. The superiority of the proposed method is verified by
   simulation analysis and applications of a centrifugal pump and a propeller

- **proposed MFBD method**, the decomposition level is three 
    - the global frequency band is separated into eight narrow bands as a list in Fig. 4(d). 
    - Five frequency bands are selected including **frequency bands 3, 4, 5, 6, and 7**. 
    - Four selected principal components are plotted

- Besides, **Autogram and Fast-SC** are two very good weak signal extraction algorithms. 
    - In order to verify the robust demodulation performance of MFBD, 
    - different white noise levels are added to signal x(t)

- MFBD provided a clearer characteristic frequency identification than Fast Kurtogram, especially under weak modulation condition.
- Carrier signals include resonance signal, line-spectrum signal and noise signal

- The monitoring signal is decomposed **into narrow bands by WPD**. 
    - the decomposition level n of WPD is derived as Eq. (2). 
    - For the multiple values n that meet the conditions, the smaller value should be selected
    - \frac{1}{2^n} > \frac{4f_{priori}}{Fs}
    - where:
        - n denotes the decompose level of wavelet packet decomposition, 
        - Fs denotes sampling frequency, 
        - fpriori denotes the prior frequency of modulation which can be predicted from rotating machinery working condition.
    - Energy: E_i^n = \lVert wpc_i^n \rVert_2^2

- **Based on PCA**, the weak modulation features can be enhanced by extracting from multiple narrow band signal components.
- The energy coefficient of frequency band can be calculated:
    - \eta_i^n = \frac{E_i^n}{\sum_{i = 2}^{2^n}E_i^n \cdot 100%}
- The components with biggest m energy coefficients are considered for weak feature extraction
- the weak modulation signals are extracted from envelope components matrix ewpmn(t) based on PCA
- The envelope signal of decomposed component wpcn i (t) can be calculated: sqrt{wpc_i^2 + Hilbert^2|}
- The criterion for multiple principal components selection can be expressed as Eq. (12). 80% could represent the main modulation components


## Novel self-adaptive vibration signal analysis: Concealed component decomposition and its application in bearing fault diagnosis
\cite{tiwari_novel_2021}
- Look at the **figure and flowchart at page 5**


-phenomenon of **mode mixing** has been a concern in every signal decomposition technique since its inception
- Concealed component decomposition (CCD)
- p.26 - results on VMD, another self-adaptive technique, which has been found promising in extracting features for classification and diagnosis in various work, is done, and some thought-provoking effects are found
- p.39 - CCD provides partially filtered and segmented derivatives having high regularity and low noise. 
    - On comparing the proposed technique with existing self-adaptive techniques such as LMD, EEMD, and VMD,
    - one may conclude the superiority of the former over the latter. 
    - The impulsive characteristics in CCD may also be a criterion of selecting the prime rotation component;
        - however, in the other approaches, this is quite difficult.
- p. 3 The abilities of CCD make it suitable for analyzing the signal from different domains. 
- It can process the non-stationary biological signal, mechanical vibration signal, earthquake, and structural vibration signatures. 

The following steps are involved in the **extraction of the concealed rotation components** from the unfiltered vibration signal:

- **Identify all local maxima and local minima**. The identification of the local minima and maxima 
- is made with nonparametric methods with minimal window length (i.e., 3). 
- join them to have the two envelope functions XM(t) and Xm(t) as the **upper and lower envelope**, respectively

- Construction of envelope may be done using **Hilbert transform**. 
- Other options such as:
    - RMS envelopes with sliding window, 
    - peak envelope with spline interpolation
    - peak separation parameter can also be recommended

- **Divide dn i into two fractions, ξ and (1 − ξ ), and place a pivot point at the junction**. 
- The fractions must be in such a way that distance between xi and the pivot point is ξ dn i. vii) 
- The same process must be repeated for all pairs of minima and maxima.

- The pivots so collected are joined with a cubic function and coined as a low-frequency function Xlj(t). ix) 
- **The first rotation component** Xrj (t) is obtained by subtraction the acquired low-frequency function Xlj(t) from X(t)

- Calculate the value of ξ for the next step utilizing the **proposed operator** (Eq.)
- Repeat the steps iv to x till a monotonic trend function is attained.

## An Improved Empirical Wavelet Transform for Noisy and Non-Stationary Signal Processing
\cite{zhuang_improved_2020}
- method combines the advantages of **piecewise cubic Hermite interpolating polynomial (PCHIP) and the EWT**, and is named PCHIP-EWT. 
- Ki(f ) = a1 +a2(f −fi)+a3(f −fi)2 +a4(f −fi)2(f −fi+1) (9) where: 
    - a1 : = Y (fi−1) 
    - a2 : = Y [fi−1, fi] = Y ′i(f ) 
    - a3 : = Y [fi−1, fi−1, fi] = (mi − Y ′i(f ))/ki 
    - a4 : = Y [fi−1, fi−1, fi, fi] = (Y ′i+1(f ) + Y ′i(f ) − 2mi)/k2 i

- select useful sub-bands from the spectrum envelope. 
    - selects the maximum points of the spectrum to reconstruct the spectrum envelope on the basis of PCHIP. 
    - a new concept and a threshold named the Local Power (LP) and λ are defined.
    - Based on the new concept LP and the λ, the useful sub-bands can be obtained

- Gilles [11] proposed a new method named EWT which can decompose the noisy and non-stationary signals into several IFMs adaptively.
- EWT is an **adaptive decomposition method which extracts narrow-band frequency components** from the analyzed signal based on the frequency information contained in the signal spectrum. 
- Compared with the EMD, the EWT performs more effectively in processing the noisy and non-stationary signals.
- some **drawbacks of EWT** have already appeared, such as **improper segmentations when the noisy** and non-stationary signals are processed.

- **PCHIP-EWT** reducing the number of parameters that need to be determined in advance
- **EWT** - key idea is to obtain the intrinsic mode of the signal through devising a proper wavelet filter bank. 
    - getting the local maximum of the spectrum; 
    - segmenting the spectrum by classifying boundaries; 
    - establishing a wavelet filter group. 
    - Gilles utilizes the Littlewood-Paley and Meyer wavelets to construct the filter group
    - it is sensitive to noise and needs to set some parameters in advance

- PCHIP-EWT detects boundaries in the spectral envelope calculated by the PCHIP algorithm 
    - **Calculate the spectrum envelope** of the K (f ) from the spectrum Y (f ) based on the **PCHIP**.
        - we use the spectrum envelope by the PCHIP instead of the Fourier spectrum to segment the boundaries
    - In Section 3.2, calculate LP pi(f )of the spectrum envelope K (f )
    - select the helpful sub-bands based on the LP and threshold λ. 
    - Process the signal through the EWT.

- DETECTING THE BOUNDARIES
    - If p_i(f) \geq \lambda, it means that this sub-band contains useful information. 
    - If p_i(f) \leq \lambda, it means this sub-band is consisted of noise
    - p_i(f) = \frac{K_{max(i)(f)}}{f_k_{min(i+1)} - f_k_{min(i)}}
    - **this is maximum of segment (n-th local maximum of spectral envelope) divided by adjacent minima**
    - \gamma = \frac{p_max}{p_min}
    - \lambda = \gamma / 10 (\gamma \geq 1000), \gamma (10, 100), 10\gamma (\gamma 0 .. 10)
    - pmax and pmin represent the maximum and minimum of the local power.

- Each boundary is the upper cut-off frequency of the useful sub-bands in the spectrum envelope K (f ). 
- If the useful sub-bands are defined as sbn and detected boundaries are defined as wn, then: wn = fkmin(n+1) 
- where fkmin(n+1) is the index of the local minimum of the K (f ) which is the nearest to Kmax(i) on the right

- In Fig.10(b), it can be observed that the **PCHIP-EWT method can generate the optimal boundaries** to identify the close frequencies accurately.
- PCHIP-EWT more effective than the EWT method for the multi-component signal that is consisted of closed frequency components.
- Noise: hese two different approaches, the **EWT generates too many boundaries**, which will lead to too many useless components in time decomposition

## An Adaptive Spectrum Segmentation Method to Optimize Empirical Wavelet Transform for Rolling Bearings Fault Diagnosis
\cite{xu_adaptive_2019}
- key spectral negentropy has stronger anti-noise ability than kurtosis.
- EWT - The method consists of four steps: 
   1. Setting parameters and choosing the way to split the spectrum; 
   2. The Fourier spectrum is adaptively segmented to obtain a set of boundaries; 
   3. Constructing filter banks based on boundaries and empirical wavelets; 
   4. A series of empirical modal components with compactly supported Fourier spectra can be obtained by filtering and reconstruction.

- EWT can achieve better results than EMD and Ensemble EMD in extracting fault features and associated harmonics. 
    - EWT has been widely used in many fields, 
    - such as wind turbine via [12], 
    - wheel-bearing of trains [13], 
    - rotor rubbing [14], 
    - and the impact vibration experiment of solar arrays

- The **root mean square, kurtosis, and skewness of empirical mode** are combined into feature vectors,
- and then **kernel density estimation and mutual information are used to classify fault features**

- new index called key spectral negentropy (KSNE)
    - The left and right boundaries of the band with the maximum KSNE were used to reconstruct the empirical modes

- When **accidental shocks occur** in the system, the new shock information breaks the balance of the original system, 
    - the **entropy of the system decreases** . 
    - entropy has the ability similar to the kurtosis to test the unbalanced disturbance in the system.
    - **When the energy flow is constant**, the **maximum** key spectral entropy appears;
    - when the energy flow is compressed **into a single pulse**, the **minimum** key spectral entropy appears.

- Key spectral negentropy is proposed to reduce the steps of empirical mode selection
- In order to reduce the selection of parameters and debugging steps,
    - this paper proposed a novel method called adaptive and **fast empirical wavelet transform (AFEWT)**


#### **Algorithm of EWT (page 4) - good description**
    - **Meyer wavelet**
    - EWT takes the minimum between the local maxima of the spectrum as boundaries, and the boundaries are used to segment the spectrum



# Empirical wavelet transform 
Jerome Gilles 
- extract the different modes of a signal by designing an **appropriate wavelet filter bank**. 
- This construction leads us to a new wavelet transform, called the empirical wavelet transform.
- some experiments [5]–[7] show that EMD behaves like an adaptive filter bank

- In 1998, Huang et al. [9] proposed an original method called **Empirical Mode Decomposition (EMD)** to decompose a signal into specific modes
    - An IMF is an amplitude modulated-frequency modulated function which can be written in the form

- most known method is the **wavelet packets** in a basis pursuit framework based on successive scale refinements of the expansion
    - they use a constant prescribed ratio in the subdivision scheme, which limits their adaptability.

- recent work of Daubechies et al. [4] entitled **“synchrosqueezed wavelets”**.
    - algorithm permits to obtain a more accurate time-frequency representation and consecutively 
    - it is possible to extract specific “modes” by choosing the appropriate information to keep

- start by assuming that the **Fourier support [0, π]** - **normalized frequencies \omega=2\pi f \omega / fs** is segmented into N contiguous segments
- empirical wavelets are defined as bandpass filters on each Λn. To do so, we utilize the idea used in the construction of both Littlewood-Paley and Meyer’s wavelets
- 1000 Hz * 2 * pi / fs je [0, pi] 

- p.5 - we know how to build a tight frame set of empirical wavelets
- The detail coefficients are given by the inner products with the empirical wavelets

- p.5 - **Fig. 6 gives an empirical filter bank** example based on the set
    -  ωn ∈ {0, 1.5, 2, 2.8, π} 
    - with γ = 0.05 (the theory tells us that γ < 0.057). 
    - (3.14 - 2.8) /  (3.14 + 2.8) = 0,057239057 je minimum


- **Empirical wavelets (eq.8, eq.9)**
    - detail coefficients (inner product) - Eq.8:
        - \mathcal{W}_f^\varepsilon(n, t) = \langle f, \psi_n \rangle = \int f(\tau) \psi_n(\tau - t) d\tau
    - approximation coeficients (with scaling function) - Eq.9:
        - \mathcal{W}_f^\varepsilon(0, t) = \langle f, \phi_n \rangle = \int f(\tau) \phi_n(\tau - t) d\tau



------



## Anomaly Detection for Data Streams Based on Isolation Forest using Scikit-multiflow
\cite{gervasi_anomaly_2020}
- **Scikit-Multiflow** [20] is the main open source machine learning framework for multi-output, multi-label and data streaming. 
- Implemented in Python language, it includes various algorithms and methods for streams mining and 
- in particular the popular **Half-space Trees algorithm**

- **IForestASD** and compare it with a well known and state of art anomaly detection algorithm for data streams called Half-Space Trees
- When compared with a state-ofthe-art method (Hoeffding Trees) its performs favorably in terms of detection accuracy and run-time performance
- expect **HS-Trees to perform better than Isolation Forest ASD** as well in term of speed,

- **Anomaly Detection in Data Stream (ADiDS)** presents many challenges due to the characteristics of this type of data. 
    - data stream treatment has to be performed in a single pass to deal with memory limits and methods have to be applied in an online way.
    - outlier detection in data stream like concept drift, uncertainty and arrival rate.
    - nomaly detection methods are based on the facts that **anomalies are rare** and have a different behavior compared to normal data

- Anomaly detection:
    - **statistics** - establish a model that characterizes the normal behavior based on the dataset. Prior knowledge not avalibale - non-parametric methods
    - **clustering**/**nearest-neighbors** - based on the proximity between observations. suffer - need to compute the distance or the density between all the observations
    - **isolation-based** - are supposed to be very different from normal ones. 
        - They are also supposed to represent a very small proportion of the whole dataset. 
        - likely to be quickly isolated

- p.6 - **Comparison of ADiDS approaches.**
- p.8 - **Classification of data stream anomaly detection methods**

- **Isolation forest (IForest)** is an isolation-based method which isolates observations by splitting dataset.
    - needs many passes over the dataset to build all the random forest. 
    - it is not adapted to data stream context

- **Evaluation**:
    - **F1 metric**: if there is an imbalanced class distribution and we search for a balance between precision and recall
    - **Running Time Ratio** (IForestASD coefficient ratio) - IRa Since **HalfSpace Trees (HST) is always faster than IForestASD**
    - **Model Size Ratio** (HSTrees coefficient ratio) 
       - HRa In the opposite of the running time, when we consider the model size, we observe that **IFA always used less memory then HST**.
- IForestASD used less memory than HSTrees (≈ 20 times less), this is explained by the fact that with IForestASD, update policy
- IForestASD is faster with a small window size while Half-Space Trees is faster with bigger window size
- testing time of IForestASD – in the right axis in red line **can be 100x longer than HSTrees testing time**
- If a fast model and especially a fast scoring time is needed, HSTrees should be the privileged option as it is still the state-ofthe-art
- approach to be more efficient for drift detection in streaming data or use existing methods in **scikit-multiflow such as ADWIN** to automatically adapt the sliding window size

- we assume that IForestASD performs better on data set with relatively high anomaly rate.

- Forest-Cover Dataset
    - we **fixed the sliding window size (50, 100, 500, 100)**
    - vary **the number of Trees (30, 50, 100)** to compare HST and IForestASD running time


## One-Class Classification with LOF and LOCI: An Empirical Comparison
\cite{janssens_one-class_2007}
- **LOF and LOCI** are two widely used **density-based outlier-detection methods**. 
- Generally, LOCI is assumed to be superior to LOF,becauseLOCI constitutes a multi-granular method.
- **LOCI does not outperform LOF.**
- We discuss possible reasons for the results obtained and argue that the multi-granularity of LOCI 
- in some cases may hamper rather than help the detection of outliers.

- **Outlier:** "a rare observation that deviates so much from other observations as to arouse suspicion that it was generated by a different mechanism."
    - “anomalies”, “novelties”, and “exceptions”

- Traditional two-class or multi-class classifiers require training objects from all classes.
- **classifier can be thought of as a function** f which maps the input object xi to some output label yi.
- in many real-world problems it may be difficult or expensive to **gather examples from one or more classes** 
    - (e.g., machine fault detection)
- Outliers: **dissimilarity measure δ and (2) a threshold θ**
    - classifiers are evaluated on a complete range of thresholds using the AUC performance measure.

p.4 - **LOF and LOCI classify an object xi by**: 
    - constructing a neighbourhood around xi, 
    - estimating the density of the neighbourhood, and 
    - comparing this density with the neighbourhood densities of the neighbouring objects.

1. Step 1: **Constructing the Neighbourhood** : Euclidean distance d from xi to its kth nearest neighbour 
    - NN(x_i,k): d_{border}(x_i,k) = d(x_i, NN(x_i,k))
    - N (xi,k)={xj ∈Dtrain \{xi} | d(xi, xj ) ≤ dborder(xi,k)}. 
2. Step 2: **Estimating the Neighbourhood Density:** 
    - the density of the constructed neighbourhood, the **reachability distance** is introduced.
    - The reachability distance dreach is formally given by: 
       d_{reach}(x_i, x_j ,k) = max{d_{border}(x_j ,k), d(x_j, x_i)}. (asymmetric measure)
    - **The neighbourhood density ρ** of object xi depends on the **number of objects in the neighbourhood, |N (xi,k)|**, and on their reachability distances.
       - \rho(\mathbf{x}_i, k) = \frac{|\mathcal{N}(x_i,k)|}{\sum_{\mathbf{x}_j \in \mathcal{N}(x_i,k)} d_{reach}(\mathbf{x}_i, \mathbf{x}_j, k)}
    - Objects xj in the neighbourhood that are further away from object xi, have a smaller impact on the neighbourhood density
3. Step 3: **Comparing the Neighbourhood Densities**
    - LOF:
    - \omega(x_i,k)= \frac{
\sum_{\mathbf{x}_j \in \mathcal{N}(\mathbf{x}_i, k)} \frac{\rho(\mathbf{x}_j, k)}{\rho(\mathbf{x}_i, k)}}{|\mathcal{N}(x_i,k)|}


- **The Local Correlation Integral (LOCI) method**
    - improvement over LOF.
    - **the authors state that the choice of the neighbourhood size k in LOF is non-trivial**
    - and may lead to erroneous outlier detections.

   1. LOCI defines two neighbourhoods for an object xi: (
       - the extended neighbourhood: N_{ext}(x_i,r) = {x_j \in D_{train} | d(x_j, x_i) \leq r} ∪ x_i,
       - local neighbourhood: N_{loc}(x_i,r,\alpha) = {xj ∈ Dtrain | d(xj , xi) ≤ αr} ∪ xi,
   2. Step 2: Estimating the Neighbourhood Density
       - average density of the local neighbourhoods of all objects in the extended neighbourhood of object xi
       - \rho(x_i,r,\alpha) = \frac{
\sum_{\mathbf{x}_j \in \mathcal{N}_{ext}(\mathbf{x}_i, k)} \rho(\mathbf{x}_j, \alpha r)}{|\mathcal{N}_{ext}(x_i,k)
   3. Step 3: Comparing the Neighbourhood Densities
       - multi-granularity deviation factor (MDEF) - MDEF (xi,r,α)=1− \frac{ρ(x_i,αr)}{ρ(x_i,r,α)}
       - o determine whether an object is an outlier, LOCI introduces the normalized MDEF
       - he dissimilarity measure δ LOCI as the maximum ratio of MDEF to σMDEF of all radii r ∈R

**Results**: 
    - LOF, LOCI,andα-LOCI achieved an average performance on all datasets of 80.84 (18.2), 74.49 (19.0), and 77.73 (17.5)
    - The first possible reason is that the multiscale analysis of LOCI can be both a blessing and a problem
    - LOCI constructs a neighbourhood with a given radius. 
       - For small radii, the extended neighbourhood may contain only one object, implying that no deviation in the density can occur 
       - LOF, does not suffer from this because it constructs a neighbourhood with a given number of objects.


## Designing a Streaming Algorithm for Outlier Detection in Data Mining - An Incremental Approach
\cite{yu_designing_2020}
- **Update LOF**: Apart from them, the lrd value of point pn itself need to be updated since the Euclidian distances 
- to every of its k neighbours have changed. 
- For update of LOF values, it is the same as insertion and deletion operations.
- we need to simplify the update operation when the change of position is tiny.
-p.12 - **Incremental LOF**
    - the result of applying their incremental LOF algorithm is the same as the result of applying 
    - the static version of LOF algorithm after receiving N data points, and it is also independent of the order of the inse


## Density-Based Clustering over an Evolving Data Stream with Noise
\cite{cao_density-based_2006} **DenStream**
- we present **DenStream**, a new approach for discovering clusters in an evolving data stream. 
The “dense” micro-cluster (named core-micro-cluster) is introduced to summarize the clusters with arbitrary shape
    - potential **core-micro-cluster and outlier micro-cluster** structures are proposed
    - to maintain and distinguish the potential clusters and outliers.
- **goal of clustering** is to group the streaming data into meaningful classes.

- evolving data streams lead to the following **requirements for stream clustering**: 
   1. No assumption on the number of clusters. The number of clusters is often unknown in advance. 
   2. Discovery of clusters with arbitrary shape. 
   3. Ability to handle outliers. - in the data stream scenario, due to the influence of various factors

-p.2 - cluster partitions on evolving data streams computed based on **certain time intervals (or windows)**. 
    - **window models: landmark window, sliding window and damped window**
    
- In **damped window model**, fading function f (t) = 2−λ·t, where λ > 0
- In static environment, the clusters with arbitrary shape are represented by **all the points which belong to the clusters**

- unrealistic to provide such a precise result, approximate result and introduce a summary representation - **core-micro-cluster**
- number of c-micro-clusters is much larger than the number of natural clusters. 
- On the other hand, it is significantly smaller than the number of points
- **(potential) p-micro-clusters** and **(outlier) o-micro-clusters** can be maintained incrementally.

1. online part of micro-cluster maintenance
2. offline part of generating the final clusters, on demand by the user.

- we should provide opportunity for an o-micro-cluster to grow into a p-micro-cluster
- When a clustering request arrives, a **variant of DBSCAN algorithm** is applied on the set of on-line maintained pmicro-clusters to get the final result of clustering
- DBSCAN: **two parameters \varepsilon and \mu.**
    - For \epsilon, if it is too large, it may mess up different clusters. 
    - If it is too small, it requires a corresponding smaller μ. 
    - However, a smaller μ will result in a larger number of micro-clusters

DenStream adopt the following setting: 
    - initial number of points InitN = 1000, 
    - stream speed v = 1000, 
    - decay factor λ = 0.25, 
    - \epsilon = 16, μ = 10, outlier threshold β = 0.2.
- **DenStream outperforms CluStream for all parameter settings**


## Review of Artificial Intelligence-based Bearing Vibration Monitoring
\cite{sheng_review_2020}
p.2 - three main machine learning methods used in health monitoring are introduced, including 
- **KNN, ANN and SVM**
- **KNN**
+Plus
- 1. Easy to understand and implement, no need to estimate parameters, and no need to train; 
- 2.Suitable for the classification of rare events; 
- 3. Especially suitable for multi-modal problems of having multiple category tags. 
-Minus
- 1. When samples are unbalanced, the number of samples will not affect the operation result; 
- 2. Heavy calculation cost;
- 3. Poor comprehensibility.

- **Main vibration features** include:
    - mean, median, Kurtosis, peakto-peak values, root mean square, nuclear density estimation, fault frequencies, time-frequency parameters

- Different:
    - **distance functions** (e.g., Euclidean, correlation and Mahalanobis distances) 
    - different **number of nearest neighbors** (K) are applied

- **how to improve the self-learning and generalization ability** of various intelligent diagnostic methods using
- such unbalanced small sample data is the focus of future research.
- Experimental results show that the ANN model is more effective than the KNN model in diagnosing multiple faults
- a new transfer learning based on pre-trained VGG-19 is proposed for fault diagnosis. 
- The proposed method was tested on the famous motor bearing data set of **Case Western Reserve University**. The accuracy obtained is about **99.175%** and the training time is only nearly 200 seconds.

## Semi-Supervised Learning on Data Streams via Temporal Label Propagation
\cite{wagner_semi-supervised_2018}
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


## State-of-the-art on clustering data streams
\cite{ghesmoune_state---art_2016}
- **Clustering** 
    - partitioning a set of observations into clusters such that the intra-cluster observations are similar 
    - and the inter-cluster observations are dissimilar.
- **Real-time processing** means that the ongoing data processing requires a very low response delay. The velocity, which refers to that Big Data

- Most of the existing algorithms (e.g. **CluStream [1], DenStream [2], StreamKM++ [3], or ClusTree [16]**) 
divide the clustering process in two phases: 
    - (a) Online, the data will be summarized; 
    - (b) Offline, the final clusters will be generated.
- Hierarchical clustering can be achieved in two different ways, namely, **bottomup and top-down clustering.**
    - once a step (merge or split) is done, it can never be undone

- **Balanced Iterative Reducing and Clustering using Hierarchies (BIRCH)**  (p.8)
    - only circle clusters
    - incrementally and dynamically clusters multi-dimensional data points 
    - to try to produce the best quality clustering with the available resources
    - BIRCH requires two user defined parameters: 
       - B the branch factor or the maximum number of entries in each non-leaf node; 
       - T the maximum diameter (or radius) of any CF in a leaf node
    - CF-tree can be performed in a similar way as the insertion in the classic B-tree. 
    - If the closest-CF in the leaf cannot absorb the data point, a new CF entry is created. If there is no room for new leaf, the parent node is split.

- **ClusTree** is a parameter-free stream clustering algorithm 
    - that is capable of processing the stream in a single pass, with limited memory usage

- **Anytime algorithms** denote approaches that are capable of delivering a result at any given point in time,
    and of using more time if it is available to refine the result.

**Density-based algorithms**
    - based on the connection between regions and density functions. 
    - dense areas of objects in the data space are considered as clusters, 
    - which are segregated by low density area (noise).
    - find clusters of arbitrary shapes and generally they **require two parameters**: 
       - the radius and 
       - the minimum number of data points within a cluster.
    **DenStream**
        - By creating two kinds of micro-clusters (potential and outlier micro-clusters), in its online phase, 
        - DenStream overcomes one of the drawbacks of CluStream, its sensitivity to noise
        - DenStream has a pruning method in which it frequently checks the weights of the 
        - outlier-micro-clusters in the outlier buffer to guarantee the recognition of the real outliers

**p. 20 - Table 1 Comparison between algorithms (WL: weighted links, 2 phases : online+offline)**

- **Apache Spark** project by adding the ability to perform online processing through a similar functional interface to Spark
- Multiple open-source software libraries use MOA to perform data stream analytics in their systems 
    - including **ADAMS, MEKA, and OpenML**. 
    - others big data streams frameworks are **SAMOA and StreamDM**

- Samples, Histograms, Wavelets, Sketches describe basic principles and recent developments 
- in building **approximate synopses (that is, lossy, compressed representations)** of massive data


## Cluster-Reduce: Compressing Sketches for Distributed Data Streams
\cite{zhao_cluster-reduce_2021}
- With the increasing volume and velocity of data streams, 
- **sketches, a type of probabilistic algorithms**, have been widely used in estimating item statistics

- Compressing the CM sketch, which is widely used in frequency estimation. 
    - A **CM sketch** contains a counter array and 𝑑 hash functions. 
    - **For each incoming item**, the CM first **locates 𝑑 counters** by calculating 𝑑 hash functions on the item ID, which are abbreviated as 𝑑-hash-counters. 
    - **Then the CM sketch increases these counters by 1**. To estimate the frequency of a given item, the **CM reports the minimum value** of the 𝑑-hash-counters

- **Cluster-Reduce** is divided into two steps: **nearness clustering and unique reducing**

- **CM sketch** is used to estimate item frequency, and can guarantee the one-side error. 
- **CU sketch** improves CM, uses conservative update strategy to achieve higher accuracy, and also guarantees the one-side error. 
- **Count sketch** achieves the unbiased estimation of item frequency, and can be combined with a heap to find top-K frequent items

- Given a compression ratio 𝜆, we divide the original sketch into 𝑤𝑐 = 𝑤𝑜 /𝜆 equal-sized groups, each containing 𝜆 continuous counters
- propose a dynamic programming (DP) method to classify counters for minimizing the compression error


## Fast Anomaly Detection for Streaming Data
\cite{tan_fast_2011}
**Streaming Half-Space-Trees (HS-Trees)**

- a fast **one-class anomaly detector** for evolving data streams. 
- It **requires only normal data for training** and works well when anomalous data are rare

- it processes data in one pass and only requires constant amount of memory to process potentially endless streaming data
- is useful when a stream contains a **significant amount of normal data**.
- **ensemble of Streaming HS-Trees** leads to a robust and accurate anomaly detector that is not too sensitive to different parameter settings.

- Each HSTree consists of a set of nodes, where each node captures the number of data items (a.k.a. **mass**) within a particular subspace of the data stream. - **Mass** is used to profile the **degree of anomaly** because it is simple
    - fast to compute in comparison to distance-based or density-based methods.

- two consecutive windows, **the reference window, followed by the latest window**. 
    - During the initial stage of the anomaly detection process, the algorithm **learns the mass profile of data in the reference window**. 
    - the learned profile is used to **infer the anomaly scores of new data** subsequently arriving **in the latest window**
    - When the latest window is full, the newly recorded profile is used to override the old profile in the reference window;

- **HS-Tree of depth h is a full binary tree** consisting of 2h+1 − 1 nodes, in which all leaves are at the same depth, h.
    - constructing a tree, the algorithm expands each node by **picking a randomly selected dimension**, (space)
    - Each node records the mass profile of data in a work space that it represents:
       - arrays min and max, which respectively store the minimum and maximum values of each dimension of the work space represented by the node
       - variables r and l, which record the mass profiles of data stream captured in the reference window and latest window, respectively;
       - variable k,which records the depth of the current node
       - two nodes representing the left child and right child of the current node, each associated with a half-space after the split

- **Constructing:** (p.3)
    - **Each internal node is formed** by randomly selecting a dimension q (Line 4) to form two half-spaces; 
    - the split point is the mid-point of the current range of q
- **Recording mass profile**
    - mass profile of normal data must be recorded in the trees before they can be employed for anomaly detection
    - These two collections of mass values at each node, r and l,represent the data profiles in the two different windows

- Streaming HSTrees is able to learn data stream profile using small samples; 
    - a **small window size of ψ = 250 is sufficient** for our experiments. 
    - The ensemble uses **25 trees as this is a moderate ensemble size**

- Hence the (average-case) amortised **time complexity for n streaming points is O(t(h+1))**; the worst-case is O(t(h+ψ)),
- **space complexity for HSTrees is O(t2h)** which is also a constant with fixed t and h.
- satisfies the key requirements for mining evolving data streams: 
    (i) it is a one-pass algorithm with O(1) amortised time complexity and O(1) space complexity

- Algorithm 1 : BuildSingleHS-Tree(min, max, k)
- Algorithm 2 : UpdateMass(x, N ode, referenceWindow)
- Algorithm 3 : Streaming HS-Trees(ψ, t)


## Classification of washing machines vibration signals using discrete wavelet analysis for feature extraction
\cite{goumas_classification_2002}
 Mapping measurement sapce into feature space
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

## Semi-Supervised Learning
\cite{chapelle_semi-supervised_2006}

## Data Clustering - Algorithms and Applications
\cite{aggarwal_data_2014}
p.113 - **DBSCAN**
- Let C1,...,Ck be the clusters of the database D with respect to **Eps and MinPts**.

- DBSCAN [11] estimates the density by counting the number of points in a fixed-radius neighborhood
- considers two points as connected if they lie within each other’s neighborhood. 
- A point is called **core point if the neighborhood of radius Eps contains at least MinPts points**, 
- i.e., the density in the neighborhood has to exceed some threshold.



## Feature-based performance of SVM and KNN classifiers for diagnosis of rolling element bearing faults
\cite{jamil_feature-based_2021}
- ML models, namely, Support Vector Machine (SVM) and K-Nearest Neighbor (KNN), are used to classify the faults associated with different ball bearing.
- Case Western Reserve University (CWRU) bearing data,
- ML classifiers are trained with extracted **time-domain and frequency-domain features**

- The results show that **frequency-domain features are more convincing for the training of ML models**, 
- and the **KNN classifier has a high level of accuracy** compared to SVM.

- The **K-Nearest Neighbor (KNN)** algorithm is one of the most basic machine-learning algorithms.
- It is a method of calculating the distance between two points.
- Due to its simplicity and ease of implementation, this is a widely used classifier
- It is a non-parametric classification
- KNN determines the **distance between two points using multiple techniques**, such as Euclidian and Manhattan [15], based on the idea of similarity based on proximity or distance
- **value of 𝐾 should be selected to decrease the number of errors** when making predictions from each run (multiple runs)

- **Fault features**
    - A total of 18 time-domain and frequency-domain fault features extracted are used in different combinations to assess the accuracy of SVM and KNN models in classifying the bearing fault categories. 

- **The time-domain features**:
    - clearance factor, crest factor, impulse factor, kurtosis, mean, peak value, RMS, SINAD, SNR, shape factor, skewness, standard deviation, approximate entropy, correlation dimension, and Lyapunov exponent. 
**The features of the frequency-domain** 
    - peak amplitude, peak frequency, and band power.
 
**Set 1**: Combination of time and frequency-domain features
    - Crest factor, impulse factor, kurtosis, RMS, SNR, skewness, peak amplitude, peak frequency, Lyapunov exponent 
    - SVM 95.0 % KNN 96.2 %
**Set 2**: Non-linear time-domain features 
    - Approximate entropy, correlation dimension, Lyapunov exponent 
    - SVM 88.8 % KNN 91.2 % 
**Set 3:** Frequency-domain features 
    - Peak amplitude, peak frequency, band power 
    - SVM 96.2 % KNN 98.8 %




