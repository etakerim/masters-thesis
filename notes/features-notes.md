 
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

#TODO --------------------------------------------------------------------

## Spectral negentropy and kurtogram performance comparison for bearing fault diagnosis
\cite{avoci_spectral_2020}
- paper is motivated by ideas borrowed from thermodynamics, where transients are seen as departures from a state of equilibrium; 
- it is proposed to measure the negentropy of the **squared envelope (SE)** and the **squared envelope spectrum (SES)** of the signal

## Application of Teager–Kaiser Energy Operator in the Early Fault Diagnosis of Rolling Bearings
\cite{shi_application_2022}



## The fast continuous wavelet transformation (fCWT) for real-time, high-quality, noise-resistant time–frequency analysis
\cite{arts_fast_2022}

## A Concentrated Time–Frequency Analysis Tool for Bearing Fault Diagnosis
\cite{yu_concentrated_2020}

## Applications of the synchrosqueezing transform in seismic time-frequency analysis
\cite{herrera_applications_2014}

## Wavelet Packet Feature Extraction for Vibration Monitoring
\cite{yen_wavelet_2000}

## A wavelet approach to dimension reduction and classiﬁcation of hyperspectral data
\cite{wickmann_wavelet_2007}
	
## The MFBD: a novel weak features extraction method for rotating machinery
\cite{song_mfbd_2021}


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

## Novel self-adaptive vibration signal analysis: Concealed component decomposition and its application in bearing fault diagnosis
\cite{tiwari_novel_2021}



## Fault Feature Extraction and Enhancement of Rolling Element Bearings Based on Maximum Correlated Kurtosis Deconvolution and Improved Empirical Wavelet Transform
	\cite{li_fault_2019}
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


## An Improved Empirical Wavelet Transform for Noisy and Non-Stationary Signal Processing
	\cite{zhuang_improved_2020}


## Time and frequency domain scanning fault diagnosis method based on spectral negentropy and its application
\cite{yonggang_time_2020}
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


## An Adaptive Spectrum Segmentation Method to Optimize Empirical Wavelet Transform for Rolling Bearings Fault Diagnosis
\cite{xu_adaptive_2019}
	
## Improved empirical wavelet transform (EWT) and its application in non‑stationary vibration signal of transformer
\cite{ni_improved_2022}





------




## Analysis of different RNN autoencoder variants for time series classification and machine prognostics
\cite{yu_analysis_2021}


## Anomaly Detection for Data Streams Based on Isolation Forest using Scikit-multiflow
\cite{gervasi_anomaly_2020}

## One-Class Classification with LOF and LOCI: An Empirical Comparison
\cite{janssens_one-class_2007}

## Designing a Streaming Algorithm for Outlier Detection in Data Mining - An Incremental Approach
\cite{yu_designing_2020}

## Density-Based Clustering over an Evolving Data Stream with Noise
\cite{cao_density-based_2006}

## A Modified Approach of OPTICS Algorithm for Data Streams
\cite{shukla_modified_2017}

## Data Clustering - Algorithms and Applications
\cite{aggarwal_data_2014}

## State-of-the-art on clustering data streams
\cite{ghesmoune_state---art_2016}

## Cluster-Reduce: Compressing Sketches for Distributed Data Streams
\cite{zhao_cluster-reduce_2021}
	

## Fast Anomaly Detection for Streaming Data
\cite{tan_fast_2011}


## Review of Artificial Intelligence-based Bearing Vibration Monitoring
\cite{sheng_review_2020}

## Semi-Supervised Learning on Data Streams via Temporal Label Propagation
\cite{wagner_semi-supervised_2018}

## Minimum covariance determinant and extensions
\cite{hubert_minimum_2018}


## Feature-based performance of SVM and KNN classifiers for diagnosis of rolling element bearing faults
\cite{jamil_feature-based_2021}

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

## A Novel Online Machine Learning Approach for Real-Time Condition Monitoring of Rotating Machines
\cite{maurya_condition-based_2021} 





