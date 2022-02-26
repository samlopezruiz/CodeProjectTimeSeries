# Multiobjective Framework for Quantile Forecasting in Financial Time Series using Transformers


Authors: Samuel López-Ruiz

### Abstract
> We present a comparison between three different artificial neural networks designed for time series forecasting. The proposed models are fully-connected neural networks, long short-term memory networks and dilated convolutional neural networks. The intent is to apply the best model(s) to financial time series predictions. The scope of this work is, however, limited to the multi-step forecast of the multivariate Lorenz attractor, which is a set of chaotic solutions of the Lorenz system and with the addition of white noise, it constitutes a set of chaotic, noisy, non-linear and easily replicable time series. 

## Models
In this work, the following neural networks architectures for time series forecasting are evaluated: CNN, CNN-LSTM, ConvLSTM, and a WaveNet architecture. Additionally, the STructured Representation On Genetic Algorithms for NOnlinear Function Fitting (Stroganoff) system is compared to the NN models. 


## Data
The 3-variate time series are derived from the Lorenz attractor, which consists of a compact invariant set in the three-dimensional phase space of a smooth flow which has the complicated topological structure mentioned below and is asymptotically stable. The concept of an attractor, that is, an attracting set, often includes only the latter of these two properties; however, both the Lorenz attractor and other practically important attractors have both these properties.

<img src="https://github.com/samlopezruiz/CodeProjectTimeSeries/blob/main/docs/lorenz-attractor-time-series.png?raw=true" width="700" height="250"/>

## Results
The superiority of GP and NN models vs. traditional approaches (i.e., ARMA) was shown in this work. Additionally, it is explained how multivariate - input models outperform univariate - input models.
Additionally, the performance achieved by multivariate - inputs models using different output steps $d$ is compared using 
different architectures.

The next figure shows the comparison between all GP and NN models, using 1, 3, 6 output steps.
The ConvLSTM architecture exhibits the best performance for 1 output steps. 
For 3 and 6 output steps, the CNN-LSTM has the best average _MinMax_ score. 
The results seems to suggest the ConvLSTM architecture is better for predicting short sequences of future values, where as the CNN-LSTM performs best when predicting longer sequences of values.
<br>
Thanks to the dilated convolutional layers, the D-CNN and WAVENET models consistently have less parameters than their counterparts.
This is beneficial to avoid overfitting and allows for better generalization.
Interestingly, the D-CNN model outperforms the WAVENET model in all three cases.
This suggests that the modifications to the WAVENET suggested by (Borovykh, 2018) allow the model to better extract patterns and predict future values more precisely. 
<br>
Concerning the NN models, the CNN model is the most simple model and has an average performance in all three cases The results suggest it is not enough to have stacked convolutional layers to efficiently extract patterns. 
When complemented with LSTM layers (CNN-LSTM), the performance greatly improves. 

Consistently along the different output steps, the NN models outperform the GP models. 
The performance difference between NN and GP models increases as the number of output steps increases. 
For 1 output step the GP models performance is outstanding, considering the small number of parameters they have. 
For larger output steps, the prediction capability of GP models greatly diminishes. 
This suggests that the model's architecture is not well suited to forecast >1 output steps. 
A better ensemble technique must be implemented to improve the performance for more than one output step.
<img src="https://github.com/samlopezruiz/CodeProjectTimeSeries/blob/main/docs/s1_s2_s3_all.png?raw=true" width="700" height="250"/>

## Forecast visualization
The next figure shows the forecast done using an average ensamble model consisting of CONV-LSTM, CNN-LSTM and D-CNN for the test dataset using 6 output steps.
It is clear the model is able to predict with good accuracy the next 6 output steps. 

<img src="https://github.com/samlopezruiz/CodeProjectTimeSeries/blob/main/docs/ensemble_forecast.png?raw=true" width="700" height="250"/>

## Conclusions
