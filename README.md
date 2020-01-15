# lstmclassification
Event Log Classification with Deep Learning


In this work, we investigate if Artificial Neural Networks can classify streams of event logs that are occurring prior to a
connection error. From production monitoring to security concerns, log data is available in almost every sector. 
Current research often focuses on predicting the next event in a sequence and the networks that are used in these studies are
capable of doing this task reliably. These models do not perform well when trying to predict events further forward in time and 
this thesis tries to find a more practical alternative, as opposed to short-term predictions. 

ABB Robotics provided a dataset that contained all the chronological changes in machine statuses.
This dataset was cleaned and used as input data for a neural network model. 
This work also aims to analyze the effect of time on the prediction and concludes that there is a trade-off to be found
when trying to do early predictions. The two methods that were compared showed that the best model is able to predict a
connection error with an f1 score of 0.91, on an average of 58 minutes before the error occurred.
The result showed that our neural network is capable of distinguishing the streams of data and that these streams
can provide the basis of root-cause analyses. 

Using techniques like Process Mining can help businesses to perform root-cause analyses and 
prevent the connection error from happening in the future.

The codes for method A en method B are found in this repository.
Method A consisted of sampling and labeling the data using a sequence of events, disregarding the time between the events.
Method B looked at the timestamps and set exact prediction windows between the training data and actual arise of the label. 

The methods were compared in the end and method A provided a more stable model.
