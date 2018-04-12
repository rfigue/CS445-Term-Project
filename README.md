# CS445-Term-Project Proposal

Project Proposal
CS445 - Spring 2018

Team Members:  Evan Salzman, Rodolfo Figueroa

# Project Description:  
For this project, we will select a number of stocks from the data available at Quandl (https://www.quandl.com/databases/WIKIP/documentation/about), where they release data each day from the stock market, and apply machine learning algorithms to forecast how they will perform in the future.  Specifically, we will make use of recurrent neural networks, as opposed to traditional neural networks, so that we can use previous results to predict those in the future.  Therefore, recurrent neural networks are ideal for this stock investigation, because much of forecasting how they will fare in the days, weeks and months to come needs to reference past performance.  The implementation of the code to complement the stock forecasting will reference the neural network classes written and provided in the course lectures and assignments as well as resources about recurrent neural networks.

# Report Outline:
* Introduction to the assignment and reasoning for choosing stock forecasting as the project topic
* Description of the dataset (i.e. origin, attributes, validity) used for this project
* Description of the algorithms and types of neural networks used to forecast the behavior of stocks, as well as all of the code written
* Explanation of all tests and experiments run to forecast stock prices, including variations in parameter values
* Plots of results most relevant to the conclusions related to parameter values and outcomes related to the objective of this project,
* Describe some of the more challenging aspects of the project and how I dealt with them, 
* Reflection on the project’s application to machine learning in addition to insight gained on recurrent neural networks and forecasting stock behavior
* Outline of additional steps to take in the case that this project continues after the semester ends.

# Project Timeline: The following dates will ensure timely progress:
* April 5: Find and ensure ability to access a dataset that we can use from the stock market.  Begin researching recurrent neural networks.
* April 9: Research is done and there is enough understanding of recurrent neural networks/forecasting being implementing the project.
* April 16: Complete the recurrent learning network class and additional methods for forecasting
* April 20: Have dataset partitioned based on specific attributes that help to forecast stocks, have identified specific stocks to highlight within the report. 
* April 24: Apply the recurrent neural network to the dataset and forecast for the chosen stocks.
* May 4: Finish the draft for the final project report, meet as a team and discuss finalizing the report
* May 9: Submit the final project report, with all individual tasks completed

# Team Member Assignments: 

Rodolfo Figueroa
* will write an introduction of the project, its significance and the connection it has to the field of machine learning.
* will find a dataset of US stocks and discuss its contents, including the author(s), origin and how the data was collected.
* will write code to read in the file of US stock data appropriately.
* will work on implementing the recurrent neural network class (RNN). 
* will apply the RNN and prediction methods to highlight 4 - 6 stocks and will consider the success of the algorithm and which of the stocks will be the best to invest in (includes graphs, explanations and multiple trials). 
* will collaborate on the discussion of further steps to take if this project continued beyond the end of this semester.

Evan Salzman
* will work on implementing the RNN class as well as forward propagation and prediction methods to go along with the RNN class.
* will partition data to build training and testing sets from the dataset.
* will provide comments, descriptions and explanations for all of the implemented code.
* will apply the RNN and prediction methods to highlight 4 - 6 stocks and will consider the success of the algorithm and which of the stocks will be the best to invest in (includes graphs, explanations and multiple trials).
* will provide insight into the successes and shortcomings of recurrent neural networks compared to what is seen as a traditional neural network
* will collaborate on the discussion of further steps to take if this project continued beyond the end of this semester.

