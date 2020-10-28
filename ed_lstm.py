from dataloader import ETHDataset
import click
import tensorboard_logger as tb_logger
from torch.utils import data
from tqdm import tqdm
from utils.helper import save_checkpoint
import torch
from model import Model
import numpy as np
import os
from datetime import datetime

from toolkit.loaders.loader_eth import load_eth

# load and plot dataset
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot

# multivariate multi-step encoder-decoder lstm
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed



@click.command()
@click.option('--experiment_name', type=str, default="naive_lstm")
@click.option('--steps', type=int, default=0)
@click.option('--num_workers', type=int, default=4)
@click.option('--last_checkpoint', type=str, default=None)
@click.option('--batch_size', type=int, default=4)
@click.option('--epoch_num', type=int, default=1)
@click.option('--lr', type=float, default=1e-3)
@click.option('--context_frame_num', type=int, default=8)
@click.option('--trajectory_interval', type=int, default=20) # each frame is 0.4 sec. It seems it takes 10 sec on average to walk on the trajectory
@click.option('--agent_buffer_size', type=int, default=8)
@click.option('--log_step', type=int, default=128)
def main(experiment_name, steps, num_workers, last_checkpoint, batch_size, epoch_num, lr, context_frame_num, trajectory_interval, agent_buffer_size, log_step):

    agent_id = 2

    # data format of ETH/seq_eth/obsmat.txt:
    # [frame_number pedestrian_ID pos_x pos_z pos_y v_x v_z v_y ]
    # pos_z and v_z are not used
    traj_dataset = load_eth('datasets/ETH/seq_eth/obsmat.txt')
    # Head : frame_id  agent_id      pos_x     pos_y     vel_x     vel_y  scene_id       label  timestamp

    if (len(traj_dataset.data['scene_id'].unique()) == 1):
        # drop a column by calling drop.
        # inplace=True means the operation would work on the original object.
        # axis=1 means we are dropping the column, not the row.
        traj_dataset.data.drop('scene_id', inplace=True, axis=1)

    if (len(traj_dataset.data['label'].unique()) == 1):
        # drop a column by calling drop.
        # inplace=True means the operation would work on the original object.
        # axis=1 means we are dropping the column, not the row.
        traj_dataset.data.drop('label', inplace=True, axis=1)

    traj_dataset.data.drop('frame_id', inplace=True, axis=1)
    traj_dataset.data.drop('timestamp', inplace=True, axis=1)

    traj_dataset.data = traj_dataset.data[traj_dataset.data['agent_id'] == agent_id ]
    traj_dataset.data.drop('agent_id', inplace=True, axis=1)

    data_predict_x = traj_dataset.data.copy()
    pos_x = data_predict_x['pos_x']
    data_predict_x.drop(labels=['pos_x'], axis=1, inplace=True)
    data_predict_x.insert(0, 'pos_x', pos_x)

    data_predict_y = traj_dataset.data.copy()
    pos_y = data_predict_y['pos_y']
    data_predict_y.drop(labels=['pos_y'], axis=1, inplace=True)
    data_predict_y.insert(0, 'pos_y', pos_y)

    # split into train and test
    train_x, test_x = split_dataset(data_predict_x)
    train_y, test_y = split_dataset(data_predict_y)
    split_dataset(data_predict_y)

    # supervised = series_to_supervised(traj_dataset.data, 1, 2)
    # print(supervised.head())
	#
    # evaluate model and get scores
    n_input = 2
    predictions_x,  score_x, scores_x = evaluate_model(train_x, test_x, n_input)
    predictions_y, score_y, scores_y = evaluate_model(train_y, test_y, n_input)

    pyplot.figure()
    pyplot.scatter(data_predict_y['pos_y'], data_predict_x['pos_x'], c='y')
    pyplot.scatter(predictions_y[:,0,0], predictions_x[:,0,0], c='g')
    pyplot.scatter(test_y[:,0,0], test_x[:,0,0], c='b')

    pyplot.figure()
    pyplot.plot(scores_x)

    pyplot.figure()
    pyplot.plot(scores_y)
    pyplot.show()


# summarize scores


# # convert time series into supervised learning problem
# def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
#     '''
#     Frame a time series as a supervised learning dataset.
#     Arguments:
#         data: Sequence of observations。 Required.
#         n_in: Number of lag observations as input (X). Values may be between [1..len(data)] Optional. Defaults to 1.
#         n_out: Number of observations as output (y). Values may be between [0..len(data)-1]. Optional. Defaults to 1.
#         dropnan: Boolean whether or not to drop rows with NaN values. Optional. Defaults to True.
#     Returns:
#         Pandas DataFrame of series framed for supervised learning
#     '''
#     col_headers = list(data.columns.values)
#     df = DataFrame(data)
#     cols, names = list(), list()
#     # input sequence (t-n, ... t-1)
#     for i in range(n_in, 0, -1):
#         cols.append(df.shift(i))
#         names += [j+('(t-%d)' % i) for j in col_headers]
#     # forecast sequence (t, t+1, ... t+n)
#     for i in range(0, n_out):
#         cols.append(df.shift(-i))
#         if i == 0:
#             names +=  [j+('(t)') for j in col_headers]
#         else:
#             names += [j+('(t+%d)' % i) for j in col_headers]
#     # put it all together
#     agg = concat(cols, axis=1)
#     agg.columns = names
#     # drop rows with NaN values
#     if dropnan:
#         agg.dropna(inplace=True)
#     return agg

# split the dataset into train/test sets
def split_dataset(data):
    last_index = data.shape[0]
    # split into 5 frames
    train, test = data[0:int(last_index*0.7)], data[int(last_index*0.7):-1]
    # restructure into windows of weekly data
    train = array(train.values.reshape(train.shape[0], 1, train.shape[1] ))
    test = array(test.values.reshape(test.shape[0], 1, test.shape[1] ))

    return train, test

# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
	scores = list()
	# calculate an RMSE score for each day
	for i in range(actual.shape[0]):
		# calculate mse
		mse = mean_squared_error(actual[i, :], predicted[i, :])
		# calculate rmse
		rmse = sqrt(mse)
		# store
		scores.append(rmse)
	# calculate overall RMSE
	s = 0
	for row in range(actual.shape[0]):
		for col in range(actual.shape[1]):
			s += (actual[row, col] - predicted[row, col])**2
	score = sqrt(s / (actual.shape[0] * actual.shape[1]))
	return score, scores


# train the model
def build_model(train, n_in):
    # prepare data
	train_x, train_y = to_supervised(train, n_in)
    # define parameters
	verbose, epochs, batch_size = 0, 100, 16
	n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # reshape output into [samples, timesteps, features]
	train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
    # define model
	model = Sequential()
	model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
	model.add(RepeatVector(n_outputs))
	model.add(LSTM(200, activation='relu', return_sequences=True))
	model.add(TimeDistributed(Dense(100, activation='relu')))
	model.add(TimeDistributed(Dense(1)))
	model.compile(loss='mse', optimizer='adam')
	# fit network
	model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
	return model

# make a forecast
def forecast(model, history, n_input):
	# flatten data
	data = array(history)
	data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
	# retrieve last observations for input data
	input_x = data[-n_input:, :]
	# reshape into [1, n_input, n]
	input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
	# forecast the next week
	yhat = model.predict(input_x, verbose=0)
	# we only want the vector forecast
	yhat = yhat[0]
	return yhat

# evaluate a single model
def evaluate_model(train, test, n_input):
	'''
	n_input: used to define the number of prior observations that the model will use as input in order to make a prediction
	'''
	# fit model
	model = build_model(train, n_input)
	# history is a list
	history = [x for x in train]
	# walk-forward validation over each week
	predictions = list()
	for i in range(len(test)):
		# predict the week
		yhat_sequence = forecast(model, history, n_input)
		# store the predictions
		predictions.append(yhat_sequence)
		# get real observation and add to history for predicting the next week
		history.append(test[i, :])
	# evaluate predictions days for each week
	predictions = array(predictions)
	score, scores = evaluate_forecasts(test[:, :, 0], predictions)
	return predictions,score, scores

# summarize scores
def summarize_scores(name, score, scores):
	s_scores = ', '.join(['%.1f' % s for s in scores])
	print('%s: [%.3f] %s' % (name, score, s_scores))

# convert history into inputs and outputs
def to_supervised(train, n_input, n_out=1):
	# flatten data
	data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
	X, y = list(), list()
	in_start = 0
	# step over the entire history one time step at a time
	for _ in range(len(data)):
		# define the end of the input sequence
		in_end = in_start + n_input
		out_end = in_end + n_out
		# ensure we have enough data for this instance
		if out_end <= len(data):
			X.append(data[in_start:in_end, :])
			y.append(data[in_end:out_end, 0])
		# move along one time step
		in_start += 1
	return array(X), array(y)

if __name__ == '__main__':
    main()
