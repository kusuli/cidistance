import os
import math
import sys
import re
import torch
import numpy
import statistics as st
#import util
import random
from multiprocessing import Pool
import importlib.resources

from .tps import TPS
from .gtps import GTPS
#import setting
from .setting import NOTE_POS_DIC
from .setting import MODE_DISTANCE_DIC
from .setting import MODE_MAJOR
from .setting import MODE_MINOR
from .setting import CHORD_TYPE_MAJ
from .setting import CHORD_TYPE_MIN
from .setting import CHORD_TYPE_DIM7

dtg = [] # dtg: degree tri-gram
dtg_weight = 0.5

#! csv (row) -> [(tonic, mode, degree)]
def load_csv(csv_line):
	arr = csv_line.split(',') # measure number, C, C#, D, D#, E, F, F#, G, G#, A, A#, B, depth, (key tonic, key mode, figure, degree, quality, inversion, root) × depth
	int_num = int(arr[13]) # depth
	ret_list = []
	for i in range(int_num):
		tonic = NOTE_POS_DIC[arr[14 + i * 7].strip()]
		mode = int(arr[15 + i * 7])
		degree = int(arr[17 + i * 7])
		ret_list.append((tonic, mode, degree))
	return ret_list

#! csv (directory) -> [[(tonic, mode, degree)]]
def load_data(data_path, ap_setting = "epsilon"):
	ret_tpl_list = []
	with open(data_path, "r") as f:
		#print(data_path)
		prev_tpl_list = []
		for line in f:
			#print(line)
			tpl_list = load_csv(line.replace("\n", ''))
			if len(tpl_list) == 0:
				pass
			else:
				if ap_setting == "local":
					ret_tpl_list.append(tpl_list[0])
				elif ap_setting == "original":
					degree = 0
					for tpl in tpl_list:
						degree += tpl[2] - 1
					degree = (degree % 7) + 1
					tpl = tpl_list[-1]
					ret_tpl_list.append((tpl[0], tpl[1], degree))
					#print(line, ret_tpl_list[-1])
				elif ap_setting == "epsilon":
					if len(prev_tpl_list) > len(tpl_list):
						tpl = (prev_tpl_list[0][0], prev_tpl_list[0][1], 1)
						ret_tpl_list.append(tpl)
					ret_tpl_list.append(tpl_list[0])
				else:
					print("load_data error: ", ap_setting)
					exit()
			prev_tpl_list = tpl_list
		if False:
			ret_tpl_list_x2 = [ret_tpl_list]
		else: # remove repetitions
			temp_tpl_list = []
			for i, tpl in enumerate(ret_tpl_list):
				if i == 0 or ret_tpl_list[i - 1] != tpl:
					temp_tpl_list.append(tpl)
			ret_tpl_list_x2 = [temp_tpl_list]
	return ret_tpl_list_x2

#! (tonic, mode, degree) -> (chord name, root_pos, chord_type)
def convert_tpl(tpl):
	root_pos = tpl[0] + MODE_DISTANCE_DIC[tpl[1]][tpl[2] - 1]
	root_pos %= 12
	chord_str = [k for k, v in NOTE_POS_DIC.items() if v == root_pos][0]
	if tpl[1] == MODE_MAJOR:
		if tpl[2] == 1 or tpl[2] == 4 or tpl[2] == 5:
			chord_str += ':maj'
			chord_type = CHORD_TYPE_MAJ
		elif tpl[2] == 2 or tpl[2] == 3 or tpl[2] == 6:
			chord_str += ':min'
			chord_type = CHORD_TYPE_MIN
		elif tpl[2] == 7:
			chord_str += ':dim'
			chord_type = CHORD_TYPE_DIM7 # dim
		else:
			print('convert_tpl_to_chord_name error: ' + tpl)
			raise Exception
	else:
		if tpl[2] == 3 or tpl[2] == 6 or tpl[2] == 7:
			chord_str += ':maj'
			chord_type = CHORD_TYPE_MAJ
		elif tpl[2] == 1 or tpl[2] == 4 or tpl[2] == 5:
			chord_str += ':min'
			chord_type = CHORD_TYPE_MIN
		elif tpl[2] == 2:
			chord_str += ':dim'
			chord_type = CHORD_TYPE_DIM7 # dim
		else:
			print('convert_tpl_to_chord_name error: ' + tpl)
			raise Exception
	return (chord_str, root_pos, chord_type)

#! add other candidates : (tonic, mode, degree) -> [(tonic, mode, degree)]
def get_chord_interpretation_list(tpl):
	(chord_str, root_pos, chord_type) = convert_tpl(tpl)
	#print(chord_str, root_pos, chord_type)
	if chord_type == CHORD_TYPE_MAJ:
		return [
				(root_pos, MODE_MAJOR, 1),
				((root_pos + 9) % 12, MODE_MINOR, 3),
				((root_pos + 7) % 12, MODE_MAJOR, 4),
				((root_pos + 4) % 12, MODE_MINOR, 6),
				((root_pos + 5) % 12, MODE_MAJOR, 5),
				((root_pos + 2) % 12, MODE_MINOR, 7)
		]
	elif chord_type == CHORD_TYPE_MIN:
		return [
				(root_pos, MODE_MINOR, 1),
				((root_pos + 3) % 12, MODE_MAJOR, 6),
				((root_pos + 7) % 12, MODE_MINOR, 4),
				((root_pos + 10) % 12, MODE_MAJOR, 2),
				((root_pos + 5) % 12, MODE_MINOR, 5),
				((root_pos + 8) % 12, MODE_MAJOR, 3)
		]
	elif chord_type == CHORD_TYPE_DIM7: # dim
		return [
				((root_pos + 1) % 12, MODE_MAJOR, 7),
				((root_pos + 10) % 12, MODE_MINOR, 2)
		]
	else:
		print('get_chord_interpretation error: ' + tpl)
		raise Exception

#! add other candidates : [(tonic, mode, degree)] -> [[(tonic, mode, degree)]]
def get_chord_interpretation_list_x2(tpl_list):
	ret_list = []
	for tpl in tpl_list:
		ret_list.append(get_chord_interpretation_list(tpl))
		#print(tpl, ret_list[-1])
	return ret_list

#! create node_list_x2 and edge_distance_list_x3 from interpretation_list_x2 (and GTPS)
def make_interpretation_graph(arg_list):
	(gtps, interpretation_list_x2) = arg_list
	node_list_x2 = [] # [layer index: [node index in the layer: [tonic, mode, degree, probability of reaching this node when considering only previous information]]]
	edge_distance_list_x3 = [] # [from layer index: [from node index: [to node index: index list for GTPS]]]
	# node_list_x2
	for i, interpretation_list in enumerate(interpretation_list_x2):
		node_list = []
		for i2, interpretation in enumerate(interpretation_list):
			node_list.append([interpretation[0], interpretation[1], interpretation[2]])
		node_list_x2.append(node_list)
	# edge_distance_list_x3
	for i, node_list in enumerate(node_list_x2):
		if i < len(node_list_x2) - 1:
			edge_distance_list_x2 = []
			for i2, node in enumerate(node_list):
				edge_distance_list = []
				for i3, next_node in enumerate(node_list_x2[i + 1]):
					sum_tpl = gtps.get_distance2(node[0], node[1], node[2], next_node[0], next_node[1], next_node[2])
					#edge_distance_list.append([tps.get_scalar_distance(sum_tpl), sum_tpl])
					edge_distance_list.append(sum_tpl)
				edge_distance_list_x2.append(edge_distance_list)
			edge_distance_list_x3.append(edge_distance_list_x2)
	# add last layer
	edge_distance_list_x2 = []
	for i, node in enumerate(node_list_x2[-1]):
		edge_distance_list_x2.append([TPS.ZERO_TPL])
	edge_distance_list_x3.append(edge_distance_list_x2)
	node_list_x2.append([[0, 0, 0, 0]]) # end node
	return (node_list_x2, edge_distance_list_x3)

#! make a backward link from an interpretation graph
#    return: [layer index: [node index: [back node index]]]
def get_back_link_list_x3(gtps, graph):
	small_value = 0.00000001
	node_list_x2 = graph[0]
	edge_distance_list_x3 = graph[1]
	back_link_list_x3 = []
	node_cost_list_x2 = [] # shortest distance for each node
	# first layer
	back_link_list_x3.append([[] for i in node_list_x2[0]]) # 
	node_cost_list_x2.append([0 for i in node_list_x2[0]])
	# other layers
	for n in range(len(node_list_x2) - 1):
		src_node_list = node_list_x2[n]
		src_node_count = len(src_node_list) # for DTG
		dest_node_list = node_list_x2[n + 1]
		back_link_list_x2 = []
		node_cost_list = []
		# find best src nodes for each dest node
		for dest_node_index, dest_node in enumerate(dest_node_list):
			min_distance = math.inf
			cand_list = []
			for src_node_index in range(src_node_count):
				src_node = src_node_list[src_node_index]
				#distance = node_cost_list_x2[n][src_node_index] + edge_distance_list_x3[n][src_node_index][dest_node_index][0]
				#print(n, src_node_index, dest_node_index, len(node_cost_list_x2[n]), len(edge_distance_list_x3[n]))
				distance = node_cost_list_x2[n][src_node_index] + gtps.get_scalar_distance(edge_distance_list_x3[n][src_node_index][dest_node_index])
				if  (min_distance - small_value) <= distance <= (min_distance + small_value):
					cand_list.append(src_node_index)
				elif distance < min_distance + small_value:
					cand_list = [src_node_index]
					min_distance = distance
				if len(dtg) == 3 and n > 0: # for DTG
					if src_node[0] == dest_node[0] and src_node[1] == dest_node[1] and src_node[2] == dtg[1] and dest_node[2] == dtg[2]:
						for src0_node_index, src0_node in enumerate(node_list_x2[n - 1]):
							if src_node[0] == src0_node[0] and src_node[1] == src0_node[1] and src0_node[2] == dtg[0]:
								new_src_node_cost = node_cost_list_x2[n - 1][src0_node_index] + gtps.get_scalar_distance(edge_distance_list_x3[n - 1][src0_node_index][src_node_index]) * dtg_weight
								distance = new_src_node_cost + gtps.get_scalar_distance(edge_distance_list_x3[n][src_node_index][dest_node_index]) * dtg_weight
								#print("c ", src0_node, src_node, dest_node, min_distance, distance)
								node_list_x2[n].append(src_node)
								middle_index = len(node_list_x2[n]) - 1
								back_link_list_x3[n].append([src0_node_index])
								#node_cost_list_x2[n].append(...
								if distance < min_distance + small_value:
									cand_list = [middle_index]
									min_distance = distance
			back_link_list_x2.append(cand_list)
			node_cost_list.append(min_distance)
		back_link_list_x3.append(back_link_list_x2)
		node_cost_list_x2.append(node_cost_list)
	return back_link_list_x3

#! calculate node probabilities based on the shortest paths
#    return: [layer index: [node index: probability]]
def get_node_probability_list_x2(back_link_list_x3):
	node_probability_list_x2 = [[0 for node_index in back_link_list_x2] for back_link_list_x2 in back_link_list_x3]
	node_path_count_list_x2 = [[0 for node_index in back_link_list_x2] for back_link_list_x2 in back_link_list_x3] # [layer index: [node index: number of paths]]
	# calculate the number of paths from each node
	if True:
		# end node has 1
		if True:
			node_path_count_list_x2[-1][0] = 1
		# other layers
		for n0 in range(len(back_link_list_x3) - 1):
			n = len(back_link_list_x3) - 2 - n0
			back_link_list_x2 = back_link_list_x3[n + 1]
			for dest_node_index, back_link_list in enumerate(back_link_list_x2):
				path_count = node_path_count_list_x2[n + 1][dest_node_index]
				for src_node_index in back_link_list:
					node_path_count_list_x2[n][src_node_index] += path_count
	#print(node_path_count_list_x2)
	# calculate node probabilities for each node
	if True:
		# first layer
		if True:
			path_sum = sum([node_probability for node_probability in node_path_count_list_x2[0]])
			for node_index, path_count in enumerate(node_path_count_list_x2[0]):
				node_probability_list_x2[0][node_index] = path_count / path_sum
		# other layers
		for n in range(len(back_link_list_x3) - 1):
			back_link_list_x2 = back_link_list_x3[n + 1]
			for src_node_index in range(len(node_path_count_list_x2[n])):
				# sum of path_counts
				path_count_sum = sum([node_path_count_list_x2[n + 1][dest_node_index] for dest_node_index, src_node_index_list in enumerate(back_link_list_x2) if src_node_index in src_node_index_list])
				if path_count_sum > 0:
					#print(n, src_node_index, path_count_sum)
					# distribute the sum
					for dest_node_index, back_link_list in enumerate(back_link_list_x2):
						if src_node_index in back_link_list:
							node_probability_list_x2[n + 1][dest_node_index] += node_probability_list_x2[n][src_node_index] * (node_path_count_list_x2[n + 1][dest_node_index] / path_count_sum)
	#print(node_probability_list_x2)
	return node_probability_list_x2

#! create tensor_edge_list
def make_computation_graph(graph, gtps, tensor_param):
	node_list_x2 = graph[0]
	edge_distance_list_x3 = graph[1]
	tensor_edge_list = []
	# first layer
	tensor_node_list = torch.tensor([1.0 / len(node_list_x2[0]) for elm in node_list_x2[0]])
	# other layers
	for n in range(len(node_list_x2) - 2): # except the last layer
		src_node_list = node_list_x2[n]
		dest_node_list = node_list_x2[n + 1]
		# calculate unnomalized edge probabilities
		tensor_edge = torch.tensor(numpy.zeros((len(src_node_list), len(dest_node_list))))
		for src_node_index, src_node in enumerate(src_node_list):
			for dest_node_index, dest_node in enumerate(dest_node_list):
				tensor_edge[src_node_index][dest_node_index] = -gtps.get_scalar_distance_tensor(edge_distance_list_x3[n][src_node_index][dest_node_index], tensor_param)
		tensor_edge = torch.exp(tensor_edge)
		tensor_Z = torch.sum(tensor_node_list * tensor_edge.sum(1))
		tensor_edge = tensor_edge / tensor_Z # normalize
		tensor_edge_list.append(tensor_edge)
		#print(n, tensor_Z, tensor_node_list, tensor_edge.sum(1), tensor_edge)
		# arrival probability of dest node
		prev_tensor_node_list = tensor_node_list
		tensor_node_list = torch.tensor([0.0 for elm in node_list_x2[n + 1]])
		for dest_node_index, dest_node in enumerate(dest_node_list):
			tensor_node_list[dest_node_index] = sum([prev_tensor_node_list[src_node_index] * tensor_edge[src_node_index][dest_node_index] for src_node_index in range(len(src_node_list))])
			#print(tensor_node_list[dest_node_index], [prev_tensor_node_list[src_node_index] * tensor_edge[src_node_index][dest_node_index] for src_node_index in range(len(src_node_list))])
		#exit()
	return tensor_edge_list

#! calculate accuracy
def get_accuracy(arg_list):
	(gtps, answer_tpl_list, interpretation_list_x2, graph) = arg_list
	#graph = make_interpretation_graph(gtps, interpretation_list_x2)
	node_list_x2 = graph[0]
	back_link_list_x3 = get_back_link_list_x3(gtps, graph)
	#print(back_link_list_x3)
	node_probability_list_x2 = get_node_probability_list_x2(back_link_list_x3)
	#print(node_probability_list_x2)
	if len(dtg) == 3: # for DTG
		for n, node_list in enumerate(node_list_x2):
			for node_index, node in enumerate(node_list):
				for node_index2 in range(node_index + 1, len(node_list)):
					if node == node_list[node_index2]:
						node_probability_list_x2[n][node_index] += node_probability_list_x2[n][node_index2]
						node_probability_list_x2[n][node_index2] = 0
	answer_index_list = []
	for n, tpl in enumerate(answer_tpl_list):
		node_list = node_list_x2[n]
		answer_index = -1
		for i, node in enumerate(node_list):
			if tpl[0] == node[0] and tpl[1] == node[1] and tpl[2] == node[2]:
				answer_index = i
				break
		if answer_index < 0:
			print('error: answer_index not found')
		answer_index_list.append(answer_index)
	correct_count = 0.0
	for n, tpl in enumerate(answer_tpl_list):
		correct_count += node_probability_list_x2[n][answer_index_list[n]]
	return (correct_count, len(answer_tpl_list))

#! average over files
def get_average_accuracy(gtps, tpl_list_x2, int_list_x3, graph_list, parallel_count=8):
	acc_list = []
	#shortest_path_count_list = []
	file_count = len(tpl_list_x2)
	if parallel_count > 1:
		with Pool(parallel_count) as p:
			acc_list = p.map(func=get_accuracy, iterable=zip([gtps] * len(tpl_list_x2), tpl_list_x2, int_list_x3, graph_list))
	else:
		acc_list = []
		for i, tpl_list in enumerate(tpl_list_x2):
			acc_list.append(get_accuracy((gtps, tpl_list, int_list_x3[i], graph_list[i])))
	acc_list2 = [v[0] / v[1] for v in acc_list]
	mean = st.mean(acc_list2)
	return round(mean, 4)

#! calculate the gradient
def calc_gradient(graph, answer_tpl_list, edge_list):
	node_list_x2 = graph[0]
	edge_distance_list_x3 = graph[1]
	answer_index_list = []
	for n, tpl in enumerate(answer_tpl_list):
		node_list = node_list_x2[n]
		answer_index = -1
		for i, node in enumerate(node_list):
			if tpl[0] == node[0] and tpl[1] == node[1] and tpl[2] == node[2]:
				answer_index = i
				break
		if answer_index < 0:
			print('error: answer_index not found')
		answer_index_list.append(answer_index)
	nl_prob = torch.tensor(0.0)
	for n in range(len(answer_index_list) - 1):
		edge = edge_list[n]
		nl_prob -= torch.log(edge[answer_index_list[n]][answer_index_list[n + 1]])
	nl_prob.backward()

#! train/validation/test set
def divide_dataset(data_dir, max_length, test_set_interval):
	# load
	filename_list = []
	tpl_list_x2 = []
	for i, filename in enumerate(sorted(os.listdir(data_dir))):
		print('\rload_data data: ', (i + 1), end='')
		temp_tpl_list_x2 = load_data(os.path.join(data_dir, filename))
		for i1, tpl_list in enumerate(temp_tpl_list_x2):
			for i2 in range((len(tpl_list) // max_length) + 1):
				if len(tpl_list[i2 * max_length : (i2 + 1) * max_length]) > 1:
					filename_list.append(filename + '_' + str(i1) + '_' + str(i2 * max_length) + ':' + str((i2 + 1) * max_length))
					tpl_list_x2.append(tpl_list[i2 * max_length : (i2 + 1) * max_length])
				else:
					pass
			#if i < 20: print("\r", i, filename)
	print('\rload_data finished ', len(tpl_list_x2), '                          ')
	# prepare candidate interpretations
	int_list_x3 = []
	for i, tpl_list in enumerate(tpl_list_x2):
		print('\rget_chord_interpretation_list_x2 data: ', (i + 1), '/', len(tpl_list_x2), end='')
		int_list_x3.append(get_chord_interpretation_list_x2(tpl_list))
	print('\rget_chord_interpretation_list_x2 finished                       ')
	# train/validation/test
	if test_set_interval > 0:
		validation_filename_list = [v for i, v in enumerate(filename_list) if i % test_set_interval == 0]
		test_filename_list = [v for i, v in enumerate(filename_list) if i % test_set_interval == 1]
		training_filename_list = [v for i, v in enumerate(filename_list) if i % test_set_interval >= 2]
		validation_tpl_list_x2 = [v for i, v in enumerate(tpl_list_x2) if i % test_set_interval == 0]
		test_tpl_list_x2 = [v for i, v in enumerate(tpl_list_x2) if i % test_set_interval == 1]
		training_tpl_list_x2 = [v for i, v in enumerate(tpl_list_x2) if i % test_set_interval >= 2]
		validation_int_list_x3 = [v for i, v in enumerate(int_list_x3) if i % test_set_interval == 0]
		test_int_list_x3 = [v for i, v in enumerate(int_list_x3) if i % test_set_interval == 1]
		training_int_list_x3 = [v for i, v in enumerate(int_list_x3) if i % test_set_interval >= 2]
	else:
		validation_filename_list = filename_list
		test_filename_list = filename_list
		training_filename_list = filename_list
		validation_tpl_list_x2 = tpl_list_x2
		test_tpl_list_x2 = tpl_list_x2
		training_tpl_list_x2 = tpl_list_x2
		validation_int_list_x3 = int_list_x3
		test_int_list_x3 = int_list_x3
		training_int_list_x3 = int_list_x3
	if False:
		print('training phrases', len(training_tpl_list_x2), 'tpls', sum([len(v) for v in training_tpl_list_x2]))
		print('validation phrases', len(validation_int_list_x3), 'tpls', sum([len(v) for v in validation_int_list_x3]))
		print('test phrases', len(test_tpl_list_x2), 'tpls', sum([len(v) for v in test_tpl_list_x2]))
	return (training_filename_list, validation_filename_list, test_filename_list, training_tpl_list_x2, validation_tpl_list_x2, test_tpl_list_x2, training_int_list_x3, validation_int_list_x3, test_int_list_x3)

#! train parameters
def train(data_dir, max_epoch, batch_size, gtps, tensor_param, optimizer, max_length=100, test_set_interval=10, wait_epoch=5, parallel_count=8):
	(training_filename_list, validation_filename_list, test_filename_list, training_tpl_list_x2, validation_tpl_list_x2, test_tpl_list_x2, training_int_list_x3, validation_int_list_x3, test_int_list_x3) = divide_dataset(data_dir, max_length, test_set_interval)
	# prepare graphs
	training_graph_list = []
	validation_graph_list = []
	test_graph_list = []
	zipped = list(zip([gtps] * len(training_int_list_x3), training_int_list_x3))
	with Pool(parallel_count) as p:
		(training_graph_list) = p.map(func=make_interpretation_graph, iterable=zipped)
	zipped = list(zip([gtps] * len(validation_int_list_x3), validation_int_list_x3))
	with Pool(parallel_count) as p:
		(validation_graph_list) = p.map(func=make_interpretation_graph, iterable=zipped)
	zipped = list(zip([gtps] * len(test_int_list_x3), test_int_list_x3))
	with Pool(parallel_count) as p:
		(test_graph_list) = p.map(func=make_interpretation_graph, iterable=zipped)
	print('\rmake_interpretation_graph finished                              ')
	# accuracy before training
	print('param: ', tensor_param)
	gtps.update_params(tensor_param)
	acc0 = get_average_accuracy(gtps, training_tpl_list_x2, training_int_list_x3, training_graph_list, parallel_count=parallel_count)
	acc1 = get_average_accuracy(gtps, validation_tpl_list_x2, validation_int_list_x3, validation_graph_list, parallel_count=parallel_count)
	acc2 = get_average_accuracy(gtps, test_tpl_list_x2, test_int_list_x3, test_graph_list, parallel_count=parallel_count)
	print('\rget_average_accuracy training mean:', acc0, ', validation mean:', acc1, ', test mean:', acc2, '          ')
	max_acc_mean = -sys.float_info.max
	max_test_acc_mean = 0.0
	max_acc_epoch = 0
	max_acc_tensor = 0
	ret_str = ''
	# start training
	for epoch in range(max_epoch + 1):
		# end training
		if (epoch > max_acc_epoch + wait_epoch) or (epoch >= max_epoch):
			ret_str = "max_test_acc: " + str(max_test_acc_mean) + " epoch: " + str(max_acc_epoch + 1) + "\n" + str(max_acc_tensor) + "\n" + ret_str
			return ret_str
		optimizer.zero_grad()
		zipped = list(zip([gtps] * len(training_tpl_list_x2), training_tpl_list_x2, training_int_list_x3, training_graph_list, [tensor_param] * len(training_tpl_list_x2)))
		random.shuffle(zipped)
		for i0 in range(math.ceil(len(training_tpl_list_x2) / batch_size)):
			with Pool(parallel_count) as p:
				grad_list = p.map(func=train2, iterable=zipped[i0 * batch_size:i0 * batch_size + batch_size])
			tensor_param.grad = sum(grad_list)
			# update
			optimizer.step()
			print('\repoch:', (epoch + 1), '-', ((i0 + 1) * batch_size), 'param:', tensor_param, ', grad:', tensor_param.grad, ',                                                ')
			optimizer.zero_grad()
			gtps.update_params(tensor_param)
			# progress
			if True:
				if test_set_interval > 0:
					acc0 = get_average_accuracy(gtps, training_tpl_list_x2, training_int_list_x3, training_graph_list, parallel_count=parallel_count)
					acc1 = get_average_accuracy(gtps, validation_tpl_list_x2, validation_int_list_x3, validation_graph_list, parallel_count=parallel_count)
					acc2 = get_average_accuracy(gtps, test_tpl_list_x2, test_int_list_x3, test_graph_list, parallel_count=parallel_count)
				else:
					acc0 = get_average_accuracy(gtps, training_tpl_list_x2, training_int_list_x3, training_graph_list, parallel_count=parallel_count)
					acc1 = acc0
					acc2 = acc0
				print('\rget_average_accuracy training mean:', acc0, ', validation mean:', acc1, ', test mean:', acc2, '          ')
				#ret_str += '\nepoch:' + str(epoch + 1) + '-' + str((i0 + 1) * batch_size) + '  training mean:' + str(acc0) + ', validation mean:' + str(acc1) + ', test mean:' + str(acc2)
				criterion_value = acc1
				if max_acc_mean < criterion_value:
					max_acc_mean = criterion_value
					max_test_acc_mean = acc2
					max_acc_epoch = epoch
					max_acc_tensor = tensor_param.clone()

def train2(arg_list):
	(gtps, tpl_list, int_list_x2, graph, tensor_param) = arg_list
	tensor_edge_list = make_computation_graph(graph, gtps, tensor_param)
	calc_gradient(graph, tpl_list, tensor_edge_list)
	return tensor_param.grad

def get_shortest_paths(tpl_list, print_paths = True, de_tpl_list = None, params = None):
	if de_tpl_list is None:
		de_tpl_list = [[5, 0, 4], [0, 5, 0]] # m_sym2 * r_min_steps + d_sym2
		params = [-3.6312,  0.8677,  1.0189,  0.9326,  0.8353, -1.6241,  0.7079, -0.7745,
				1.6572,  0.4125,  0.5329, -0.7823, -0.6915,  0.5387,  1.4548, -0.7100,
				0.2862, -0.4906, -1.4794, -0.3454, -1.0922,  0.6207,  0.2275, -0.5879,
				0.6052,  0.0277,  0.2120,  0.2958,  0.8515,  0.7324, -0.4900,  0.4104,
				0.1301, -0.3750, -0.3427,  0.0591] # from epsilon setting
	b_fixed_list = [0, 0, 0]
	gtps = GTPS(de_tpl_list, _b_fixed_list=b_fixed_list)
	gtps.update_params(torch.tensor(params))
	int_list_x2 = get_chord_interpretation_list_x2(tpl_list)
	graph = make_interpretation_graph([gtps, int_list_x2])
	back_link_list_x3 = get_back_link_list_x3(gtps, graph)
	if print_paths:
		debug_show_shortest_paths(tpl_list, graph, back_link_list_x3)
		acc = get_accuracy([gtps, tpl_list, int_list_x2, graph])
		print('accuracy', str(acc[0] / acc[1]))
	return (graph[0], back_link_list_x3)

def debug_show_shortest_paths(answer_tpl_list, graph, back_link_list_x3):
	node_list_x2 = graph[0]
	edge_distance_list_x3 = graph[1]
	correct_count = 0.0
	node_probability_list_x2 = get_node_probability_list_x2(back_link_list_x3)
	for n, answer_tpl in enumerate(answer_tpl_list):
		answer_tpl2 = (answer_tpl[0], True if answer_tpl[1] == MODE_MAJOR else False, answer_tpl[2]) # mode → b_major
		print('chord', n, ':', debug_tpl_str(answer_tpl2))
		node_list = node_list_x2[n]
		back_link_list_x2 = back_link_list_x3[n]
		node_probability_list = node_probability_list_x2[n]
		for n2, node in enumerate(node_list):
			node2 = (node[0], True if node[1] == MODE_MAJOR else False, node[2]) # mode → b_major
			back_link_list = back_link_list_x2[n2]
			node_probability = node_probability_list[n2]
			print('  interpretation', n2, ':', debug_tpl_str(node2), 'probability:', node_probability, 'prev: ', end='')
			if debug_tpl_str(answer_tpl2) == debug_tpl_str(node2):
				correct_count += node_probability / len(answer_tpl_list)
			for back_link in back_link_list:
				print(back_link, end=' ')
			print('')
		print('    accuracy:', (correct_count * len(answer_tpl_list) / (n + 1)))
	back_link_list = back_link_list_x3[-1][0]
	print('from end node: ', end='')
	for back_link in back_link_list:
		print(back_link, end=' ')
	print('final accuracy: ', correct_count)
	return correct_count

def debug_tpl_str(tpl):
	tonic = tpl[0]
	b_major = tpl[1]
	degree = tpl[2]
	arr = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
	ret = ''
	if b_major:
		ret += arr[tonic]
	else:
		ret += arr[tonic].lower()
	ret += ':' + str(degree)
	return ret
