import torch
import copy

from cidistance import main
from cidistance.gtps import GTPS

if __name__ == '__main__':

	def exp_i(title, index, gtps, tensor_param=None):
		data_dir = './sample_data'
		out_file = 'experiment1_result.txt'
		if tensor_param is None:
			tensor_param = torch.tensor([0.0 for _ in range(gtps.get_param_count_all() + gtps.get_pi_param_count_all())], requires_grad=True)
		op = torch.optim.SGD([tensor_param], lr=0.001, momentum=0.0)
		torch.set_printoptions(edgeitems=tensor_param.size()[0])
		ret_str = main.train(data_dir, 200, 100, gtps, tensor_param, op, max_length=50, wait_epoch=2, parallel_count=2)
		with open(out_file, mode='a') as f:
			f.write('\n\n' + title + ': '+ str(index) + ' de_tpl_list: ' + str(gtps.de_tpl_list) + ' params: ' + str(gtps.get_param_count_all()) + ' redundancy: ' + str(len(gtps.de_tpl_list)) + ' ' + ret_str + '\n')

	b_fixed_list = [0, 0, 0]
	# prepare all combinations
	g_tpl_list_x2 = [] # combination patterns of elemental functions
	for i1 in range(2): # s_mode (term index. ignore if 0)
		for i2 in range(i1 + 2): # d_mode
			for i3 in range(max(i1, i2) + 2): # s_degree
				for i4 in range(max(i1, i2, i3) + 2): # d_degree
					for i5 in range(max(i1, i2, i3, i4) + 2): # tonic(region)
						g_tpl_list_x2.append([i1, i2, i3, i4, i5])
	# g_tpl_list_x2 -> de_tpl_list_x2
	de_tpl_list_x2 = [] # combination patterns compatible with GTPS
	for g_tpl_list in g_tpl_list_x2:
		de_tpl_list = []
		for i in range(1, 6): # at most five terms
			if i <= max(g_tpl_list): # if the term is used
				de_tpl = [0, 0, 0]
				# mode
				if g_tpl_list[0] == i and g_tpl_list[1] == i:
					de_tpl[0] = 3 # expand later
				elif g_tpl_list[0] == i:
					de_tpl[0] = 1
				elif g_tpl_list[1] == i:
					de_tpl[0] = 2
				# degree
				if g_tpl_list[2] == i and g_tpl_list[3] == i:
					de_tpl[1] = 3 # expand later
				elif g_tpl_list[2] == i:
					de_tpl[1] = 1
				elif g_tpl_list[3] == i:
					de_tpl[1] = 2
				# tonic(region)
				if g_tpl_list[4] == i:
					de_tpl[2] = 1 # expand later
				de_tpl_list.append(de_tpl)
		de_tpl_list_x2.append(de_tpl_list)
	de_tpl_list_x2_ = de_tpl_list_x2
	de_tpl_list_x2 = []
	for de_tpl_list in de_tpl_list_x2_: # expand about mode
		de_tpl_list_x2.append(copy.deepcopy(de_tpl_list))
		for i in range(len(de_tpl_list)):
			if de_tpl_list[i][0] == 3:
				de_tpl_list[i][0] = 4
				de_tpl_list_x2.append(copy.deepcopy(de_tpl_list))
				de_tpl_list[i][0] = 5
				de_tpl_list_x2.append(de_tpl_list)
				break
	de_tpl_list_x2_ = de_tpl_list_x2
	de_tpl_list_x2 = []
	for de_tpl_list in de_tpl_list_x2_: # expand about degree
		de_tpl_list_x2.append(copy.deepcopy(de_tpl_list))
		for i in range(len(de_tpl_list)):
			if de_tpl_list[i][1] == 3:
				de_tpl_list[i][1] = 4
				de_tpl_list_x2.append(copy.deepcopy(de_tpl_list))
				de_tpl_list[i][1] = 5
				de_tpl_list_x2.append(copy.deepcopy(de_tpl_list))
				de_tpl_list[i][1] = 6
				de_tpl_list_x2.append(copy.deepcopy(de_tpl_list))
				de_tpl_list[i][1] = 7
				de_tpl_list_x2.append(de_tpl_list)
				break
	de_tpl_list_x2_ = de_tpl_list_x2
	de_tpl_list_x2 = []
	for de_tpl_list in de_tpl_list_x2_: # expand about tonic(region)
		de_tpl_list_x2.append(copy.deepcopy(de_tpl_list))
		for i in range(len(de_tpl_list)):
			if de_tpl_list[i][2] == 1:
				de_tpl_list[i][2] = 2
				de_tpl_list_x2.append(copy.deepcopy(de_tpl_list))
				de_tpl_list[i][2] = 3
				de_tpl_list_x2.append(copy.deepcopy(de_tpl_list))
				de_tpl_list[i][2] = 4
				de_tpl_list_x2.append(de_tpl_list)
				break
	for i, de_tpl_list in enumerate(de_tpl_list_x2):
		if i > 0:
			gtps = GTPS(de_tpl_list, _b_fixed_list=b_fixed_list)
			exp_i('#!Exp ', i, gtps)
