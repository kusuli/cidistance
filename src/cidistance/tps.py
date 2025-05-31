import heapq
import math

#import util
#import setting
from .setting import NOTE_POS_DIC
from .setting import MODE_DISTANCE_DIC
from .setting import MODE_MAJOR
from .setting import MODE_MINOR
from .setting import CHORD_TYPE_MAJ
from .setting import CHORD_TYPE_MIN
from .setting import CHORD_TYPE_DIM7

class TPS:
	INF_TPL = (999999,)
	ZERO_TPL = (999998,)
	
	def __init__(self):
		self.coef_i = 1/3
		self.coef_j = 1/3
		#self.coef_k = 1.0
		self.coef_sum = 1.0
		self.region_mode = 1 # 1: normal, 2: relative_key_cost
		self.relative_key_cost = 0.5
		self.stored_distances = {} # to improve speed. make sure to clear when i,j,k are updated
	
	def get_distance(self, tonic_note_1, b_major_1, degree_1, tonic_note_2, b_major_2, degree_2):
		return self.get_scalar_distance(self.get_distance_bs(tonic_note_1, b_major_1, degree_1, [], tonic_note_2, b_major_2, degree_2, []))
	
	#get_distance_dic = {'A': 0, 'A#': 1, 'Bb': 1, 'B': 2, 'Cb': 2, 'C': 3, 'B#': 3, 'C#': 4, 'Db': 4, 'D': 5, 'D#': 6, 'Eb': 6, 'E': 7, 'Fb': 7, 'F': 8, 'E#': 8, 'F#': 9, 'Gb': 9, 'G': 10, 'G#': 11, 'Ab': 11}
	def get_distance_bs(self, tonic_note_1, b_major_1, degree_1, dummy_1, tonic_note_2, b_major_2, degree_2, dummy_2):
		#dic = {'A': 0, 'A#': 1, 'Bb': 1, 'B': 2, 'Cb': 2, 'C': 3, 'B#': 3, 'C#': 4, 'Db': 4, 'D': 5, 'D#': 6, 'Eb': 6, 'E': 7, 'Fb': 7, 'F': 8, 'E#': 8, 'F#': 9, 'Gb': 9, 'G': 10, 'G#': 11, 'Ab': 11}
		tonic_1 = NOTE_POS_DIC[tonic_note_1]
		tonic_2 = NOTE_POS_DIC[tonic_note_2]
		if b_major_1:
			mode_1 = MODE_MAJOR
		else:
			mode_1 = MODE_MINOR
		if b_major_2:
			mode_2 = MODE_MAJOR
		else:
			mode_2 = MODE_MINOR
		return self.get_distance2(tonic_1, mode_1, degree_1, dummy_1, tonic_2, mode_2, degree_2, dummy_2)
	
	def mode_to_b_major(self, mode):
		return mode == MODE_MAJOR
	
	# set b_close=True if you already know they are close keys
	def get_distance2(self, tonic_1, mode_1, degree_1, dummy_1, tonic_2, mode_2, degree_2, dummy_2, b_close = False):
		if (tonic_1, mode_1, degree_1, tonic_2, mode_2, degree_2) in self.stored_distances:
			return self.stored_distances[(tonic_1, mode_1, degree_1, tonic_2, mode_2, degree_2)]
		b_major_1 = self.mode_to_b_major(mode_1)
		b_major_2 = self.mode_to_b_major(mode_2)
		if b_close or self.is_close_key(tonic_1, b_major_1, tonic_2, b_major_2): # close keys
			print("close", tonic_1, b_major_1, tonic_2, b_major_2)
			sum1 = self.get_region_distance(tonic_1, b_major_1, tonic_2, b_major_2)
			sum2 = self.get_chord_distance(tonic_1, mode_1, degree_1, tonic_2, mode_2, degree_2)
			sum3 = self.get_basicspace_distance(tonic_1, mode_1, degree_1, dummy_1, tonic_2, mode_2, degree_2, dummy_2)
			sum_relative_key = self.get_relative_key_value(tonic_1, b_major_1, tonic_2, b_major_2)
			sum_other = self.get_other_distance(tonic_1, mode_1, degree_1, dummy_1, tonic_2, mode_2, degree_2, dummy_2)
			return (sum1, sum2, sum3, sum_relative_key, sum_other)
		else: # distant keys (shortest path must be searched)
			print("distant", tonic_1, b_major_1, tonic_2, b_major_2)
			sum_tpl = (0, 0, 0, 0, 0)
			reached = {}
			for close_key, _ in self.get_close_key_list(tonic_1, mode_1).items():
				reached[close_key] = self.get_distance2(tonic_1, mode_1, degree_1, dummy_1, close_key[0], close_key[1], 1, [], True)
			for i in range(100):
				k1 = min(reached, key=lambda x: max(0, self.get_scalar_distance(reached[x])))
				v1 = reached[k1]
				if (tonic_2, b_major_2) in reached and reached[(tonic_2, b_major_2)] == v1:
					sum_tpl = (sum_tpl[0] + v1[0], sum_tpl[1] + v1[1], sum_tpl[2] + v1[2], sum_tpl[3] + v1[3], sum_tpl[4] + v1[4])
					break
				close_key_list = self.get_close_key_list(k1[0], k1[1], True, reached)
				reached[k1] = TPS.INF_TPL
				for k2, v2 in close_key_list.items():
					if k2 == (tonic_2, mode_2 == MODE_MAJOR) and degree_2 != 1:
						v2 = self.get_distance2(k1[0], k1[1], 1, [], tonic_2, mode_2, degree_2, dummy_2, True)
					if (not k2 in reached) or (reached[k2] != TPS.INF_TPL and max(0, self.get_scalar_distance(reached[k2])) > max(0, self.get_scalar_distance(v1)) + max(0, self.get_scalar_distance(v2))):
						reached[k2] = (v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2], v1[3] + v2[3], v1[4] + v2[4])
		self.stored_distances[(tonic_1, mode_1, degree_1, tonic_2, mode_2, degree_2)] = sum_tpl
		print(reached)
		return sum_tpl
	
	def get_scalar_distance(self, dist_tpl):
		if dist_tpl == TPS.ZERO_TPL:
			return 0
		elif dist_tpl == TPS.INF_TPL:
			return 99999
		else:
			if self.region_mode == 2: # relative_key_cost mode
				return dist_tpl[0] * self.coef_i + dist_tpl[1] * self.coef_j + dist_tpl[2] * (self.coef_sum - self.coef_i - self.coef_j) + dist_tpl[3] * self.relative_key_cost
			else: # normal mode
				return dist_tpl[0] * self.coef_i + dist_tpl[1] * self.coef_j + dist_tpl[2] * (self.coef_sum - self.coef_i - self.coef_j)

	def is_close_key(self, tonic_1, b_major_1, tonic_2, b_major_2):
		if (tonic_1, b_major_1) == (tonic_2, b_major_2):
			return True
		else:
			close_key_list = self.get_close_key_list(tonic_1, b_major_1)
			for k in close_key_list:
				if (tonic_2, b_major_2) == (k[0], k[1]):
					return True
			return False

	# b_with_distance: whether to calculate distance
	# reached: for computational efficiency in get_distance2()
	def get_close_key_list(self, tonic, b_major, b_with_distance = False, reached = []):
		ret = {(tonic, not b_major): -1, ((tonic + 7) % 12, b_major): -1, ((tonic + 5) % 12, b_major): -1};
		if b_major:
			ret[((tonic - 3) % 12, not b_major)] = -1
			ret[((tonic + 7 - 3) % 12, not b_major)] = -1
			ret[((tonic + 5 - 3) % 12, not b_major)] = -1
		else:
			ret[((tonic + 3) % 12, not b_major)] = -1
			ret[((tonic + 7 + 3) % 12, not b_major)] = -1
			ret[((tonic + 5 + 3) % 12, not b_major)] = -1
		if b_with_distance:
			for k in ret:
				if k not in reached or reached[k] != TPS.INF_TPL:
					mode_1 = MODE_MAJOR if b_major else MODE_MINOR
					mode_2 = MODE_MAJOR if k[1] else MODE_MINOR
					ret[k] = self.get_distance2(tonic, mode_1, 1, [], k[0], mode_2, 1, [], True)
		return ret

	# tps_region
	get_region_distance_arr = [9, 2, 7, 0, 5, 10, 3, 8, 1, 6, 11, 4]
	def get_region_distance(self, tonic_1, b_major_1, tonic_2, b_major_2):
		modpos_1 = tonic_1
		if not b_major_1:
			modpos_1 = (modpos_1 + 3) % 12
		modpos_2 = tonic_2
		if not b_major_2:
			modpos_2 = (modpos_2 + 3) % 12
		mod = (self.get_region_distance_arr[modpos_1] - self.get_region_distance_arr[modpos_2]) % 12;
		return min(mod, 12 - mod)

	# tps_chord
	tonic_to_digree = {0: 0, 2: 1, 3: 2, 4: 2, 5: 3, 7: 4, 8: 5, 9: 5, 10: 6}
	get_chord_distance_arr = [0, 5, 3, 1, 6, 4, 2]
	def get_chord_distance(self, tonic_1, mode_1, degree_1, tonic_2, mode_2, degree_2):
		mod = (self.get_chord_distance_arr[((degree_2 + self.tonic_to_digree[(tonic_2 - tonic_1) % 12]) % 7) - (degree_1 % 7)]) % 7 
		return min(mod, 7 - mod)

	# tps_basicspace
	def get_basicspace_distance(self, tonic_1, mode_1, degree_1, dummy_1, tonic_2, mode_2, degree_2, dummy_2):
		bs1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		bs2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		# scale level
		scale1 = MODE_DISTANCE_DIC[mode_1]
		scale2 = MODE_DISTANCE_DIC[mode_2]
		bs1[(scale1[0] + tonic_1) % 12] += 1
		bs1[(scale1[1] + tonic_1) % 12] += 1
		bs1[(scale1[2] + tonic_1) % 12] += 1
		bs1[(scale1[3] + tonic_1) % 12] += 1
		bs1[(scale1[4] + tonic_1) % 12] += 1
		bs1[(scale1[5] + tonic_1) % 12] += 1
		bs1[(scale1[6] + tonic_1) % 12] += 1
		bs2[(scale2[0] + tonic_2) % 12] += 1
		bs2[(scale2[1] + tonic_2) % 12] += 1
		bs2[(scale2[2] + tonic_2) % 12] += 1
		bs2[(scale2[3] + tonic_2) % 12] += 1
		bs2[(scale2[4] + tonic_2) % 12] += 1
		bs2[(scale2[5] + tonic_2) % 12] += 1
		bs2[(scale2[6] + tonic_2) % 12] += 1
		# root level
		bs1[(scale1[(0 + degree_1 - 1) % 7] + tonic_1) % 12] += 3
		bs1[(scale1[(2 + degree_1 - 1) % 7] + tonic_1) % 12] += 1
		bs1[(scale1[(4 + degree_1 - 1) % 7] + tonic_1) % 12] += 2
		bs2[(scale2[(0 + degree_2 - 1) % 7] + tonic_2) % 12] += 3
		bs2[(scale2[(2 + degree_2 - 1) % 7] + tonic_2) % 12] += 1
		bs2[(scale2[(4 + degree_2 - 1) % 7] + tonic_2) % 12] += 2
		#print(bs1)
		#print(bs2)
		sum = 0
		for i in range(12):
			sum += max(0, bs2[i] - bs1[i])
		return sum
	
	def get_relative_key_value(self, tonic_1, b_major_1, tonic_2, b_major_2):
		modpos_1 = tonic_1
		if not b_major_1:
			modpos_1 = (modpos_1 + 3) % 12
		modpos_2 = tonic_2
		if not b_major_2:
			modpos_2 = (modpos_2 + 3) % 12
		mod = (self.get_region_distance_arr[modpos_1] - self.get_region_distance_arr[modpos_2]) % 12;
		if modpos_1 == modpos_2 and b_major_1 != b_major_2:
			return 1
		else:
			return 0
	
	def get_other_distance(self, tonic_1, mode_1, degree_1, dummy_1, tonic_2, mode_2, degree_2, dummy_2):
		return 0
