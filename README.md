# CIDistance
Calculates the distance between two chord interpretations.

This is a program implementation for https://doi.org/10.1080/09298215.2025.2523245 <br>
Yamamoto, H., & Tojo, S. (2024). Effective features to define harmonic distance models. Journal of New Music Research, 53(5), 399–419. https://doi.org/10.1080/09298215.2025.2523245

<hr size="1" />
experiment1.py
	Example of parameter training (Section 5).

experiment2.py
	Example of prediction (Figure 10).

main.py
	Prediction and Parameter training.

gtps.py
	An implementations of proposed distance model (Section 3).
	Set _de_tpl_list to specify the combination of learnable elemental functions.
	_de_tpl_list is a list of triplets (mode index, degree index, tonic index).
	
	Mode index:
		1: m_src
		2: m_dest
		3: m_src * m_dest = m_asym # treated like an elemental function
		4: m_sym1
		5: m_sym2
	Degree index:
		1: d_src
		2: d_est
		3: d_src * d_dest = d_asym # treated like an elemental function
		4: d_sym1
		5: d_sym2
		6: d_oneway_steps
		7: d_min_steps
	Tonic index:
		1: t_oneway_steps
		2: t_min_steps
		3: r_oneway_steps
		4: r_min_steps
	
	For "multiplication", set several indices within a triplet, and for "addition", in each triplet.
	For example, [(1, 0, 2)] represents m_src * t_min_steps while [(1, 0, 0), (0, 0, 2)] represents m_src + t_min_steps.
	And both operations can be used at the same time, e.g., [(5, 0, 4), (0, 5, 0)] represents m_sym2 * r_min_steps + d_sym2.


sample_data/
	Learning / experimental data.

results/
	Learning results.
