import numpy as np

# all length in mm
w = 2.5 * 6e-3

C_input_pitch = 0.127
C_pitch_array = np.array([1, 2, 3, 4]) * C_input_pitch

mode_field_diameter = 0.0056

D_output_pitch = 0.2
D_output_non_redundant_array = np.array([0, 4, 1, 2])*D_output_pitch

D_output_angles_deg = np.array([1.72, 0, -0.43, -1.29])
D_output_angles_rad = np.array([0.0299910048568779, 0, -0.0074998593797459025, -0.022496204277883902])