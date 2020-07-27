import numpy as np

# all length in mm
#w = 2.5 * 6e-3

C_input_pitch = 0.127 #250
C_pitch_array = np.array([1, 2, 3, 4]) * C_input_pitch

mode_field_diameter = 0.0056

fine_output_pitch = 0.2
fine_output_non_redundant_array = np.array([0, 4, 5, 7]) * fine_output_pitch

fine_output_angles_deg = np.array([-1.29, -0.43, 0, 1.72])
fine_output_angles_rad = np.array([ -0.022496204277883902, -0.0074998593797459025, 0, 0.0299910048568779])
sidestep = 8.33