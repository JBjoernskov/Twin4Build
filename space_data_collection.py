
class SpaceDataCollection:
    def __init__(self):
        self.space_name = None
        self.has_sufficient_data = None
        self.lower_limit = {"sw_radiation": 0, 
                            "lw_radiation": 0, 
                            "OAT": -100, 
                            "temperature": 0, 
                            "CO2": 200, 
                            "r_valve": 0, 
                            "v_valve": 0,
                            "shades": 0}
        self.upper_limit = {"sw_radiation": 5000, 
                            "lw_radiation": 5000, 
                            "OAT": 50, 
                            "temperature": 50, 
                            "CO2": 4000, 
                            "r_valve": 1, 
                            "v_valve": 1,
                            "shades": 1}
        self.time = None
        self.raw_data_dict = {}

        self.n_sequence = 144 #72
        self.nan_interpolation_gap_limit = 12
        self.n_sequence_repeat = 32
        self.n_data_sequence_min = 1
        

        self.clean_data_dict = {}
        self.n_data_points = None
        self.n_data_sequence = None
        self.has_sequence_vec = None

        self.adjacent_spaces_no_data_list = []

        self.data_matrix = None
        self.adjacent_space_data_frac = None


        self.data_min_vec = None
        self.data_max_vec = None

