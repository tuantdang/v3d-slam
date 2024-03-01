import os
import configargparse

class ConfigParser(configargparse.ArgParser):
    """
    Parse argurment from command line or from file
    """
    def __init__(self):
        super().__init__(default_config_files=['config.yml']) #Load from yaml instead from command lines
        self.add_argument('--dataset_path', type=str)
        self.add_argument('--dataset_name', type=str)
        self.add_argument('--associations', type=str)
        self.add_argument('--width', type=int)
        self.add_argument('--height', type=int)
        self.add_argument('--fx', type=float)
        self.add_argument('--fy', type=float)
        self.add_argument('--cx', type=float)
        self.add_argument('--cy', type=float)
        self.add_argument('--depth_scale', type=float)
        self.add_argument('--yolo_model', type=str)
        self.add_argument('--conf_threshold', type=float)
        self.add_argument('--seg_threshold', type=float)
        self.add_argument('--ratio_threshold', type=float)
        self.add_argument('--kernel_size', type=float)
        self.add_argument('--eps', type=float)
        self.add_argument('--min_points', type=int)
        self.add_argument('--vote_threshold', type=float) 
        self.add_argument('--deform_threshold', type=float)
        
        self.add_argument('--use_visualization', type=bool, 
                          default= True)
        
    
if __name__ == '__main__':
    cfg = ConfigParser().parse_args()
    cfg_dict = vars(cfg) # Convert Namespace into dictionary
    for k in cfg_dict:
        print(f'{k} : {cfg_dict[k]}')