import torch


class SolarConfig:
    """Accurate configuration for Solar dataset based on existing checkpoints"""

    def __init__(self, pred_len=96):
        # Basic config
        self.is_training = 0  # Set to 0 for testing/inference
        self.model_id = f'solar_96_{pred_len}'
        self.model = 'iTransformer'

        # Data loader
        self.data = 'Solar'
        self.root_path = './dataset/Solar/'
        self.data_path = 'solar_AL.txt'
        self.features = 'M'  # multivariate predict multivariate
        self.target = 'OT'
        self.freq = 't'  # minutely frequency for solar data
        self.checkpoints = './checkpoints/'

        # Forecasting task
        self.seq_len = 96
        self.label_len = 48
        self.pred_len = pred_len  # 96, 192, or 336

        # Model define - EXACT VALUES from checkpoint names
        self.enc_in = 137  # This needs to match the Solar dataset
        self.dec_in = 137
        self.c_out = 137
        self.d_model = 512  # From checkpoint: pl512
        self.n_heads = 8  # From checkpoint: dm8
        self.e_layers = 2  # From checkpoint: nh2
        self.d_layers = 1  # From checkpoint: el1
        self.d_ff = 512  # From checkpoint: dl512
        self.moving_avg = 25
        self.factor = 1  # From checkpoint: df1
        self.distil = True  # From checkpoint: ebTrue
        self.dropout = 0.1
        self.embed = 'timeF'  # From checkpoint: fctimeF
        self.activation = 'gelu'
        self.output_attention = False
        self.do_predict = False

        # Optimization
        self.num_workers = 0  # Set to 0 to avoid multiprocessing issues
        self.itr = 1
        self.train_epochs = 10
        self.batch_size = 32
        self.patience = 3
        self.learning_rate = 0.0001
        self.des = 'Exp'  # From checkpoint: dtExp
        self.loss = 'MSE'
        self.lradj = 'type1'
        self.use_amp = False

        # GPU
        self.use_gpu = True if torch.cuda.is_available() else False
        self.gpu = 0
        self.use_multi_gpu = False
        self.devices = '0,1,2,3'

        # iTransformer specific
        self.exp_name = 'RolloutHeads'
        self.channel_independence = False
        self.inverse = False
        self.class_strategy = 'projection'  # From checkpoint: projection
        self.target_root_path = './dataset/Solar/'
        self.target_data_path = 'solar_AL.txt'
        self.efficient_training = False
        self.use_norm = True
        self.partial_start_index = 0
        self.output_attention_map = True

        # Update GPU settings
        self._setup_gpu()

    def _setup_gpu(self):
        """Setup GPU configuration"""
        if self.use_gpu and torch.cuda.is_available():
            if self.use_multi_gpu:
                self.devices = self.devices.replace(' ', '')
                device_ids = self.devices.split(',')
                self.device_ids = [int(id_) for id_ in device_ids]
                self.gpu = self.device_ids[0]
        else:
            self.use_gpu = False

    def get_setting_string(self):
        """Generate the exact setting string used in checkpoints"""
        return f'{self.model_id}_{self.model}_{self.data}_{self.features}_ft{self.seq_len}_sl{self.label_len}_ll{self.pred_len}_pl{self.d_model}_dm{self.n_heads}_nh{self.e_layers}_el{self.d_layers}_dl{self.d_ff}_df{self.factor}_fc{self.embed}_eb{self.distil}_dt{self.des}_{self.class_strategy}_0'


# Additional configs for other datasets (if needed later)
class ElectricityConfig:
    """Configuration for Electricity dataset - common parameters"""

    def __init__(self, pred_len=96):
        # Basic config
        self.is_training = 0
        self.model_id = f'electricity_96_{pred_len}'
        self.model = 'iTransformer'

        # Data loader
        self.data = 'custom'
        self.root_path = './dataset/electricity/'
        self.data_path = 'electricity.csv'
        self.features = 'M'
        self.target = 'OT'
        self.freq = 'h'  # hourly frequency
        self.checkpoints = './checkpoints/'

        # Forecasting task
        self.seq_len = 96
        self.label_len = 48
        self.pred_len = pred_len

        # Model define - typical values for electricity
        self.enc_in = 321  # Electricity dataset variates
        self.dec_in = 321
        self.c_out = 321
        self.d_model = 512
        self.n_heads = 8
        self.e_layers = 2
        self.d_layers = 1
        self.d_ff = 2048
        self.moving_avg = 25
        self.factor = 1
        self.distil = True
        self.dropout = 0.1
        self.embed = 'timeF'
        self.activation = 'gelu'
        self.output_attention = False
        self.do_predict = False

        # Optimization
        self.num_workers = 0
        self.itr = 1
        self.train_epochs = 10
        self.batch_size = 32
        self.patience = 3
        self.learning_rate = 0.0001
        self.des = 'Exp'
        self.loss = 'MSE'
        self.lradj = 'type1'
        self.use_amp = False

        # GPU
        self.use_gpu = True if torch.cuda.is_available() else False
        self.gpu = 0
        self.use_multi_gpu = False
        self.devices = '0,1,2,3'

        # iTransformer specific
        self.exp_name = 'MTSF'
        self.channel_independence = False
        self.inverse = False
        self.class_strategy = 'projection'
        self.target_root_path = './dataset/electricity/'
        self.target_data_path = 'electricity.csv'
        self.efficient_training = False
        self.use_norm = True
        self.partial_start_index = 0

        self._setup_gpu()

    def _setup_gpu(self):
        """Setup GPU configuration"""
        if self.use_gpu and torch.cuda.is_available():
            if self.use_multi_gpu:
                self.devices = self.devices.replace(' ', '')
                device_ids = self.devices.split(',')
                self.device_ids = [int(id_) for id_ in device_ids]
                self.gpu = self.device_ids[0]
        else:
            self.use_gpu = False

    def get_setting_string(self):
        """Generate the setting string"""
        return f'{self.model_id}_{self.model}_{self.data}_{self.features}_ft{self.seq_len}_sl{self.label_len}_ll{self.pred_len}_pl{self.d_model}_dm{self.n_heads}_nh{self.e_layers}_el{self.d_layers}_dl{self.d_ff}_df{self.factor}_fc{self.embed}_eb{self.distil}_dt{self.des}_{self.class_strategy}_0'


# Configuration factory
def get_config(dataset_name, pred_len=96):
    """Factory function to get configuration for a specific dataset"""
    configs = {
        'solar': SolarConfig,
        'electricity': ElectricityConfig,
    }

    if dataset_name.lower() not in configs:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(configs.keys())}")

    return configs[dataset_name.lower()](pred_len=pred_len)
