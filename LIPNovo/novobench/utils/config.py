"""
generate config  for different models 
"""
import yaml

class Config:
    def __init__(self, config_file, model_name):
        self.model_name = model_name
        # load config file
        with open(config_file) as f_in:
            self.config = yaml.safe_load(f_in)
        self.check_config_type()
        # print config
        print(self.config)

    def check_config_type(self):
        config_types = dict(
            random_seed=int,
            n_peaks=int,
            min_mz=float,
            max_mz=float,
            min_intensity=float,
            remove_precursor_tol=float,
            max_charge=int,
            precursor_mass_tol=float,
            isotope_error_range=lambda min_max: (int(min_max[0]), int(min_max[1])),
            warmup_iters=int,
            max_iters=int,
            num_sanity_val_steps=int,
            learning_rate=float,
            weight_decay=float,
            max_epochs=int,
            train_batch_size=int,
            save_weights_only=bool,
            model_save_folder_path=str,
            logger_save_path=str,
            predict_batch_size=int,
            val_check_interval=int,
            check_val_every_n_epoch=int,
            n_workers=int,
            save_top_k=int,
            devices=int,
        )
        for k, t in config_types.items():
            try:
                if self.config[k] is not None:
                    self.config[k] = t(self.config[k])
            except (TypeError, ValueError) as e:
                print("Incorrect type for configuration value %s: %s", k, e)
                raise TypeError(f"Incorrect type for configuration value {k}: {e}")
        # check model config type
        for key, value in self.config.items():
            setattr(self, key, value)
        self.check_model_config_type(self.model_name)

    def check_model_config_type(self, model_name):
        if model_name == 'casanovo':
            self.check_casanovo_config_type()
        elif model_name == 'adanovo':
            self.check_adanovo_config_type()
        elif model_name == 'helixnovo':
            self.check_helixnovo_config_type()
        elif model_name == 'instanovo':
            self.check_instanovo_config_type()
        elif model_name == 'impnovo':
            self.check_impnovo_config_type()



    def check_impnovo_config_type(self):
        config_types = dict(
            dim_model=int,
            n_head=int,
            dim_feedforward=int,
            n_layers=int,
            dropout=float,
            dim_intensity=int,
            max_length=int,
            min_peptide_len=int,
            train_label_smoothing=float,
            calculate_precision=bool,
            residues=dict,
            n_beams=int,
            top_match=int,
            accelerator=str,
            gen_num=int,
            gen_enc_layers=int,
            gen_dec_layers=int,
            gen_threshold=float
        )
        for k, t in config_types.items():
            try:
                if self.config['impnovo'][k] is not None:
                    self.config['impnovo'][k] = t(self.config['impnovo'][k])
            except (TypeError, ValueError) as e:
                print("Incorrect type for configuration value %s: %s", k, e)
                raise TypeError(f"Incorrect type for configuration value {k}: {e}")
        self.config['impnovo']["residues"] = {
            str(aa): float(mass) for aa, mass in self.config['impnovo']["residues"].items()}
        
        for key, value in self.config['impnovo'].items():
            setattr(self, key, value)


    def check_casanovo_config_type(self):
        config_types = dict(
            dim_model=int,
            n_head=int,
            dim_feedforward=int,
            n_layers=int,
            dropout=float,
            dim_intensity=int,
            max_length=int,
            min_peptide_len=int,
            train_label_smoothing=float,
            calculate_precision=bool,
            residues=dict,
            n_beams=int,
            top_match=int,
            accelerator=str,
        )
        for k, t in config_types.items():
            try:
                if self.config['casanovo'][k] is not None:
                    self.config['casanovo'][k] = t(self.config['casanovo'][k])
            except (TypeError, ValueError) as e:
                print("Incorrect type for configuration value %s: %s", k, e)
                raise TypeError(f"Incorrect type for configuration value {k}: {e}")
        self.config['casanovo']["residues"] = {
            str(aa): float(mass) for aa, mass in self.config['casanovo']["residues"].items()}
        
        for key, value in self.config['casanovo'].items():
            setattr(self, key, value)
    
    def check_adanovo_config_type(self):
        config_types = dict(
            dim_model=int,
            n_head=int,
            dim_feedforward=int,
            n_layers=int,
            dropout=float,
            dim_intensity=int,
            max_length=int,
            min_peptide_len=int,
            train_label_smoothing=float,
            calculate_precision=bool,
            residues=dict,
            accelerator=str,
            n_beams=int,
            top_match=int,
            s1=float,
            s2=float,
        )
        for k, t in config_types.items():
            try:
                if self.config['adanovo'][k] is not None:
                    self.config['adanovo'][k] = t(self.config['adanovo'][k])
            except (TypeError, ValueError) as e:
                print("Incorrect type for configuration value %s: %s", k, e)
                raise TypeError(f"Incorrect type for configuration value {k}: {e}")
        self.config['adanovo']["residues"] = {
            str(aa): float(mass) for aa, mass in self.config['adanovo']["residues"].items()}
        
        for key, value in self.config['adanovo'].items():
            setattr(self, key, value)

    def check_helixnovo_config_type(self):
        config_types = dict(
            dim_model=int,
            n_head=int,
            dim_feedforward=int,
            n_layers=int,
            dropout=float,
            dim_intensity=int,
            custom_encoder=str,
            max_length=int,
            residues=dict,
            decoding = str,
            n_beams=int
        )
        for k, t in config_types.items():
            try:
                if self.config['helixnovo'][k] is not None:
                    self.config['helixnovo'][k] = t(self.config['helixnovo'][k])
            except (TypeError, ValueError) as e:
                print("Incorrect type for configuration value %s: %s", k, e)
                raise TypeError(f"Incorrect type for configuration value {k}: {e}")
        self.config['helixnovo']["residues"] = {
            str(aa): float(mass) for aa, mass in self.config['helixnovo']["residues"].items()}
        for key, value in self.config['helixnovo'].items():
            setattr(self, key, value)

    def check_instanovo_config_type(self):
        config_types = dict(
            dim_model=int,
            n_head=int,
            dim_feedforward=int,
            n_layers=int,
            dropout=float,
            dim_intensity=int,
            max_length=int,
            custom_encoder=str,
            use_depthcharge=bool,
            enc_type=str,
            dec_type=str,
            dec_precursor_sos=bool,
            residues=dict,
            n_beams=int,
            grad_accumulation=int,
            gradient_clip_val=float,
            save_model=bool,
            knapsack_path=str,
        )
        for k, t in config_types.items():
            try:
                if self.config['instanovo'][k] is not None:
                    self.config['instanovo'][k] = t(self.config['instanovo'][k])
            except (TypeError, ValueError) as e:
                print("Incorrect type for configuration value %s: %s", k, e)
                raise TypeError(f"Incorrect type for configuration value {k}: {e}")
        self.config['instanovo']["residues"] = {
            str(aa): float(mass) for aa, mass in self.config['instanovo']["residues"].items()}
        
        for key, value in self.config['instanovo'].items():
            setattr(self, key, value)