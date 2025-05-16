from experiments.experiment_runner import ExperimentRun

ecl_96 = ExperimentRun("ECL_96_96", config={
    # Data
    "root_path": "data_dir/",
    "data_path": "electricity.csv",
    "data": "custom",
    "features": "M",
    "target": "OT",
    "freq": "h",
    "num_workers": 5,
    "feature_dimension": 321,

    # Model
    "conv_dims": [[3, 1, 2], [6, 1, 3], [12, 3, 4], [48, 6, 12]],
    "h": 16,
    "N": 1,
    "encoding_size": 16,
    "attention_func": None,
    "dropout": 0.3,
    "time_embedding": "timeF",
    "time_embedding_size": 8,
    "representation_module": "cnn_3",
    "lstm_layer": 2,
    "tsrm_fc": False,
    "glu_layer": True,
    "revin": True,
    "seq_len": 96,
    "pred_len": 96,

})


ettm1_96 = ExperimentRun("ETTm1_96_96", config={
    # Data
    "root_path": "data_dir/",
    "data_path": "ETTm1.csv",
    "data": "ETTm1",
    "features": "M",
    "target": "OT",
    "freq": "t",
    "num_workers": 5,
    "feature_dimension": 7,

    # Model
    "conv_dims": [[3, 1, 2], [6, 1, 3], [12, 3, 4], [48, 6, 12]],
    "h": 32,
    "N": 1,
    "encoding_size": 64,
    "attention_func": None,
    "dropout": 0.5,
    "time_embedding": "",
    "representation_module": "linear",
    "lstm_layer": 0,
    "tsrm_fc": False,
    "glu_layer": True,
    "revin": True,
    "seq_len": 96,
    "pred_len": 96,

})


etth1_96 = ExperimentRun("ETTh1_96_96", config={
    # Data
    "root_path": "data_dir/",
    "data_path": "ETTh1.csv",
    "data": "ETTh1",
    "features": "M",
    "target": "OT",
    "freq": "h",
    "num_workers": 5,
    "feature_dimension": 7,

    # Model
    "conv_dims": [[3, 1, 1], [10, 2, 4], [15, 3, 5]],
    "h": 16,
    "N": 3,
    "encoding_size": 16,
    "attention_func": None,
    "dropout": 0.3,
    "time_embedding": "posEmb",
    "time_embedding_size": 16,
    "representation_module": "linear",
    "lstm_layer": 0,
    "tsrm_fc": False,
    "glu_layer": True,
    "revin": True,
    "seq_len": 96,
    "pred_len": 96,

})
