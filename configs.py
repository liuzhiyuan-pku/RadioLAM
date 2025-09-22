import torch

TRAIN_3D_UNet = {
    'dataset_type': 'SpectrumNet',
    'model_path': '',
    'batch_size': 32,
    'n_epochs': 999999,
    'scheduler_cfg': {
        'lr': 5e-7,
        'milestones': 1200,
        'gamma': 0.5,
    },
    'pretrain_weight_path': "",
    'target_height': 'X',
    'sparse_num': 50,
    "point_cloud_height" : 'mix',
    'channel_num': 'f03',
    'Geographic_type':  '',
}


test_noise_UNet = {
    'dataset_type': 'SpectrumNet_test',
    'weight_path':{
        'urban': "",
        '05.Suburban': "",
        '06.DenseUrban': "",
        '07.Rural': "",
        '08.OrdinaryUrban': "",
    },
    'sparse_num': 50,
    'sample_num': 64,
    "point_cloud_height" : 'mix',
    'channel_num': 'f03',
}

FT_CONFIG = {
    'dataset_type': 'SpectrumNet',
    'model_path': '',
    'batch_size': 4,
    'n_epochs': 999999,
    'scheduler_cfg': {
        'lr': 4e-7,
        'milestones': 1200,
        'gamma': 0.5,
    },
    'router_path':"",
    'weight_path':[
        ('urban', ""),
        ('05.Suburban', ""),
        ('06.DenseUrban', ""),
        ('07.Rural', ""),
        ('08.OrdinaryUrban', ""),
    ],
    'target_height': 'X',
    'sparse_num': 50,
    "point_cloud_height" : 'mix',
    'channel_num': 'f03',
    'Geographic_type':  'urban',
}


ROUTER_CONFIG = {
    'dataset_type': 'SpectrumNet',
    'model_path': '',
    'gen_num': 5,
    'batch_size': 128,
    'n_epochs': 999999,
    'scheduler_cfg': {
        'lr': 1e-6,
        'milestones': 800,
        'gamma': 0.5,
    },
    'pretrain_weight_path': "",
    'channel_num': 'f03',
    'Geographic_type':  'urban',
}

MoG_TEST_CONFIG = {
    'dataset_type': 'SpectrumNet_test',
    'router_path':"",
    'weight_path':[
        ('urban', ""),
        ('05.Suburban', ""),
        ('06.DenseUrban', ""),
        ('07.Rural', ""),
        ('08.OrdinaryUrban', ""),
    ],
    'sparse_num': 50,
    'sample_num': 64,
    "point_cloud_height" : 'mix',
    'channel_num': 'f03',
}



# 535,651,969
UNet_535M_cfg = { 'image_size': 128,
        'in_channels': 4,
        'out_channels': 1,
        'model_channels': 192,
        'attention_resolutions':[8, 4, 2],
        'num_res_blocks': 2,
        'channel_mult':[1, 2, 3, 5],
        'num_head_channels': 32,
        'use_spatial_transformer': True,
        'transformer_depth': 2,
        'context_dim': 64}

# 174,357,377
UNet_174M_cfg = { 'image_size': 128,
        'in_channels': 3,
        'out_channels': 1,
        'model_channels': 128,
        'attention_resolutions':[8, 4, 2],
        'num_res_blocks': 2,
        'channel_mult':[1, 2, 3, 5],
        'num_head_channels': 32,
        'use_spatial_transformer': True,
        'transformer_depth': 1,
        'context_dim': 64}

# 230,139,425
UDiT_230M_cfg = {'down_factor':2, 
             'hidden_size':192, 
             'num_heads':4, 
             'depth':[2, 5, 8, 5, 2], 
             'ffn_type':'rep', 
             'rep':1, 
             'mlp_ratio':2,
             'attn_type':'v2', 
             'posemb_type':'rope2d', 
             'downsampler':'dwconv5', 
             'down_shortcut':1
             }

# 914,111,553
UDiT_914M_cfg = {'down_factor':2, 
             'hidden_size':384, 
             'num_heads':16, 
             'depth':[2, 5, 8, 5, 2], 
             'ffn_type':'rep', 
             'rep':1, 
             'mlp_ratio':2,
             'attn_type':'v2', 
             'posemb_type':'rope2d', 
             'downsampler':'dwconv5', 
             'down_shortcut':1
             }

# 121,876,225
UNet_cfg = {
    'img_shape': [5, 128,128],
    'channels': [128, 128, 256, 256, 512, 512, 1024],
    'pe_dim': 256,
    'with_attn': [False, False, False, False, True, True, True],
    'norm_type': 'gn'
}

# 121,876,225
EXP_UNet_cfg = {
    'img_shape': [5, 128,128],
    'channels': [128, 128, 256, 256, 512, 512, 1024],
    'pe_dim': 256,
    'with_attn': [False, False, False, False, True, True, True],
    'norm_type': 'gn'
}

CONFIG = TRAIN_3D_UNet

TEST_CONFIG = test_noise_UNet