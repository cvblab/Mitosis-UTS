# Stain normalization image
PATH_NORM_IMAGE = "./local_data/color_norm/sample.tif"

# Path with datasets
PATH_DATASETS = "./local_data/datasets/"

# TUPAC16
PATH_TUPAC = PATH_DATASETS + "TUPAC16/"
PATH_TUPAC_RAW_IMAGES = PATH_DATASETS + "TUPAC16/mitoses_train_image_data/"
PATH_TUPAC_RAW_GT = PATH_DATASETS + "TUPAC16/mitoses_ground_truth/"
PATH_TUPAC_PROCESSED = PATH_DATASETS + "TUPAC16/processed/"
TUPAC16_ID_TRAIN = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                    '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '32', '33', '34', '35', '36', '39',
                    '40', '41', '42', '43', '46', '47', '48', '49', '50', '53', '54', '55', '56', '57', '60', '61', '62',
                    '63', '64', '67', '68', '69', '70', '71']
TUPAC16_ID_VAL = ['30', '37', '44', '51', '58', '65', '72']
TUPAC16_ID_TEST = ['31', '38', '45', '52', '59', '66', '73']  # This test set is Li et al. (2019) validation set.

# MITOS14
PATH_MITOS14 = PATH_DATASETS + "MITOS14/"
PATH_MITOS14_RAW_IMAGES = PATH_DATASETS + "MITOS14/images/"
PATH_MITOS14_RAW_GT = PATH_DATASETS + "MITOS14/MIDOG.json"
PATH_MITOS14_PROCESSED = PATH_DATASETS + "MITOS14/processed/"
MITOS14_ID_VAL = ['A03', 'H03']
MITOS14_ID_TEST = ['A04', 'H04']

# MIDOG21
PATH_MIDOG21 = PATH_DATASETS + "MIDOG21/"
PATH_MIDOG21_RAW_IMAGES = PATH_DATASETS + "MIDOG21/images/"
PATH_MIDOG21_RAW_GT = PATH_DATASETS + "MIDOG21/MIDOG.json"
PATH_MIDOG21_PROCESSED = PATH_DATASETS + "MIDOG21/processed/"
MIDOG21_ID_VAL = ['41', '42', '43', '44', '45', '91', '92', '93', '94', '95', '141', '142', '143', '144', '145']
MIDOG21_ID_TEST = ['46', '47', '48', '49', '50', '96', '97', '98', '99', '100', '146', '147', '148', '149', '150']

# CCMCT
PATH_CCMCT = PATH_DATASETS + "CCMCT/"
PATH_CCMCT_RAW_IMAGES = PATH_DATASETS + "CCMCT/WSI/"
PATH_CCMCT_RAW_GT = PATH_DATASETS + "CCMCT/MITOS_WSI_CCMCT_MEL.json"
PATH_CCMCT_PROCESSED = PATH_DATASETS + "CCMCT/processed/"
CCMCT_ID_TEST = ['be10fa37ad6e88e1f406', 'f3741e764d39ccc4d114', 'c86cd41f96331adf3856', '552c51bfb88fd3e65ffe',
                 '8c9f9618fcaca747b7c3', 'c91a842257ed2add5134', 'dd4246ab756f6479c841', 'f26e9fcef24609b988be',
                 '96274538c93980aad8d6', 'add0a9bbc53d1d9bac4c', '1018715d369dd0df2fc0']
CCMCT_ID_VAL = ["3f2e034c75840cb901e6", "2efb541724b5c017c503", "70ed18cd5f806cf396f0"]
CCMCT_wsi2id_lookup_dict = {1: 'c91a842257ed2add5134',   2: 'add0a9bbc53d1d9bac4c',
                            3: '96274538c93980aad8d6',   4: 'c3eb4b8382b470dd63a9',
                            6: 'f26e9fcef24609b988be',   7: 'fff27b79894fe0157b08',
                            8: '2f17d43b3f9e7dacf24c',   9: '8c9f9618fcaca747b7c3',
                            11: 'be10fa37ad6e88e1f406', 12: 'ac1168b2c893d2acad38',
                            13: 'a0c8b612fe0655eab3ce', 14: '34eb28ce68c1106b2bac',
                            15: '3f2e034c75840cb901e6', 17: '8bebdd1f04140ed89426',
                            18: 'dd4246ab756f6479c841', 19: '39ecf7f94ed96824405d',
                            20: '1018715d369dd0df2fc0', 21: '20c0753af38303691b27',
                            22: '2efb541724b5c017c503', 23: '2f2591b840e83a4b4358',
                            24: '91a8e57ea1f9cb0aeb63', 25: '066c94c4c161224077a9',
                            26: '9374efe6ac06388cc877', 27: '552c51bfb88fd3e65ffe',
                            28: 'dd6dd0d54b81ebc59c77', 29: '285f74bb6be025a676b6',
                            30: 'c86cd41f96331adf3856', 31: 'f3741e764d39ccc4d114',
                            32: '2e611073cff18d503cea', 34: 'ce949341ba99845813ac',
                            35: '70ed18cd5f806cf396f0', 36: '0e56fd11a762be0983f0'}


# Path for results
PATH_RESULTS = "./local_data/results/"