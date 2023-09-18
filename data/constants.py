# Stain normalization image
PATH_NORM_IMAGE = "./local_data/color_norm/sample.jpg"

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


# Path for results
PATH_RESULTS = "./local_data/results/"