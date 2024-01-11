We suggest putting the following dataset organization to ease management  to avoid modifying the source code.
The file structure looks like:

### TUPAC16

```
Mitosis-UTS/
└── local_data/
    └── datasets/
        └── TUPAC16/
            ├── mitoses-train_image_data/
            │   ├── mitoses_train_image_data_part_1/
            │   │   ├── 01/
            │   │   │   ├── 01.tiff
            │   │   │   ├── 02.tif
            │   │   │   └── ...
            │   │   ├── 02/
            │   │   ├── 03/
            │   │   └── ...
            │   ├── mitoses_train_image_data_part_2/
            │   │   └── ...
            │   └── mitoses_train_image_data_part_3/
            │       └── ...
            └── mitoses_ground_truth/
                ├── 01/
                │   ├── 01.csv
                │   ├── 02.csv
                │   └── ...
                ├── 02/
                └── ...
```

### MIDOG21

```
Mitosis-UTS/
└── local_data/
    └── datasets/
        └── MIDOG21/
            ├── images/
            │   ├── 001.tiff
            │   ├── 002.tiff
            │   └── ...
            └── MIDOG.json
```
