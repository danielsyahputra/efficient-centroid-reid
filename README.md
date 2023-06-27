# Efficient Centroid Re-ID
Experiment with Centroid Re-ID by Changing Backbone Using EfficientNet-V2 and Adding Re-Identification Based Data Augmentation

# Requirements


# Table Of Contents
-  [In a Nutshell](#in-a-nutshell)
-  [In Details](#in-details)
-  [Future Work](#future-work)
-  [Contributing](#contributing)
-  [Acknowledgments](#acknowledgments)

# In a Nutshell   


# In Details
```
├──  config
│    └── defaults.py  - here's the default config file.
│
│
├──  configs  
│    └── train_mnist_softmax.yml  - here's the specific config file for specific model or dataset.
│ 
│
├──  data  
│    └── datasets  - here's the datasets folder that is responsible for all data handling.
│    └── transforms  - here's the data preprocess folder that is responsible for all data augmentation.
│    └── build.py  		   - here's the file to make dataloader.
│    └── collate_batch.py   - here's the file that is responsible for merges a list of samples to form a mini-batch.
│
│
├──  engine
│   ├── trainer.py     - this file contains the train loops.
│   └── inference.py   - this file contains the inference process.
│
│
├── layers              - this folder contains any customed layers of your project.
│   └── conv_layer.py
│
│
├── modeling            - this folder contains any model of your project.
│   └── example_model.py
│
│
├── solver             - this folder contains optimizer of your project.
│   └── build.py
│   └── lr_scheduler.py
│   
│ 
├──  tools                - here's the train/test model of your project.
│    └── train_net.py  - here's an example of train model that is responsible for the whole pipeline.
│ 
│ 
└── utils
│    ├── logger.py
│    └── any_other_utils_you_need
│ 
│ 
└── tests					- this foler contains unit test of your project.
     ├── test_data_sampler.py
```

# Future Work

# Contributing
Any kind of enhancement or contribution is welcomed.

# Acknowledgments

```
@article{Wieczorek2021OnTU,
  title={On the Unreasonable Effectiveness of Centroids in Image Retrieval},
  author={Mikolaj Wieczorek and Barbara Rychalska and Jacek Dabrowski},
  journal={ArXiv},
  year={2021},
  volume={abs/2104.13643}
}
```
```
@article{ReIDAugmentations,
      title={A Person Re-identification Data Augmentation Method with Adversarial Defense Effect}, 
      author={Yunpeng Gong and Zhiyong Zeng and Liwen Chen and Yifan Luo and Bin Weng and Feng Ye},
      year={2021},
      journal={ArXiv}
      eprint={2101.08783},
}
```


