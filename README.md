# An Unofficial Implementation of MAE-AD

The idea of using ViT-based masked autoencoder to detect anomalies in images seems very intriguing. Hence, I implemented the paper [*Image Anomaly Detection and Localization Using Masked Autoencoder*](https://link.springer.com/chapter/10.1007/978-981-99-1645-0_33) by Xiaohuo Yu *et al.* published at ICONIP 2022.

I have to say that the authors haven't provided their code, so this idea (i.e., using MAE to detect anomaly) might be less credible. Anyway, I trained the model on `zipper` category of MVTec-AD and got the best validation results of 96.03% AUC-ROC and 98.86% AUC-PR, compared with the official result of 99.1%.

- The MVTec-AD dataset is available on [Kaggle](https://www.kaggle.com/datasets/ipythonx/mvtec-ad).