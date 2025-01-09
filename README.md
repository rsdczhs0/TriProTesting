# TriProTesting

## 1. Installation

```bash
conda create -n tri_pro_test python=3.8 -y
conda activate tri_pro_test

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113
pip install transformers tqdm pillow
pip install git+https://github.com/openai/CLIP.git
```

## 2. Dataset Download and Processing

We use four datasets in this project. Please download them from the following links and place them in the `./data/` directory:

- **CelebA**: [Download here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
    - Place the image files in `./data/CelebA/img_align_celeba/` and the attributes file `list_attr_celeba.txt` in `./data/CelebA/`.

- **UTKFace**: [Download here](https://susanqq.github.io/UTKFace/)
    - Place the dataset in `./data/UTKFace/`.

- **FairFace**: [Download here](https://github.com/joojs/fairface)
    - Place the training and validation images in `./data/FairFace/train/` and `./data/FairFace/val/` respectively. Place the CSV files in `./data/FairFace/`.

- **IdenProf**: [Download here](https://github.com/OlafenwaMoses/IdenProf)
    - Place the training and testing images in `./data/IdenProf/train/` and `./data/IdenProf/test/` respectively.

### Dataset Processing

After placing the datasets in the appropriate directories, preprocess them by running the following script:

```bash
python data_process.py
```

This will preprocess the datasets and save the processed data in the `./processed_data/` 

## 3. Run the Logits Extraction Script

Before running the script, download the required models and place them in the `./model/` directory:

- **ALIGN**: [Download here](https://huggingface.co/kakaobrain/align-base)
    - Place the files in `./model/ALIGN/`.

- **BridgeTower**: [Download here](https://huggingface.co/BridgeTower/bridgetower-large-itm-mlm-itc)
    - Place the files in `./model/BridgeTower/`.

- **OWLv2**: [Download here](https://huggingface.co/google/owlv2-base-patch16-ensemble)
    - Place the files in `./model/OWLv2/`.

Use `get_logits.py` to extract logits for the datasets using the pre-trained models. This script supports multiple models and probes.
- The results are stored in `./results/` 



## 4. Generate Mixed Labels

Run `get_mixed_label.py` to infer and add gender labels to the datasets.
The updated results will include a new field `gender_label`, which indicates the predicted gender for each image.

## 5. Run Bias Testing Scripts

### 5.1 Single Bias Test

Run `Single_Bias_Test.py` to perform Single Bias Testing and apply AdaLogAdjustment to improve the results.

```bash
python Single_Bias_Test.py
```
The results are stored in `./LogitsAdjustmentResults/Single_LogitsAdjustment/`. 


### 5.2 Mixed Bias Test

Run `Mixed_Bias_Test.py` to perform Mixed Bias Testing and apply AdaLogAdjustment.

```bash
python Single_Bias_Test.py
```
The results are stored in `./LogitsAdjustmentResults/Mixed_LogitsAdjustment/`.
