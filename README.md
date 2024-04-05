# GPT-2
Contlo On-Campus Placement Assignment: Implementation of GPT-2 Architecture

## Description
GPT-2, short for "Generative Pre-trained Transformer 2," is a state-of-the-art natural language processing (NLP) model developed by OpenAI. This repository provide a simplified code for the GPT-2 architecture, modifications and with large scale training stategy. The [Contlo Placement Assigment](https://contlo.notion.site/Assignment-32610c8f37dd4435b1f97ecaff93bdaf) provided taks to develope the architecture along with modification. The code focused on understanding the Transformer architecture, modifying its structures for improved performance, and implementing efficient training loops suitable for distributed training across multiple GPUs.


### 1 | GPT-2 Model & Checkpoints
Implement the `GPT2-small` model (with 125 million parameters) using Python and PyTorch. Touch upon the key aspects of the model like multi-head self-attention mechanism, feed-forward networks and positional encoding.

Key points:
- Follow the original GPT-2 design of using both token and positional embeddings.
- Implement the transformer layers with multi-head self-attention and point-wise feed-forward network.
- Required to abstain from using pre-built transformer libraries.

To validate your implementation, load the original GPT-2 125M model checkpoints and run a sample prediction.

### 2 | Transformer Architectural Changes (40 Points)
Add alterations to the original GPT-2 model architecture to experiment and assess the potential of improvements. Here's what you need to do:

- **Rotary Positional Embedding:** Replace the original positional embeddings in the GPT-2 model with Rotary embeddings. You may refer to [Su et. al. RoFormer](https://arxiv.org/pdf/2104.09864.pdf).
- **Group Query Attention:** Equip your model with the Group Query Attention mechanism following the insights from the [Ainslie et. al. GQA: Training Generalized Multi-Query Transformer](https://arxiv.org/pdf/2305.13245v2.pdf). Analyze how this mechanism can modify the model's operation compared to the standard attention mechanism.
- **Sliding Window Attention:** Imbibe the Sliding Window Attention mechanism in your model and observe its effects on model performance. Refer to the work by [Beltagy et. al. Longformer](https://arxiv.org/pdf/2004.05150v2.pdf) for better comprehension of its implementation and advantages.


## 3 | Training Loop Implementation (40 Points)
Finally, create a training loop considering these following requirements:

1. **Single GPU Training Loop:** Your base implementation should be equipped to train your model on a single GPU setup.
2. **Distributed Data Parallel (DDP):** Extend the single GPU training loop to support training across multiple GPUs using DDP. Revisit the [PyTorch's DDP tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) for guidance.
3. **Fully Sharded Data Parallel (FSDP):** Implement FSDP as a part of the training loop to shard the model parameters, gradients, and optimizer state. You can follow [Gupta et al., 2020, Training GPT-3 Like Models on a Single Machine](https://arxiv.org/pdf/2101.06840.pdf) for a comprehensive understanding of it.

## Environment Setup

### Conda Setup
To ensure a consistent environment, it is recommended to use conda for managing dependencies. Follow these steps to set up the environment using the provided `environment.yml` file.

1. Clone the repository:
   ```bash
   git clone https://github.com/omm-prakash/MemesNet.git
   cd MemesNet
   ```

2. Create conda environment:
   ```bash
   conda env create -f environment.yml
   ```

3. Activate the environment:
   ```bash
   conda activate gpt2
   ```

4. Create directoris:
   ```bash
        mkdir data assets
   ```

### Dataset Download
Download the dataset from [Hateful Memes Challenge](https://hatefulmemeschallenge.com/#download) and follow the steps below to organize your project structure:

1. Move to the `data` directory from the project root:
   ```bash
   cd ./data
   ```

2. Save the downloaded dataset file into the `data` directory.

3. Unzip the dataset:
   ```bash
   cd ..
   unzip data/your-dataset.zip -d data/
   ```

Now the environment is set up, and the dataset is ready for use in the project.

### Usage

#### 1. Object Detection

To perform object detection on images using YOLO (You Only Look Once) model, run the following command:

```bash
python object_detection.py \
  --directory ./data/hateful_memes/img/ \
  --file <path_to_image_file> \
  --pick_random <True/False> \
  --plot <True/False> \
  --config data/yolov3.cfg \
  --weight data/yolov3.weights \
  --classes data/coco.names
```

- `--directory`: Path of the folder containing all images.
- `--file`: Image file path (optional, if not using random image).
- `--pick_random`: Select a random image for detection (default: True).
- `--plot`: Show the image with detected objects (default: True).
- `--config`: Path for YOLO model configuration file (default: data/yolov3.cfg).
- `--weight`: Path for YOLO model weight file (default: data/yolov3.weights).
- `--classes`: Path for COCO class name file (default: data/coco.names).

#### 2. Optical Character Recognition (OCR)

To extract text from images using OCR, use the following command:

```bash
python ocr.py \
  --directory ./data/hateful_memes/img/ \
  --file <path_to_image_file> \
  --draw_anno <True/False> \
  --pick_random <True/False> \
  --plot <True/False>
```

- `--directory`: Path of the folder containing all images.
- `--file`: Image file path (optional, if not using random image).
- `--draw_anno`: Draw annotations on the image (default: False).
- `--pick_random`: Select a random image for OCR (default: True).
- `--plot`: Show the image with extracted text (default: True).

#### 3. Training

To initiate the training process with a modified configuration in `config.yml,` execute the following command:

```bash
python train.py
```
Ensure that you have updated the necessary configurations in the `config.yml` file before starting the training process.

## License
The project has not been licensed till now.

## Acknowledgements
The cross-modality encoder part is referred from [LXMERT](https://arxiv.org/abs/1908.07490). 

## Contact
Please contact me at ommprakash.sahoo.eee20@iitbhu.ac.in for any query related to the code.
