# CSE398 Lab 1: Object Detection with YOLOv11 and YOLOE

This repository contains the three experiments completed for Lab 1 in CSE398/498: Generative AI for Healthcare and Robotics.

The lab explores three approaches to object detection:

1. Applying pretrained YOLOv11 to everyday objects.
2. Fine-tuning YOLOv11 on a custom nail-clipper dataset.
3. Using YOLOE for open-set detection with text and visual prompts.

## Repository Contents

```text
CSE398-Lab1-Object-Detection/
├── Lab1_YOLOv11_Step1.ipynb
├── Lab1_YOLOv11_FineTuning_Step2.ipynb
├── Lab1_YOLOE_Step3.ipynb
├── CSE398_Lab1_Object_Detection_Report.pdf
├── models/
│   └── best.pt
└── results/
    └── step2/
        ├── results.png
        ├── results.csv
        ├── confusion_matrix.png
        ├── PR_curve.png
        ├── F1_curve.png
        ├── val_batch0_pred.jpg
        └── test_predictions/
```

## Experiment 1: Pretrained YOLOv11

A pretrained YOLOv11n model was applied to nine self-captured images of everyday objects.

### Configuration

- Model: `yolo11n.pt`
- Number of test images: 9
- Input size: `640 × 640`
- Confidence threshold: `0.25`
- Runtime: Google Colab with an NVIDIA Tesla T4 GPU

The model correctly detected several common objects, including cups, keyboards, bottles, and a monitor. However, it also produced incorrect classifications. For example, a mouse was classified as a vase, a spoon as a toothbrush, and a cat as a bear. The nail clipper was classified as a knife and a cell phone.

These results motivated the selection of the nail clipper as the custom object for fine-tuning.

## Experiment 2: YOLOv11 Fine-Tuning

A custom dataset of 50 nail-clipper images was collected and annotated in Roboflow.

### Dataset Split

- Training: 40 images
- Validation: 5 images
- Testing: 5 images
- Class: `nail_clipper`

### Training Configuration

- Initial model: `yolo11s.pt`
- Epochs: 10
- Image size: `640 × 640`
- Test confidence threshold: `0.25`
- Runtime: Google Colab with an NVIDIA Tesla T4 GPU

### Validation Results

| Metric | Value |
|---|---:|
| Precision | 0.989 |
| Recall | 1.000 |
| mAP@0.5 | 0.995 |
| mAP@0.5:0.95 | 0.995 |

After fine-tuning, the model successfully predicted the new `nail_clipper` category in all five test images. Some test images contained additional or overlapping detections, and the validation set was small. Therefore, the high validation metrics should not be interpreted as evidence of broad generalization.

The best trained weights are available at [`models/best.pt`](models/best.pt).

## Experiment 3: YOLOE Open-Set Detection

YOLOE was used to test text-prompt and visual-prompt detection without retraining the model for the nail-clipper category.

### Image Experiment Results

| Prompt type | Prompt or setting | Result |
|---|---|---|
| Text | `nail clipper`, `conf=0.02` | No detection |
| Text | `fingernail clipper`, `conf=0.02` | No detection |
| Text | `toenail clipper`, `conf=0.02` | One accurate detection |
| Fine-grained text | `lever of a nail clipper`, `conf=0.01` | Incorrect localization |
| Visual | `conf=0.01`, `imgsz=1280` | Multiple false positives |
| Visual | `conf=0.05`, `imgsz=1280` | No detection |

The successful result with `toenail clipper` demonstrates that YOLOE can detect a custom category without additional model training. However, the different outcomes produced by similar phrases also show that open-set detection can be sensitive to prompt wording.

The cross-image visual prompt did not generalize reliably because of differences in object scale, orientation, and background between the source and target images.

The official suitcase and croissant video examples were also completed to test text-prompt and visual-prompt video processing.

## Running the Notebooks

The notebooks were developed and executed in Google Colab.

1. Open the required `.ipynb` file in Google Colab.
2. Select a GPU runtime.
3. Run the cells in order.
4. Upload the required input images when prompted.
5. Store any required API tokens in Colab Secrets rather than writing them directly in the notebook.

Step 2 requires access to the Roboflow dataset and a `ROBOFLOW_API_KEY`. Step 3 may require a Hugging Face token for downloading the YOLOE model. Tokens are not included in this repository.

## Notes

- The notebooks retain their original execution outputs as a record of the experimental process.
- The Roboflow hosted-inference step was not completed because the `inference` package was incompatible with the Python 3.13 Colab runtime used in the experiment.
- Local inference using the fine-tuned `best.pt` weights was completed successfully.
- The large Step 3 notebook may take additional time to load in the GitHub preview.

## Report

The complete six-page assignment discussion, including the YOLOv11 workflow, fine-tuning lessons, open-set object detection, and YOLOE architecture, is available in:

[CSE398 Lab 1 Object Detection Report](CSE398_Lab1_Object_Detection_Report.pdf)

## Author

Ruoxi Wang
