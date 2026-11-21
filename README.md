# BRISC 2025 / Figshare: duplication, leakage, inter-annotator variability and CLAHE

Code for the article *Duplication, leakage and inter-annotator variability in public brain tumour MRI datasets* (Halloum K., Ez-Zahraouy H., submitted to Computers in Biology and Medicine).

## What it does

| Part | Content |
|---|---|
| A | Matches every BRISC 2025 tumour image to the 3,064 Figshare slices (Cheng et al., 2015) with a 64-bit perceptual hash over four orientations, confirmed by pixel correlation. Counts copies and train/test leakage in the official BRISC split. |
| B | Compares the Figshare and BRISC tumour masks of the same slice (Dice, IoU, HD95, ASSD; 0.49 mm/pixel at 512 px). |
| C | Trains ResNet50, DenseNet121 and EfficientNet-B0 on a patient-level Figshare split under four CLAHE settings (none, clip 1, 2, 4), three seeds. Reports accuracy, F1, AUC, calibration (with temperature scaling) and the localisation of Grad-CAM, Grad-CAM++ and Integrated Gradients against both expert masks. |

## Data

Nothing is redistributed here. The scripts download:
- Figshare brain tumour dataset (article 1512427) from api.figshare.com;
- BRISC 2025 from Kaggle (`briscdataset/brisc2025`) through `kagglehub`.

## How to run

The analysis was run on Google Colab with a T4 GPU.

1. Open `CLAHE_Figshare_BRISC.ipynb` in Colab (GPU runtime) and run all cells, or run `python pipeline.py`.
2. Results are written to `CLAHE_BRISC/results/` (audit pairs, mask agreement, classification and XAI tables).
3. `saliency_figure6.py`, run after the pipeline in the same session, retrains ResNet50 (no CLAHE, seed 0) and exports the saliency examples; `plot_figure_6_saliency.py` draws Figure 6.
4. `extract_figure_data.py`, `plot_figures_1_2_3_5.py` and `plot_figure_4_masks.py` produce Figures 1 to 5.

Set `SMOKE_TEST = True` at the top of `pipeline.py` for a quick check on 150 images.

## Files

- `*.py` scripts
- `classification_all_seeds.csv` classification and calibration metrics, 3 models × 4 conditions × 3 seeds

## License

MIT. Datasets keep their own licences.
