# Future-Proofing Multilingual Fake Speech Detection

The rapid advancements in generative-AI models have significantly improved the realism of synthetic audio, making it increasingly difficult to distinguish between real and AI-generated speech. This research investigates how well machine learning models can detect AI-generated speech in unknown languages and from previously unseen generative models. 

## Goals
* Determine whether AI-generated speech can be detected based on language characteristics or model-specific artifacts.
* Evaluate detection accuracy when facing speech from unknown languages and unseen AI models.
* Develop a robust detection approach that remains effective against future generative-AI advancements. 

## Method

### Feature Extraction - Bispectrum Analysis
We utilize bispectrum analysis instead of conventional features like MFCC to capture higher-order correlations in audio signals. Bispectrum reveals unique spectral patterns indicative of synthetic speech.  
* Extracted features: absolute, angle, cum3, real, and imag  
* A "signature image" is created from these features for model input.

## Models
Three CNN architectures were trained:
* ResNet50 
* GoogLeNet 
* Mid-CNN (a custom CNN architecture that improved version from the previous paper 'AI Generated Speech Detection Using CNN') 
Each model was tested on individual feature sets and multi-form data combinations.

## Dataset
* English datasets: [‘In-the-Wild'](https://deepfake-total.com/in_the_wild), [Fake or Real (FoR)](https://bil.eecs.yorku.ca/datasets/), [WaveFake](https://zenodo.org/records/5642694), and [VCTK](https://doi.org/10.7488/ds/2645.). 
* Multilingual datasets: [CommonLanguage](https://zenodo.org/records/5036977) and [ELTOLSM](https://drive.google.com/drive/u/1/folders/1SVSou6rZkQYgmZhVCFCOj6bPEkrZrBvT) (31 languages, 7 AI methods).  
* Hypothesis datasets: [HULTEL](https://drive.google.com/drive/u/6/folders/1EXysipBgs3tpQOTU7hL_3RPjB3ZqDkIW) (28 languages, generated using a new AI model).

## Conclusion
* Model Performance: The best test accuracy scores achieved were:  
  * 94.92% for known-language, unknown-AI  
  * 98.44% for unknown-language, known-AI  
  * 95.18% for unknown-language, unknown-AI   
* Detection is AI-Model Dependent: The results suggest that detection models learn AI-specific artifacts rather than linguistic characteristics.  
* Augmentations Improve Robustness: Gaussian noise and background noise augmentations impacted performance, highlighting their role in training effective models.  
* Data Balance is Critical: Models trained on unbalanced datasets struggle to generalize across languages.  

## Future Work
* Exploring higher-order spectral features like trispectrum. 
* Training sequence-aware models like MAMBA for segment-based detection. 
* Developing adaptive models for different sample rates and audio qualities.

## Paper and Citation
[Go to paper](materials/Future-Proofing&#32;Multilingual&#32;Fake&#32;Speech&#32;Detection.pdf)
```bibtex
Meriç Demirörs, Ahmet Murat Özbayoğlu and Toygar Akgün, "Future-Proofing Multilingual Fake Speech Detection", 2025
```
