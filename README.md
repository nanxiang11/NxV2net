# NxV2net: A Nested Multiscale Network for Robust Crack Segmentation
![image](https://github.com/user-attachments/assets/34d60842-4c8b-468f-b40d-ec472c861240)


**NxV2net** is a novel deep learning architecture for robust crack segmentation in complex real-world scenarios. It addresses the challenges of multiscale feature extraction, fusion, and generalization in the presence of lighting variations, water infiltration, and human interference.

---

## 🔍 Motivation

Crack detection plays a crucial role in the safety assessment and maintenance of infrastructure. However, automatic crack segmentation remains challenging due to:

- Diverse crack patterns and complex topologies  
- Variable imaging conditions (e.g., lighting, blur)  
- Inadequate generalization of existing models  
- Insufficiently diverse benchmark datasets  

---

## 🚀 What We Propose

We introduce **V2net**, a nested multiscale segmentation network built upon cascaded VNet submodules, designed for improved generalization and feature representation.

### ✨ Key Contributions

- **🔗 V2net Architecture**: Cascaded VNet-style modules that progressively enhance multiscale representation.
- **💡 Multichannel Fusion Attention (MCFA)**: A lightweight yet effective module for feature extraction and channel-wise fusion.
- **📦 SUES-CRACK Dataset**: A real-world crack segmentation dataset featuring:
  - Lighting variation
  - Water-blurred boundaries
  - Human interference

---

## 📊 Results

| Dataset        | mIoU (%) |
|----------------|----------|
| CrackTree200   | 66.51    |
| DeepCrack      | 44.23    |
| **SUES-CRACK** | **50.91** |

Our model demonstrates superior performance in both quantitative and qualitative evaluations compared to existing crack segmentation methods.

---

## 📁 Resources

- 📄 **Paper**: _To be released upon publication_
- 💻 **Code & Dataset**: [https://github.com/nanxiang11/NxV2net](https://github.com/nanxiang11/NxV2net)

---

## 🧪 Coming Soon

- [ ] Training instructions  
- [ ] Inference demo  
- [ ] Dataset format guide  
- [ ] Model weights  
- [ ] BibTeX citation  

---

> For any questions or collaborations, feel free to open an issue or contact the author via GitHub.

