# 🧠 Simple CRNN OCR Model

A minimal and easy-to-understand **CRNN model** for character sequence recognition in **OCR (Optical Character Recognition)** tasks.  
The model combines **CNN** for feature extraction and **BiLSTM (RNN)** for sequence modeling, with **CTC loss** for alignment-free training.

This project is intended as a learning demo and can be extended into more advanced OCR architectures.

---

## 🗂Structure
```
crnn_demo/
│
├── README.md
├── requirements.txt
│
├── models/
│ └── crnn_model.py # CRNN architecture definition (CNN + BiLSTM + FC)
│
├── train.py # Training script
├── predict.py # Prediction script
└── utils.py # Helper functions
```
---

# 📦 Install requirements:
```bash
pip install -r requirements.txt
```
## 📬Feedback
For feedback or questions, contact: [barkin.adiguzel@gmail.com](mailto:barkin.adiguzel@gmail.com)
