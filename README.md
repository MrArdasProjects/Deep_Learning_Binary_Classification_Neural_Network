# 🧠 Deep Learning – Binary Classification Neural Network

This project was developed as a course assignment for the **MSc (Thesis) Computer Engineering** program at **İzmir University of Economics**, within the **Deep Learning Methods and Applications** course.

The study focuses on **neural network–based binary classification** using **structured (tabular) data**, with an emphasis on model design, hyperparameter optimization, and performance analysis.

---

## 🎯 Project Objectives

The main objectives of this project are:

- Define and solve a **binary classification problem** on a structured dataset  
- Design and train a **Neural Network model** using TensorFlow & Keras  
- Systematically explore different **hyperparameters**  
- Analyze model performance using training and validation metrics  
- Compare different network configurations through visual evaluation  

---

## 📊 Dataset Information

- **Dataset:** HR Employee Attrition Dataset  
- **Problem Type:** Binary Classification (Attrition: Yes / No)  
- **Inputs:** Numerical and categorical employee-related features  
- **Output:** Employee attrition status (0 / 1)  

The dataset provides a realistic, structured learning problem suitable for evaluating neural network performance on real-world data.

---

## 🛠️ Technology Stack

- **Python**
- **TensorFlow & Keras**
- **NumPy / Pandas**
- **Matplotlib**
- **Scikit-learn**

---

## 🔬 Model Design and Experiments

The following experiments were conducted during the project:

- Training with different **epoch values**  
- **Learning rate** tuning  
- Varying the **number of hidden layers and neurons**  
- **Dropout and regularization** experiments  
- Comparison of multiple neural network configurations  

All experiments were performed consistently on the same dataset to ensure fair comparison.

---

## 📈 Model Evaluation

Model performance was evaluated using:

- Training and validation **accuracy curves**  
- Training and validation **loss curves**  
- Comparative analysis across different models  

All generated figures are included in the repository:

- `final_accuracy_curve.png`
- `final_loss_curve.png`
- `comparison_accuracy.png`
- `comparison_loss.png`

---

## 📁 Repository Structure

├── deep_learning_attrition.py        # Main training script for binary classification
├── compare_models.py                 # Model comparison and hyperparameter experiments
├── WA_Fn-UseC_-HR-Employee-Attrition.csv  # Structured dataset used for training and testing
├── final_accuracy_curve.png          # Final training vs validation accuracy curve
├── final_loss_curve.png              # Final training vs validation loss curve
├── comparison_accuracy.png           # Accuracy comparison across different models
├── comparison_loss.png               # Loss comparison across different models
├── EEE517_Rapor.pdf                  # Detailed academic project report


- **Python files:** Model training and comparison scripts  
- **PNG files:** Training and evaluation visualizations  
- **PDF report:** Detailed academic report of the experiments  

---

## 📝 Conclusion

This project presents a comprehensive analysis of **neural networks for binary classification on structured data**.

Through systematic hyperparameter optimization and model comparison, the impact of architectural and training choices on performance is quantitatively evaluated.

The repository is designed as both an **academic submission** and a **portfolio-ready deep learning project**.

---

📌 *This repository contains an academic course assignment prepared for graduate-level study.*
