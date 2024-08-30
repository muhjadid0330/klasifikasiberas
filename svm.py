import streamlit as st
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from utils import load_dataset, classify_uploaded_image

def app():
    st.title("Support Vector Machine")

    # Load the dataset
    data, labels = load_dataset()
    
    # Correct syntax for checking if data or labels are None
    if data is None or labels is None:
        st.error("Error: Dataset could not be loaded. Please check the dataset path.")
        return

    # Reshape the data and split
    X_train, X_test, y_train, y_test = train_test_split(data.reshape(len(data), -1), labels, test_size=0.2, random_state=42)

    # Train the SVM model
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Display metrics
    st.write(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
    st.write("Classification Report:")
    st.text(classification_report(y_test, predictions))

    # Confusion matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, predictions)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    st.pyplot(fig)

    # File upload and classification (with flattening for SVM)
    classify_uploaded_image(model, flatten_image=True)
