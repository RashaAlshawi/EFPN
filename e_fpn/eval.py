import numpy as np
import tensorflow as tf
from utils import calculate_fwiou, calculate_pixel_acc, calculate_balanced_accuracy, map_to_rgb, calculate_iou,calculate_f1_score, calculate_pixel_acc, calculate_balanced_accuracy
from dataset import load_data, prepare_data

def evaluate_model():
    # Load the trained model
    model = tf.keras.models.load_model('/Users/rashaalshawi/Documents/Research_PhD23/E-FPN-Segmentation/experiments/final_model.h5')

    # Load and prepare the dataset
    X_datas, y_datas = load_data()
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(X_datas, y_datas)
    
    # Create test dataset for evaluation
    test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)  # Set your batch size

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_data, verbose=1)
    print(f'Test Loss: {test_loss}')
    print(f'Test Accuracy: {test_accuracy}')

    # Predict on the test set
    y_predict = model.predict(X_test)
    y_predict_ = np.argmax(y_predict, axis=-1)
    y_test_ = np.argmax(y_test, axis=-1)

    # Class frequencies for FWIoU calculation
    class_frequencies = {
        0: 0.0, 1: 1, 2: 1, 3: 1, 4: 0.1622, 5: 0.7100, 6: 0.3518, 7: 0.6419, 8: 0.5419
    }

    # Calculate and print metrics
    fwiou = calculate_fwiou(y_predict_, y_test_, class_frequencies)
    print(f"FWIoU: {fwiou}")

    IoU=calculate_iou(y_predict_, y_test_)
    print(f"IoU: {IoU}")

    F1=calculate_f1_score(y_predict_, y_test_)
    print(f"F1: {F1}")

    pixel_acc=calculate_pixel_acc(y_predict_, y_test_)
    print(f"pixel_acc: {pixel_acc}")

    balanced_accuracy=calculate_balanced_accuracy(y_predict_, y_test_)
    print(f"balanced_accuracy: {balanced_accuracy}")


    # More metrics like IoU, Pixel Accuracy, etc. can be calculated here similarly

if __name__ == "__main__":
    evaluate_model()
