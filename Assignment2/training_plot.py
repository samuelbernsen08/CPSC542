import matplotlib.pyplot as plt

def plot(model_history, epochs):
    epochs_range = range(1, epochs+1)
    train_acc = model_history.history['accuracy']
    val_acc = model_history.history['val_accuracy']

    # Plotting the accuracy
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig("training_plot")
    # plt.show()