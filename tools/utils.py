import matplotlib
matplotlib.use('Agg')   
import matplotlib.pyplot as plt
import os


def plot_acc_curve(train_acc, val_acc, save_path):
    plt.figure(figsize=(6,4))
    plt.plot(train_acc, label="train_acc")
    plt.plot(val_acc, label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()



def plot_all_curves(train_loss, train_acc, val_acc, val_prec, val_recall, save_path):
    fig, axes = plt.subplots(2,2, figsize=(12,10))

    axes[0,0].plot(train_loss)
    axes[0,0].set_title("Training Loss Curve")
    axes[0,0].set_xlabel("Epochs")
    axes[0,0].set_ylabel("Loss")
    axes[0,0].set_xlim(left=0)

    axes[0,1].plot(train_acc, label="Training Accuracy", color="green")
    axes[0,1].plot(val_acc, label="Test Accuracy", color="red")
    axes[0,1].set_title("Accuracy Curve")
    axes[0,1].set_xlabel("Epochs")
    axes[0,1].set_ylabel("Accuracy")
    axes[0,1].set_xlim(left=0)
    axes[0,1].legend()

    axes[1,0].plot(val_prec, color="purple")
    axes[1,0].set_title("Test Precision Curve")
    axes[1,0].set_xlabel("Epochs")
    axes[1,0].set_ylabel("Precision")
    axes[1,0].set_xlim(left=0)

    axes[1,1].plot(val_recall, color="gold")
    axes[1,1].set_title("Test Recall Curve")
    axes[1,1].set_xlabel("Epochs")
    axes[1,1].set_ylabel("Recall")
    axes[1,1].set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
