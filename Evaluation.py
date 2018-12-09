import matplotlib.pyplot as plt

def evaluation(history):
    fig, ax = plt.subplots(1, 2, figsize=(15,5))
    ax[0].set_title('loss')
    ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
    ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax[1].set_title('acc')
    ax[1].plot(history.epoch, history.history["acc"], label="Train acc")
    ax[1].plot(history.epoch, history.history["val_acc"], label="Validation acc")
    ax[0].legend()
    ax[1].legend()
    
def evaluation_f1(history):
    fig, ax = plt.subplots(1,figsize=(15,10))
    ax.set_title('f1')
    ax.plot(history.epoch, history.history["f1"], label="f1")
    ax.plot(history.epoch, history.history["val_f1"], label="val f1")
    ax.legend()

