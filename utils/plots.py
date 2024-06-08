import matplotlib.pyplot as plt
import skorch
''' 
def line_plot(x, y, z, label_1, label_2, title):
    # Create a new figure
    fig, ax = plt.subplots()

    # Plot x vs y with a blue line
    ax.plot(x, y, color='blue', label= label_1)

    # Plot x vs z with a red line
    ax.plot(x, z, color='red', label=label_2)

    # Add legends for each plot
    ax.legend(loc='upper left')

    # Set the title
    ax.set_title(title)

    # Return the plot object
    return fig

def model_plots(model, title):
    epochs = []

    train_acc = []
    val_acc = []

    train_loss = []
    val_loss = []


    for x in model.history:
        epoch_num = x["epoch"]
        epochs.append(epoch_num)

        train_acc.append(x['train_acc'])
        val_acc.append(x['valid_acc'])

        train_loss.append(x["train_loss"])
        val_loss.append(x["valid_loss"])

    fig_1 = line_plot(epochs, train_acc, val_acc, "train_acc", "val_acc", f"{title}_Accuracy")
    fig_2 = line_plot(epochs,train_loss,val_loss, "train_loss", "val_loss", f"{title}_Loss")
    
    #plt.show(fig_1)
    #plt.show(fig_2)

    return fig_1, fig_2

    

'''

def line_plot(ax, x, y, z, label_1, label_2, title):
    # Plot x vs y with a blue line
    ax.plot(x, y, color='blue', label=label_1)

    # Plot x vs z with a red line
    ax.plot(x, z, color='red', label=label_2)

    # Add legends for each plot
    ax.legend(loc='upper left')

    # Set the title
    ax.set_title(title)

def model_plots(model, title):
    epochs = []

    train_acc = []
    val_acc = []

    train_loss = []
    val_loss = []

    for x in model.history:
        epoch_num = x["epoch"]
        epochs.append(epoch_num)

        train_acc.append(x['train_acc'])
        val_acc.append(x['valid_acc'])

        train_loss.append(x["train_loss"])
        val_loss.append(x["valid_loss"])

    # Create a figure with 1 row and 2 columns
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Plot the accuracy and loss on the respective subplots
    line_plot(axs[0], epochs, train_acc, val_acc, "train_acc", "val_acc", f"{title}_Accuracy")
    line_plot(axs[1], epochs, train_loss, val_loss, "train_loss", "val_loss", f"{title}_Loss")

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Return the combined figure
    return fig
