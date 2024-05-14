import matplotlib.pyplot as plt
import skorch

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

    fig_1 = line_plot(epochs, train_acc, val_acc, "train_acc", "val_acc", title)
    fig_2 = line_plot(epochs,train_loss,val_loss, "train_loss", "val_loss", title)
    
    #plt.show(fig_1)
    #plt.show(fig_2)

    return fig_1, fig_2

    

