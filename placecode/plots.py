import seaborn as sns
import pandas as pd


def plot_fluorescence_per_round(binned_rounds, ax) -> None:
    """
    binned_rounds: array[rounds x bins]
    ax: matplotlib axis object
    """
    for i,raw_trace in enumerate(binned_rounds):
        ax.plot(raw_trace+i*100)
    ax.set_title('Raw fluorescence per round')
    ax.set_xlabel('Distance on the belt(cm)')
    hide_ticks_and_spines(ax)


def plot_average_raw_fluorescence(data, ax):
    ax.plot(data)
    ax.set_title('Average raw fluorescence Cell')
    ax.set_xlabel('Distance on the belt(cm)')
    hide_ticks_and_spines(ax)


def plot_zscore_fluorescence_per_round(data,ax,num_rounds):
    sns.heatmap(data,ax=ax,cbar_kws={'label': 'z-score fluorescence'})
    ax.set_ylabel('Rounds')
    ax.set_title("Activity per round")
    ax.set_xlabel('Position belt (cm)')
    ax_labels=range(1,num_rounds+1)
    ax.set_yticklabels(ax_labels)
    custom_ticks = [0, 50, 100, 150]  # Specify the positions where you want the ticks
    custom_labels = ['0', '50', '100', '150']  # Specify the labels for the ticks
    ax.set_xticks(custom_ticks)
    ax.set_xticklabels(custom_labels)


def plot_average_zscore_activity(data,ax):  
    data=pd.DataFrame(data,columns=[''])  
    sns.heatmap(data.transpose(),ax,cbar_kws={'label': 'z-score fluorescence'})
    ax.set_title("Average activity")
    ax.set_ylabel('')
    ax.set_ylim(0,4)
    ax.set_xlabel('Position belt (cm)')
    custom_ticks = [0, 50, 100, 150]  # Specify the positions where you want the ticks
    custom_labels = ['0', '50', '100', '150']  # Specify the labels for the ticks
    ax.set_xticks(custom_ticks)
    ax.set_xticklabels(custom_labels)




def hide_ticks_and_spines(ax):
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
