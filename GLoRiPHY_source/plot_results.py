import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def populate_node_data(node_name, csv_file, node_dict):
    df = pd.read_csv(csv_file)
    node_dict[node_name]["LoRaPHY"] = df.loc[df['Testing'] == 'LoRaPHY', 'Accuracy'].values[0]
    node_dict[node_name]["NELoRa"] = 100 - df.loc[df['Testing'] == 'NELoRa', 'Accuracy'].values[0]
    node_dict[node_name]["Our"] =  100 - df.loc[df['Testing'] == 'GLoRiPHY', 'Accuracy'].values[0]

if __name__ == "__main__":
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams.update({'font.size': 24})
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Load the first CSV file (for SF8)
    curr_path = os.path.join(os.getcwd(), 'GLoRiPHY_source/testing')
    sf8_file = os.path.join(curr_path, 'test_SF8_sim_8_test/testing_log.csv')
    sf9_file = os.path.join(curr_path,'test_SF9_sim_9_test/testing_log.csv')
    
    data = {
        "SF8": {},
        "SF9": {}
    }

    populate_node_data("SF8", sf8_file, data)
    populate_node_data("SF9", sf9_file, data)
    
    df = pd.DataFrame(data).T  # Transpose to have SFs as rows

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.2
    index = np.arange(len(df.index))

    ax.bar(index + 0 * bar_width, df['LoRaPHY'], bar_width, label='LoRaPHY',color = 'antiquewhite',hatch='x',edgecolor='tan')
    ax.bar(index + 1 * bar_width, df['NELoRa'], bar_width, label='NELoRa',color = color_cycle[1], hatch='-',edgecolor='bisque')
    ax.bar(index + 2 * bar_width, df['Our'], bar_width, label='GLoRiPHY',color = color_cycle[3])

    ax.set_xlabel('Spreading Factor (SF)')
    ax.set_ylabel('Symbol Error Rate (SER %)')
    # ax.set_title('SER by SF and Model')
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(df.index)
    ax.legend(loc='upper right', fontsize = 23, bbox_to_anchor=(0.625, 1.032))


    plt.grid(True, which='both', axis ='y', linestyle='--', linewidth=0.5, color='grey')
    plt.tight_layout()
    plt.savefig(os.path.join(curr_path,'testing_sim.pdf'))


    new_node_data = {
        "Node D": {},
        "Node E": {},
        "Node F": {},
        "Node G": {},
        "Node H": {},
    }

    nodeD_file = os.path.join(curr_path, 'test_Node_D_8_test/testing_log.csv')
    nodeE_file = os.path.join(curr_path, 'test_Node_E_8_test/testing_log.csv')
    nodeF_file = os.path.join(curr_path, 'test_Node_F_8_test/testing_log.csv')
    nodeG_file = os.path.join(curr_path, 'test_Node_G_8_test/testing_log.csv')
    nodeH_file = os.path.join(curr_path, 'test_Node_H_8_test/testing_log.csv')

    populate_node_data("Node D", nodeD_file, new_node_data)
    populate_node_data("Node E", nodeE_file, new_node_data)
    populate_node_data("Node F", nodeF_file, new_node_data)
    populate_node_data("Node G", nodeG_file, new_node_data)
    populate_node_data("Node H", nodeH_file, new_node_data)

    df = pd.DataFrame(new_node_data).T  # Transpose to have SFs as rows

    # Plotting
    fig, ax = plt.subplots(figsize=(15, 6))
    bar_width = 0.25
    index = np.arange(len(df.index))

    # Shade background regions
    ax.axvspan(-0.25, 2.75, facecolor='#F9E0A2', alpha=0.1)  # Region 1 (first 3 bars)
    ax.axvspan(2.75, 4.75, facecolor='#F9E0A2', alpha=0.4)  # Region 2 (next 2 bars)

    ax.bar(index + 0 * bar_width, df['LoRaPHY'], bar_width, label='LoRaPHY',color = 'antiquewhite',hatch='x',edgecolor='tan')
    ax.bar(index + 1 * bar_width, df['NELoRa'], bar_width, label='NELoRa',color = color_cycle[1], hatch='-',edgecolor='bisque')
    ax.bar(index + 2 * bar_width, df['Our'], bar_width, label='GLoRiPHY',color = color_cycle[3])

    # Add labels for the regions
    ax.text(1.25,  max(list(df.max())) *1.07, 'Outdoor \n(Unseen)', horizontalalignment='center')
    ax.text(3.75,  max(list(df.max())) *1.07, 'Outdoor LOS \n(Unseen)', horizontalalignment='center')

    ax.set_ylabel('SER (%)')
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(df.index)
    ax.legend(loc='upper right', fontsize = 27, bbox_to_anchor=(1.01, 1.04))

    plt.xlim(-0.25, 4.75)
    plt.grid(True, which='both', axis ='y', linestyle='--', linewidth=0.5, color='grey')
    plt.tight_layout()
    plt.savefig(os.path.join(curr_path,'real_nodes_unseen.pdf'))
