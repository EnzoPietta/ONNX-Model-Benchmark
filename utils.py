import pickle
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class utils():
    def __init__(self):
        super(utils, self).__init__()
        
    def save_object(filename, object):
        with open(filename, 'wb') as outp:
            pickle.dump(object, outp, pickle.HIGHEST_PROTOCOL)
    
    def read_object(filename):
        with open(filename, 'rb') as inp:
            return pickle.load(inp)

    def get_model_output_filename (topology,quant):
        return f"./pytorch_models/sat6-cnn-t{topology}w{quant}.pt"

    def get_hardware_config_name (topology,quant,target_fps):
        return f"t{topology}w{quant}_{target_fps}fps"

    def save_csv_table(results,csv_pathname):
        df = pd.DataFrame(results)
        print(df.to_string(header=None, index=False))
        df.to_csv(csv_pathname) 
        print(f"succesfully saved at {csv_pathname}")
        
    def read_json_dict(filename):
        with open(filename, "r") as f:
            ret = json.load(f)
        return ret


class plots():
    def __init__(self):
        super(plots, self).__init__()
        
    def time_plot(numpy_array, var_name, unit):

        mean = np.mean(numpy_array)
        std  = np.std(numpy_array)

        plt.figure(figsize=(8, 6))
        plt.plot(numpy_array, label=var_name, color='b', linewidth=1.5)
        plt.axhline(mean, color='r', linestyle='--', label=f'Média: {mean:.2f} {unit}')
        plt.fill_between(range(len(numpy_array)), mean - std, 
                            mean + std, color='r', alpha=0.2, label=f'Desvio padrão: {std:.2f} {unit}')
        plt.title(f'{var_name} ao Longo do Tempo')
        plt.xlabel('Iteração')
        plt.ylabel(f'{var_name} ({unit})')
        plt.legend()
        plt.show()
    
    def histogram_plot(numpy_array, var_name, unit):
        mean = np.mean(numpy_array)
        std  = np.std(numpy_array)
        
        # hist_array = numpy_array[numpy_array < 3*std]
        # hist_array = numpy_array[(numpy_array > (mean - 3*std)) & (numpy_array < (mean + 3*std))]
        # hist_array = np.clip(numpy_array, mean - 3*std, mean + 3*std)
        plt.figure(figsize=(8, 6))
        sns.set_theme(style="whitegrid")
        plt.figure()
        sns.histplot(numpy_array, bins=30, kde=True, color='b', stat='density')
        plt.axvline(mean, color='r', linestyle='--', label=f'Média: {mean:.2f} {var_name}')
        plt.title(f'Distribuição da {var_name}')
        plt.xlabel(f'{var_name} ({unit})')
        plt.ylabel('Densidade')
        plt.xlim(3*std)
        plt.legend()
        plt.tight_layout()
        plt.show()
