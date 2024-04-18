import os.path
import sys
import warnings
import time
import glob
# supress pandas deprication warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd
import torch
import torch.optim as optim
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

from data_loader import get_dataloader
from model import *
DATA_DIR = 'data'
OUT_DIR = 'outputs'
os.makedirs(OUT_DIR, exist_ok=True)

def plot_metrics(metrics, title='Training Metrics'):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    for i, (k, v) in enumerate(metrics[0].items()):
        ax = axs[i // 2, i % 2]
        ax.plot([m[k] for m in metrics])
        ax.set_title(k)
    # add sup title
    fig.suptitle(title)
    save_path = f'{OUT_DIR}/{title}.png'
    plt.savefig(save_path)
    print(f"Saved plot of to {os.getcwd()}/{save_path}")
    plt.show()


def get_performance_metrics(predictions, actuals):
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    spearman = spearmanr(actuals, predictions)
    return {'mse': mse, 'mae': mae, 'r2': r2, 'spearman_r': spearman.correlation}

def train_model(model, train_loader, validation_loader, plot=False):
    num_epochs = 100
    early_stopping = EarlyStopping(patience=5, verbose=True)

    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    train_epoch_metrics = []
    val_epoch_metrics = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        pred_vals = []
        true_vals = []
        for i, data in enumerate(train_loader):
            labels = data['DMS_score']
            optimizer.zero_grad()
            outputs = model(data).squeeze()
            # recon_batch, mu, logvar = model(batch)
            pred_vals.extend(outputs.detach().numpy())
            true_vals.extend(labels.numpy())
            loss = criterion(outputs, labels)
            # loss = loss_function(recon_batch,batch['embedding'], mu, logvar)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_metrics = get_performance_metrics(pred_vals, true_vals)
        train_epoch_metrics.append(train_metrics)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, MAE: {train_metrics["mae"]:.4f}, '
              f'R2: {train_metrics["r2"]:.4f}, Spearman R: {train_metrics["spearman_r"]:.4f}')
        val_metrics = evaluate_model(model, validation_loader)
        val_epoch_metrics.append(val_metrics)
        

            # Call early stopping to monitor validation loss
        early_stopping(loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    if plot:
        plot_metrics(train_epoch_metrics, title='Training Metrics')
        plot_metrics(val_epoch_metrics, title='Validation Metrics')


def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for data in test_loader:
            labels = data['DMS_score']
            outputs = model(data)
            predictions.extend(outputs.numpy())
            actuals.extend(labels.numpy())
    metrics = get_performance_metrics(predictions, actuals)
    return metrics


# Define loss function
def loss_function(recon_x, x, mu, logvar):
    '''loss function for VAE'''
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# add new imformation
def blossum_score(mutant):
    blossum_dict = {'AA': 4, 'AR': -1, 'AN': -2, 'AD': -2, 'AC': 0, 'AQ': -1, 'AE': -1, 'AG': 0, 'AH': -2, 'AI': -1,
                'AL': -1, 'AK': -1, 'AM': -1, 'AF': -2, 'AP': -1, 'AS': 1, 'AT': 0, 'AW': -3, 'AY': -2, 'AV': 0,
                'AB': -2, 'AZ': -1, 'RA': -1, 'RR': 5, 'RN': 0, 'RD': -2, 'RC': -3, 'RQ': 1, 'RE': 0, 'RG': -2, 'RH': 0,
                'RI': -3, 'RL': -2, 'RK': 2, 'RM': -1, 'RF': -3, 'RP': -2, 'RS': -1, 'RT': -1, 'RW': -3, 'RY': -2,
                'RV': -3, 'RB': -1, 'RZ': 0, 'NA': -2, 'NR': 0, 'NN': 6, 'ND': 1, 'NC': -3, 'NQ': 0, 'NE': 0, 'NG': 0,
                'NH': 1, 'NI': -3, 'NL': -3, 'NK': 0, 'NM': -2, 'NF': -3, 'NP': -2, 'NS': 1, 'NT': 0, 'NW': -4,
                'NY': -2, 'NV': -3, 'NB': 3, 'NZ': 0, 'DA': -2, 'DR': -2, 'DN': 1, 'DD': 6, 'DC': -3, 'DQ': 0, 'DE': 2,
                'DG': -1, 'DH': -1, 'DI': -3, 'DL': -4, 'DK': -1, 'DM': -3, 'DF': -3, 'DP': -1, 'DS': 0, 'DT': -1,
                'DW': -4, 'DY': -3, 'DV': -3, 'DB': 4, 'DZ': 1, 'CA': 0, 'CR': -3, 'CN': -3, 'CD': -3, 'CC': 9,
                'CQ': -3, 'CE': -4, 'CG': -3, 'CH': -3, 'CI': -1, 'CL': -1, 'CK': -3, 'CM': -1, 'CF': -2, 'CP': -3,
                'CS': -1, 'CT': -1, 'CW': -2, 'CY': -2, 'CV': -1, 'CB': -3, 'CZ': -3, 'QA': -1, 'QR': 1, 'QN': 0,
                'QD': 0, 'QC': -3, 'QQ': 5, 'QE': 2, 'QG': -2, 'QH': 0, 'QI': -3, 'QL': -2, 'QK': 1, 'QM': 0, 'QF': -3,
                'QP': -1, 'QS': 0, 'QT': -1, 'QW': -2, 'QY': -1, 'QV': -2, 'QB': 0, 'QZ': 3, 'EA': -1, 'ER': 0, 'EN': 0,
                'ED': 2, 'EC': -4, 'EQ': 2, 'EE': 5, 'EG': -2, 'EH': 0, 'EI': -3, 'EL': -3, 'EK': 1, 'EM': -2, 'EF': -3,
                'EP': -1, 'ES': 0, 'ET': -1, 'EW': -3, 'EY': -2, 'EV': -2, 'EB': 1, 'EZ': 4, 'GA': 0, 'GR': -2, 'GN': 0,
                'GD': -1, 'GC': -3, 'GQ': -2, 'GE': -2, 'GG': 6, 'GH': -2, 'GI': -4, 'GL': -4, 'GK': -2, 'GM': -3,
                'GF': -3, 'GP': -2, 'GS': 0, 'GT': -2, 'GW': -2, 'GY': -3, 'GV': -3, 'GB': -1, 'GZ': -2, 'HA': -2,
                'HR': 0, 'HN': 1, 'HD': -1, 'HC': -3, 'HQ': 0, 'HE': 0, 'HG': -2, 'HH': 8, 'HI': -3, 'HL': -3, 'HK': -1,
                'HM': -2, 'HF': -1, 'HP': -2, 'HS': -1, 'HT': -2, 'HW': -2, 'HY': 2, 'HV': -3, 'HB': 0, 'HZ': 0,
                'IA': -1, 'IR': -3, 'IN': -3, 'ID': -3, 'IC': -1, 'IQ': -3, 'IE': -3, 'IG': -4, 'IH': -3, 'II': 4,
                'IL': 2, 'IK': -3, 'IM': 1, 'IF': 0, 'IP': -3, 'IS': -2, 'IT': -1, 'IW': -3, 'IY': -1, 'IV': 3,
                'IB': -3, 'IZ': -3, 'LA': -1, 'LR': -2, 'LN': -3, 'LD': -4, 'LC': -1, 'LQ': -2, 'LE': -3, 'LG': -4,
                'LH': -3, 'LI': 2, 'LL': 4, 'LK': -2, 'LM': 2, 'LF': 0, 'LP': -3, 'LS': -2, 'LT': -1, 'LW': -2,
                'LY': -1, 'LV': 1, 'LB': -4, 'LZ': -3, 'KA': -1, 'KR': 2, 'KN': 0, 'KD': -1, 'KC': -3, 'KQ': 1, 'KE': 1,
                'KG': -2, 'KH': -1, 'KI': -3, 'KL': -2, 'KK': 5, 'KM': -1, 'KF': -3, 'KP': -1, 'KS': 0, 'KT': -1,
                'KW': -3, 'KY': -2, 'KV': -2, 'KB': 0, 'KZ': 1, 'MA': -1, 'MR': -1, 'MN': -2, 'MD': -3, 'MC': -1,
                'MQ': 0, 'ME': -2, 'MG': -3, 'MH': -2, 'MI': 1, 'ML': 2, 'MK': -1, 'MM': 5, 'MF': 0, 'MP': -2, 'MS': -1,
                'MT': -1, 'MW': -1, 'MY': -1, 'MV': 1, 'MB': -3, 'MZ': -1, 'FA': -2, 'FR': -3, 'FN': -3, 'FD': -3,
                'FC': -2, 'FQ': -3, 'FE': -3, 'FG': -3, 'FH': -1, 'FI': 0, 'FL': 0, 'FK': -3, 'FM': 0, 'FF': 6,
                'FP': -4, 'FS': -2, 'FT': -2, 'FW': 1, 'FY': 3, 'FV': -1, 'FB': -3, 'FZ': -3, 'PA': -1, 'PR': -2,
                'PN': -2, 'PD': -1, 'PC': -3, 'PQ': -1, 'PE': -1, 'PG': -2, 'PH': -2, 'PI': -3, 'PL': -3, 'PK': -1,
                'PM': -2, 'PF': -4, 'PP': 7, 'PS': -1, 'PT': -1, 'PW': -4, 'PY': -3, 'PV': -2, 'PB': -2, 'PZ': -1,
                'SA': 1, 'SR': -1, 'SN': 1, 'SD': 0, 'SC': -1, 'SQ': 0, 'SE': 0, 'SG': 0, 'SH': -1, 'SI': -2, 'SL': -2,
                'SK': 0, 'SM': -1, 'SF': -2, 'SP': -1, 'SS': 4, 'ST': 1, 'SW': -3, 'SY': -2, 'SV': -2, 'SB': 0, 'SZ': 0,
                'TA': 0, 'TR': -1, 'TN': 0, 'TD': -1, 'TC': -1, 'TQ': -1, 'TE': -1, 'TG': -2, 'TH': -2, 'TI': -1,
                'TL': -1, 'TK': -1, 'TM': -1, 'TF': -2, 'TP': -1, 'TS': 1, 'TT': 5, 'TW': -2, 'TY': -2, 'TV': 0,
                'TB': -1, 'TZ': -1, 'WA': -3, 'WR': -3, 'WN': -4, 'WD': -4, 'WC': -2, 'WQ': -2, 'WE': -3, 'WG': -2,
                'WH': -2, 'WI': -3, 'WL': -2, 'WK': -3, 'WM': -1, 'WF': 1, 'WP': -4, 'WS': -3, 'WT': -2, 'WW': 11,
                'WY': 2, 'WV': -3, 'WB': -4, 'WZ': -3, 'YA': -2, 'YR': -2, 'YN': -2, 'YD': -3, 'YC': -2, 'YQ': -1,
                'YE': -2, 'YG': -3, 'YH': 2, 'YI': -1, 'YL': -1, 'YK': -2, 'YM': -1, 'YF': 3, 'YP': -3, 'YS': -2,
                'YT': -2, 'YW': 2, 'YY': 7, 'YV': -1, 'YB': -3, 'YZ': -2, 'VA': 0, 'VR': -3, 'VN': -3, 'VD': -3,
                'VC': -1, 'VQ': -2, 'VE': -2, 'VG': -3, 'VH': -3, 'VI': 3, 'VL': 1, 'VK': -2, 'VM': 1, 'VF': -1,
                'VP': -2, 'VS': -2, 'VT': 0, 'VW': -3, 'VY': -1, 'VV': 4, 'VB': -3, 'VZ': -2, 'BA': -2, 'BR': -1,
                'BN': 3, 'BD': 4, 'BC': -3, 'BQ': 0, 'BE': 1, 'BG': -1, 'BH': 0, 'BI': -3, 'BL': -4, 'BK': 0, 'BM': -3,
                'BF': -3, 'BP': -2, 'BS': 0, 'BT': -1, 'BW': -4, 'BY': -3, 'BV': -3, 'BB': 4, 'BZ': 1, 'ZA': -1,
                'ZR': 0, 'ZN': 0, 'ZD': 1, 'ZC': -3, 'ZQ': 3, 'ZE': 4, 'ZG': -2, 'ZH': 0, 'ZI': -3, 'ZL': -3, 'ZK': 1,
                'ZM': -1, 'ZF': -3, 'ZP': -1, 'ZS': 0, 'ZT': -1, 'ZW': -3, 'ZY': -2, 'ZV': -2, 'ZB': 1, 'ZZ': 4,
                'XA': 0, 'XR': -1, 'XN': -1, 'XD': -1, 'XC': -2, 'XQ': -1, 'XE': -1, 'XG': -1, 'XH': -1, 'XI': -1,
                'XL': -1, 'XK': -1, 'XM': -1, 'XF': -1, 'XP': -2, 'XS': 0, 'XT': 0, 'XW': -2, 'XY': -1, 'XV': -1,
                'XB': -1, 'XZ': -1}
    ref_aa = mutant[0]  # Extract reference amino acid from mutant, e.g., "A" from "A101C"
    mutant_aa = mutant[-1]  # Extract alt amino acid
    return blossum_dict.get(ref_aa + mutant_aa, 0)  # Returns the value for key if key is in the dictionary, else 0


def extract_wt_sequence(mutant, mutated_sequence):
    # Extract the position and reference amino acid from the mutant
    position = int(mutant[1:-1])  # Extract position from mutant, e.g., "101" from "A101C"
    ref_aa = mutant[0]  # Extract reference amino acid from mutant, e.g., "A" from "A101C"
    # Retrieve the WT sequence
    wt_sequence = mutated_sequence[:position - 1] + ref_aa + mutated_sequence[position:]
    return wt_sequence

    for batch in loader:
        # Assuming batch['embedding'] and batch['mutant'] are correctly formatted and exist
        # print("Before embedding shape:", batch['embedding'].shape)
        
        # Calculate blosum scores and update embeddings
        blosum_vals = [blossum_score(mut) for mut in batch['mutant']]
        # print("Mutants:", batch['mutant'])
#         print("Blosum values:", blosum_vals)
#         print("Num mutants in batch", len(batch['mutant']))
        
        blosum_scores_tensor = torch.tensor(blosum_vals, dtype=batch['embedding'].dtype)
        #print('CHECK', blosum_scores_tensor.shape)
        # I want (batch_size, seq_size, 1281), (broadcasting across seq_size)
        
        blosum_expanded = blosum_scores_tensor.unsqueeze(-1).unsqueeze(-1).expand(-1, batch['embedding'].shape[1], 1)
        batch['embedding'] = torch.cat((batch['embedding'], blosum_expanded), dim=-1)

        # print("After adding blossum_score, embedding shape:", batch['embedding'].shape)
        updated_batches.append(batch)

    return updated_batches



def main(experiment_path, train_folds=[1,2,3], validation_folds=[4], test_folds=[5], plot=True):
    print(f"\nTraining model on {experiment_path}")
      
    print(f"\nTraining model on {experiment_path}")
    train_loader = get_dataloader(experiment_path=experiment_path, folds=train_folds, return_logits=True, return_wt=True)
    train_loader=update_embedding(train_loader)
    
    val_loader = get_dataloader(experiment_path=experiment_path, folds=validation_folds, return_logits=True)
    val_loader=update_embedding(val_loader)

    test_loader = get_dataloader(experiment_path=experiment_path, folds=test_folds, return_logits=True)
    test_loader=update_embedding(test_loader)

    # model = ProteinModel()
    model = EmbeddingModel()
    start = time.time()
    train_model(model, train_loader, val_loader, plot=plot)
    train_time = time.time() - start
    metrics = evaluate_model(model, test_loader)
    metrics['train_time_secs'] = round(train_time, 1)
    print("Test performance metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    metrics['DMS_id'] = experiment_path.split('/')[-1]
    return metrics

if __name__ == "__main__":
    start = time.time()
    rows = []
    # if command line argument provided, use that as the experiment name
    # otherwise, loop over all experiments
    if len(sys.argv) > 1:
        experiments = sys.argv[1].split(',')
    else:
        experiments = list(set([fname.split('.')[0] for fname in os.listdir(DATA_DIR) if not fname.startswith('.')]))
    for experiment in experiments:
        # try:
        experiment_path = f"{DATA_DIR}/{experiment}"
        new_row = main(experiment_path=experiment_path)
        rows.append(new_row)
        # except Exception as e:
        #     print(f"Error with {experiment}: {e}")
        #     continue
    df = pd.DataFrame(rows)
    df_savepath = f'{OUT_DIR}/supervised_results.csv'
    df.to_csv(df_savepath, index=False)
    print(f"Metrics for {len(df)} experiments saved to {os.getcwd()}/{df_savepath}")
    print(df.head())
    end = time.time()
    print(f"Total time: {(end-start)/60:.2f} minutes")
