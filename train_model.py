import argparse



from data import load_data_set
from data import SequenceDataModule


SequenceDataModule(train_sequences=)


## Globals #######

## X train path
x_train_data_path = r"P:\workspace\jan\fire_detection\dl\prepocessed_ref_tables\05_df_x_8_250smps.csv"
y_train_data_path = r"P:\workspace\jan\fire_detection\dl\prepocessed_ref_tables\05_df_y_8_250smps.csv"

## path to safe model
save_model_path = r"P:\workspace\jan\fire_detection\dl\models_store\06_LSTM_Light"

if os.path.exists(base_path):
    print("path exists")
else:
    os.mkdir(base_path)


## model params
N_EPOCHS = 100
BATCH_SIZE = 64

## training logger path
logger_path = "P:/workspace/jan/fire_detection/dl/models_store/06_LSTM_Light/tl_logger/"
logger_name = "Disturbance_predictor_9"