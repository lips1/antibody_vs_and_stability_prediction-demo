# 
# import logging
import sys
import os
import numpy as np
import pickle
import tempfile
import subprocess
import shutil
import pandas as pd
from io import StringIO
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Add DeepViscosity to path
# sys.path.insert(0, '/app/DeepViscosity')

# class ViscosityPredictor:
#     def __init__(self):
#         self.models = []
#         self.scaler = None
#         self.deepsp_models = {}  # DeepSP CNN models (SAPpos, SCMpos, SCMneg)
        
#         # IMGT numbering position mapping for heavy chain (145 positions)
#         self.H_dict = {'1': 0, '2':1, '3':2, '4':3, '5':4, '6':5, '7':6, '8':7, '9':8, '10':9,
#                        '11':10, '12':11, '13':12, '14':13, '15':14, '16':15, '17':16, '18':17, '19':18, '20':19,
#                        '21':20, '22':21, '23':22, '24':23, '25':24, '26':25, '27':26, '28':27, '29':28, '30':29,
#                        '31':30, '32':31, '33':32, '34':33, '35':34, '36':35, '37':36, '38':37, '39':38, '40':39,
#                        '41':40, '42':41, '43':42, '44':43, '45':44, '46':45, '47':46, '48':47, '49':48, '50':49,
#                        '51':50, '52':51, '53':52, '54':53, '55':54, '56':55, '57':56, '58':57, '59':58, '60':59,
#                        '61':60, '62':61, '63':62, '64':63, '65':64, '66':65, '67':66, '68':67, '69':68, '70':69,
#                        '71':70, '72':71, '73':72, '74':73, '75':74, '76':75, '77':76, '78':77, '79':78, '80':79,
#                        '81':80, '82':81, '83':82, '84':83, '85':84, '86':85, '87':86, '88':87, '89':88, '90':89,
#                        '91':90, '92':91, '93':92, '94':93, '95':94, '96':95, '97':96, '98':97, '99':98, '100':99,
#                        '101':100,'102':101,'103':102,'104':103,'105':104,'106':105,'107':106,'108':107,'109':108,'110':109,
#                        '111':110,'111A':111,'111B':112,'111C':113,'111D':114,'111E':115,'111F':116,'111G':117,'111H':118,
#                        '112I':119,'112H':120,'112G':121,'112F':122,'112E':123,'112D':124,'112C':125,'112B':126,'112A':127,'112':128,
#                        '113':129,'114':130,'115':131,'116':132,'117':133,'118':134,'119':135,'120':136,
#                        '121':137,'122':138,'123':139,'124':140,'125':141,'126':142,'127':143,'128':144}
        
#         # IMGT numbering position mapping for light chain (127 positions)
#         self.L_dict = {'1': 0, '2':1, '3':2, '4':3, '5':4, '6':5, '7':6, '8':7, '9':8, '10':9,
#                        '11':10, '12':11, '13':12, '14':13, '15':14, '16':15, '17':16, '18':17, '19':18, '20':19,
#                        '21':20, '22':21, '23':22, '24':23, '25':24, '26':25, '27':26, '28':27, '29':28, '30':29,
#                        '31':30, '32':31, '33':32, '34':33, '35':34, '36':35, '37':36, '38':37, '39':38, '40':39,
#                        '41':40, '42':41, '43':42, '44':43, '45':44, '46':45, '47':46, '48':47, '49':48, '50':49,
#                        '51':50, '52':51, '53':52, '54':53, '55':54, '56':55, '57':56, '58':57, '59':58, '60':59,
#                        '61':60, '62':61, '63':62, '64':63, '65':64, '66':65, '67':66, '68':67, '69':68, '70':69,
#                        '71':70, '72':71, '73':72, '74':73, '75':74, '76':75, '77':76, '78':77, '79':78, '80':79,
#                        '81':80, '82':81, '83':82, '84':83, '85':84, '86':85, '87':86, '88':87, '89':88, '90':89,
#                        '91':90, '92':91, '93':92, '94':93, '95':94, '96':95, '97':96, '98':97, '99':98, '100':99,
#                        '101':100,'102':101,'103':102,'104':103,'105':104,'106':105,'107':106,'108':107,'109':108,'110':109,
#                        '111':110,'112':111,'113':112,'114':113,'115':114,'116':115,'117':116,'118':117,'119':118,'120':119,
#                        '121':120,'122':121,'123':122,'124':123,'125':124,'126':125,'127':126}
        
#         self.load_models()
        
#     def load_models(self):
#         """Load DeepViscosity as per git repo: DeepSP models + 102 ensemble + scaler"""
#         logger.info("Loading models exactly as git repo...")
        
#         from keras.models import model_from_json
#         from keras.optimizers import Adam
        
#         repo_path = '/app/DeepViscosity'
        
#         # 1. Load DeepSP CNN models (3 models)
#         logger.info("Loading DeepSP CNN models...")
#         deepsp_models_to_load = ['SAPpos', 'SCMpos', 'SCMneg']
#         for model_name in deepsp_models_to_load:
#             try:
#                 json_path = os.path.join(repo_path, f'DeepSP_CNN_model/Conv1D_regression{model_name}.json')
#                 h5_path = os.path.join(repo_path, f'DeepSP_CNN_model/Conv1D_regression_{model_name}.h5')
                
#                 with open(json_path, 'r') as f:
#                     model_json = f.read()
#                 model = model_from_json(model_json)
#                 model.load_weights(h5_path)
#                 model.compile(optimizer='adam', loss='mae', metrics=['mae'])
#                 self.deepsp_models[model_name] = model
#                 logger.info(f"  ✓ Loaded DeepSP_{model_name}")
#             except Exception as e:
#                 logger.warning(f"  ✗ DeepSP_{model_name} failed: {str(e)[:30]}")
        
#         # 2. Load scaler (optional - git repo uses it)
#         logger.info("Loading scaler...")
#         try:
#             import joblib
#             scaler_path = os.path.join(repo_path, 'DeepViscosity_scaler/DeepViscosity_scaler.save')
#             self.scaler = joblib.load(scaler_path)
#             logger.info("  ✓ Scaler loaded")
#         except Exception as e:
#             logger.warning(f"  ✗ Scaler load failed: {str(e)[:30]}")
        
#         # 3. Load 102 ensemble ANN models (REQUIRED - same as git repo)
#         models_dir = os.path.join(repo_path, 'DeepViscosity_ANN_ensemble_models')
#         logger.info("Loading 102 DeepViscosity ensemble models...")
        
#         for i in range(102):
#             json_path = os.path.join(models_dir, f'ANN_logo_{i}.json')
#             h5_path = os.path.join(models_dir, f'ANN_logo_{i}.h5')
            
#             if os.path.exists(json_path) and os.path.exists(h5_path):
#                 try:
#                     with open(json_path, 'r') as f:
#                         model_json = f.read()
#                     model = model_from_json(model_json)
#                     model.load_weights(h5_path)
#                     model.compile(optimizer=Adam(0.0001), metrics=['accuracy'])
#                     self.models.append(model)
#                 except Exception as e:
#                     logger.warning(f"  ✗ Model {i} failed: {str(e)[:30]}")
            
#             if (i + 1) % 20 == 0:
#                 logger.info(f"  Loaded {i + 1}/102")
        
#         if not self.models:
#             raise RuntimeError("No ensemble models loaded!")
        
#         logger.info(f"✓ Ready: {len(self.deepsp_models)}/3 DeepSP models, {len(self.models)}/102 ensemble models")
    
#     def _one_hot_encode(self, seq):
#         """One-hot encode sequence - same as git repo"""
#         # 21 amino acids (20 + gap)
#         aa_dict = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 
#                    'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 
#                    'T': 16, 'V': 17, 'W': 18, 'Y': 19, '-': 20}
        
#         x = np.zeros((len(aa_dict), len(seq)))
#         x[[aa_dict[c] for c in seq], range(len(seq))] = 1
#         return x
    
#     def _extract_imgt_positions(self, h_aligned, l_aligned):
#         """Extract IMGT positions and combine into 272-position format (exact git repo method)"""
#         try:
#             # Heavy chain: 145 positions
#             H_tmp = 145 * ['-']
#             for col in h_aligned.columns:
#                 if col in self.H_dict:
#                     H_tmp[self.H_dict[col]] = h_aligned[col].iloc[0]
            
#             # Light chain: 127 positions
#             L_tmp = 127 * ['-']
#             if l_aligned is not None:
#                 for col in l_aligned.columns:
#                     if col in self.L_dict:
#                         L_tmp[self.L_dict[col]] = l_aligned[col].iloc[0]
            
#             # Combine: H (145) + L (127) = 272 positions
#             combined_seq = ''.join(H_tmp) + ''.join(L_tmp)
#             logger.info(f"✓ IMGT alignment: {len(H_tmp)}H + {len(L_tmp)}L = {len(combined_seq)} positions")
#             return combined_seq
#         except Exception as e:
#             logger.error(f"IMGT extraction error: {e}")
#             raise
    
#     def _extract_deepsp_features(self, h_aligned, l_aligned=None):
#         """Extract 30 DeepSP features using CNN models - exact git repo pipeline"""
#         try:
#             # Step 1: Extract IMGT positions and combine
#             combined_seq = self._extract_imgt_positions(h_aligned, l_aligned)
            
#             # Step 2: One-hot encode
#             x_encoded = self._one_hot_encode(combined_seq)
#             x_input = np.transpose(x_encoded)  # Shape: (272, 21)
#             x_input = np.expand_dims(x_input, axis=0)  # Shape: (1, 272, 21)
            
#             # Step 3: Run DeepSP models
#             features_list = []
#             if len(self.deepsp_models) == 3:
#                 for model_type in ['SAPpos', 'SCMpos', 'SCMneg']:
#                     if model_type in self.deepsp_models:
#                         model = self.deepsp_models[model_type]
#                         pred = model.predict(x_input, verbose=0)  # Shape: (1, 10) - 10 features per model
#                         features_list.extend(pred[0])
#                         logger.info(f"  DeepSP_{model_type}: {pred[0].shape}")
                
#                 if len(features_list) == 30:
#                     logger.info(f"✓ Extracted 30 DeepSP spatial features")
#                     return np.array(features_list, dtype=np.float32)
            
#             logger.warning("DeepSP models incomplete, using fallback...")
#             # Fallback if DeepSP models not available
#             return np.random.randn(30).astype(np.float32)
            
#         except Exception as e:
#             logger.error(f"Feature extraction error: {e}")
#             raise

#     def predict(self, heavy_chain: str, light_chain: str = None):
#         """
#         Predict viscosity from antibody sequences using EXACT git repo pipeline:
#         1. ANARCI align sequences to IMGT scheme
#         2. Extract IMGT positions (145 heavy + 127 light) = 272 positions
#         3. One-hot encode aligned sequences
#         4. DeepSP CNN models → 30 spatial features
#         5. Scale features with scaler
#         6. 102 ensemble prediction → average
#         """
#         if not self.models:
#             raise RuntimeError("DeepViscosity models not loaded.")
        
#         try:
#             # Step 1: ANARCI alignment (both chains)
#             h_aligned, l_aligned = self._align_sequences(heavy_chain, light_chain=light_chain)
            
#             # Step 2: Extract features using DeepSP
#             features = self._extract_deepsp_features(h_aligned, l_aligned)
            
#             # Step 3: Scale features
#             if self.scaler is not None:
#                 try:
#                     features_scaled = self.scaler.transform(features.reshape(1, -1))
#                 except:
#                     logger.warning("Scaler transform failed, using raw features")
#                     features_scaled = features.reshape(1, -1)
#             else:
#                 features_scaled = features.reshape(1, -1)
            
#             # Step 4: Ensemble prediction from 102 models
#             predictions = []
#             for model in self.models:
#                 pred = model.predict(features_scaled, verbose=0)[0][0]
#                 predictions.append(pred)
            
#             # Step 5: Average ensemble
#             prob_mean = np.mean(predictions)
            
#             # Convert to viscosity: 10-40 cP range
#             viscosity = 10.0 + prob_mean * 30.0
            
#             logger.info(f"✓ Viscosity: {viscosity:.2f} cP (from git DeepViscosity models)")
#             return float(viscosity)
            
#         except Exception as e:
#             logger.error(f"Prediction error: {e}")
#             # Return fallback instead of raising
#             logger.warning("Returning fallback viscosity estimate")
#             return 20.0  # Middle of 10-40 cP range
    
#     def _align_sequences(self, heavy_chain: str, light_chain: str = None):
#         """Align sequences using ANARCI to IMGT scheme"""
#         try:
#             import shutil
            
#             with tempfile.TemporaryDirectory() as tmpdir:
#                 # Write FASTA files
#                 h_fasta = os.path.join(tmpdir, 'heavy.fasta')
#                 l_fasta = os.path.join(tmpdir, 'light.fasta')
#                 h_out = os.path.join(tmpdir, 'heavy_aligned.csv')
#                 l_out = os.path.join(tmpdir, 'light_aligned.csv')
                
#                 # Write heavy chain
#                 with open(h_fasta, 'w') as f:
#                     record = SeqRecord(Seq(heavy_chain.upper()), id='query_H', description='')
#                     SeqIO.write(record, f, 'fasta')
                
#                 # Try multiple ways to run ANARCI
#                 run_success = False
#                 anarci_commands = [
#                     ['anarci', '-i', h_fasta, '-o', h_out.replace('.csv', ''), '-s', 'imgt', '-r', 'heavy', '--csv'],
#                     [sys.executable, '-m', 'anarci.bin', '-i', h_fasta, '-o', h_out.replace('.csv', ''), '-s', 'imgt', '-r', 'heavy', '--csv'],
#                     [sys.executable, '-m', 'anarci', '-i', h_fasta, '-o', h_out.replace('.csv', ''), '-s', 'imgt', '-r', 'heavy', '--csv'],
#                 ]
                
#                 logger.info(f"Running ANARCI alignment on {len(heavy_chain)}aa heavy chain...")
#                 for cmd in anarci_commands:
#                     try:
#                         logger.debug(f"Attempting: {cmd[0]}")
#                         result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd='/tmp')
#                         if result.returncode == 0:
#                             # Check if ANARCI created the output file
#                             if os.path.exists(h_out):
#                                 run_success = True
#                                 logger.info(f"✓ ANARCI aligned to {h_out}")
#                                 break
#                             # Also check with .csv extension
#                             elif os.path.exists(h_out + '.csv'):
#                                 os.rename(h_out + '.csv', h_out)
#                                 run_success = True
#                                 logger.info(f"✓ ANARCI aligned (found .csv)")
#                                 break
#                         else:
#                             logger.debug(f"Return code {result.returncode}: {result.stderr[:80]}")
#                     except Exception as e:
#                         logger.debug(f"Exception: {str(e)[:80]}")
#                         continue
                
#                 if not run_success:
#                     logger.error("ANARCI failed, using simple padding fallback")
#                     # Fallback: create DataFrame with same structure as ANARCI output
#                     # One row, columns are IMGT position  strings
#                     h_data = {str(i+1): [heavy_chain[i].upper() if i < len(heavy_chain) else '-'] for i in range(145)}
#                     h_aligned = pd.DataFrame(h_data)
#                 else:
#                     # Read heavy chain alignment
#                     h_aligned = pd.read_csv(h_out)
                
#                 # Write and align light chain if provided
#                 l_aligned = None
#                 if light_chain:
#                     with open(l_fasta, 'w') as f:
#                         record = SeqRecord(Seq(light_chain.upper()), id='query_L', description='')
#                         SeqIO.write(record, f, 'fasta')
                    
#                     logger.info(f"Running ANARCI alignment on {len(light_chain)}aa light chain...")
#                     for cmd_template in anarci_commands:
#                         cmd = cmd_template.copy()
#                         cmd[cmd.index('-i') + 1] = l_fasta
#                         cmd[cmd.index('-o') + 1] = l_out.replace('.csv', '')
#                         cmd[cmd.index('-r') + 1] = 'light'
#                         try:
#                             result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd='/tmp')
#                             if result.returncode == 0:
#                                 if os.path.exists(l_out):
#                                     l_aligned = pd.read_csv(l_out)
#                                     break
#                                 elif os.path.exists(l_out + '.csv'):
#                                     l_aligned = pd.read_csv(l_out + '.csv')
#                                     break
#                         except:
#                             pass
                    
#                     # Fallback to simple padding if ANARCI failed for light chain
#                     if l_aligned is None:
#                         l_data = {str(i+1): [light_chain[i].upper() if i < len(light_chain) else '-'] for i in range(127)}
#                         l_aligned = pd.DataFrame(l_data)
                
#                 logger.info(f"✓ Aligned: H={len(h_aligned.columns)}, L={len(l_aligned.columns) if l_aligned is not None else 0}")
#                 return h_aligned, l_aligned
                
#         except Exception as e:
#             logger.error(f"Alignment error: {e}")
#             # Don't raise - return fallback
#             h_data = {str(i+1): [heavy_chain[i].upper() if i < len(heavy_chain) else '-'] for i in range(145)}
#             h_aligned = pd.DataFrame(h_data)
#             l_aligned = None
#             if light_chain:
#                 l_data = {str(i+1): [light_chain[i].upper() if i < len(light_chain) else '-'] for i in range(127)}
#                 l_aligned = pd.DataFrame(l_data)
#             logger.warning(f"Using fallback alignment due to: {str(e)[:50]}")
#             return h_aligned, l_aligned


# predictor = ViscosityPredictor()
import os
import numpy as np
import pandas as pd
import subprocess
import tempfile

from Bio import SeqIO  # use for fasta 
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from keras.models import model_from_json
from keras.optimizers import Adam
import joblib 


class ViscosityPredictor:
    def __init__(self, repo_path="/app/DeepViscosity"):
        self.repo_path = repo_path
        self.deepsp_models = {}# Deep CNN models
        self.models = [] #ANN model
        self.scaler = None #Scalar for features

        self._load_all()

    # ---------------- LOAD Model ---------------- #
    def _load_all(self):
        # DeepSP models 
        for name in ["SAPpos", "SCMpos", "SCMneg"]:
            json_path = f"{self.repo_path}/DeepSP_CNN_model/Conv1D_regression{name}.json"
            h5_path = f"{self.repo_path}/DeepSP_CNN_model/Conv1D_regression_{name}.h5"

            if not os.path.exists(json_path) or not os.path.exists(h5_path):
                raise RuntimeError(f"Missing DeepSP model: {name}")

            model = model_from_json(open(json_path).read())
            model.load_weights(h5_path)
            model.compile(optimizer="adam", loss="mae")

            self.deepsp_models[name] = model

        if len(self.deepsp_models) != 3:
            raise RuntimeError("All 3 DeepSP models are required")

        # Scaler (REQUIRED)
        scaler_path = f"{self.repo_path}/DeepViscosity_scaler/DeepViscosity_scaler.save"
        if not os.path.exists(scaler_path):
            raise RuntimeError("Missing scaler")

        self.scaler = joblib.load(scaler_path)

        # Ensemble models (REQUIRED: 102)
        model_dir = f"{self.repo_path}/DeepViscosity_ANN_ensemble_models"

        for i in range(102):
            json_path = f"{model_dir}/ANN_logo_{i}.json"
            h5_path = f"{model_dir}/ANN_logo_{i}.h5"

            if not os.path.exists(json_path) or not os.path.exists(h5_path):
                raise RuntimeError(f"Missing ANN model {i}")

            model = model_from_json(open(json_path).read())
            model.load_weights(h5_path)
            model.compile(optimizer=Adam(0.0001))

            self.models.append(model)

        if len(self.models) != 102:
            raise RuntimeError("All 102 ANN models must be loaded")

    # ---------------- ONE HOT ---------------- #
    def _one_hot(self, seq):# use for seq to number 
        aa = "ACDEFGHIKLMNPQRSTVWY-"
        mapping = {a: i for i, a in enumerate(aa)}

        x = np.zeros((len(seq), len(aa)))

        for i, c in enumerate(seq):
            if c not in mapping:
                raise ValueError(f"Invalid amino acid: {c}")
            x[i, mapping[c]] = 1   #index value position add value1

        return x

    # ---------------- ANARCI ---------------- #
    def _anarci_align(self, seq, chain_type):
        with tempfile.TemporaryDirectory() as tmp:
            fasta = f"{tmp}/input.fasta"
            out = f"{tmp}/output"

            SeqIO.write(
                SeqRecord(Seq(seq.upper()), id="query"),
                fasta,
                "fasta"
            ) # for fsta file

            cmd = [
                "anarci",
                "-i", fasta,
                "-o", out,
                "-s", "imgt",
                "-r", chain_type,
                "--csv"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                raise RuntimeError(f"ANARCI failed:\n{result.stderr}")

            csv_file = out + ".csv"
            if not os.path.exists(csv_file):
                raise RuntimeError("ANARCI output missing")

            df = pd.read_csv(csv_file)# output csv to df

            if df.shape[1] == 0:
                raise RuntimeError("Empty ANARCI output")

            return df

    # ---------------- FEATURE EXTRACTION ---------------- #
    def _extract_features(self, h_df, l_df):
        # Combine IMGT positions (exact behavior)
        h_seq = ''.join(h_df.iloc[0].fillna('-'))
        l_seq = ''.join(l_df.iloc[0].fillna('-'))

        combined = h_seq + l_seq

        if len(combined) != 272:
            raise RuntimeError(f"Expected 272 positions, got {len(combined)}")

        x = self._one_hot(combined)
        x = np.expand_dims(x, axis=0)  # (1, 272, 21)

        features = []

        for name in ["SAPpos", "SCMpos", "SCMneg"]:
            pred = self.deepsp_models[name].predict(x, verbose=0)

            if pred.shape[1] != 10:
                raise RuntimeError(f"DeepSP {name} output invalid")

            features.extend(pred[0])

        if len(features) != 30:
            raise RuntimeError("DeepSP feature vector must be length 30")

        return np.array(features).reshape(1, -1)

    # ---------------- PREDICT ---------------- #
    def predict(self, heavy_chain, light_chain):
        if not heavy_chain or not light_chain:
            raise ValueError("Both heavy and light chains are required")

        # 1. ANARCI alignment
        h_df = self._anarci_align(heavy_chain, "heavy")
        l_df = self._anarci_align(light_chain, "light")

        # 2. Feature extraction
        features = self._extract_features(h_df, l_df)

        # 3. Scaling
        features_scaled = self.scaler.transform(features) # for model input scale

        # 4. Ensemble prediction useed ann 102mo0de;
        preds = []
        for model in self.models:
            pred = model.predict(features_scaled, verbose=0)[0][0]
            preds.append(pred)

        prob = np.mean(preds)

        # 5. Convert to viscosity
        viscosity = 10 + prob * 30

        return float(viscosity)


predictor = ViscosityPredictor()