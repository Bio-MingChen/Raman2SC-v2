# CELL 0 [markdown]
          +-------------------+         +-------------------+
          |                   |         |                   |
          |  Single-Cell Data  |         |   Raman Data      |
          |      (x_ref)       |         |     (x_raman)     |
          +---------+---------+         +---------+---------+
                    |                             |
        +-----------v-----------+     +-----------v-----------+
        |      Ref VAE (宸茶缁?   |     |       Raman VAE        |
        | Encoder + Decoder      |     | Encoder + Decoder      |
        +-----------+-----------+     +-----------+-----------+
                    |                             |
              +-----v------+              +------v------+
              |   z_ref    |              |   z_raman   |
              +-----+------+              +------+------+
                    |                            |
                    +-------------+--------------+
                                  |
                           +------v------+
                           | Discriminator|  <--- 鍒ゅ埆鏄惁鏉ヨ嚜ref or raman
                           +------+-------+
                                  |
                +-----------------+-----------------+
                |                                   |
     +----------v----------+           +-----------v-----------+
     |  缁嗚優绫诲瀷鍒嗙被鍣?(宸茶缁?  |           |  Raman 鏁版嵁閲嶅缓 (Decoder) |
     +---------------------+           +----------------------+


# CELL 1 [code]
import scanpy as sc
from torch.utils.data import TensorDataset, DataLoader
import torch

# CELL 2 [code]
# 妫€鏌ユ槸鍚︽湁鍙敤鐨?GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
# 鎵撳嵃璁惧淇℃伅
print(f"Using device: {device}")

# 鍒涘缓涓€涓紶閲忓苟灏嗗叾绉诲姩鍒版寚瀹氳澶嘰ntensor = torch.randn(3, 3).to(device)
print(tensor)

# CELL 3 [code]
###### BELOW are more models and more losses

# we now have information regarding the spatiality of the VIM gene

import os


# no tesnorflow, being very uncooperative
import torch
# import torchvision
# import torchsummary

import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


import numpy as np
import scipy as sp
import pandas as pd

import sys
sys.path.extend([".", ".."])

def turn_on_model(model):
    for param in model.parameters():
        param.requires_grad = True
        
def turn_off_model(model):
    for param in model.parameters():
        param.requires_grad = False

        
# better new arch    
class StandardEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=512):
        super(StandardEncoder, self).__init__()
        self.part1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  # x tra here
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),  # x tra end 
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
        )
        
        self.to_mean = nn.Linear(hidden_dim, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim, latent_dim)
        
        self.latent_dim = latent_dim
    
    def forward(self, x):
        x = self.part1(x)
        return self.to_mean(x), self.to_logvar(x)
    
    
class StandardDecoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=512, no_final_relu=False):
        super(StandardDecoder, self).__init__()
     
        
        # this is for the case of non-zinb
        if no_final_relu:
            self.net = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),  # x tra start here
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),  # xtra end
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
#                 nn.ReLU(),  # do the activation here
            )
            
        else:
            self.net = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),  # x tra start here
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),  # xtra end
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
                nn.ReLU(),  # do the activation here
            )
        
        self.latent_dim = latent_dim
    
    # returns a tuple regardless
    def forward(self, x):
        res = self.net(x)
        return res


class Discriminator(nn.Module):
    def __init__(self, latent_dim, spectral=True, end_dim=2):
        super(Discriminator, self).__init__()
        if spectral:
            self.net = nn.Sequential(
                U.spectral_norm(nn.Linear(latent_dim, 1<<6)),
                nn.ReLU(),
                U.spectral_norm(nn.Linear(1<<6, 1<<5)),
                nn.ReLU(),
                U.spectral_norm(nn.Linear(1<<5, 1<<5)),
                nn.ReLU(),
                U.spectral_norm(nn.Linear(1<<5, end_dim)),
    #             nn.Sigmoid(), just do w logits for now 
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(latent_dim, 1<<6),
                nn.ReLU(),
                nn.Linear(1<<6, 1<<5),
                nn.ReLU(),
                nn.Linear(1<<5, 1<<5),
                nn.ReLU(),
                nn.Linear(1<<5, end_dim),
    #             nn.Sigmoid(), just do w logits for now 
            )
        
    def forward(self, x):
        return self.net(x)
    
class VAE(nn.Module):
    def __init__(self, encoder, decoder, is_vae=True, use_latent_norm=True):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.is_vae = is_vae
        self.latent_normalizer = (
            nn.BatchNorm1d(self.encoder.latent_dim) if 1
            else nn.Sigmoid()
        )
        self.use_latent_norm = use_latent_norm 
        
    def reparam_trick(self, mean, logvar):
        sigma = torch.exp(0.5*logvar)
        eps = torch.randn_like(sigma)
        res = (
            mean + eps*sigma if self.is_vae
            else mean
        )
        return res
        # below is stupid garbo
        # this was a massive BUG
#         return mean + eps*sigma
#         return mean  # for non variational version, uncomment
    
    def get_latent(self, x):
        mean, logvar = self.encoder(x)
            
        if self.use_latent_norm:
            mean = self.latent_normalizer(mean)
            logvar = self.latent_normalizer(logvar)
            
        return self.reparam_trick(mean, logvar)
    
    def forward(self, x, noise_latent_lambda=0.):
        mean, logvar = self.encoder(x)
        
        if self.use_latent_norm:
            if 0:
                mean = self.latent_normalizer(mean)
                logvar = self.latent_normalizer(logvar)
            
            
                latent = self.reparam_trick(mean, logvar)
            else:
                latent = self.reparam_trick(mean, logvar)
                latent = self.latent_normalizer(latent)
        else:
            latent = self.reparam_trick(mean, logvar)
            
        if noise_latent_lambda:
            latent = latent + noise_latent_lambda*torch.randn_like(latent)
            
        
#         m_bar, pi, theta = self.decoder(latent)
        # return everything , last 3 are mean, logvar, latent

        recon_x = self.decoder(latent)
        return recon_x, mean, logvar, latent
       
    
#### LOSS FUNCTIONS
# gives option for VAE type of loss
def old_mse_loss(x, recon_x, weights=None):
    return F.mse_loss(
        recon_x, x, 
    ) # * 1e5  cm: 鍘绘帀杩欎釜涔樻暟锛屼笉鐒秎oss浼氬彉寰楅潪甯稿ぇ


def discrim_criter(pred, true):
    return F.binary_cross_entropy_with_logits(
        pred, true,
    ) # * 1e5


def weighted_mse(a, b, weights=None):
    return (
        torch.sum(((a-b)**2)*weights) if (weights is not None)
        else F.mse_loss(a, b)
    ) # * 1e5

def old_vae_loss(x, recon_x, mean, logvar, weights=None, this_lambda=0.,):
    if weights is None:
        bce = F.mse_loss(
            recon_x, x, 
        ) # * 1e5  # poss comment out last part 
    else:
        bce = weighted_mse(recon_x, x, weights=weights)
   
    kl_div = -.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    
    return bce + this_lambda*kl_div

def discrim_loss(pred, true):
    return F.binary_cross_entropy_with_logits(
        pred, true, 
    ) # * 1e5

# don't do any requires_grad stuff in here
def adv_vae_loss(
    x, recon_x, 
    mean, logvar, discrim_preds,
    alpha, beta, weights=None,
):
    vae_part_loss = old_vae_loss(
        x, recon_x, mean, logvar, weights=weights)
    source_label = [1., 0.]
    target_label = [0., 1.]
    discrim_labels = torch.tensor([source_label] * x.shape[0]).to(device)
    total_discrim_loss = F.binary_cross_entropy_with_logits(
        discrim_preds, discrim_labels, 
    ) # * 1e5
    
    discrim_part_loss = beta * total_discrim_loss
    return alpha * vae_part_loss + discrim_part_loss, vae_part_loss, total_discrim_loss

# CELL 4 [code]
import scanpy as sc
abc_ad = sc.read_h5ad("./data/abc_B_lineage_processed.h5ad")
abc_ad

# CELL 5 [code]
abc_ad.obs["cell_type"].value_counts()

# CELL 6 [code]
abc_ad.var_names[abc_ad.var['highly_variable']]

# CELL 7 [code]
raman_ad = sc.read_h5ad("./data/raman_B.h5ad")
raman_ad

# CELL 8 [code]
# 涓嶈褰掍竴鍖栨媺鏇肩殑宄板浘锛屼笉鐒剁粨鏋滅殑umap鍥句細鐗瑰埆绂绘暎
# raman_ad.layers["data"] = raman_ad.X.copy()
# sc.pp.scale(raman_ad)

# CELL 9 [code]
raman_ad.obs.head()

# CELL 10 [code]
import anndata
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 鍋囪 abc_ad 鏄竴涓?AnnData 瀵硅薄
# 鑾峰彇 adata.X 鏁版嵁鍜岀粏鑳炵被鍨嬫爣绛綷nhighly_variable_genes = abc_ad.var_names[abc_ad.var['highly_variable']]

# data = abc_ad.X
data = abc_ad[:, highly_variable_genes].X
labels = abc_ad.obs['cell_type']

# 灏嗙粏鑳炵被鍨嬫爣绛捐浆鎹负鏁板€肩紪鐮乗nlabel_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# 灏嗘暟鎹垎鎴愯缁冮泦鍜屾祴璇曢泦
train_data, test_data, train_labels, test_labels = train_test_split(data, labels_encoded, test_size=0.2, random_state=42)

# 灏嗘暟鎹浆鎹负 PyTorch 寮犻噺
train_tensor = torch.tensor(train_data, dtype=torch.float32)
test_tensor = torch.tensor(test_data, dtype=torch.float32)
train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

# 鍒涘缓 TensorDataset
train_dataset = TensorDataset(train_tensor, train_labels_tensor)
test_dataset = TensorDataset(test_tensor, test_labels_tensor)

# 鍒涘缓 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

train_loader.dataset.tensors[0].shape

# CELL 11 [code]

epochs = 30  
input_dim = train_data.shape[1]

# ODD FINDING - DEEPER MAKES LATENT SPACE LOOK BETTER
# ALSO LARGER HIDDEN DIM BY FACTOR OF 2
ref_vae = VAE(
    StandardEncoder(input_dim, 1<<7, hidden_dim=1<<11),  # hidden was 1<<10
    StandardDecoder(input_dim, 1<<7, hidden_dim=1<<11,),
    is_vae=False,
    use_latent_norm=True,  # was True for all else 
).to(device)

ref_vae_opt = optim.Adam( #5-5 is 181
    ref_vae.parameters(), lr=1e-3, #betas=(.5,.999), 5e get .074, after 10, same is avg .057
)

ref_vae.to(device)

epoch_losses = []
 
# got to < .115 avg loss after 150 epochs
# best was ====> Epoch: 1000 Average loss: 0.0979890559
for epoch in range(1, epochs+1):
    epoch_loss = 0.0
    for _id, [batch,labels] in enumerate(train_loader):
        batch = batch.to(device)
        labels = labels.to(device)
        ref_vae_opt.zero_grad()

        recon_x, mean, logvar, latent = ref_vae(batch)
        batch_loss = old_vae_loss(
            batch, recon_x, mean, logvar, weights=None,
        )


        batch_loss.backward()
        epoch_loss += batch_loss.item()
        ref_vae_opt.step()
        if not (_id % 10):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}'.format(
                epoch, 
                _id * len(batch), 
                len(train_loader.dataset),
                25. * _id / len(train_loader),
                batch_loss.item() / len(batch),
            ))

    print('====> Epoch: {} Average loss: {:.10f}'.format(
                epoch, epoch_loss / len(train_loader.dataset)))

    epoch_losses.append(epoch_loss)


# CELL 12 [code]
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn

# -----------------------------
# 1. 鍒╃敤宸茶缁冪殑 VAE 鎻愬彇璁粌鏁版嵁鐨?latent 琛ㄧず
# -----------------------------
latent_dim = 1 << 7  # 128
# 灏?VAE 妯″瀷绉诲埌 CPU 涓婅繘琛岀壒寰佹彁鍙栵紙濡傛灉浣犵殑鏁版嵁閲忎笉澶э紝涔熷彲浠ョ洿鎺ョ敤 GPU锛塡nref_vae = ref_vae.to('cpu')

# 杩欓噷鐩存帴浣跨敤涔嬪墠鏋勯€犵殑 train_tensor锛堝叾鍐呭涓?train_data 杞崲鍚庣殑 torch.tensor锛塡norig_cells_dataset = train_tensor.to('cpu')

# 閫氳繃 VAE 寰楀埌閲嶆瀯缁撴灉銆佸潎鍊笺€乴ogvar 浠ュ強 latent 琛ㄧず锛堟澶勫彧鍙?latent锛塡n_, _, _, latent = ref_vae(orig_cells_dataset)
latent = latent.detach().numpy()  # 灏?latent 琛ㄧず杞崲涓?NumPy 鏁扮粍

# -----------------------------
# 2. 鏋勯€犵粏鑳炵被鍨嬪垎绫诲櫒鎵€闇€鐨勬暟鎹甛n# -----------------------------
# 鍥犱负浣犲凡鐢?LabelEncoder 澶勭悊杩囨爣绛撅紝杩欓噷鐩存帴浣跨敤 train_labels
# 璁＄畻绫诲埆鏁帮紙鍒嗙被鍣ㄧ殑杈撳嚭缁村害锛塡nfinal_output_shape = len(np.unique(train_labels))
print("缁嗚優绫诲瀷绫诲埆鏁?", final_output_shape)

# 鏋勯€犲垎绫诲櫒锛圖iscriminator锛夛紝杈撳叆缁村害涓?latent_dim锛岃緭鍑虹淮搴︿负绫诲埆鏁癨ncelltype_classifier = Discriminator(latent_dim, end_dim=final_output_shape).to(device)
celltype_classifier_opt = optim.Adam(celltype_classifier.parameters(), lr=1e-3)

# 璁剧疆鎵规澶у皬锛岃繖閲屼笌鏂囩尞浠ｇ爜淇濇寔涓€鑷达紙1 << 5 = 32锛塡nbatch_size = 1 << 5

# 鏋勯€犲垎绫诲櫒璁粌鎵€闇€鐨勬爣绛炬暟缁刓n# 姝ゅ train_labels 宸茬粡鏄?numpy 鏁扮粍锛屽涓嶆槸璇风敤 np.array(train_labels) 杞崲
celltype_train_list = train_labels

# 鏋勯€?DataLoader锛屽皢 latent 琛ㄧず鍜屽搴旂殑鏍囩鎵撳寘
celltype_data_loader = DataLoader(
    TensorDataset(
        torch.from_numpy(latent),
        torch.tensor(celltype_train_list, dtype=torch.long)
    ),
    batch_size=batch_size,
    shuffle=True, num_workers=1, pin_memory=True,
)

# -----------------------------
# 3. 璁剧疆绫诲埆鏉冮噸涓庢崯澶卞嚱鏁帮紙澶勭悊绫诲埆涓嶅钩琛★級
# -----------------------------
num_cells = len(celltype_train_list)
# 璁＄畻姣忎釜绫诲埆鐨勬潈閲嶏細鎬绘牱鏈暟闄や互璇ョ被鍒牱鏈暟
class_weights = torch.tensor([
    float(num_cells) / np.sum(celltype_train_list == class_label)
    for class_label in range(final_output_shape)
], dtype=torch.float32).to(device)

# 瀹氫箟浜ゅ弶鐔垫崯澶卞嚱鏁帮紝骞朵紶鍏ョ被鍒潈閲峔ncriter = nn.CrossEntropyLoss(weight=class_weights)

# -----------------------------
# 4. 璁粌鍒嗙被鍣細鍒╃敤 latent 琛ㄧず杩涜缁嗚優绫诲瀷棰勬祴
# -----------------------------
epochs = 32
celltype_classifier = celltype_classifier.to(device)

for epoch in range(epochs):
    epoch_loss = 0.0
    for _id, (this_batch, this_label) in enumerate(celltype_data_loader):
        this_batch = this_batch.to(device)
        this_label = this_label.to(device)
        celltype_classifier_opt.zero_grad()

        # 鍓嶅悜浼犳挱锛氳緭鍏?latent 琛ㄧず寰楀埌绫诲埆寰楀垎
        predicted_labels = celltype_classifier(this_batch.float())
        # 璁＄畻鎹熷け
        this_batch_loss = criter(predicted_labels, this_label)
        # 鍙嶅悜浼犳挱涓庡弬鏁版洿鏂癨n        this_batch_loss.backward()
        epoch_loss += this_batch_loss.item()
        celltype_classifier_opt.step()

    # 杩欓噷璁＄畻鐨勬槸鏁翠釜璁粌闆嗕笂姣忎釜鏍锋湰鐨勫钩鍧囨崯澶盶n    avg_loss = epoch_loss / len(celltype_data_loader.dataset)
    print('====> Epoch: {} Average loss: {:.10f}'.format(epoch+1, avg_loss))

# -----------------------------
# 5. 璇勪及鍒嗙被鍣ㄥ湪璁粌闆嗕笂鐨勫噯纭巼
# -----------------------------
celltype_classifier = celltype_classifier.to('cpu')
orig_pred_labels = celltype_classifier(torch.from_numpy(latent)).detach().numpy()
orig_pred_labels = np.argmax(orig_pred_labels, axis=1)
num_final_correct = np.sum(orig_pred_labels == celltype_train_list)
print(f'final_accuracy: { num_final_correct / float(num_cells) }')


# CELL 13 [code]
from sklearn.model_selection import train_test_split

# 鍙栧嚭鏁版嵁鍜屾爣绛綷nraman_X = raman_ad.X
raman_y = label_encoder.transform(raman_ad.obs['cell_type'].values)

# 鎷嗗垎锛?0% 璁粌锛?0% 娴嬭瘯
raman_X_train, raman_X_test, raman_y_train, raman_y_test = train_test_split(
    raman_X, raman_y, test_size=0.2, random_state=42, stratify=raman_y  # 鎸夌粏鑳炵被鍨嬪垎灞俓n)
# 鑾峰彇 Raman 鏁版嵁鐨?index锛堢粏鑳炲悕瀛楋級
raman_index = raman_ad.obs.index

# 鎷嗗垎 index锛屼繚鎸?stratify 鎸夌収缁嗚優绫诲瀷鍒嗗眰
raman_index_train, raman_index_test = train_test_split(
    raman_index,
    test_size=0.2,
    random_state=42,
    stratify=raman_y
)

# 鍐嶆彁鍙栨暟鎹甛n# raman_X_train = raman_ad[raman_index_train].X
# raman_X_test = raman_ad[raman_index_test].X

# Raman Train DataLoader
raman_train_loader = DataLoader(
    TensorDataset(
        torch.from_numpy(raman_X_train).float(),
        torch.from_numpy(raman_y_train).long()
    ),
    batch_size=batch_size,
    shuffle=True,
    num_workers=1,
    pin_memory=True,
)

# Raman Test DataLoader
raman_test_loader = DataLoader(
    TensorDataset(
        torch.from_numpy(raman_X_test).float(),
        torch.from_numpy(raman_y_test).long()
    ),
    batch_size=batch_size,
    shuffle=False,
    num_workers=1,
    pin_memory=True,
)


# CELL 14 [code]
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import scipy  # 鑻ラ渶瑕佸垽鏂█鐤忕被鍨嬶紝鍙敤 scipy.sparse
import scanpy as sc

# ======================================================
# 1. 璇诲彇骞堕澶勭悊鏁版嵁
# ======================================================

# 1.1 鍙傝€冩ā鎬佹暟鎹紙渚嬪 RNA 鏁版嵁锛塡n# abc_ad = sc.read_h5ad("path/to/reference_data.h5ad")  # 璇锋浛鎹负瀹為檯鏂囦欢璺緞
# 閫夋嫨楂樺彉鍩哄洜
highly_variable_genes = abc_ad.var_names[abc_ad.var['highly_variable']]
# 鎻愬彇鏁版嵁锛屽苟纭繚鏁版嵁涓?dense 鏍煎紡
train_feature = abc_ad[:, highly_variable_genes].X
# if hasattr(train_feature, "todense"):
    # train_feature = train_feature.todense()
# train_feature = np.array(train_feature) / 0.1  # 褰掍竴鍖栧鐞嗭紙闄や互0.1锛塡n
# 1.2 鎷夋浖鏁版嵁
# raman_ad = sc.read_h5ad("./data/raman_B.h5ad")  # 鏂囦欢璺緞鏍规嵁瀹為檯鎯呭喌璋冩暣
# train_feature_raman = raman_ad.X
# if hasattr(train_feature_raman, "todense"):
    # train_feature_raman = train_feature_raman.todense()
# train_feature_raman = np.array(train_feature_raman) / 0.1

# 1.3 鏋勯€犵粏鑳炵被鍨嬫爣绛炬槧灏刓n# 鍋囪鍙傝€冩暟鎹笌鎷夋浖鏁版嵁鍏变韩鐩稿悓鐨勭粏鑳炵被鍨嬫爣绛綷n# raman_labels_encoded = label_encoder.transform(raman_ad.obs['cell_type'].values)
# raman_labels_encoded

# ======================================================
# 2. 鏋勯€?DataLoader
# ======================================================

# 璁惧畾鎵规澶у皬锛屼笌鍘熸枃涓?"1<<5" 涓€鑷达紝鍗?2
batch_size = 1 << 5  # 32

# DataLoader锛氬弬鑰冩ā鎬佹暟鎹紙杩欓噷鍙娇鐢ㄦ暟鎹紝涓嶇敤鏍囩锛塡nref_data_loader = DataLoader(
    TensorDataset(torch.from_numpy(train_feature).float()),
    batch_size=batch_size,
    shuffle=True,
    num_workers=1,
    pin_memory=True,
)

# DataLoader锛氭媺鏇兼暟鎹紝鍚屾椂鍖呭惈鍏夎氨鏁版嵁鍜岀粏鑳炵被鍨嬫爣绛綷n# raman_data_loader = DataLoader(
#     TensorDataset(
#         torch.from_numpy(train_feature_raman).float(),
#         torch.from_numpy(raman_labels_encoded).long(),
#     ),
#     batch_size=batch_size,
#     shuffle=True,
#     num_workers=1,
#     pin_memory=True,
# )

# ======================================================
# 3. 瀹氫箟妯″瀷鍜屼紭鍖栧櫒
# ======================================================

latent_dim = 1 << 7  # 128
# 鍙傝€冩ā鎬?VAE锛堝亣瀹氫綘宸茬粡璁粌濂藉苟淇濆瓨鍦ㄥ彉閲?ref_vae 涓級
ref_vae = ref_vae.to(device)

# 鎷夋浖妯℃€?VAE锛氳緭鍏ョ淮搴﹁涓哄弬鑰冩暟鎹殑鐗瑰緛鏁帮紙渚嬪鑻?input_cell_dim 涓?30锛塡n# input_cell_dim = train_feature.shape[1]
# input_cell_dim = train_feature_raman.shape[1]
input_cell_dim = raman_X.shape[1]
raman_vae = VAE(
    StandardEncoder(input_cell_dim, latent_dim, hidden_dim=1<<11),  # hidden_dim=2^11=2048
    StandardDecoder(input_cell_dim, latent_dim, hidden_dim=1<<11, no_final_relu=True),
    is_vae=False,
    use_latent_norm=True,
).to(device)
raman_opt = optim.Adam(raman_vae.parameters(), lr=5e-5)
raman_vae.to(device)

# 鍒ゅ埆鍣細鐢ㄤ簬鍖哄垎鍙傝€冧笌鎷夋浖鐨?latent 鍒嗗竷
raman_discrim = Discriminator(latent_dim).to(device)
raman_discrim_opt = optim.Adam(raman_discrim.parameters(), lr=4e-3)

# 浠ヤ笅鍑犱釜瓒呭弬鏁帮紙鍙牴鎹渶瑕佽皟鑺傦級
alpha = 1e0       # 瀵规姉鎹熷け涓噸鏋?姝ｅ垯椤圭殑鏉冮噸
beta = 3e-4       # 鍙︿竴涓崯澶遍」鐨勬潈閲嶏紙渚嬪 KL 鏁ｅ害绛夛級
raman_beta = 5e1  # 缁嗚優绫诲瀷鍒嗙被鎹熷け鐨勬潈閲峔n
# 鍋囪缁嗚優绫诲瀷鍒嗙被鍣ㄥ凡瀹氫箟骞堕鍏堣缁冨ソ
celltype_classifier = celltype_classifier.to(device)

print("begin_raman_latent_train")

# ======================================================
# 4. 瀵规姉璁粌 鈥斺€?璁粌鎷夋浖 VAE 浠ヤ笌鍙傝€?VAE 瀵归綈锛屽悓鏃跺姞鍏ョ粏鑳炵被鍨嬬洃鐫n# ======================================================

# 璁惧畾鏄惁閲嶆柊璁粌鎷夋浖 VAE锛堣嫢涓?0 鍒欑洿鎺ュ姞杞介璁粌妯″瀷锛塡nneed_retrain_raman = 1   # 鑻ラ渶瑕侀噸鏂拌缁冿紝璁句负1锛涘惁鍒欒涓?

if need_retrain_raman:
    epochs = 30  # 鍙牴鎹渶瑕佽皟鏁磋缁冭疆鏁癨n    for epoch in range(1, epochs + 1):
        # 绱鍚勯儴鍒嗘崯澶盶n        discrim_epoch_loss = 0.0
        vae_part_epoch_loss = 0.0
        raman_vae_epoch_loss = 0.0
        celltype_part_epoch_loss = 0.0
        print(f"begin epoch {epoch}")

        # 鍚屾椂閬嶅巻鍙傝€冩暟鎹拰鎷夋浖鏁版嵁鐨?DataLoader
        for _id, ([ref_batch,], [raman_batch, raman_celltypes]) in enumerate(
            zip(ref_data_loader, raman_train_loader)
        ):
            # 娓呴浂姊害
            raman_opt.zero_grad()
            raman_discrim_opt.zero_grad()
            # 灏嗘暟鎹Щ鍒?device 涓奬n            ref_batch = ref_batch.to(device)
            raman_batch = raman_batch.to(device)
            raman_celltypes = raman_celltypes.to(device)

            # 閫氳繃鍙傝€?VAE 鍜屾媺鏇?VAE 鍒嗗埆鑾峰緱 latent 琛ㄧず锛坉etach 闃叉姊害鍙嶄紶鍒?ref_vae /鏃╂湡鍙傛暟锛塡n            ref_encoded = ref_vae.get_latent(ref_batch).detach()
            raman_encoded = raman_vae.get_latent(raman_batch).detach()

            # 鎷兼帴涓や釜妯℃€佺殑 latent 琛ㄧず锛屽苟鏋勯€犲垽鍒櫒鐨勬爣绛綷n            source_label, target_label = [1., 0.], [0., 1.]  # 閲囩敤 one-hot 缂栫爜
            encodeds = torch.cat((ref_encoded, raman_encoded), axis=0)
            discrim_labels = torch.tensor(
                [source_label] * ref_encoded.shape[0] + [target_label] * raman_encoded.shape[0]
            ).to(device)

            # 鍒ゅ埆鍣ㄥ墠鍚戜紶鎾強鎹熷け璁＄畻
            pred_discrim_labels = raman_discrim(encodeds)
            batch_discrim_loss = discrim_loss(pred_discrim_labels, discrim_labels)
            batch_discrim_loss.backward()
            discrim_epoch_loss += batch_discrim_loss.item()
            raman_discrim_opt.step()

            # ================================
            # 绗簩閮ㄥ垎锛氭洿鏂版媺鏇?VAE
            # ================================
            # 鍐荤粨鍒ゅ埆鍣ㄥ弬鏁帮紝闃叉鍏舵洿鏂癨n            for param in raman_discrim.parameters():
                param.requires_grad = False

            # 閫氳繃鎷夋浖 VAE 寰楀埌閲嶆瀯缁撴灉鍙?latent 琛ㄧず
            recon_raman_batch, raman_batch_mean, raman_batch_logvar, raman_batch_latent = raman_vae(raman_batch)
            raman_vae_discrim_preds = raman_discrim(raman_batch_latent)

            # 璁＄畻瀵规姉寮?VAE 鎹熷け锛堣嚜瀹氫箟鎹熷け鍑芥暟 adv_vae_loss锛塡n            raman_vae_batch_loss, vae_part_batch_loss, _ = adv_vae_loss(
                raman_batch.detach(), recon_raman_batch,
                raman_batch_mean, raman_batch_logvar, raman_vae_discrim_preds,
                alpha, beta,
            )

            # ================================
            # 鍔犲叆缁嗚優绫诲瀷鍒嗙被鎹熷け
            # ================================
            raman_celltype_preds = celltype_classifier(raman_batch_latent)
            raman_celltype_loss = criter(raman_celltype_preds, raman_celltypes)
            # 鎬绘崯澶?= 瀵规姉寮?VAE 鎹熷け + 缁嗚優绫诲瀷鍒嗙被鎹熷け锛堜箻浠ヨ緝澶ф潈閲嶏級
            raman_vae_batch_loss = raman_vae_batch_loss + raman_beta * raman_celltype_loss

            # 鍙嶅悜浼犳挱骞舵洿鏂版媺鏇?VAE 鍙傛暟
            raman_vae_batch_loss.backward()
            celltype_part_epoch_loss += raman_celltype_loss.item()
            raman_vae_epoch_loss += raman_vae_batch_loss.item()
            vae_part_epoch_loss += vae_part_batch_loss.item()
            raman_opt.step()

            # 鎭㈠鍒ゅ埆鍣ㄥ弬鏁扮殑姊害璁＄畻鐘舵€乗n            for param in raman_discrim.parameters():
                param.requires_grad = True

            # 姣忛殧涓€瀹?batch 杈撳嚭涓€娆℃棩蹇梊n            if _id % 500 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLosses: {:.6f} {:.6f}'.format(
                    epoch,
                    _id * len(raman_batch),
                    len(raman_train_loader.dataset),
                    100. * _id / len(raman_train_loader),
                    batch_discrim_loss.item() / len(raman_batch),
                    raman_vae_batch_loss.item() / len(raman_batch)
                ))

        # 姣忎釜 epoch 缁撴潫鍚庯紝鎵撳嵃鍚勯儴鍒嗗钩鍧囨崯澶盶n        print('====> Epoch: {} Average adv vae loss: {:.10f}'.format(
            epoch, raman_vae_epoch_loss / len(raman_train_loader.dataset)))
        print('====> Epoch: {} Average vae part loss: {:.10f}'.format(
            epoch, vae_part_epoch_loss / len(raman_train_loader.dataset)))
        print('====> Epoch: {} Average celltype part loss: {:.10f}'.format(
            epoch, celltype_part_epoch_loss / len(raman_train_loader.dataset)))
        print('====> Epoch: {} Average discrim vae loss: {:.10f}'.format(
            epoch, discrim_epoch_loss / len(raman_train_loader.dataset)))
        print(f"end epoch {epoch}")
    print("end_raman_latent_train")
else:
    # 鑻ユ棤闇€閲嶆柊璁粌锛屽垯鍔犺浇棰勮缁冨ソ鐨勬媺鏇?VAE 妯″瀷鍙傛暟
    raman_vae.load_state_dict(torch.load('raman_vae.pt'))


# CELL 15 [code]
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import scanpy as sc

train_feature_ad = abc_ad[:, highly_variable_genes]
# 鍋囪锛歕n#   - 浣犵殑鍗曠粏鑳炴暟鎹瓨鍌ㄥ湪 abc_ad 涓紝
#   - 鍙傝€?VAE (ref_vae) 鍜?Raman VAE (raman_vae) 宸茶缁冨ソ锛孿n#   - celltype_classifier銆乺aman_vae.encoder銆乺ef_vae.decoder 鍧囧凡瀹氫箟锛孿n#   - 璁惧鍙橀噺 device 宸插畾涔夛紙渚嬪 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")锛夈€俓n
# ---------------------------------------------------------------------
# 1. 鏋勯€犵敤浜庢瘮杈冪殑妯″瀷锛氳縼绉?VAE
# ---------------------------------------------------------------------
# 杩欓噷鏋勯€犵殑 transfer_vae 閲囩敤 Raman VAE 鐨勭紪鐮佸櫒鍜屽弬鑰?VAE 鐨勮В鐮佸櫒锛孿n# 鐢ㄤ簬灏嗏€淩aman鈥濇暟鎹浆鎹㈠埌鍙傝€冩ā鎬佺┖闂翠腑銆俓n
#杩欎釜 杩佺Щ VAE锛歕n
#缂栫爜鍣細鐢ㄧ殑鏄?Raman VAE 鐨?encoder锛屾彁鍙?Raman 鏁版嵁鐨勬綔灞傝〃绀恒€俓n
#瑙ｇ爜鍣細鐢ㄧ殑鏄?鍙傝€冨崟缁嗚優 VAE 鐨?decoder锛屾妸娼滃眰琛ㄧず杞崲鍥炲崟缁嗚優 RNA 鏁版嵁鐨勭┖闂淬€俓n
#馃敆 鐩殑锛歕n
#鎶?Raman 鏁版嵁 缁忚繃缂栫爜鍣?鈫?寰楀埌 latent锛堟綔灞傝〃绀猴級

#鍐嶇敤 鍗曠粏鑳炵殑 decoder 閲嶆瀯 鈫?鐢熸垚涓€涓被浼煎崟缁嗚優 RNA 鐨勮〃杈剧煩闃点€俓n
#杩欏氨鏄皢 Raman 鏁版嵁鈥滅炕璇戔€濆埌鍗曠粏鑳炴暟鎹殑绌洪棿銆俓n
transfer_vae = VAE(
    raman_vae.encoder,
    ref_vae.decoder,
    is_vae=False,
    use_latent_norm=True,
)
transfer_vae = transfer_vae.to('cpu')
ref_vae = ref_vae.to('cpu')

# ---------------------------------------------------------------------
# 2. 浠庡崟缁嗚優鏁版嵁鏋勯€犺緭鍏ュ紶閲? 
#    杩欓噷鍥犱负浣犵殑鏁版嵁閮藉湪 abc_ad 涓紝鎵€浠ユ垜浠洿鎺ョ敤 abc_ad.X 浣滀负杈撳叆  
# ---------------------------------------------------------------------
# 濡傛灉浣犵殑鏁版嵁闇€瑕佸綊涓€鍖栨垨杞崲锛屽彲鍦ㄨ繖閲岃皟鏁达紝渚嬪闄や互涓€涓瘮渚嬪洜瀛怽norig_cells_dataset = (
    torch.from_numpy(train_feature).float().to('cpu')
)
# 寰楀埌鍙傝€冮噸鏋勭粨鏋滐紙浣跨敤 ref_vae锛塡nrecon, _, _, _ = ref_vae(orig_cells_dataset)

# 濡傛灉浣犳湁鐙珛鐨?Raman 鏁版嵁锛岃浣跨敤鐩稿簲鐨勫彉閲忥紱  
# 鍚﹀垯锛岃繖閲屾垜浠篃浣跨敤 abc_ad.X 鏉ユā鎷熲€滆縼绉烩€濇暟鎹殑杈撳叆
orig_cells_dataset_raman = (
    # torch.from_numpy(train_feature_raman).float().to('cpu')
    torch.from_numpy(raman_X).float().to('cpu')
)
recon_raman, _, _, _ = transfer_vae(orig_cells_dataset_raman)

# ---------------------------------------------------------------------
# 3. 灏嗛噸鏋勭粨鏋滆浆鎹负 AnnData 瀵硅薄锛屽苟璁剧疆娉ㄩ噴淇℃伅
# ---------------------------------------------------------------------
recon_adata = sc.AnnData(recon.detach().numpy())
# 鐢ㄥ師濮嬬殑缁嗚優娉ㄩ噴瑕嗙洊锛岀‘淇濅袱鑰呭彲浠ュ榻? 
recon_adata.obs = train_feature_ad.obs.copy()

recon_adata_raman = sc.AnnData(recon_raman.detach().numpy())
recon_adata_raman.obs = raman_ad.obs.copy()

# 鍚堝苟涓や釜 AnnData 瀵硅薄锛孉nnData.concatenate 浼氬湪 .obs 涓坊鍔犱竴涓?"batch" 鍒楁爣璁版潵婧怽ntogether_recon = recon_adata.concatenate(recon_adata_raman)

# ---------------------------------------------------------------------
# 4. 闄嶇淮銆佽绠楅偦鍩熴€佸苟鐢熸垚 UMAP 鍥? 
# ---------------------------------------------------------------------
sc.pp.pca(together_recon, n_comps=30)
sc.pp.neighbors(together_recon, n_neighbors=30)
sc.tl.umap(together_recon)

# CELL 16 [code]
import numpy as np
import torch
import scanpy as sc

# ------------------------------------------------
# 1. 鏋勯€犺縼绉?VAE
# ------------------------------------------------
transfer_vae = VAE(
    raman_vae.encoder,
    ref_vae.decoder,
    is_vae=False,
    use_latent_norm=True,
).to('cpu')
ref_vae = ref_vae.to('cpu')

# ------------------------------------------------
# 2. 鍗曠粏鑳炴暟鎹浆鎹n# ------------------------------------------------
orig_cells_dataset = torch.from_numpy(train_feature).float().to('cpu')
recon_singlecell, _, _, _ = ref_vae(orig_cells_dataset)
recon_adata_singlecell = sc.AnnData(recon_singlecell.detach().numpy())
recon_adata_singlecell.obs = train_feature_ad.obs.copy()
recon_adata_singlecell.obs['batch'] = 'SingleCell'  # 娣诲姞 batch 鏍囩

# ------------------------------------------------
# 3. Raman 璁粌闆嗚浆鎹n# ------------------------------------------------
orig_cells_dataset_raman_train = torch.from_numpy(raman_X_train).float().to('cpu')
recon_raman_train, _, _, _ = transfer_vae(orig_cells_dataset_raman_train)
recon_adata_raman_train = sc.AnnData(recon_raman_train.detach().numpy())
recon_adata_raman_train.obs = raman_ad.obs.loc[raman_index_train].copy()  # 鎸?index 瀵归綈
recon_adata_raman_train.obs['batch'] = 'Raman_Train'  # 娣诲姞 batch 鏍囩

# ------------------------------------------------
# 4. Raman 娴嬭瘯闆嗚浆鎹n# ------------------------------------------------
orig_cells_dataset_raman_test = torch.from_numpy(raman_X_test).float().to('cpu')
recon_raman_test, _, _, _ = transfer_vae(orig_cells_dataset_raman_test)
recon_adata_raman_test = sc.AnnData(recon_raman_test.detach().numpy())
recon_adata_raman_test.obs = raman_ad.obs.loc[raman_index_test].copy()  # 鎸?index 瀵归綈
recon_adata_raman_test.obs['batch'] = 'Raman_Test'  # 娣诲姞 batch 鏍囩

# ------------------------------------------------
# 5. 鍚堝苟涓変釜 AnnData 瀵硅薄
# ------------------------------------------------
together_recon = recon_adata_singlecell.concatenate(
    recon_adata_raman_train,
    recon_adata_raman_test,
    batch_key=None  # 鍙栨秷鑷姩鐢熸垚 batch 鍒楋紝浣跨敤鑷畾涔夌殑 batch
)

# ------------------------------------------------
# 6. 闄嶇淮 + UMAP
# ------------------------------------------------
sc.pp.pca(together_recon, n_comps=30)
sc.pp.neighbors(together_recon, n_neighbors=30)
sc.tl.umap(together_recon)


# CELL 17 [code]
# ------------------------------------------------
# 7. 缁樺浘
# ------------------------------------------------
sc.pl.umap(together_recon, color=['batch', 'cell_type'],ncols=1,save="_Raman2SC.pdf")

# CELL 18 [code]
together_recon.obs.head()

# CELL 19 [code]
sc.pl.umap(together_recon[together_recon.obs["cell_type"] == "Pre B"],color=["batch"],title="Pre B distribution of Single cell and Raman transfered data",save="_Raman2SC_PreB.pdf")

# CELL 20 [code]
together_recon.write("./data/Raman2SC_four_normal_celltype_abc.h5ad")

# CELL 21 [code]
def test_vae(model, test_loader, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            recon_x, mean, logvar, latent = model(data)
            test_loss += F.mse_loss(recon_x, data, reduction='mean').item()

    test_loss /= len(test_loader.dataset)
    print(f'====> Test set loss: {test_loss:.4f}')
    return test_loss

# import umap
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt

# def visualize_latent_space(model, train_loader, test_loader, device):
#     model.eval()
#     train_latent_vectors = []
#     train_labels = []
#     test_latent_vectors = []
#     test_labels = []
    
#     with torch.no_grad():
#         for data, label in train_loader:
#             data = data.to(device)
#             _, _, _, latent = model(data)
#             train_latent_vectors.append(latent.cpu().numpy())
#             train_labels.append(label.cpu().numpy())
        
#         for data, label in test_loader:
#             data = data.to(device)
#             _, _, _, latent = model(data)
#             test_latent_vectors.append(latent.cpu().numpy())
#             test_labels.append(label.cpu().numpy())

#     train_latent_vectors = np.concatenate(train_latent_vectors, axis=0)
#     train_labels = np.concatenate(train_labels, axis=0)
#     test_latent_vectors = np.concatenate(test_latent_vectors, axis=0)
#     test_labels = np.concatenate(test_labels, axis=0)

#     # UMAP
#     umap_model = umap.UMAP(n_components=2, random_state=42)
#     train_latent_2d_umap = umap_model.fit_transform(train_latent_vectors)
#     test_latent_2d_umap = umap_model.transform(test_latent_vectors)

#     plt.figure(figsize=(10, 8))
#     plt.scatter(train_latent_2d_umap[:, 0], train_latent_2d_umap[:, 1], c=train_labels, cmap='viridis', marker='o', label='Train')
#     plt.scatter(test_latent_2d_umap[:, 0], test_latent_2d_umap[:, 1], c=test_labels, cmap='viridis', marker='x', label='Test')
#     plt.colorbar()
#     plt.title('Latent Space Visualization with UMAP')
#     plt.xlabel('UMAP Component 1')
#     plt.ylabel('UMAP Component 2')
#     plt.legend()
#     plt.show()

#     # t-SNE
#     # 1. 灏嗚缁冮泦鍜屾祴璇曢泦鐨勫悜閲忔嫾鍒颁竴璧穃n#     all_latent_vectors = np.concatenate([train_latent_vectors, test_latent_vectors], axis=0)

#     # 2. 涓€娆?fit_transform 瀛﹀埌鍏叡鐨?t-SNE 绌洪棿
#     tsne_model = TSNE(n_components=2, random_state=42)
#     all_latent_2d_tsne = tsne_model.fit_transform(all_latent_vectors)

#     # 3. 灏嗙粨鏋滄媶寮€
#     train_size = train_latent_vectors.shape[0]
#     train_latent_2d_tsne = all_latent_2d_tsne[:train_size]
#     test_latent_2d_tsne  = all_latent_2d_tsne[train_size:]


#     plt.figure(figsize=(10, 8))
#     plt.scatter(train_latent_2d_tsne[:, 0], train_latent_2d_tsne[:, 1], c=train_labels, cmap='viridis', marker='o', label='Train')
#     plt.scatter(test_latent_2d_tsne[:, 0], test_latent_2d_tsne[:, 1], c=test_labels, cmap='viridis', marker='x', label='Test')
#     plt.colorbar()
#     plt.title('Latent Space Visualization with t-SNE')
#     plt.xlabel('t-SNE Component 1')
#     plt.ylabel('t-SNE Component 2')
#     plt.legend()
#     plt.show()
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import numpy as np
import torch

def visualize_latent_space_discrete(model, train_loader, test_loader, device, class_names=None):
    model.eval()
    train_latent_vectors = []
    train_labels = []
    test_latent_vectors = []
    test_labels = []
    
    # 鑾峰彇娼滃湪绌洪棿琛ㄧず鍜屾爣绛綷n    with torch.no_grad():
        for data, label in train_loader:
            data = data.to(device)
            _, _, _, latent = model(data)
            train_latent_vectors.append(latent.cpu().numpy())
            train_labels.append(label.cpu().numpy())
        
        for data, label in test_loader:
            data = data.to(device)
            _, _, _, latent = model(data)
            test_latent_vectors.append(latent.cpu().numpy())
            test_labels.append(label.cpu().numpy())

    train_latent_vectors = np.concatenate(train_latent_vectors, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    test_latent_vectors = np.concatenate(test_latent_vectors, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)

    # UMAP 闄嶇淮
    umap_model = umap.UMAP(n_components=2, random_state=42)
    train_latent_2d_umap = umap_model.fit_transform(train_latent_vectors)
    test_latent_2d_umap = umap_model.transform(test_latent_vectors)

    # 璁剧疆绂绘暎璋冭壊鏉縗n    num_classes = len(np.unique(train_labels))
    palette = sns.color_palette("tab10", num_classes)
    colors = np.array(palette)

    plt.figure(figsize=(10, 8))

    # 缁樺埗 Train
    for class_idx in range(num_classes):
        idx = train_labels == class_idx
        plt.scatter(train_latent_2d_umap[idx, 0], train_latent_2d_umap[idx, 1], 
                    color=colors[class_idx], label=f'Train {class_names[class_idx] if class_names else class_idx}', marker='o', alpha=0.7)

    # 缁樺埗 Test
    for class_idx in range(num_classes):
        idx = test_labels == class_idx
        plt.scatter(test_latent_2d_umap[idx, 0], test_latent_2d_umap[idx, 1], 
                    edgecolor=colors[class_idx], facecolor='none', 
                    label=f'Test {class_names[class_idx] if class_names else class_idx}', 
                    marker='o', alpha=0.9, s=50, linewidth=1.5)


    plt.title('Latent Space Visualization with UMAP (Discrete Labels)')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


# CELL 22 [code]
# 娴嬭瘯 VAE 妯″瀷
test_loss = test_vae(ref_vae, test_loader, device)

# CELL 23 [code]
# 鍙鍖栭殣钘忓眰鏁版嵁
visualize_latent_space_discrete(ref_vae, train_loader, test_loader, device)

# CELL 24 [code]

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class VAEOutput:
    def __init__(self, z_dist, z_sample, x_recon, loss, loss_recon, loss_kl):
        self.z_dist = z_dist
        self.z_sample = z_sample
        self.x_recon = x_recon
        self.loss = loss
        self.loss_recon = loss_recon
        self.loss_kl = loss_kl

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # 鍧囧€糪n        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # 鏂瑰樊
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
    
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return Normal(self.fc21(h1), torch.exp(0.5 * self.fc22(h1)))
    
    def reparameterize(self, dist):
        return dist.rsample()
    
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)
    
    def forward(self, x, compute_loss=True):
        dist = self.encode(x)
        z = self.reparameterize(dist)
        recon_x = self.decode(z)

        if not compute_loss:
            return VAEOutput(
                z_dist=dist,
                z_sample=z,
                x_recon=recon_x,
                loss=None,
                loss_recon=None,
                loss_kl=None,
            )
        
        loss_recon = F.mse_loss(recon_x, x, reduction='sum')
        std_normal = Normal(torch.zeros_like(z, device=z.device), torch.ones_like(z, device=z.device))
        loss_kl = torch.distributions.kl.kl_divergence(dist, std_normal).sum()

        loss = loss_recon + loss_kl

        return VAEOutput(
            z_dist=dist,
            z_sample=z,
            x_recon=recon_x,
            loss=loss,
            loss_recon=loss_recon,
            loss_kl=loss_kl,
        )

# CELL 25 [code]
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class VAEOutput:
    def __init__(self, z_dist, z_sample, x_recon, loss, loss_recon, loss_kl):
        self.z_dist = z_dist
        self.z_sample = z_sample
        self.x_recon = x_recon
        self.loss = loss
        self.loss_recon = loss_recon
        self.loss_kl = loss_kl

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # 鍧囧€糪n        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # 鏂瑰樊
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, input_dim)
    
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return Normal(self.fc21(h2), torch.exp(0.5 * self.fc22(h2)))
    
    def reparameterize(self, dist):
        return dist.rsample()
    
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h4 = F.relu(self.fc4(h3))
        return self.fc5(h4)
    
    def forward(self, x, compute_loss=True):
        dist = self.encode(x)
        z = self.reparameterize(dist)
        recon_x = self.decode(z)

        if torch.isnan(dist.loc).any() or torch.isnan(dist.scale).any():
            print("NaN detected in dist parameters")
            print("dist.loc:", dist.loc)
            print("dist.scale:", dist.scale)

        if not compute_loss:
            return VAEOutput(
                z_dist=dist,
                z_sample=z,
                x_recon=recon_x,
                loss=None,
                loss_recon=None,
                loss_kl=None,
            )
        
        loss_recon = F.mse_loss(recon_x, x, reduction='sum')
        std_normal = Normal(torch.zeros_like(z), torch.ones_like(z))
        loss_kl = torch.distributions.kl.kl_divergence(dist, std_normal).sum()

        loss = loss_recon + loss_kl

        return VAEOutput(
            z_dist=dist,
            z_sample=z,
            x_recon=recon_x,
            loss=loss,
            loss_recon=loss_recon,
            loss_kl=loss_kl,
        )

# 鍒濆鍖栨ā鍨嬪弬鏁癨ndef weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# CELL 26 [code]
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

class VAEOutput:
    def __init__(self, z_dist, z_sample, x_recon, loss, loss_recon, loss_kl):
        self.z_dist = z_dist
        self.z_sample = z_sample
        self.x_recon = x_recon
        self.loss = loss
        self.loss_recon = loss_recon
        self.loss_kl = loss_kl

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) class

    Args:
        input_dim(int): Dimensionality of the input data.
        hidden_dim(int): Dimensionality of the hidden layer.
        latent_dim(int): Dimensionality of the latent space.
    """

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.SiLU(),
            nn.Linear(hidden_dim // 8, 2 * latent_dim),  # 2 for mean and variance.
        )
        self.softplus = nn.Softplus()

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 8),
            nn.SiLU(),
            nn.Linear(hidden_dim // 8, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x, eps: float = 1e-8):
        """
        Encodes the input data into the latent space
        
        Args:
            x (torch.Tensor): input data.
            eps (float): Small value to avoid numerical instability
        
        Returns:
            torch.distributions.MultivariateNormal: Normal distribution of the encoded data.
        """
        x = self.encoder(x)
        mu, logvar = torch.chunk(x, 2, dim=-1)
        scale = self.softplus(logvar) + eps
        scale_tril = torch.diag_embed(scale)

        return MultivariateNormal(mu, scale_tril=scale_tril)

    def reparameterize(self, dist):
        """
        Reparameterizes the encoded data to sample from the latent space

        Args:
            dist (torch.distributions.MultivariateNormal): Normal distributions of the encoded data.
        
        Returns:
            torch.Tensor: Sampled data from the latent space.
        """
        return dist.rsample()
    
    def decode(self, z):
        """
        Decodes the data from the latent space to the original input space.

        Args:
            z (torch.Tensor): Data in the latent space.
        
        Returns:
            torch.Tensor: Reconstructed data in the original input space.
        """
        return self.decoder(z)
    
    def forward(self, x, compute_loss: bool = True):
        """
        Performs a forward pass of the VAE.

        Args:
            x (torch.Tensor): Input data.
            compute_loss (bool): Whether to compute the loss or not.
        
        Returns:
            VAEOutput: VAE output dataclass.
        """
        dist = self.encode(x)
        z = self.reparameterize(dist)
        recon_x = self.decode(z)

        if not compute_loss:
            return VAEOutput(
                z_dist=dist,
                z_sample=z,
                x_recon=recon_x,
                loss=None,
                loss_recon=None,
                loss_kl=None,
            )
        
        loss_recon = F.mse_loss(recon_x, x, reduction='mean')
        std_normal = MultivariateNormal(
            torch.zeros_like(z, device=z.device),
            scale_tril=torch.eye(z.shape[-1], device=z.device).unsqueeze(0).expand(z.shape[0], -1, -1),
        )
        loss_kl = torch.distributions.kl.kl_divergence(dist, std_normal).mean()

        loss = loss_recon + loss_kl

        return VAEOutput(
            z_dist=dist,
            z_sample=z,
            x_recon=recon_x,
            loss=loss,
            loss_recon=loss_recon,
            loss_kl=loss_kl,
        )

# CELL 27 [code]
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.utils as utils

device = torch.device("cpu")

input_dim = train_data.shape[1]
hidden_dim = 512
latent_dim = 2

model = VAE(input_dim, hidden_dim, latent_dim).to(device)
model.apply(weights_init)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

epochs = 30  # 澧炲姞璁粌杞暟
for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = output.loss
        loss.backward()
        
        # 姊害瑁佸壀
        utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
    
    avg_train_loss = train_loss / len(train_loader.dataset)
    print(f'====> Epoch: {epoch} Average loss: {avg_train_loss:.4f}')
    
    # 璋冩暣瀛︿範鐜嘰n    scheduler.step(avg_train_loss)

# CELL 28 [code]
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        z = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x

# CELL 29 [code]
import anndata
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 鍋囪 abc_ad 鏄竴涓?AnnData 瀵硅薄
# 閫夋嫨楂樺彉鍩哄洜
highly_variable_genes = abc_ad.var_names[abc_ad.var['highly_variable']]

# 鑾峰彇楂樺彉鍩哄洜鐨勬暟鎹甛ndata = abc_ad[:, highly_variable_genes].X
labels = abc_ad.obs['cell_type']

# 鏍囧噯鍖栨暟鎹甛nscaler = StandardScaler()
data = scaler.fit_transform(data)

# 灏嗙粏鑳炵被鍨嬫爣绛捐浆鎹负鏁板€肩紪鐮乗nlabel_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# 灏嗘暟鎹垎鎴愯缁冮泦鍜屾祴璇曢泦
train_data, test_data, train_labels, test_labels = train_test_split(data, labels_encoded, test_size=0.2, random_state=42)

# 灏嗘暟鎹浆鎹负 PyTorch 寮犻噺
train_tensor = torch.tensor(train_data, dtype=torch.float32)
test_tensor = torch.tensor(test_data, dtype=torch.float32)
train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

# 鍒涘缓 TensorDataset
train_dataset = TensorDataset(train_tensor, train_labels_tensor)
test_dataset = TensorDataset(test_tensor, test_labels_tensor)

# 鍒涘缓 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# CELL 30 [code]
import torch.optim as optim

device = torch.device("cpu")

input_dim = train_data.shape[1]
hidden_dim = 400
latent_dim = 20

model = Autoencoder(input_dim, hidden_dim, latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

epochs = 30
for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_data = model(data)
        loss = criterion(recon_data, data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
    
    avg_train_loss = train_loss / len(train_loader.dataset)
    print(f'====> Epoch: {epoch} Average loss: {avg_train_loss:.4f}')

