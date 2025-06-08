# %%
import numpy as np
import os
import glob
import cv2
from osgeo import gdal
import matplotlib.pyplot as plt
import torch
import time

from sklearn.decomposition import PCA
import sys

#from livelossplot import PlotLosses

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')

# %% [markdown]
# #### Auto-Encoderの定義
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super(Encoder, self).__init__()
        self.L = latent_size # 潜在空間の次元数 LxN_i
        self.fc1 = nn.Linear(input_size, input_size//2)
        self.fc2 = nn.Linear(input_size//2, self.L) # L = (band_num//3)//2
        # 重みを初期化
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = x.float()
        #x = torch.tensor(x)
        x = torch.tanh(self.fc1(x)) # Xavierの初期値を使用
        x = self.fc2(x)
        return x

class Decoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super(Decoder, self).__init__()
        self.L = latent_size # 潜在空間の次元数 LxN_i
        self.fc1 = nn.Linear(self.L, input_size//2)
        self.fc2 = nn.Linear(input_size//2, input_size) # band_num = ((L*2)*3)
        # 重みを初期化
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = x.float()
        #x = torch.tensor(x)
        x = torch.tanh(self.fc1(x)) # Xavierの初期値を使用
        x = self.fc2(x)
        return x

class AutoEncoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super().__init__()
        self.L = L
        self.enc = Encoder(input_size, latent_size).to(device)
        self.dec = Decoder(input_size, latent_size).to(device)

    def forward(self, x):
        #x = torch.tensor(x)
        encoded = self.enc(x) # Encode
        decoded = self.dec(encoded) # Decode
        return encoded, decoded


# %% ファイルの読み込み
data_name = 'Schrodinger_2'
#file_path = '../Dataset/M3/'+data_name+'/data/'+data_name+'/'+data_name+'_map.tif' # 前処理前
#file_path = '../Dataset/M3/'+data_name+'/data/'+data_name+'/'+data_name+'_rm_outlier_2.tif' # 前処理1
#file_path = '../Dataset/M3/'+data_name+'/data/'+data_name+'/'+data_name+'_norm_750_rmoutliner_11.tif'　# 前処理3
#file_path = '../../Dataset/M3/'+data_name+'/data/'+data_name+'/'+data_name+'_norm_750_hullquotient.tif' #　前処理4

# %% 変更する値
file_path_s = [#'../Dataset/M3/'+data_name+'/data/'+data_name+'/'+data_name+'_map.tif',
               '../Dataset/M3/Schrodinger_2/data/Schrodinger_2/Schrodinger_2_norm_750_2.tif', # Prepro 3
               '../Dataset/M3/Schrodinger_2/data/Schrodinger_2/Schrodinger_2_norm_750_hullquotient_2.tif', # Prepro 4
                '../Dataset/M3/'+data_name+'/data/'+data_name+'/'+data_name+'_rm_outlier_2.tif'] # Prepro 1
#path_name = ['Prepro0', 'Prepro1', 'Prepro2', 'Prepro3', 'Prepro4']
#path_name = ['Prepro0', 'Prepr1', 'Prepro2']
path_name = ['Prepro3_2', 'Prepro4_2', 'Prepro1_2']
epochs = 500 # epoch数

# 削減チャネル数
L_s = [8, 16]

# %%
for index_file in range(len(file_path_s)):
    save_path = data_name+'/'+path_name[index_file] # Schrodinger_2/Prepro0/
    for L in L_s:
        img_save_path = save_path+'/'+data_name+'_DR_'+str(L)+'.tif'
        weight_path = save_path+'/weight_'+str(L)

        print(f'path: {path_name[index_file]}, L: {L}')

        imgs_ = gdal.Open(file_path_s[index_file])
        imgs = np.array(imgs_.ReadAsArray(), np.float32) # [0~1]にスケーリング正規化かつfloat64->float32へ変換
        #print(f'dtype: {imgs.dtype}, shape: {imgs.shape}, min: {np.min(imgs)}, max: {np.max(imgs)}')
        #imgs = imgs[3:] # band 1, 2は真っ黒なため削除
        #imgs = imgs[1:]
        #print(f'dtype: {imgs.dtype}, shape: {imgs.shape}, min: {np.min(imgs)}, max: {np.max(imgs)}')
        #imgs = imgs[:-1]
        band_size = imgs.shape[0] # band number
        print(f'dtype: {imgs.dtype}, shape: {imgs.shape}, min: {np.min(imgs)}, max: {np.max(imgs)}')

        import scipy.stats
        imgs = scipy.stats.zscore(imgs) # 平均0, 分散1に標準化
        print(f'aft mean:{np.mean(imgs)}, std:{np.std(imgs)}')
        print(f'dtype: {imgs.dtype}, shape: {imgs.shape}, min: {np.min(imgs)}, max: {np.max(imgs)}')

        # plot
        # plt.figure(figsize=(10,10))
        # plt.title(f"M3 {data_name} first band image")
        # plt.imshow(imgs[0], cmap='gray')
        # plt.colorbar()
        # plt.show


        # 波長 82 band
        imgs_pca_input = imgs.reshape(band_size, -1).transpose()
        print(imgs_pca_input.shape)


        # %%
        # 画像180x1794x1891（band, h, w）にPCAを適用
        # chanel: 82　->　3（主成分）
        pca = PCA(3)
        imgs_pca_output = pca.fit_transform(imgs_pca_input)

        imgs_pca_output.shape

        # %%
        # PCAによって180から3chanelにした擬似カラー画像
        imgs_slic_input = imgs_pca_output.reshape(imgs.shape[1], imgs.shape[2], -1)

        # BGR to RGB
        #imgs_slic_input_rgb = imgs_slic_input[:, :, [2, 1, 0]]
        imgs_slic_input_rgb = imgs_slic_input

        print(imgs_slic_input_rgb.shape)
        #plt.imshow(imgs_slic_input_rgb)


        # %%
        # Superpixel Segmentaion のパラメータ
        height, width, channels = imgs_slic_input.shape[:3]
        region_size = 35
        ruler = 5
        min_element_size = 30
        num_iterations = 20  # 反復回数

        # 64ビットから32ビットへの変換
        imgs_slic_input_32f = np.float32(imgs_slic_input_rgb)
        imgs_slic_input_hsv = cv2.cvtColor(imgs_slic_input_32f, cv2.COLOR_BGR2HSV)

        slic = cv2.ximgproc.createSuperpixelSLIC(imgs_slic_input_hsv, cv2.ximgproc.SLIC, region_size, float(ruler))

        # 画像のスーパーピクセルセグメンテーションを計算
        # 入力画像は,HSVまたはL*a*b*
        slic.iterate(num_iterations)
        slic.enforceLabelConnectivity(min_element_size)

        labels = slic.getLabels()

        # スーパーピクセルセグメンテーションの境界を取得
        contour_mask = slic.getLabelContourMask(False)
        imgs_slic_output = imgs_slic_input_rgb.copy()
        imgs_slic_output[0 < contour_mask] = (255, 255, 0)

        # セグメンテーション数の取得
        # ColAE論文中の J (nseg) にあたる（Jが100前後の時が性能が良い）
        # 論文中のDataset（Indian: 145x145, UoPavia: 610x340, Salinas: 512x217）
        ## と比較して今回のデータは　1640x998　と3倍ほど大きいので J も少し大きい値でもいいか？
        J = slic.getNumberOfSuperpixels()

        lbavgimg = np.zeros((height, width, channels), dtype=np.uint8)
        meanimg = np.zeros((J, channels+1), dtype=np.float32)

        # 画像表示
        print(f'J (nSeg) = {J}')
        #plt.imshow(imgs_slic_output)


        # %% [markdown]
        # ## 2. Collaborative AEs
        # %%
        print(f'size: {labels.shape}')
        print(f'画像pixelごとのseglabel: {labels}')


        # %%
        band_num, height, width = imgs.shape[:3]

        print(f'band:{band_num}, h:{height}, w:{width}')

        # ベクトル N_j x 波長180

        # セグメンテーション毎のベクトル X_i 作成 (J, N_j, B)
        class_data = [[] for _ in range(J)]
        save_pixel_index = [[] for _ in range(J)]

        # ピクセルごとにクラスごとのデータを追加
        for y in range(height):
            for x in range(width):
                class_idx = labels[y, x]  # クラスを取得
                pixel_data = imgs[:, y, x]  # 長さがBの一次元配列を生成（仮のデータ）
                class_data[class_idx].append(pixel_data)
                save_pixel_index[class_idx].append([y, x])

        # クラスごとのデータを NumPy 配列に変換
        X = np.array([np.array(class_data[j]) for j in range(J)], dtype=object)
        pixel = np.array([np.array(save_pixel_index[j]) for j in range(J)], dtype=object)

        # セグメンテーション毎（クラスごと）のpixel数 N_j
        N = [0]*J
        for index, x_i in enumerate(X):
            N[index] = X[index].shape[0]
            #print(f'X[{i}].shape: {X[i].shape[0]}', end=' ')

        # X の形状を表示
        print(f'X.shape（J個のベクトル）: {X.shape}')
        i = 0 # i番目のSuperpixel
        print(f'{i}番目のsuperpixelのサンプル数: {N[i]}')
        print(f'{i}番目のsuperpixelのサンプル: {X[i]}')
        print(f'X[i].shape: {X[i].shape}')

        N = np.array(N)

        # %%
        for i in range(J):
            print(f'{i}:{N[i]}', end=' ')
        print()

        # %% [markdown]
        # #### 各Superpixel内の平均ベクトルを作成

        # %%
        # i番目のSuperpixelの平均ベクトル x J
        #mu = [0]*J
        mu = [[] for _ in range(J)]
        for i in range(J):
            mu[i] = np.mean(X[i], axis=0)

        mu = np.array(mu)
        print(f'平均ベクトル: {mu.shape}')

        # %%
        # LLE
        sys.path.append(os.path.join(os.path.dirname(__file__), 'Generative-LLE'))
        #from Generative-LLS.functions import my_LLE
        from functions import my_LLE

        # 潜在空間のサイズ
        #L = 8
        # K-最近傍の数 (固定比率 R=0.2 のとき)
        K = J * 0.2
        K = int(K)
        print(f'K={K}')

        # My_LLEの入力のため転置
        mu_transpose = mu.transpose()

        embedding = my_LLE.My_LLE(X=mu_transpose, n_neighbors=K, n_components=L,
                                path_save=data_name+'/LLE')


        # %%
        X_transformed = embedding.fit_transform(calculate_again=True)
        print(f'X shape: {mu_transpose.shape}')
        print(f'Y shape: {X_transformed.shape}')

        # 式（2）に従った各平均ベクトルの重み
        W = embedding.W_linearEmbedding
        print(f'W.shape: {W.shape}')
        W = torch.from_numpy(W).to(dtype=torch.float32, device=device)
        W.requires_grad = True

        # K-最近傍のインデックス
        neighbor_indices = embedding.neighbor_indices
        print(f'neighbor_indices.shape: {neighbor_indices.shape}')


        # %% [markdown]
        # ### Auto-Encoder (AE)
        #
        # 各SuperpixelごとにAutoencoderを実装
        #
        # 余裕があれば、Variational Autoencoder（変分AE）にする、他も試してみる

        # %%
        # 入力サイズ BxN_i
        i = 0 # i番目のSuperpixel
        print(f'Input size ({band_num}x{N[i]}): {band_num*N[i]}')
        # 潜在空間のサイズ
        #L = 30
        print(f'Latent space size ({L}x{N[i]}): {L*N[i]}')


        # %% [markdown]
        # #### 損失関数, 最適化手法の定義

        # %%
        # MSE Loss funcion (再構成誤差)
        AE_loss_function = torch.nn.MSELoss()
        monifold_loss_function = torch.nn.MSELoss()

        # 乱数のシードを固定して再現性を確保
        torch.manual_seed(0)


        # %%
        #import dadaptation

        model = [[] for _ in range(J)]
        optimizer = [[] for _ in range(J)]
        scheduler = [[] for _ in range(J)]

        #epochs = 200 # epoch数

        for i in range(J):
            # i番目のModel
            model[i] = AutoEncoder(band_num, L).to(device)

            # i番目のOptimizer
            optimizer[i] = torch.optim.Adam(model[i].parameters(), lr=1e-2, weight_decay=1e-5)
            #optimizer[i] = dadaptation.dadapt_adam.DAdaptAdam(model[i].parameters(), lr=1.0, decouple=False, weight_decay=1e-5)
            scheduler[i] = torch.optim.lr_scheduler.LinearLR(optimizer[i], start_factor=1.0, end_factor=1e-2, total_iters=(epochs-5))


        # %% [markdown]
        # ##### 学習

        # %%
        #save_path = '~/BachelorResearch/ColAE/'+data_name
        #%cd {save_path}
        #weight_path = f'weight'

        if not os.path.exists(weight_path):
            print(f'{weight_path} is not exists')
            print(f'path: {weight_path}')
            os.makedirs(weight_path, exist_ok=True)
        # %%
        #from sklearn.manifold import LocallyLinearEmbedding

        #L = 30 # 符号空間（code space）の次元数
        #eta =  0.75 # 損失のバランス係数
        eta =  1.25 # 損失のバランス係数

        # i-th Superpixelの符号空間の平均ベクトル x J
        mu_y_ = [[] for _ in range(J)]
        mu_y = [[] for _ in range(J)]

        # K-近傍
        y_ik_ = [[0] for _ in range(J)]
        y_ik = [[0] for _ in range(J)]

        start_ColAE = time.time() #
        #liveloss = PlotLosses() # livelossplot
        plot_AE_loss = []
        plot_monifold_loss = []

        # 学習
        for epoch in range(epochs):
            print(f'Epoch {epoch}/{epochs-1}\t(J={J})')
            print('\t', end='')
            #logs = {} # livelossplot

            AE_loss = 0.0 # AEの損失
            monifold_loss = 0.0 # 多様体損失(LLE)
            loss = 0.0 # ColAEの損失

            # 潜在空間(Latent Space)ベクトル
            Y_ = [[] for _ in range(J)]
            Y = [[] for _ in range(J)]

            for i in range(J):
        # ========== train ===========
                # modelのパラメータの読み込み
                #weight_path = f'~/BachelorResearch/ColAE/weight/ColAE_weight_{i}.pth'
                if os.path.exists(os.path.join(weight_path, f'ColAE_weight_{i}.pth')):
                    model[i].load_state_dict(torch.load(os.path.join(weight_path, f'ColAE_weight_{i}.pth')))

                model[i].train() # モデルを訓練モードに設定

                for j, input in enumerate(X[i]): # BxN_i から 長さBのベクトルごとに

                    # Encoder input
                    input = torch.from_numpy(input).to(dtype=torch.float32, device=device)
                    input.requires_grad = True

                    # Encoder output (Latent Space) and Decoder output
                    latent, reconstructed = model[i](input)

                    # L(latentの次元) x N_i
                    latent = np.array(latent.detach().cpu())
                    Y_[i].append(latent)

                    #  ======= 損失計算 =======
                    # Loss function for the i-th AE (式9, 10)
                    AE_loss_ = AE_loss_function(reconstructed, input)
                    AE_loss_.retain_grad()
                    AE_loss += AE_loss_
                    AE_loss.retain_grad()
                    # ======================

                if epoch==0:
                    #print(f'AE_{i} Train Loss: {AE_loss_:.5f},', end=' ')
                    print(f'{i},', end='')
                # i番目のSuperpixelの潜在空間ベクトル
                Y[i] = np.array(Y_[i])

            print()

            # ======= 損失計算 =======
            # 再構成誤差, 多様体損失 (式8)
            # i-th Superpixelの符号空間の平均ベクトル x J
            for i in range(J):
                mu_y_[i] = np.mean(Y[i], axis=0)
                mu_y_[i] = np.array(mu_y_[i])
                mu_y[i] = torch.from_numpy(mu_y_[i]).to(dtype=torch.float32, device=device)
                mu_y[i].requires_grad = True

            for i in range(J):
                for k in neighbor_indices[i].astype('int32'):
                    # y_i のK-最近傍の線型結合
                    y_ik_[i] = np.array(y_ik_[i])
                    y_ik[i] = torch.from_numpy(y_ik_[i]).to(dtype=torch.float32, device=device)
                    y_ik[i].requires_grad = True
                    y_ik[i] = y_ik[i] + W[i][k] * mu_y[k]
                    y_ik[i].retain_grad()

                monifold_loss_ = monifold_loss_function(mu_y[i], y_ik[i])
                monifold_loss_.retain_grad()
                # LLEの再構成誤差（多様体損失）
                monifold_loss +=  monifold_loss_
                monifold_loss.retain_grad()

            # ColAEの損失(式11)
            loss = AE_loss + eta*monifold_loss
            loss.retain_grad()
            # ======================

            for i in range(J):
                # 全Superpixelの多様体構造を保つようにパラメータ更新
                optimizer[i].zero_grad()
            loss.backward() # 誤差逆伝播法に基づいて各パラメータの勾配を計算
            #monifold_loss.backward()
            for i in range(J):
                optimizer[i].step() # 勾配の値に基づいて選択した最適化手法によりパラメータ W, b を更新
                scheduler[i].step()

                # modelのパラメータの保存（epochごとに使う）
                #weight_path = f'~/BachelorResearch/ColAE/weight/ColAE_weight_{i}.pth'
                if not(epoch==0):
                    torch.save(model[i].state_dict(), os.path.join(weight_path, f'ColAE_weight_{i}.pth'))
            print(f'\tAE Loss: {AE_loss:.6f}, Monifold Loss: {monifold_loss:.6f}, ColAE Loss: {loss:.6f}')
            #logs['AE_loss'] = AE_loss.to('cpu').detach().numpy()
            #logs['Monifold_loss'] = monifold_loss.to('cpu').detach().numpy()
            #liveloss.update(logs)
            #liveloss.send()
            plot_AE_loss.append(AE_loss.to('cpu').detach().numpy())
            plot_monifold_loss.append(monifold_loss.to('cpu').detach().numpy())

            if epoch==0 or epoch==1: # epochあたりの実行時間を表示
                end_each_ColAE = time.time()
                print(f'Running time of 1 epoch ColAE is {end_each_ColAE-start_ColAE}')


        end_ColAE = time.time() #
        time_diff = end_ColAE-start_ColAE
        print(f'AE Loss: {AE_loss/J:.6f}, Monifold Loss: {monifold_loss/K:.6f} (Loss / number of superpixel)')
        print(f'Running time of ColAE is {time_diff}')
        # ========== train ===========


        # %%
        #%cd ~/BachelorResearch/ColAE
        #img_save_path = data_name
        # with open('info_dadaptation.txt','a') as f_dadaptation:
        #     for i in range(J):
        #         f_dadaptation.write(f'{i}-th Params')
        #         f_dadaptation.write(f'{optimizer[i].param_groups}')
        #         f_dadaptation.write('\n')
        #
        with open(f'{save_path}/info_{L}.txt','a') as f:
            f.write(f'Number of Epoch (J) = {J}, L={L}, K={K}, eta={eta}\n')
            for i in range(epochs):
                f.write(f'Epoch {i}: AE Loss: {plot_AE_loss[i]}, Monifold Loss: {plot_monifold_loss[i]}')
                f.write('\n')
            f.write(f'AE Loss: {AE_loss/J:.6f}, Monifold Loss: {monifold_loss/K:.6f} (Loss / number of superpixel)')
            f.write(f'Running time: {time_diff}')


        # %%
        # Auto-Encoder plot
        fig, ax = plt.subplots()
        t = np.linspace(0,epochs,epochs)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('AE loss')
        ax.grid()
        ax.plot(t, plot_AE_loss)
        fig.tight_layout()
        plt.savefig(f'{save_path}/AE_loss_{L}.pdf')
        #plt.show()

        # Monifolod loss plot
        fig, ax = plt.subplots()
        t = np.linspace(0,epochs,epochs)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Monifolod loss')
        ax.grid()
        ax.plot(t, plot_monifold_loss)
        fig.tight_layout()
        plt.savefig(f'{save_path}/monifold_loss_{L}.pdf')
        #plt.show()

        # %%
        i = 0 # i番目のSuperpixel
        #print(f'imgs: {len(imgs)}')
        #print(f'imgs[{i}]: {len(imgs[i])}x{len(imgs[i][i])}')
        # X の形状を表示
        print(f'X.shape（{J}個のベクトル）: {X.shape}')
        print(f'X[{i}].shape  : {X[i].shape}')

        # Y の形状を表示
        print(f'Y.shape（{J}個のベクトル）: {len(Y)}')
        i = 0 # i番目のSuperpixel
        #print(f'{i}番目のsuperpixelのサンプル数: {N[i]}')
        #print(f'{i}番目のsuperpixelの潜在表現: {Y[i]}')
        print(f'Y[{i}].shape: {np.array(Y[i]).shape}')


        # %%
        # セグメンテーション毎のベクトル X_i 作成 (J, N_j, B)
        class_data_ = [[] for _ in range(J)]

        imgs_DR_output_ = [[[[0] for _ in range(height)] for _ in range(width)] for _ in range(L)]
        print(f'imgs_DR_output_: {len(imgs_DR_output_)}')
        #print(f'imgs_DR_output_[{0}]: {len(imgs_DR_output_[0])}x{len(imgs_DR_output_[0][0])}')

        for i in range(J):
            #for y in range(height):
                #for x in range(width):
            for j in range(pixel[i].shape[0]):
                y, x = pixel[i][j]
                #print(f'{i}:({y},{x})', end=' ')
                for l in range(L):
                    imgs_DR_output_[l][y][x] = Y[i][j][l]

        imgs_DR_output = np.array(imgs_DR_output_)
        #print(f'imgs_DR_output[0]: {imgs_DR_output[0]}')


        # %%
        #print(pixel.shape) # Jの数
        #print(pixel[0].shape) # ラベルiの数, (y,x)座標
        print(f'h:{height}, w:{width}, l:{L}')


        # %%
        #plt.imshow(imgs_DR_output[0], cmap='gray')


        # %%
        # 0~1の範囲にスケーリング
        img_max = np.amax(imgs_DR_output); # 3Dの最大値
        img_min = np.amin(imgs_DR_output); # 3Dの最小値
        for i in range(height):
            for j in range(width):
                for l in range(L): # channel数
                    #imgs_DR_output[i][j][k] = 2*(imgs_DR_output[i][j][k]-img_min)/(img_max-img_min) - 1
                    imgs_DR_output[l][j][i] = (imgs_DR_output[l][j][i]-img_min)/(img_max-img_min) # 0~1の範囲
        print(f'max:{np.amax(imgs_DR_output)}, min:{np.amin(imgs_DR_output)}')

        ######## GeoTfii画像に保存 #########
        #%cd ~/BachelorResearch/ColAE
        #img_save_path = data_name
        #save_path = data_name+'_DR.tif'

        dtype = gdal.GDT_Float32
        output = gdal.GetDriverByName('GTiff').Create(img_save_path, height, width, L, dtype) # 空の出力ファイル

        for l in range(L): # バンド後にとに書き出し
            output.GetRasterBand(l+1).WriteArray(imgs_DR_output[l,:,:])
        output.FlushCache() # ディスクに書き出し
        output = None
        ##################################


        # # %%
        # # plot
        # #plt.figure(figsize=(6,6))
        # #plt.title(f"M3 {data_name} first band Dimensionality Reduced image")
        # plt.imshow(imgs_DR_output[0], cmap='gray')
        # #plt.colorbar()
        # #plt.show


        # # %%
        # plt.imshow(imgs_DR_output[1], cmap='gray')


        # # %%
        # plt.imshow(imgs_DR_output[L-1], cmap='gray')
