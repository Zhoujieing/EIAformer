import torch
import torch.nn as nn
import numpy as np
from statsmodels.tsa.seasonal import STL
from scipy import stats
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


class HybridDecomposition(nn.Module):
    def __init__(self, moving_avg_window=25):
        super().__init__()
        self.moving_avg_window = moving_avg_window
        self.padding = (moving_avg_window - 1) // 2

    def forward(self, x):
        batch_size, seq_len, n_channels = x.shape
        trend = self._moving_average(x)
        initial_seasonal = self._dft_decompose(x)
        d_t, d_s = self._calculate_degrees(x, trend, initial_seasonal)
        
        seasonal = torch.zeros_like(x)
        for b in range(batch_size):
            for c in range(n_channels):
                if d_t[b, c] > 0.5 or d_s[b, c] > 0.5:
                    seasonal[b, :, c] = x[b, :, c] / (trend[b, :, c] + 1e-8)
                else:
                    seasonal[b, :, c] = x[b, :, c] - trend[b, :, c]
        return trend, seasonal

    def _moving_average(self, x):
        # use stride=1 to keep original sequence length
        avg_pool = nn.AvgPool1d(kernel_size=self.moving_avg_window, stride=1, padding=self.padding)
        return avg_pool(x.transpose(1, 2)).transpose(1, 2)

    def _dft_decompose(self, x):
        x_np = x.detach().cpu().numpy()
        dft = np.fft.fft(x_np, axis=1)
        half_len = x_np.shape[1] // 2
        dft[:, :half_len, :] = 0
        return torch.tensor(np.fft.ifft(dft, axis=1).real, dtype=x.dtype, device=x.device)

    def _calculate_degrees(self, x, trend, seasonal):
        batch_size, seq_len, n_channels = x.shape
        d_t = torch.zeros(batch_size, n_channels, device=x.device)
        d_s = torch.zeros(batch_size, n_channels, device=x.device)
        
        for b in range(batch_size):
            for c in range(n_channels):
                stl = STL(x[b, :, c].detach().cpu().numpy(), period=max(2, seq_len//10))
                try:
                    resid_np = stl.fit().resid
                    # Guard against unexpected resid length
                    if resid_np.shape[0] != seq_len:
                        residual = x[b, :, c] - trend[b, :, c] - seasonal[b, :, c]
                    else:
                        residual = torch.tensor(resid_np, dtype=x.dtype, device=x.device)
                except Exception:
                    residual = x[b, :, c] - trend[b, :, c] - seasonal[b, :, c]
                var_res = torch.var(residual)
                var_t_res = torch.var(trend[b, :, c] + residual)
                var_s_res = torch.var(seasonal[b, :, c] + residual)
                d_t[b, c] = torch.clamp(1 - var_res / (var_t_res + 1e-8), 0, 1)
                d_s[b, c] = torch.clamp(1 - var_res / (var_s_res + 1e-8), 0, 1)
        return d_t, d_s


class DynamicPatching(nn.Module):
    def __init__(self, patch_lengths=[2,4,8,16], dilation_rates=[1,2,4,8]):
        super().__init__()
        self.patch_lengths = patch_lengths
        self.dilation_rates = dilation_rates
        self.convs = nn.ModuleList([
            nn.Conv1d(1, 1, kernel_size=p, dilation=d, padding=(p-1)*d)
            for p, d in zip(patch_lengths, dilation_rates)
        ])

    def forward(self, seasonal):
        batch_size, seq_len, n_channels = seasonal.shape
        all_patches = []
        
        for c in range(n_channels):
            channel_data = seasonal[:, :, c].unsqueeze(1)
            layer_patches = []
            for conv, p in zip(self.convs, self.patch_lengths):
                conv_out = conv(channel_data)[:, :, :seq_len]
                patches = self._slide_window(conv_out.squeeze(1), p)
                layer_patches.append(patches)
            optimal_patches = self._select_optimal_patches(layer_patches)
            all_patches.append(optimal_patches.unsqueeze(-1))
        return torch.cat(all_patches, dim=-1)

    def _slide_window(self, x, window_size):
        batch_size, seq_len = x.shape
        n_patches = seq_len - window_size + 1
        return torch.stack([x[:, i:i+window_size] for i in range(n_patches)], dim=1)

    def _calculate_score(self, patch1, patch2):
        batch_size, p = patch1.shape
        score = torch.zeros(batch_size, device=patch1.device)
        for b in range(batch_size):
            x = np.arange(2*p)
            y = torch.cat([patch1[b], patch2[b]]).detach().cpu().numpy()
            slope, intercept, r_value = stats.linregress(x, y)[:3]
            score[b] = torch.tensor(r_value**2, dtype=patch1.dtype, device=patch1.device)
        return score

    def _select_optimal_patches(self, layer_patches):
        batch_size = layer_patches[0].shape[0]
        all_candidate = torch.cat(layer_patches, dim=1)
        n_candidates = all_candidate.shape[1]
        selected = []
        
        for b in range(batch_size):
            candidates = all_candidate[b]
            used = torch.zeros(n_candidates, dtype=torch.bool, device=candidates.device)
            for i in range(n_candidates):
                if used[i]:
                    continue
                if i < n_candidates - 1 and not used[i+1]:
                    score = self._calculate_score(candidates[i], candidates[i+1])
                    if score >= 0.5:
                        selected.append((candidates[i] + candidates[i+1])/2)
                        used[i] = used[i+1] = True
                    else:
                        selected.append(candidates[i])
                        used[i] = True
                else:
                    selected.append(candidates[i])
                    used[i] = True
        
        max_len = max([p.shape[0] for p in selected[:batch_size]])
        return torch.stack([
            torch.cat([p, torch.zeros(max_len - p.shape[0], device=p.device)]) 
            for p in selected
        ], dim=0).reshape(batch_size, -1, max_len)


class ChannelFusion(nn.Module):
    def __init__(self, d_model=512, lambda1=0.5, lambda2=0.5):
        super().__init__()
        self.d_model = d_model
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.projection = nn.Linear(1, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, d_model))
        self.channel_encoding = nn.Parameter(torch.randn(1, 1, 100, d_model))

    def forward(self, patches):
        batch_size, n_patches, patch_len, n_channels = patches.shape
        self._preprocess(patches)
        
        ic_matrix = self._calculate_intrinsic_connection(patches)
        ec_matrix = self._calculate_extrinsic_connection(patches)
        
        fused_channels = []
        used = torch.zeros(n_channels, dtype=torch.bool)
        for i in range(n_channels):
            if used[i]:
                continue
            partners = [j for j in range(i+1, n_channels) if not used[j] and ic_matrix[i,j]==1 and ec_matrix[i,j]==1]
            if partners:
                fused = self._fuse_channels(patches[:, :, :, [i] + partners])
                fused_channels.append(fused)
                used[i] = True
                used[partners] = True
            else:
                fused_channels.append(patches[:, :, :, i].unsqueeze(-1))
        return torch.cat(fused_channels, dim=-1)

    def _preprocess(self, patches):
        batch_size, n_patches, patch_len, n_channels = patches.shape
        flat = patches.permute(0,1,3,2).reshape(-1, patch_len, 1)
        projected = self.projection(flat)
        pos_enc = self.pos_encoding[:, :n_patches, :].repeat(batch_size*n_channels, 1, 1)
        channel_enc = self.channel_encoding[:, :, :n_channels, :].permute(0,2,1,3).reshape(-1, 1, self.d_model)
        return projected + pos_enc + channel_enc

    def _calculate_intrinsic_connection(self, patches):
        n_channels = patches.shape[-1]
        ic_matrix = torch.zeros(n_channels, n_channels, dtype=torch.int)
        from statsmodels.tsa.stattools import grangercausalitytests
        
        for i in range(n_channels):
            for j in range(n_channels):
                if i == j:
                    continue
                x = patches[0, :, :, i].detach().cpu().numpy().reshape(-1)
                y = patches[0, :, :, j].detach().cpu().numpy().reshape(-1)
                try:
                    test = grangercausalitytests(np.vstack([x, y]).T, maxlag=1, verbose=False)
                    ic_matrix[i, j] = 1 if test[1][0]['ssr_chi2test'][1] < 0.05 else 0
                except:
                    ic_matrix[i, j] = 0
        return ic_matrix

    def _calculate_extrinsic_connection(self, patches):
        n_channels = patches.shape[-1]
        ec_matrix = torch.zeros(n_channels, n_channels, dtype=torch.int)
        
        for i in range(n_channels):
            for j in range(n_channels):
                if i == j:
                    continue
                x = patches[:, :, :, i].detach().cpu().numpy()
                y = patches[:, :, :, j].detach().cpu().numpy()
                mu_i, sigma_i = np.mean(x), np.std(x)
                mu_j, sigma_j = np.mean(y), np.std(y)
                th_d = self.lambda1*(mu_i+mu_j) + self.lambda2*(sigma_i+sigma_j)
                dtw_dist, _ = fastdtw(x.reshape(-1), y.reshape(-1), dist=euclidean)
                ec_matrix[i, j] = 1 if dtw_dist < th_d else 0
        return ec_matrix

    def _fuse_channels(self, channels):
        sigma = torch.std(channels, dim=(1,2), keepdim=True)
        weights = sigma / (torch.sum(sigma, dim=-1, keepdim=True) + 1e-8)
        return torch.sum(channels * weights, dim=-1, keepdim=True)
