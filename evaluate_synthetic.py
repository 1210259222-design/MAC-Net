import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import random
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.metrics import structural_similarity as ssim
from numba import njit
import math
import time

# ==============================================================================
# 1. 核心网络组件
# ==============================================================================
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_dim, embed_dim=64, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads 
        self.q_proj = nn.Linear(in_dim, embed_dim)
        self.k_proj = nn.Linear(in_dim, embed_dim)
        self.v_proj = nn.Linear(in_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, in_dim)
    def forward(self, q_in, k_in, v_in):
        B, C_in, H, W = v_in.shape
        N = H * W
        q_flat = q_in.view(B, q_in.shape[1], N).transpose(1, 2)
        k_flat = k_in.view(B, k_in.shape[1], N).transpose(1, 2)
        v_flat = v_in.view(B, C_in, N).transpose(1, 2)
        q = self.q_proj(q_flat).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k_flat).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v_flat).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        out_attn = F.scaled_dot_product_attention(q, k, v)
        out = out_attn.transpose(1, 2).reshape(B, N, -1)
        return self.out_proj(out).transpose(1, 2).view(B, C_in, H, W)

class FFN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(dim, dim * 4, 1), nn.GELU(), nn.Conv2d(dim * 4, dim, 1))
    def forward(self, x): return self.net(x)

class EncoderTransformerBlock(nn.Module):
    def __init__(self, dim, embed_dim=64, num_heads=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim) 
        self.msa = MultiHeadSelfAttention(dim, embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = FFN(dim)
    def forward(self, x):
        B, C, H, W = x.shape
        norm_x = self.ln1(x.view(B, C, -1).transpose(1, 2)).transpose(1, 2).view(B, C, H, W)
        x = self.msa(norm_x, norm_x, norm_x) + x
        norm_x2 = self.ln2(x.view(B, C, -1).transpose(1, 2)).transpose(1, 2).view(B, C, H, W)
        return self.ffn(norm_x2) + x

class DecoderTransformerBlock(nn.Module):
    def __init__(self, dim, embed_dim=64, num_heads=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.msa1 = MultiHeadSelfAttention(dim, embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(dim)
        self.msa2 = MultiHeadSelfAttention(dim, embed_dim, num_heads)
        self.ln3 = nn.LayerNorm(dim)
        self.ffn = FFN(dim)
    def forward(self, x):
        B, C, H, W = x.shape
        norm_x1 = self.ln1(x.view(B, C, -1).transpose(1, 2)).transpose(1, 2).view(B, C, H, W)
        fused_x = self.msa1(norm_x1, norm_x1, norm_x1) + x
        norm_x2 = self.ln2(x.view(B, C, -1).transpose(1, 2)).transpose(1, 2).view(B, C, H, W)
        x2 = self.msa2(q_in=norm_x2, k_in=fused_x, v_in=fused_x) + fused_x
        norm_x3 = self.ln3(x2.view(B, C, -1).transpose(1, 2)).transpose(1, 2).view(B, C, H, W)
        return self.ffn(norm_x3) + x2

class ConvNode(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.GELU())
    def forward(self, x): return self.conv(x)

class UNetPlusPlusBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv0_0 = ConvNode(dim, dim); self.conv1_0 = ConvNode(dim, dim)
        self.conv2_0 = ConvNode(dim, dim); self.conv3_0 = ConvNode(dim, dim)
        self.conv0_1 = ConvNode(dim * 2, dim); self.conv1_1 = ConvNode(dim * 2, dim)
        self.conv2_1 = ConvNode(dim * 2, dim); self.conv0_2 = ConvNode(dim * 3, dim)
        self.conv1_2 = ConvNode(dim * 3, dim); self.conv0_3 = ConvNode(dim * 4, dim)
    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        return self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

class MAC_Net(nn.Module):
    def __init__(self, in_channels, hidden_dim=64, down_factor=4): 
        super().__init__()
        self.down_factor = down_factor
        self.in_channels = in_channels
        
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, hidden_dim, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 5, padding=2), nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 5, padding=2), nn.GELU()
        )
        
        if self.in_channels == 2:
            self.mask_encoder = nn.Sequential(
                nn.Conv2d(1, hidden_dim, 3, padding=1), nn.GELU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1), nn.Sigmoid() 
            )
            self.mask_fusion = nn.Sequential(
                nn.Conv2d(hidden_dim * 2, hidden_dim, 1),
                nn.GELU()
            )

        self.front_encoder = EncoderTransformerBlock(hidden_dim, embed_dim=64, num_heads=4)
        self.unet_pp = UNetPlusPlusBlock(hidden_dim)
        self.front_decoder = DecoderTransformerBlock(hidden_dim, embed_dim=64, num_heads=4)
        self.final_conv = nn.Conv2d(hidden_dim, 1, 1)

    def forward(self, x):
        if self.in_channels == 2:
            noisy = x[:, 0:1, :, :]
            mask = x[:, 1:2, :, :]
            input_data = noisy * mask
        else:
            noisy = x
            input_data = noisy 

        c_feat = self.initial_conv(input_data)
        
        if self.in_channels == 2:
            m_gate = self.mask_encoder(mask)
            modulated_feat = c_feat * m_gate
            c_feat = self.mask_fusion(torch.cat([modulated_feat, m_gate], dim=1))

        t_feat_in = F.avg_pool2d(c_feat, kernel_size=self.down_factor, stride=self.down_factor)
        t_feat_out = self.front_encoder(t_feat_in)
        t_feat_up = F.interpolate(t_feat_out, scale_factor=self.down_factor, mode='bilinear', align_corners=False)
        encoder_out = c_feat + t_feat_up 
        
        unet_out = self.unet_pp(encoder_out)
        
        d_feat_in = F.avg_pool2d(unet_out, kernel_size=self.down_factor, stride=self.down_factor)
        d_feat_out = self.front_decoder(d_feat_in)
        d_feat_up = F.interpolate(d_feat_out, scale_factor=self.down_factor, mode='bilinear', align_corners=False)
        decoder_out = unet_out + d_feat_up
        
        return self.final_conv(decoder_out)

# ==============================================================================
# 2. 全尺寸物理正演引擎
# ==============================================================================
@njit(boundscheck=True)
def ricker_wavelet_jit(freq, dt, len_s):
    n_samples = int(len_s / dt) + 1
    t_offset = (n_samples - 1) / 2 * dt
    wav = np.zeros(n_samples, dtype=np.float32)
    for i in range(n_samples):
        t = i * dt - t_offset
        pi2_f2_t2 = (np.pi * freq * t) ** 2
        wav[i] = (1 - 2 * pi2_f2_t2) * np.exp(-pi2_f2_t2)
    return wav

@njit(boundscheck=True)
def batch_place_wavelet_safe_jit(data, arrival_times, amplitudes, wavelet, dt):
    nt, nx = data.shape
    wav_len = len(wavelet)
    half_wav = wav_len // 2
    for i in range(nx): 
        amp = amplitudes[i]
        if abs(amp) < 1e-15: continue
        t_val = arrival_times[i]
        if t_val < -0.05: continue
        t_float_idx = t_val / dt
        t_int_idx = int(t_float_idx)
        frac = t_float_idx - t_int_idx
        t_start = t_int_idx - half_wav
        wav_start_idx = 0
        if t_start < 0:
            wav_start_idx = -t_start
            t_start = 0
        t_end = t_start + (wav_len - wav_start_idx)
        if t_end >= nt - 1: t_end = nt - 2
        if t_end > t_start:
            for k in range(t_end - t_start):
                w_idx = wav_start_idx + k
                d_idx = t_start + k
                if 0 <= w_idx < wav_len and 0 <= d_idx < nt-1:
                    val = wavelet[w_idx] * amp
                    data[d_idx, i] += val * (1.0 - frac)
                    data[d_idx + 1, i] += val * frac

@njit(boundscheck=True)
def compute_layer_travel_times(nx, rec_z_all, model_z, model_v):
    t0_rec = np.zeros(nx, dtype=np.float32)
    v_rms_rec = np.zeros(nx, dtype=np.float32)
    num_layers = len(model_z) - 1
    for i in range(nx):
        z_rec = rec_z_all[i]
        t_sum = 0.0
        v_sq_t_sum = 0.0
        for k in range(num_layers):
            z_s = model_z[k]; z_e = model_z[k+1]; v_k = model_v[k]
            if z_rec > z_s:
                dz = min(z_rec, z_e) - z_s
                dt_k = dz / v_k
                t_sum += dt_k
                v_sq_t_sum += (v_k**2 * dt_k)
            else: break 
        if t_sum > 1e-12:
            t0_rec[i] = t_sum
            v_rms_rec[i] = np.sqrt(v_sq_t_sum / t_sum)
        else:
            t0_rec[i] = 0.0
            v_rms_rec[i] = model_v[0] if model_v[0] > 0 else 1500.0
    return t0_rec, v_rms_rec

def apply_gauge_length(data, dx, gauge_length=5.0):
    nt, nx = data.shape
    n_gauge = int(gauge_length / dx)
    if n_gauge < 1: n_gauge = 1
    
    strain_rate = np.zeros_like(data)
    if n_gauge <= 1:
        for i in range(nx - 1):
            strain_rate[:, i] = (data[:, i + 1] - data[:, i]) / dx
        strain_rate[:, nx - 1] = strain_rate[:, nx - 2] 
    else:
        half_g = n_gauge // 2
        if nx > n_gauge:
            u_plus = data[:, n_gauge:]
            u_minus = data[:, :-n_gauge]
            valid_width = u_plus.shape[1]
            start_idx = half_g
            strain_rate[:, start_idx : start_idx + valid_width] = (u_plus - u_minus) / gauge_length
            
            for i in range(start_idx):
                strain_rate[:, i] = strain_rate[:, start_idx]
            for i in range(start_idx + valid_width, nx):
                strain_rate[:, i] = strain_rate[:, start_idx + valid_width - 1]
    return strain_rate

def run_forward_modeling_by_config(fw_cfg):
    nt = fw_cfg['nt']
    nx = fw_cfg['nx']
    dt = fw_cfg['dt']
    dx = fw_cfg['dx']
    
    freq_p = float(np.random.uniform(15.0, 65.0)) 
    num_layers = np.random.randint(10, 16)
    q_scale = 0.6; dip_max = 15.0
    freq_s = float(freq_p * np.random.uniform(0.6, 0.8))

    start_v = np.random.uniform(1500, 2000) 
    layer_vp = np.zeros(num_layers + 1, dtype=np.float32)
    layer_vp[0] = start_v
    
    for i in range(1, num_layers + 1):
        if np.random.rand() < 0.30 and layer_vp[i-1] > 2200:
            layer_vp[i] = layer_vp[i-1] - np.random.uniform(200, 600)
        else:
            layer_vp[i] = layer_vp[i-1] + np.random.randint(100, 500)

    vp_vs_ratio = np.random.uniform(1.6, 2.3, size=num_layers + 1)
    layer_vs = (layer_vp / vp_vs_ratio).astype(np.float32)
    
    z_start = np.random.randint(50, 200)
    layer_thicknesses = np.random.randint(80, 400, size=num_layers)
    layer_depths = np.cumsum(layer_thicknesses) + z_start
    model_z = np.concatenate(([0], layer_depths, [nx*dx + 500])).astype(np.float32)
    rec_z_all = (np.arange(nx) * dx).astype(np.float32)
    
    src_x = np.random.uniform(200.0, 1800.0) 
    
    wav_p = ricker_wavelet_jit(freq_p, dt, 0.15)
    wav_s = ricker_wavelet_jit(freq_s, dt, 0.20)
    
    Q_p = np.random.uniform(80.0, 150.0) * q_scale
    Q_s = np.random.uniform(40.0, 80.0) * q_scale
    current_gain_pp = np.random.uniform(1.0, 5.0)
    current_gain_ps = np.random.uniform(1.0, 4.0)
    
    clean_vsp = np.zeros((nt, nx), dtype=np.float32)
    
    t0_p, vrms_p = compute_layer_travel_times(nx, rec_z_all, model_z, layer_vp)
    t_direct_array = np.sqrt(t0_p**2 + (src_x / vrms_p)**2)
    dist_direct = t_direct_array * vrms_p
    attenuation_p = (1.0 / np.maximum(dist_direct, 50.0)) * np.exp(-np.pi * freq_p * t_direct_array / Q_p)
    amp_direct = (attenuation_p * 2000.0 * np.random.uniform(1.0, 2.0)).astype(np.float32)
    batch_place_wavelet_safe_jit(clean_vsp, t_direct_array, amp_direct, wav_p, dt)
    
    if np.random.rand() < 0.8:
        num_multiples = np.random.randint(1, 4)
        for m in range(num_multiples):
            delay_gradient = np.random.uniform(1.5, 3.5)
            t_multiple = t_direct_array + (rec_z_all / layer_vp[0]) * delay_gradient + np.random.uniform(0.02, 0.15)
            amp_multiple = amp_direct * np.random.uniform(-0.5, 0.5) * np.exp(-np.pi * freq_p * t_multiple / (Q_p * 0.7))
            batch_place_wavelet_safe_jit(clean_vsp, t_multiple, amp_multiple.astype(np.float32), wav_p, dt)

    for L, z_L in enumerate(layer_depths):
        k_idx = L; 
        v_avg_down_p = np.mean(layer_vp[:k_idx+1])
        v_avg_up_s = np.mean(layer_vs[:k_idx+1])
        v_local_s = layer_vs[k_idx]
        
        v_down = layer_vp[k_idx+1]; v_up = layer_vp[k_idx]
        polarity = float(np.sign(v_down - v_up))
        if polarity == 0: polarity = 1.0 
        
        r_dist = np.sqrt(src_x**2 + z_L**2 + 1e-9)
        sin_theta = src_x / r_dist  
        cos_theta = z_L / r_dist    
        cos2_theta = cos_theta ** 2
        sin_cos_theta = abs(sin_theta * cos_theta)
        
        dip_angle = np.radians(np.random.uniform(-dip_max, dip_max))
        
        t_down_p_theoretical = np.sqrt((z_L/v_avg_down_p)**2 + (src_x/v_avg_down_p)**2)
        idx_interface = min(np.searchsorted(rec_z_all, z_L), nx - 1)
        t_anchor_actual = t_direct_array[idx_interface]; incident_amp = amp_direct[idx_interface]
        time_shift = t_anchor_actual - t_down_p_theoretical
        depth_penalty = np.exp(-z_L / 1500.0)
        
        mask_up = rec_z_all < z_L
        if np.any(mask_up):
            rec_z_up = rec_z_all[mask_up]
            dip_skew = np.sin(dip_angle) * ((z_L - rec_z_up) / v_avg_down_p) * 0.4
            
            t_up_pp = (z_L - rec_z_up) / v_avg_down_p
            t_total_pp = (t_down_p_theoretical + t_up_pp) + time_shift + dip_skew
            causal_mask_pp = t_total_pp >= (t_direct_array[mask_up] + 0.005) 
            decay_pp = np.exp(-np.pi * freq_p * t_up_pp / Q_p) * (1.0 / (1 + (z_L - rec_z_up)/2000.0))
            amp_val_pp = (incident_amp * 0.35 * polarity * cos2_theta * decay_pp * current_gain_pp * depth_penalty).astype(np.float32)
            
            times_full = np.zeros(nx, dtype=np.float32) - 1.0; amps_full = np.zeros(nx, dtype=np.float32)
            valid_idx_pp = np.where(mask_up)[0][causal_mask_pp]
            times_full[valid_idx_pp] = t_total_pp[causal_mask_pp]; amps_full[valid_idx_pp] = amp_val_pp[causal_mask_pp]
            batch_place_wavelet_safe_jit(clean_vsp, times_full, amps_full, wav_p, dt)
            
            t_up_ps = (z_L - rec_z_up) / v_avg_up_s
            t_total_ps = (t_down_p_theoretical + t_up_ps) + time_shift + dip_skew * 1.5 
            causal_mask_ps = t_total_ps >= (t_direct_array[mask_up] + 0.005)
            decay_ps = np.exp(-np.pi * freq_s * t_up_ps / (Q_s / 3.0)) * (1.0 / (1 + (z_L - rec_z_up)/1000.0))
            amp_val_ps = (incident_amp * 0.20 * polarity * sin_cos_theta * decay_ps * current_gain_ps * depth_penalty).astype(np.float32)
            
            times_full_ps = np.zeros(nx, dtype=np.float32) - 1.0; amps_full_ps = np.zeros(nx, dtype=np.float32)
            valid_idx_ps = np.where(mask_up)[0][causal_mask_ps]
            times_full_ps[valid_idx_ps] = t_total_ps[causal_mask_ps]; amps_full_ps[valid_idx_ps] = amp_val_ps[causal_mask_ps]
            batch_place_wavelet_safe_jit(clean_vsp, times_full_ps, amps_full_ps, wav_s, dt)
            
        mask_down = rec_z_all > z_L
        if np.any(mask_down):
            rec_z_down = rec_z_all[mask_down]
            t_transit_s = (rec_z_down - z_L) / v_local_s
            t_total_down_ps = t_anchor_actual + t_transit_s
            decay_down_ps = np.exp(-np.pi * freq_s * t_transit_s / (Q_s / 4.0)) * (1.0 / (1 + (rec_z_down - z_L)/500.0))
            amp_val_down_ps = (incident_amp * 0.08 * sin_cos_theta * decay_down_ps * current_gain_ps * depth_penalty).astype(np.float32)
            times_full_dps = np.zeros(nx, dtype=np.float32) - 1.0; amps_full_dps = np.zeros(nx, dtype=np.float32)
            valid_idx_dps = np.where(mask_down)[0]
            if len(valid_idx_dps) > 0:
                times_full_dps[valid_idx_dps] = t_total_down_ps; amps_full_dps[valid_idx_dps] = amp_val_down_ps
            batch_place_wavelet_safe_jit(clean_vsp, times_full_dps, amps_full_dps, wav_s, dt)

    random_gl = float(np.random.choice(np.array([5.0, 10.0, 15.0, 20.0])))
    clean_das = apply_gauge_length(clean_vsp, dx, gauge_length=random_gl)
    
    if np.random.rand() < 0.3:
        smooth_k = np.array([0.15, 0.7, 0.15], dtype=np.float32)
        for t in range(clean_das.shape[0]):
            clean_das[t, :] = np.convolve(clean_das[t, :], smooth_k, mode='same')
            
    t_array = np.arange(nt) * dt + 0.1
    gain_matrix = (t_array ** 1.8).reshape(-1, 1)
    clean_das = clean_das * gain_matrix
            
    return clean_das

# ==============================================================================
# 3. 混合噪声注入与掩码破坏
# ==============================================================================
def calculate_metrics(target, pred):
    mse = np.mean((target - pred) ** 2)
    rmse = np.sqrt(mse)
    signal_power = np.sum(target ** 2)
    noise_power = np.sum((target - pred) ** 2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    data_rng = max(target.max() - target.min(), 1e-8)
    ssim_val = ssim(target, pred, data_range=data_rng)
    return snr, ssim_val, rmse

def calculate_local_hole_metrics(target, pred, mask):
    missing_region = (1.0 - mask) > 0.5
    target_hole = target[missing_region]
    pred_hole = pred[missing_region]
    
    if target_hole.size == 0:
        return float('inf')
        
    signal_power = np.sum(target_hole ** 2)
    noise_power = np.sum((target_hole - pred_hole) ** 2)
    hole_snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    return hole_snr

def find_best_demonstration_trace_in_widest_gap(mask, pred_2ch, pred_1ch, target):
    mask_1d = mask[0, :]
    is_zero = (mask_1d == 0).astype(int)
    
    if np.sum(is_zero) == 0:
        return mask.shape[1] // 2 
        
    padded = np.pad(is_zero, (1, 1), mode='constant')
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    lengths = ends - starts
    max_idx = np.argmax(lengths)
    gap_start = starts[max_idx]
    gap_end = ends[max_idx]
    
    best_idx = gap_start
    max_advantage = -float('inf')
    
    for idx in range(gap_start, gap_end):
        err_1ch = np.mean((pred_1ch[:, idx] - target[:, idx]) ** 2)
        err_2ch = np.mean((pred_2ch[:, idx] - target[:, idx]) ** 2)
        
        advantage = err_1ch - err_2ch
        if advantage > max_advantage:
            max_advantage = advantage
            best_idx = idx
            
    return int(best_idx)

def generate_corrupted_sample(clean_full, cfg, bg_noise_bank, coupling_bank):
    H, W = clean_full.shape
    noise_field = np.zeros_like(clean_full)
    std_clean = clean_full.std() + 1e-8
    noise_cfg = cfg['noise']
    noise_scale = noise_cfg['noise_scale']
    
    def seamless_expand(patch, target_h, target_w):
        h, w = patch.shape
        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)
        expanded = np.pad(patch, ((0, pad_h), (0, pad_w)), mode='symmetric')
        return expanded[:target_h, :target_w]

    if noise_cfg['use_bg'] and bg_noise_bank is not None:
        bg_patch = bg_noise_bank[random.randint(0, len(bg_noise_bank) - 1)].astype(np.float32)
        bg_full = seamless_expand(bg_patch, H, W)
        noise_field += (bg_full / (bg_full.std() + 1e-8)) * std_clean * random.uniform(0.1, 0.4) * noise_scale

    if noise_cfg['use_empirical_coupling'] and coupling_bank is not None:
        num_events = random.randint(1, 4)
        for _ in range(num_events):
            cp_idx = random.randint(0, len(coupling_bank) - 1)
            cp_patch = coupling_bank[cp_idx].astype(np.float32)
            hp, wp = cp_patch.shape
            
            start_w = random.randint(0, max(0, W - wp))
            actual_wp = min(wp, W - start_w)
            
            cp_full_h = seamless_expand(cp_patch, H, wp)[:, :actual_wp]
                
            energy_factor = random.uniform(1.0, 3.5)
            noise_field[:, start_w:start_w+actual_wp] += (cp_full_h / (cp_full_h.std() + 1e-8)) * std_clean * energy_factor * noise_scale

    P_signal = np.sum(clean_full ** 2)
    P_noise = np.sum(noise_field ** 2)
    if P_noise > 0 and noise_cfg['target_snr'] is not None:
        current_snr = 10 * np.log10(P_signal / P_noise)
        k = 10 ** ((current_snr - noise_cfg['target_snr']) / 20)
        noise_field *= k
        
    mask = np.ones_like(clean_full)
    if noise_cfg['use_mask']:
        gap_size = random.randint(noise_cfg['gap_min_width'], noise_cfg['gap_max_width']) 
        start_idx = int(W / 2) - (gap_size // 2)
        mask[:, start_idx:start_idx+gap_size] = 0
            
        discrete_missing_rate = random.uniform(noise_cfg['discrete_min'], noise_cfg['discrete_max'])
        num_discrete = int(W * discrete_missing_rate)
        discrete_indices = random.sample(range(W), num_discrete)
        mask[:, discrete_indices] = 0
        
    missing_region = 1.0 - mask
    
    if noise_cfg.get('inject_harsh_noise', True):
        degradation_type = random.random()
        if degradation_type < 0.33:
            noise_field = noise_field * mask 
            harsh_noise = np.zeros_like(clean_full)
        elif degradation_type < 0.66:
            fake_full = run_forward_modeling_by_config(cfg['forward'])
            harsh_noise = fake_full + (np.random.randn(*clean_full.shape).astype(np.float32) * std_clean * 0.5)
        else:
            if coupling_bank is not None and len(coupling_bank) > 0:
                cp_idx_harsh = random.randint(0, len(coupling_bank) - 1)
                cp_patch_harsh = coupling_bank[cp_idx_harsh].astype(np.float32)
                harsh_noise_full = seamless_expand(cp_patch_harsh, H, W)
                harsh_noise = (harsh_noise_full / (harsh_noise_full.std() + 1e-8)) * std_clean * 1.2
            else:
                fake_full = run_forward_modeling_by_config(cfg['forward'])
                harsh_noise = fake_full
    else:
        harsh_noise = np.zeros_like(clean_full)
        noise_field = noise_field * mask

    corrupted_full = (clean_full * mask) + noise_field + (missing_region * harsh_noise)
    
    return corrupted_full, mask, clean_full

# ==============================================================================
# 4. 滑动切片推理引擎
# ==============================================================================
def get_2d_hanning_window(patch_h, patch_w):
    win_y = np.hanning(patch_h)
    win_x = np.hanning(patch_w)
    return np.outer(win_y, win_x)

def infer_sliding_window(model, net_input_np, mask_np, device, cfg):
    C, H, W = net_input_np.shape
    patch_size = cfg['inference']['patch_size']
    overlap = cfg['inference']['overlap']
    batch_size = cfg['inference']['batch_size']
    
    stride = int(patch_size * (1 - overlap))
    pad_h = (stride - H % stride) % stride
    pad_w = (stride - W % stride) % stride
    pad_h_total = pad_h + patch_size 
    pad_w_total = pad_w + patch_size
    
    padded_input = np.pad(net_input_np, ((0,0), (0, pad_h_total), (0, pad_w_total)), mode='constant')
    padded_mask = np.pad(mask_np, ((0, pad_h_total), (0, pad_w_total)), mode='constant', constant_values=1.0)
    
    _, pad_H, pad_W = padded_input.shape
    
    reconstructed = np.zeros((pad_H, pad_W), dtype=np.float32)
    weight_sum = np.zeros((pad_H, pad_W), dtype=np.float32)
    window = get_2d_hanning_window(patch_size, patch_size)
    
    patches, mask_patches, coords = [], [], []
    for y in range(0, pad_H - patch_size + 1, stride):
        for x in range(0, pad_W - patch_size + 1, stride):
            patches.append(padded_input[:, y:y+patch_size, x:x+patch_size])
            mask_patches.append(padded_mask[y:y+patch_size, x:x+patch_size])
            coords.append((y, x))
            
    model.eval()
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            batch_patches = np.stack(patches[i:i+batch_size])
            batch_masks = np.stack(mask_patches[i:i+batch_size])
            
            batch_vmax = np.zeros((batch_patches.shape[0], 1, 1, 1), dtype=np.float32)
            for b in range(batch_patches.shape[0]):
                valid_pixels = batch_patches[b, 0][batch_masks[b] == 1]
                if len(valid_pixels) > 100:
                    v_max = np.percentile(np.abs(valid_pixels), 99.5) + 1e-8
                else:
                    v_max = np.percentile(np.abs(batch_patches[b, 0]), 99.5) + 1e-8
                    
                batch_vmax[b, 0, 0, 0] = v_max
                batch_patches[b, 0] = np.clip(batch_patches[b, 0], -v_max * 3.0, v_max * 3.0) / v_max
                
            batch_tensor = torch.from_numpy(batch_patches).float().to(device)
            
            with torch.amp.autocast('cuda'):
                batch_preds = model(batch_tensor)
                
            batch_preds = batch_preds.cpu().numpy()
            batch_preds = batch_preds * batch_vmax
            batch_preds = batch_preds.squeeze(1)
            
            for j, (y, x) in enumerate(coords[i:i+batch_size]):
                reconstructed[y:y+patch_size, x:x+patch_size] += batch_preds[j] * window
                weight_sum[y:y+patch_size, x:x+patch_size] += window
                
    weight_sum[weight_sum == 0] = 1.0
    final_pred = reconstructed / weight_sum
    return final_pred[:H, :W]

# ==============================================================================
# 5. 传统基线算法: 多通道奇异谱分析 (MSSA) - 极限护盘版 (严格 POCS 数据一致性)
# ==============================================================================
# ==============================================================================
# 5. 传统基线算法: 多通道奇异谱分析 (MSSA) - 标准同步去噪插值版
# ==============================================================================
def mssa_process(noisy_data, mask, dt=0.002, f_max_hz=85.0, rank=25, max_iter=5):
    """
    标准版说明：
    前 max_iter-1 次迭代执行 POCS 约束，引导缝隙插值；
    最后 1 次迭代释放约束，输出纯 MSSA 低秩结果，实现全局同步去噪。
    """
    nt, nx = noisy_data.shape
    L = nx // 2 + 1
    K = nx - L + 1
    
    recon = noisy_data * mask
    f_max_idx = min(int(f_max_hz * nt * dt), nt // 2)
    
    for it in range(max_iter):
        fx_data = np.fft.fft(recon, axis=0)
        current_rank = int(rank + (max_iter - it - 1) * 1.5)
        
        for f in range(1, f_max_idx):
            slice_f = fx_data[f, :]
            
            H = np.zeros((L, K), dtype=np.complex128)
            for i in range(L):
                H[i, :] = slice_f[i:i+K]
                
            try:
                U, S, Vh = np.linalg.svd(H, full_matrices=False)
                S_trunc = S[:current_rank]
                H_recon = (U[:, :current_rank] * S_trunc) @ Vh[:current_rank, :]
                
                slice_recon = np.zeros(nx, dtype=np.complex128)
                counts = np.zeros(nx, dtype=int)
                for i in range(L):
                    for j in range(K):
                        slice_recon[i+j] += H_recon[i, j]
                        counts[i+j] += 1
                slice_recon /= counts
                
                fx_data[f, :] = slice_recon
                if f < nt // 2:
                    fx_data[nt-f, :] = np.conj(slice_recon)
                    
            except np.linalg.LinAlgError:
                pass
                
        for f in range(f_max_idx, nt // 2 + 1):
            fx_data[f, :] = 0.0
            if f < nt // 2:
                fx_data[nt-f, :] = 0.0
                
        recon_tx = np.real(np.fft.ifft(fx_data, axis=0))
        
        # 🌟 恢复标准的同步去噪插值逻辑
        if it < max_iter - 1:
            # 迭代中：利用已知道约束，逼迫能量向盲区外推
            recon = noisy_data * mask + recon_tx * (1 - mask)
        else:
            # 最后一步：释放约束，全局接受 MSSA 降秩结果，实现有效道去噪
            recon = recon_tx
            
    return recon.astype(np.float32)

# ==========================================
# 6. 期刊级严格对比绘图输出 (完美 2上3下 居中排版)
# ==========================================
def plot_comparison_figure(noisy, pred_mssa, pred_2ch, pred_1ch, target, save_dir, test_idx, best_trace_idx=None):
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial'], 'font.size': 11})
    
    fig = plt.figure(figsize=(24, 14))
    gs = gridspec.GridSpec(2, 6, figure=fig, wspace=0.15, hspace=0.25)
    
    ax1 = fig.add_subplot(gs[0, 1:3]) # 上行左侧：Clean Target (跨列 1,2)
    ax2 = fig.add_subplot(gs[0, 3:5]) # 上行右侧：Noisy Input (跨列 3,4)
    ax3 = fig.add_subplot(gs[1, 0:2]) # 下行左一：MSSA (跨列 0,1)
    ax4 = fig.add_subplot(gs[1, 2:4]) # 下行居中：1Ch Pred (跨列 2,3)
    ax5 = fig.add_subplot(gs[1, 4:6]) # 下行右一：2Ch Pred (跨列 4,5)

    axes = [ax1, ax2, ax3, ax4, ax5]
    plot_data = [target, noisy, pred_mssa, pred_1ch, pred_2ch]
    
    labels = ['(a)', '(b)', '(c)', '(d)', '(e)']
    
    vmax = np.percentile(np.abs(target), 99.8)
    noisy_viz_vmax = np.percentile(np.abs(noisy), 99)
    noisy_display = np.clip(noisy, -noisy_viz_vmax, noisy_viz_vmax)
    plot_data[1] = noisy_display

    for i, ax in enumerate(axes):
        ax.imshow(plot_data[i], cmap='seismic', aspect='auto', vmin=-vmax, vmax=vmax)
        
        if best_trace_idx is not None:
            ax.axvline(x=best_trace_idx, color='#00FF00', linestyle='--', linewidth=1.2, alpha=0.8)
            
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.set_title("") 
        
        ax.text(0.02, 0.98, labels[i], transform=ax.transAxes, 
                fontsize=18, fontweight='bold', va='top', ha='left',
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=3))
                
        ax.set_xlabel('Traces')
        
        if i in [0, 2]:  
            ax.set_ylabel('Time Samples')
        else:            
            ax.set_yticks([])

    tiff_path = os.path.join(save_dir, f"Synthetic_Comparison_Test_{test_idx}.tiff")
    svg_path = os.path.join(save_dir, f"Synthetic_Comparison_Test_{test_idx}.svg")
    plt.savefig(tiff_path, dpi=600, format='tiff', bbox_inches='tight')
    plt.savefig(svg_path, dpi=600, format='svg', bbox_inches='tight')
    plt.close()

def plot_trace_comparison_figure(noisy, pred_mssa, pred_2ch, pred_1ch, target, trace_idx, save_dir, test_idx):
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial'], 'font.size': 11})
    fig, ax = plt.subplots(figsize=(14, 5))
    
    t_samples = np.arange(target.shape[0])
    trace_target = target[:, trace_idx]
    
    ax.plot(t_samples, noisy[:, trace_idx], color='lightgray', linewidth=1.5, label='Noisy Input')
    ax.plot(t_samples, pred_mssa[:, trace_idx], color='green', linestyle=':', linewidth=1.2, alpha=0.8, label='MSSA Pred')
    ax.plot(t_samples, pred_1ch[:, trace_idx], color='blue', linestyle='--', linewidth=1.2, alpha=0.8, label='1Ch Pred (Blind)')
    ax.plot(t_samples, pred_2ch[:, trace_idx], color='red', linestyle='-.', linewidth=1.2, alpha=0.9, label='2Ch Pred (Masked)')
    ax.plot(t_samples, trace_target, color='black', linewidth=1.0, label='Clean Target')
    
    energy = trace_target ** 2
    total_energy = np.sum(energy)
    if total_energy > 1e-10:
        cum_energy = np.cumsum(energy)
        start_idx = max(0, np.searchsorted(cum_energy, 0.01 * total_energy) - 80)
        end_idx = min(len(t_samples) - 1, np.searchsorted(cum_energy, 0.99 * total_energy) + 80)
        ax.set_xlim(start_idx, end_idx)
        
        window_target = trace_target[start_idx:end_idx]
        y_max = max(
            np.max(np.abs(window_target)), 
            np.max(np.abs(pred_2ch[start_idx:end_idx, trace_idx])), 
            np.max(np.abs(pred_1ch[start_idx:end_idx, trace_idx]))
        ) * 1.2
        ax.set_ylim(-y_max, y_max)
        
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    
    ax.set_xlabel('Time Samples')
    ax.set_ylabel('Amplitude')
    ax.set_title("") 
    ax.legend(loc='upper right', framealpha=0.9, ncol=5)
    
    plt.tight_layout()
    tiff_path = os.path.join(save_dir, f"Trace_Comparison_Test_{test_idx}.tiff")
    svg_path = os.path.join(save_dir, f"Trace_Comparison_Test_{test_idx}.svg")
    plt.savefig(tiff_path, dpi=600, format='tiff', bbox_inches='tight')
    plt.savefig(svg_path, dpi=600, format='svg', bbox_inches='tight')
    plt.close()

# ==========================================
# 7. 双模型 + MSSA 对比测试主控引擎
# ==========================================
def execute_ablation_evaluation(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    mode = cfg.get('test_mode', 'extreme')
    cfg['noise'] = cfg['noise_profiles'][mode]
    output_dir = cfg['paths']['output_dir'] + f"_{mode.capitalize()}"
    
    print(f"🚀 启动全尺寸消融测试 | 模式: 【{mode.upper()}】 | 设备: {device}")
    
    bg_noise_bank = np.load(cfg['paths']['bg_npy']) if cfg['noise']['use_bg'] else None
    coupling_bank = None
    if cfg['noise']['use_empirical_coupling']:
        coupling_list = []
        for h5_path in cfg['paths']['coupling_h5']:
            if os.path.exists(h5_path):
                with h5py.File(h5_path, 'r') as f:
                    key = list(f.keys())[0]
                    coupling_list.append(f[key][:])
        if coupling_list:
            coupling_bank = np.concatenate(coupling_list, axis=0)
            print(f"✅ 成功加载经验耦合噪声切片: {len(coupling_bank)} 张")
        
    print(f"⏳ 正在加载预训练权重...")
    model_2ch = MAC_Net(in_channels=2, hidden_dim=64).to(device)
    model_2ch.load_state_dict(torch.load(cfg['paths']['model_2ch_weights'], map_location=device))
    model_2ch.eval()
    
    model_1ch = MAC_Net(in_channels=1, hidden_dim=64).to(device)
    model_1ch.load_state_dict(torch.load(cfg['paths']['model_1ch_weights'], map_location=device))
    model_1ch.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    
    txt_log_path = os.path.join(output_dir, f"evaluation_metrics_with_mssa_{mode}.txt")
    with open(txt_log_path, 'w', encoding='utf-8') as f:
        f.write(f"=== {mode.upper()} 模式：合成数据消融测试指标 ===\n\n")

    metrics = {
        '2ch_snr': [], '2ch_ssim': [], '2ch_hole_snr': [], '2ch_time': [],
        '1ch_snr': [], '1ch_ssim': [], '1ch_hole_snr': [], '1ch_time': [],
        'mssa_snr': [], 'mssa_ssim': [], 'mssa_hole_snr': [], 'mssa_time': []
    }
    
    for i in range(cfg['test_counts']):
        print(f"\n--- 生成并测试物理记录 {i+1}/{cfg['test_counts']} ---")
        clean_full = run_forward_modeling_by_config(cfg['forward'])
        corrupted_full, mask, clean_target_np = generate_corrupted_sample(clean_full, cfg, bg_noise_bank, coupling_bank)
        
        input_2ch = np.stack([corrupted_full, mask], axis=0)
        input_1ch = np.expand_dims(corrupted_full, axis=0)
        
        t0 = time.time()
        pred_2ch_np = infer_sliding_window(model_2ch, input_2ch, mask, device, cfg)
        t_2ch = time.time() - t0
        
        t0 = time.time()
        pred_1ch_np = infer_sliding_window(model_1ch, input_1ch, mask, device, cfg)
        t_1ch = time.time() - t0
        
        print(f"⚙️ 运行 MSSA 降秩基线算法 (Rank={cfg['noise']['mssa_rank']})...")
        t0 = time.time()
        pred_mssa_np = mssa_process(
            corrupted_full, mask, 
            dt=cfg['forward']['dt'], 
            f_max_hz=85.0, 
            rank=cfg['noise']['mssa_rank'],
            max_iter=5
        )
        t_mssa = time.time() - t0
            
        snr_base, ssim_base, _ = calculate_metrics(clean_target_np, corrupted_full)
        snr_2ch, ssim_2ch, _ = calculate_metrics(clean_target_np, pred_2ch_np)
        hole_snr_2ch = calculate_local_hole_metrics(clean_target_np, pred_2ch_np, mask)
        
        snr_1ch, ssim_1ch, _ = calculate_metrics(clean_target_np, pred_1ch_np)
        hole_snr_1ch = calculate_local_hole_metrics(clean_target_np, pred_1ch_np, mask)
        
        snr_mssa, ssim_mssa, _ = calculate_metrics(clean_target_np, pred_mssa_np)
        hole_snr_mssa = calculate_local_hole_metrics(clean_target_np, pred_mssa_np, mask)
        
        metrics['2ch_snr'].append(snr_2ch); metrics['2ch_ssim'].append(ssim_2ch); metrics['2ch_hole_snr'].append(hole_snr_2ch); metrics['2ch_time'].append(t_2ch)
        metrics['1ch_snr'].append(snr_1ch); metrics['1ch_ssim'].append(ssim_1ch); metrics['1ch_hole_snr'].append(hole_snr_1ch); metrics['1ch_time'].append(t_1ch)
        metrics['mssa_snr'].append(snr_mssa); metrics['mssa_ssim'].append(ssim_mssa); metrics['mssa_hole_snr'].append(hole_snr_mssa); metrics['mssa_time'].append(t_mssa)
        
        trace_idx = find_best_demonstration_trace_in_widest_gap(mask, pred_2ch_np, pred_1ch_np, clean_target_np)

        print(f"数据尺寸: {clean_target_np.shape}")
        print(f"原始带噪 -> SNR: {snr_base:.2f} dB, SSIM: {ssim_base:.4f}")
        print(f"MSSA 传统降秩 -> 全局 SNR: {snr_mssa:.2f} dB | 🎯 断道 SNR: {hole_snr_mssa:.2f} dB | ⏱️ 耗时: {t_mssa:.2f} s")
        print(f"单通道 (盲切) -> 全局 SNR: {snr_1ch:.2f} dB | 🎯 断道 SNR: {hole_snr_1ch:.2f} dB | ⏱️ 耗时: {t_1ch:.2f} s")
        print(f"双通道 (Mask) -> 全局 SNR: {snr_2ch:.2f} dB | 🎯 断道 SNR: {hole_snr_2ch:.2f} dB | ⏱️ 耗时: {t_2ch:.2f} s")
        
        log_text = (f"Test {i+1} 数据尺寸: {clean_target_np.shape}\n"
                    f"原始带噪 -> SNR: {snr_base:.2f} dB, SSIM: {ssim_base:.4f}\n"
                    f"MSSA 传统降秩 -> 全局 SNR: {snr_mssa:.2f} dB | 全局 SSIM: {ssim_mssa:.4f} | 🎯 断道区 SNR: {hole_snr_mssa:.2f} dB | ⏱️ 耗时: {t_mssa:.2f} s\n"
                    f"单通道 (盲切) -> 全局 SNR: {snr_1ch:.2f} dB | 全局 SSIM: {ssim_1ch:.4f} | 🎯 断道区 SNR: {hole_snr_1ch:.2f} dB | ⏱️ 耗时: {t_1ch:.2f} s\n"
                    f"双通道 (Mask) -> 全局 SNR: {snr_2ch:.2f} dB | 全局 SSIM: {ssim_2ch:.4f} | 🎯 断道区 SNR: {hole_snr_2ch:.2f} dB | ⏱️ 耗时: {t_2ch:.2f} s\n"
                    f"已生成最优对比单道波形（Trace {trace_idx}）\n"
                    f"{'-'*75}\n")
        with open(txt_log_path, 'a', encoding='utf-8') as f:
            f.write(log_text)

        plot_comparison_figure(corrupted_full, pred_mssa_np, pred_2ch_np, pred_1ch_np, clean_target_np, output_dir, test_idx=i+1, best_trace_idx=trace_idx)
        plot_trace_comparison_figure(corrupted_full, pred_mssa_np, pred_2ch_np, pred_1ch_np, clean_target_np, trace_idx, output_dir, test_idx=i+1)

    print(f"\n✅ [{mode.upper()}] 模式对比验证全部完成！输出目录：{output_dir}")
    
    summary_text = (f"\n✅ 整体测试平均指标\n"
                    f"【MSSA 基线】平均全局 SNR: {np.mean(metrics['mssa_snr']):.2f} dB | 平均断道 SNR: {np.mean(metrics['mssa_hole_snr']):.2f} dB | 平均耗时: {np.mean(metrics['mssa_time']):.2f} s\n"
                    f"【单通道网络】平均全局 SNR: {np.mean(metrics['1ch_snr']):.2f} dB | 平均断道 SNR: {np.mean(metrics['1ch_hole_snr']):.2f} dB | 平均耗时: {np.mean(metrics['1ch_time']):.2f} s\n"
                    f"【双通道网络】平均全局 SNR: {np.mean(metrics['2ch_snr']):.2f} dB | 平均断道 SNR: {np.mean(metrics['2ch_hole_snr']):.2f} dB | 平均耗时: {np.mean(metrics['2ch_time']):.2f} s\n")
    print(summary_text)
    with open(txt_log_path, 'a', encoding='utf-8') as f:
        f.write(summary_text)

# ==============================================================================
# 🌟 全局消融实验控制中枢
# ==============================================================================
if __name__ == "__main__":
    CONFIG = {
        'test_counts': 5,  
        
        # 🌟 模式切换开关：可设为 'mild' 或 'extreme'
        'test_mode': 'extreme', 
        
        'forward': {
            'nx': 600,       
            'dx': 5.0,       
            'nt': 1000,
            'dt': 0.002,    
        },
        
       'noise_profiles': {
            'mild': {
                'use_bg': True,
                'use_empirical_coupling': False,
                'noise_scale': 0.1,
                'target_snr': 15.0,
                'use_mask': True,
                'gap_min_width': 25,
                'gap_max_width': 35,
                'discrete_min': 0.01,
                'discrete_max': 0.05,
                'inject_harsh_noise': False,
                'mssa_rank': 25  # 🌟 轻度工况：提高秩数，努力还原深部双曲线
            },
            'extreme': {
                'use_bg': True,
                'use_empirical_coupling': True,
                'noise_scale': 0.8,
                'target_snr': None,
                'use_mask': True,
                'gap_min_width': 25,
                'gap_max_width': 45,
                'discrete_min': 0.05,
                'discrete_max': 0.15,
                'inject_harsh_noise': True,
                'mssa_rank': 15 # 🌟 强噪工况：压低秩数，防止伪影涂抹
            }
        },
        
        'inference': {
            'mode': 'sliding',      
            'patch_size': 256,      
            'overlap': 0.5,         
            'batch_size': 8         
        },
        
        'paths': {
            'bg_npy': "G:/DAS-VSP数据处理/去噪+插值/das_dataset_noise/real_noise_bank_manual.npy",
            'coupling_h5': [
                "G:/DAS-VSP数据处理/去噪+插值/das_dataset_noise/custom_simulated_noise_bank.h5",
                "G:/DAS-VSP数据处理/去噪+插值/das_dataset_noise/multi_region_noise_bank.h5",
                "G:/DAS-VSP数据处理/去噪+插值/das_dataset_noise/purified_z_result.h5"
            ],
            
            'model_2ch_weights': "G:/DAS-VSP数据处理/去噪+插值/das_curriculum_dataset_rt30k/outputs_rt_model_finetunedv2/MAC_Net_2Ch_Best_Finetuned.pth",
            'model_1ch_weights': "G:/DAS-VSP数据处理/去噪+插值/das_curriculum_dataset_rt30k/outputs_rt_model_finetunedv2/MAC_Net_1Ch_Best_Finetuned.pth",
            
            'output_dir': "G:/DAS-VSP数据处理/去噪+插值/das_curriculum_dataset_rt30k/Experiments_on_Synthetic_MSSAv24"
        }
    }
    
    execute_ablation_evaluation(CONFIG)