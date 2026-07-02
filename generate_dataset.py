import numpy as np
import h5py
import time
import os
from numba import njit

# ==============================================================================
# 1. 核心计算
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

# ==============================================================================
# 2. 按难度分级的随机地质与正演引擎
# ==============================================================================
def run_single_random_shot(difficulty):
    nt = 1000
    nx = 350
    dt = 0.002
    dx = 5.0
    
    if difficulty == 'easy':
        freq_p = float(np.random.uniform(30.0, 45.0))
        num_layers = np.random.randint(4, 7)
        q_scale = 1.5; dip_max = 0.0
    elif difficulty == 'medium':
        freq_p = float(np.random.uniform(25.0, 55.0))
        num_layers = np.random.randint(7, 12)
        q_scale = 1.0; dip_max = 5.0
    else: 
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

    # 让每一层的 Vp/Vs 比例独立随机变化
    vp_vs_ratio = np.random.uniform(1.6, 2.3, size=num_layers + 1)
    layer_vs = (layer_vp / vp_vs_ratio).astype(np.float32)
    
    z_start = np.random.randint(50, 200)
    layer_thicknesses = np.random.randint(80, 400, size=num_layers)
    layer_depths = np.cumsum(layer_thicknesses) + z_start
    model_z = np.concatenate(([0], layer_depths, [nx*dx + 500])).astype(np.float32)
    rec_z_all = (np.arange(nx) * dx).astype(np.float32)
    
    src_x = np.random.uniform(0.0, 1800.0) 
    
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
    
    # 生成 1 到 3 条斜率和延迟各异的多次波，极大地增加波场干涉复杂度
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
# 3. 【核心物理隔离】生成独立的 Train 和 Val 数据集
# ==============================================================================
def fill_dataset_split(dset, target_patches, difficulty, split_name, patch_h, patch_w):
    saved_count = 0
    shot_count = 0
    
    while saved_count < target_patches:
        shot_count += 1
        try:
            full_wavefield = run_single_random_shot(difficulty)
        except Exception as e:
            continue
            
        H, W = full_wavefield.shape
        
        patches_from_this_shot = 0
        attempts = 0
        
        while patches_from_this_shot < 10 and attempts < 100:
            attempts += 1
            if saved_count >= target_patches: break
                
            r_start = np.random.randint(0, H - patch_h)
            c_start = np.random.randint(0, W - patch_w)
            patch = full_wavefield[r_start:r_start+patch_h, c_start:c_start+patch_w]
            
            if np.max(np.abs(patch)) > 1e-6 and np.std(patch) > 1e-8: 
                vmax = np.percentile(np.abs(patch), 99.5) + 1e-9
                patch_normalized = np.clip(patch / vmax, -1.0, 1.0)
                
                if np.random.rand() > 0.5:
                    patch_normalized = -patch_normalized
                
                dset[saved_count] = patch_normalized
                saved_count += 1
                patches_from_this_shot += 1
        
        print(f"   [{split_name.upper()} - {difficulty}] 炮集 #{shot_count} 提取 {patches_from_this_shot} 张 -> 进度: {saved_count}/{target_patches}")

def build_isolated_curriculum_datasets(save_dir, plan_dict, val_split=0.2, patch_h=256, patch_w=256):
    print(f"\n🚀 启动全自动物理隔离数据集工厂 (最终版)...")
    start_time = time.time()
    os.makedirs(save_dir, exist_ok=True)
    
    train_path = os.path.join(save_dir, "train_dataset.h5")
    val_path = os.path.join(save_dir, "val_dataset.h5")
    
    with h5py.File(train_path, 'w') as f_train, h5py.File(val_path, 'w') as f_val:
        for difficulty, total_patches in plan_dict.items():
            if total_patches == 0: continue
            
            val_patches = int(total_patches * val_split)
            train_patches = total_patches - val_patches
            
            print(f"\n==================================================")
            print(f"🔥 开始生成 [{difficulty.upper()}] 难度阵列 (Train: {train_patches}, Val: {val_patches})")
            print(f"==================================================")
            
            dset_name = f'clean_data_{difficulty}'
            dset_train = f_train.create_dataset(dset_name, shape=(train_patches, patch_h, patch_w), dtype='f4', chunks=(1, patch_h, patch_w))
            dset_val = f_val.create_dataset(dset_name, shape=(val_patches, patch_h, patch_w), dtype='f4', chunks=(1, patch_h, patch_w))
            
            print(f"\n--- 🏭 阶段 1：构建 Train 隔离集 ---")
            fill_dataset_split(dset_train, train_patches, difficulty, "train", patch_h, patch_w)
            
            print(f"\n--- 🏭 阶段 2：构建 Val (Test) 隔离集 ---")
            fill_dataset_split(dset_val, val_patches, difficulty, "val", patch_h, patch_w)
                
    print(f"\n✅ 全部物理隔离数据构建完成!")
    print(f"   📂 训练集: {train_path}")
    print(f"   📂 验证集: {val_path}")
    print(f"⏱️ 总耗时: {(time.time() - start_time) / 60:.2f} 分钟")

# ==============================================================================
if __name__ == "__main__":
    curriculum_plan = {
        'easy': 3000,   
        'medium': 9000,  
        'hard': 18000     
    }
    
    save_directory = "G:/DAS-VSP数据处理/去噪+插值/das_curriculum_dataset_rt30k"
    build_isolated_curriculum_datasets(save_directory, plan_dict=curriculum_plan, val_split=0.2)