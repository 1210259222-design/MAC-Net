import os
# 解决 PyTorch 与 Matplotlib 画图时的 OpenMP 底层冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import random
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.fft
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json
import gc

# 🌟 引入多尺度结构相似度
try:
    from pytorch_msssim import MS_SSIM
except ImportError:
    os.system("pip install pytorch-msssim")
    from pytorch_msssim import MS_SSIM

# ==============================================================================
# 1. 基础组件与网络架构 (含特征空间门控迟融合)
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


# ==========================================
# 2. 统一的数据集生成器 (🌟 锚点修复版)
# ==========================================
class DASUnifiedDataset(Dataset):
    def __init__(self, clean_h5_path, noise_paths, difficulties, patch_size=256):
        super().__init__()
        self.patch_size = patch_size
        self.clean_patches = []
        
        with h5py.File(clean_h5_path, 'r') as f:
            for diff in difficulties:
                dataset_name = f'clean_data_{diff}'
                if dataset_name in f:
                    self.clean_patches.append(f[dataset_name][:])
        self.clean_patches = np.concatenate(self.clean_patches, axis=0)
        self.num_samples = len(self.clean_patches)
        self.bg_noise_data = np.load(noise_paths['bg_npy'])
        
        self.coupling_noises = []
        for h5_path in noise_paths['coupling_h5']:
            if os.path.exists(h5_path):
                with h5py.File(h5_path, 'r') as f:
                    key = list(f.keys())[0]
                    self.coupling_noises.append(f[key][:])
        if self.coupling_noises:
            self.coupling_noises = np.concatenate(self.coupling_noises, axis=0)

    def __len__(self): return self.num_samples

    def __getitem__(self, idx):
        clean_patch = torch.from_numpy(self.clean_patches[idx]).float()
        std_clean = clean_patch.std() + 1e-8

        # 底噪
        noise_field = torch.zeros_like(clean_patch)
        bg_idx = random.randint(0, len(self.bg_noise_data) - 1)
        bg_noise = torch.from_numpy(self.bg_noise_data[bg_idx]).float()
        bg_energy = random.uniform(0.1, 0.4) 
        noise_field += (bg_noise / (bg_noise.std() + 1e-8)) * std_clean * bg_energy

        if len(self.coupling_noises) > 0 and random.random() < 0.6:
            cp_idx = random.randint(0, len(self.coupling_noises) - 1)
            cp_noise_np = self.coupling_noises[cp_idx]
            H, W = cp_noise_np.shape
            r = random.randint(0, max(0, H - self.patch_size))
            c = random.randint(0, max(0, W - self.patch_size))
            cp_crop = cp_noise_np[r:r+self.patch_size, c:c+self.patch_size]
            cp_noise = torch.from_numpy(cp_crop).float()
            cp_energy = random.uniform(0.8, 2.0)
            noise_field += (cp_noise / (cp_noise.std() + 1e-8)) * std_clean * cp_energy

        # 掩码
        mask = torch.ones_like(clean_patch)
        if random.random() < 0.6:
            gap_size = random.randint(15, 45) 
            start_idx = random.randint(10, self.patch_size - gap_size - 10)
            mask[:, start_idx:start_idx+gap_size] = 0
            
        discrete_missing_rate = random.uniform(0.1, 0.3)
        num_discrete = int(self.patch_size * discrete_missing_rate)
        discrete_indices = random.sample(range(self.patch_size), num_discrete)
        mask[:, discrete_indices] = 0

        missing_region = 1.0 - mask
        degradation_type = random.random()
        
        if degradation_type < 0.33:
            noise_field = noise_field * mask 
            harsh_noise = torch.zeros_like(clean_patch)
        elif degradation_type < 0.66:
            fake_idx = random.randint(0, self.num_samples - 1)
            fake_patch = torch.from_numpy(self.clean_patches[fake_idx]).float()
            harsh_noise = fake_patch + (torch.randn_like(clean_patch) * std_clean * 0.5)
        else:
            if len(self.coupling_noises) > 0:
                cp_idx_harsh = random.randint(0, len(self.coupling_noises) - 1)
                cp_patch_harsh = self.coupling_noises[cp_idx_harsh]
                r_h = random.randint(0, max(0, cp_patch_harsh.shape[0] - self.patch_size))
                c_h = random.randint(0, max(0, cp_patch_harsh.shape[1] - self.patch_size))
                harsh_noise_np = cp_patch_harsh[r_h:r_h+self.patch_size, c_h:c_h+self.patch_size]
                harsh_noise = torch.from_numpy(harsh_noise_np).float()
                harsh_noise = (harsh_noise / (harsh_noise.std() + 1e-8)) * std_clean * 1.2
            else:
                fake_idx = random.randint(0, self.num_samples - 1)
                harsh_noise = torch.from_numpy(self.clean_patches[fake_idx]).float()

        corrupted_patch = (clean_patch * mask) + noise_field + (missing_region * harsh_noise)
        
        # 🌟 修复归一化：强制以“干净数据”为缩放锚点，防止恶劣噪声摧毁弱波场的动态范围
        vmax = torch.quantile(torch.abs(clean_patch), 0.995) + 1e-8
        
        # 允许破损数据的噪声稍稍溢出 [-1, 1] (至 [-3, 3])，保证网络看到真实的高差梯度
        corrupted_patch = torch.clamp(corrupted_patch, min=-vmax*3.0, max=vmax*3.0) / vmax
        clean_patch = torch.clamp(clean_patch, min=-vmax, max=vmax) / vmax

        return {
            "noisy": corrupted_patch.unsqueeze(0),
            "clean": clean_patch.unsqueeze(0),
            "mask": mask.unsqueeze(0) 
        }

# ==========================================
# 3. 绘图与可视化
# ==========================================
def plot_research_loss(history, save_dir, stage_boundaries, current_epoch, prefix):
    plt.style.use('seaborn-v0_8-paper')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    epochs = range(1, current_epoch + 1)
    
    ax1 = axes[0]
    ax1.plot(epochs, history['train_total'], label='Train Total Loss', color='#1f77b4', linewidth=2)
    ax1.plot(epochs, history['val_total'], label='Validation Total Loss', color='#ff7f0e', linewidth=2.5, linestyle='--')
    for sb in stage_boundaries:
        if current_epoch > sb:
            ax1.axvline(x=sb, color='gray', linestyle=':', linewidth=1.5)
            ax1.text(sb + 0.5, max(history['train_total']) * 0.9, f'Stage Shift', color='gray', fontsize=10)
    ax1.set_title(f'Learning Curve: Total Loss ({prefix})', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs'); ax1.set_ylabel('Loss Value')
    ax1.legend(); ax1.grid(True, alpha=0.3); ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True)) 
    
    ax2 = axes[1]
    ax2.plot(epochs, history['train_base'], label='Log-L1 Loss', color='#2ca02c', linewidth=1.5)
    ax2.plot(epochs, history['train_fft'], label='FFT Frequency Loss', color='#9467bd', linewidth=1.5)
    ax2.plot(epochs, history['train_ssim'], label='MS-SSIM Loss', color='#d62728', linewidth=1.5)
    
    ax2.set_title('Training Components Decomposition', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs'); ax2.set_ylabel('Component Loss Value')
    ax2.set_yscale('log'); ax2.legend(); ax2.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"MAC_Net_{prefix}_Loss_Curve.png"), bbox_inches='tight')
    plt.close()

def save_visual_results(model, fixed_batch, device, epoch, save_dir, prefix, in_channels):
    model.eval()
    with torch.no_grad():
        noisy = fixed_batch["noisy"].to(device, non_blocking=True)
        mask = fixed_batch["mask"].to(device, non_blocking=True)
        targets = fixed_batch["clean"].to(device, non_blocking=True)
        
        net_inputs = torch.cat([noisy, mask], dim=1) if in_channels == 2 else noisy
        
        with torch.amp.autocast('cuda'):
            preds = model(net_inputs)
            
    noisy_input_np = noisy[0, 0].detach().cpu().to(torch.float32).numpy()
    pred_output_np = preds[0, 0].detach().cpu().to(torch.float32).numpy()
    target_clean_np = targets[0, 0].detach().cpu().to(torch.float32).numpy()
    
    pred_output_np = np.nan_to_num(pred_output_np, nan=0.0)
    fig, axes = plt.subplots(1, 4, figsize=(20, 6), dpi=200)
    
    vmax = np.percentile(np.abs(target_clean_np), 98)
    visual_vmax = min(max(vmax, 1e-8), 1.5) 
        
    axes[0].imshow(noisy_input_np, cmap='seismic', aspect='auto', vmin=-visual_vmax, vmax=visual_vmax)
    axes[0].set_title("Input (Noisy & Incomplete)")
    axes[1].imshow(pred_output_np, cmap='seismic', aspect='auto', vmin=-visual_vmax, vmax=visual_vmax)
    axes[1].set_title(f"Prediction (Cleaned)")
    axes[2].imshow(target_clean_np, cmap='seismic', aspect='auto', vmin=-visual_vmax, vmax=visual_vmax)
    axes[2].set_title("Ground Truth")
    
    noise_removed = noisy_input_np - pred_output_np
    axes[3].imshow(noise_removed, cmap='seismic', aspect='auto', vmin=-visual_vmax, vmax=visual_vmax)
    axes[3].set_title("Removed Noise")
    
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"MAC_Net_{prefix}_Visual_Epoch_{epoch:03d}.png"))
    plt.close()

# ==============================================================================
# 4. 核心单次训练引擎 (🌟 MS-SSIM + 高压 FFT + Log-L1 保幅版)
# ==============================================================================
def train_single_model(cfg, in_channels):
    prefix = f"{in_channels}Ch"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n" + "="*60)
    print(f"🚀 启动 MAC-Net [{prefix}版本] 保幅去噪同步插值训练")
    print("="*60)

    model = MAC_Net(in_channels=in_channels, hidden_dim=64).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['total_epochs'], eta_min=1e-6)

    # 损失函数初始化
    l1_loss_fn = nn.L1Loss() 
    ms_ssim_loss_fn = MS_SSIM(data_range=1.0, size_average=True, channel=1).to(device)

    def update_dataloader(difficulties, is_train=True):
        h5_file = cfg['train_h5'] if is_train else cfg['val_h5']
        dataset = DASUnifiedDataset(h5_file, cfg['noise_paths'], difficulties, patch_size=cfg['patch_size'])
        
        data_ratio = cfg.get('data_ratio', 1.0)
        if data_ratio < 1.0:
            subset_size = max(1, int(len(dataset) * data_ratio))
            rng = np.random.RandomState(42)
            indices = rng.choice(len(dataset), subset_size, replace=False)
            dataset = torch.utils.data.Subset(dataset, indices)
            
        return DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=is_train, num_workers=4, pin_memory=True)

    start_epoch = 1
    best_loss = float('inf')
    history = {'train_total': [], 'val_total': [], 'train_base': [], 'train_fft': [], 'train_ssim': []}

    checkpoint_file = os.path.join(cfg['output_dir'], f"MAC_Net_{prefix}_Latest.pth")
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        history = checkpoint['history']
        
        if start_epoch > cfg['total_epochs']:
            print(f"✅ 发现 {prefix} 进度已达满级，自动跳过！")
            return
            
        print(f"📥 发现 {prefix} 断点，从 Epoch {start_epoch} 恢复训练。")

    val_loader = update_dataloader(['easy', 'medium', 'hard'], is_train=False)
    fixed_val_batch = next(iter(val_loader))

    for epoch in range(start_epoch, cfg['total_epochs'] + 1):
        curr_diff = ['easy', 'medium', 'hard']

        if epoch == start_epoch or epoch == cfg['stage_boundaries'][0] + 1 or epoch == cfg['stage_boundaries'][1] + 1:
            train_loader = update_dataloader(curr_diff, is_train=True)
            
        model.train()
        epoch_losses = {'total': 0.0, 'base': 0.0, 'fft': 0.0, 'ssim': 0.0}
        progress_bar = tqdm(train_loader, desc=f"[{prefix}] Train Epoch {epoch}/{cfg['total_epochs']}", leave=False)

        for batch in progress_bar:
            noisy = batch["noisy"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            targets = batch["clean"].to(device, non_blocking=True)

            net_inputs = torch.cat([noisy, mask], dim=1) if in_channels == 2 else noisy

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                preds = model(net_inputs)
                # 强制退出 autocast，在纯 Float32 域下执行高精度 Loss 计算
            preds_f32 = preds.float()
            targets_f32 = targets.float()
            
            sign_target = torch.sign(targets_f32)
            log_target = sign_target * torch.log1p(torch.abs(targets_f32) * 10.0)
            sign_pred = torch.sign(preds_f32)
            log_pred = sign_pred * torch.log1p(torch.abs(preds_f32) * 10.0)
            l_base = l1_loss_fn(log_pred, log_target) * 3.0
            
            pred_fft = torch.fft.rfft2(preds_f32, norm="ortho")
            target_fft = torch.fft.rfft2(targets_f32, norm="ortho")
            l_fft = F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft)) * 3.0
            
            preds_norm = torch.clamp((preds_f32 + 1.0) / 2.0, 0.0, 1.0)
            targets_norm = torch.clamp((targets_f32 + 1.0) / 2.0, 0.0, 1.0)
               
            raw_ssim = ms_ssim_loss_fn(preds_norm, targets_norm)
            safe_ssim = torch.clamp(raw_ssim, min=0.0, max=1.0) # 强制最高分为1.0
            l_ssim = (1.0 - safe_ssim) * 2.0
                
            total_loss = l_base + l_fft + l_ssim
            
            scaler.scale(total_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0) 
            scaler.step(optimizer)
            scaler.update()

            epoch_losses['total'] += total_loss.item(); epoch_losses['base'] += l_base.item()
            epoch_losses['fft'] += l_fft.item(); epoch_losses['ssim'] += l_ssim.item()
            progress_bar.set_postfix({'loss': f"{total_loss.item():.4f}"})

        num_batches = len(train_loader)
        history['train_total'].append(epoch_losses['total'] / num_batches)
        history['train_base'].append(epoch_losses['base'] / num_batches)
        history['train_fft'].append(epoch_losses['fft'] / num_batches)
        history['train_ssim'].append(epoch_losses['ssim'] / num_batches)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                noisy = batch["noisy"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                targets = batch["clean"].to(device, non_blocking=True)
                net_inputs = torch.cat([noisy, mask], dim=1) if in_channels == 2 else noisy
                
                with torch.amp.autocast('cuda'):
                    preds = model(net_inputs)
                    # 同样必须退出 autocast
                preds_f32 = preds.float()
                targets_f32 = targets.float()
                
                s_t = torch.sign(targets_f32)
                log_t = s_t * torch.log1p(torch.abs(targets_f32) * 10.0)
                s_p = torch.sign(preds_f32)
                log_p = s_p * torch.log1p(torch.abs(preds_f32) * 10.0)
                l_b = l1_loss_fn(log_p, log_t) * 3.0
                
                p_f = torch.fft.rfft2(preds_f32, norm="ortho")
                t_f = torch.fft.rfft2(targets_f32, norm="ortho")
                l_f = F.l1_loss(torch.abs(p_f), torch.abs(t_f)) * 3.0
                
                p_n = torch.clamp((preds_f32 + 1.0) / 2.0, 0.0, 1.0)
                t_n = torch.clamp((targets_f32 + 1.0) / 2.0, 0.0, 1.0)
                    
                raw_s = ms_ssim_loss_fn(p_n, t_n)
                safe_s = torch.clamp(raw_s, min=0.0, max=1.0) # 强制最高分为1.0
                l_s = (1.0 - safe_s) * 2.0
                    
                val_loss += (l_b + l_f + l_s).item()
        avg_val_loss = val_loss / len(val_loader)
        history['val_total'].append(avg_val_loss)

        print(f"✅ Epoch [{epoch}/{cfg['total_epochs']}] | LR: {optimizer.param_groups[0]['lr']:.2e} | Train: {history['train_total'][-1]:.4f} | Val: {avg_val_loss:.4f}")

        checkpoint_state = {
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'scaler_state_dict': scaler.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(), 'best_loss': best_loss, 'history': history
        }
        torch.save(checkpoint_state, checkpoint_file)
        with open(os.path.join(cfg['output_dir'], f"loss_history_{prefix}.json"), 'w') as f:
            json.dump(history, f)

        is_best = avg_val_loss < best_loss
        if epoch % 2 == 0 or is_best:
            plot_research_loss(history, cfg['output_dir'], cfg['stage_boundaries'], epoch, prefix)
        if epoch % 5 == 0:
            save_visual_results(model, fixed_val_batch, device, epoch, cfg['output_dir'], prefix, in_channels)

        if is_best:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(cfg['output_dir'], f"MAC_Net_{prefix}_Best.pth"))
            print(f"   🏆 突破新低！最优 {prefix} 模型已保存。")

# ==============================================================================
# 5. 自动化集成调用入口 
# ==============================================================================
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    
    DATA_DIRECTORY = "G:/DAS-VSP数据处理/去噪+插值/das_curriculum_dataset_rt30k" 
    # 为保底起见建立一个新的权重文件夹输出，防止覆盖你上一个版本的模型
    OUTPUT_DIRECTORY = os.path.join(DATA_DIRECTORY, "outputs_rt_modelv6_fide_final")
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    
    CONFIG = {
        'data_dir': DATA_DIRECTORY,
        'output_dir': OUTPUT_DIRECTORY,
        'train_h5': os.path.join(DATA_DIRECTORY, "train_dataset.h5"),
        'val_h5': os.path.join(DATA_DIRECTORY, "val_dataset.h5"),
        'noise_paths': {
            'bg_npy': "G:/DAS-VSP数据处理/去噪+插值/das_dataset_noise/real_noise_bank_manual.npy",
            'coupling_h5': [
                "G:/DAS-VSP数据处理/去噪+插值/das_dataset_noise/custom_simulated_noise_bank.h5",
                "G:/DAS-VSP数据处理/去噪+插值/das_dataset_noise/multi_region_noise_bank.h5",
                "G:/DAS-VSP数据处理/去噪+插值/das_dataset_noise/purified_z_result.h5"
            ]
        },
        'batch_size': 8,
        'patch_size': 256,
        
        # 跑满全部数据获取最高精度
        'data_ratio': 0.3,           
        'total_epochs': 100,          
        'stage_boundaries': [999, 999] 
    }

    # 🌟 重新对 2Ch 和 1Ch 发起全面训练 (基于全新公平、高保幅 Loss)
    for channels in [2, 1]:
        train_single_model(CONFIG, in_channels=channels)
        torch.cuda.empty_cache()
        gc.collect()
        
    print("\n🎉 极强保幅绝杀版模型训练完毕！")


