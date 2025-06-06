"""
Image Quality Assessment (IQA) Metrics Implementation
====================================================

This module implements various IQA methods spanning from classical statistical measures 
to modern deep learning-based approaches.

Full-Reference Metrics:
- MSE, MAE, RMSE, PSNR (Classical)
- UIQI, SSIM, MS-SSIM (Structural)
- VIF, MAD, FSIM, GMSD (Advanced)
- LPIPS, DISTS (Deep Learning)

No-Reference Metrics:
- BLINDS-II, PIQUE, BRISQUE, NIQE
- IS, FID, Cross-IQA

Author: Image Quality Assessment Implementation
Date: 2024
"""

import numpy as np
from scipy import ndimage
from scipy.stats import entropy
from scipy.ndimage import gaussian_filter
from skimage import feature, filters
from skimage.util import img_as_float
from skimage.metrics import structural_similarity
import warnings
warnings.filterwarnings('ignore')

try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False
    warnings.warn("OpenCV not available. Some metrics may have limited functionality.")

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    import torchvision.models as models
    from PIL import Image
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Deep learning metrics will be limited.")

# ============================================================================
# CLASSICAL FULL-REFERENCE METRICS (18th-20th century)
# ============================================================================

def mse(img1, img2) -> float:
    """Mean Squared Error (18th-19th century)
        output: float - MSE in [0,1] for images in [0,1]
    """
    img1 = img_as_float(img1)
    img2 = img_as_float(img2)
    score = np.mean((img1 - img2) ** 2)
    # Normalize: MSE in [0,1] for images in [0,1]
    score = np.clip(score, 0, 1)
    return float(score)

def mae(img1, img2) -> float:
    """Mean Absolute Error (18th-19th century)
        output: float - MAE in [0,1] for images in [0,1]
    """
    img1 = img_as_float(img1)
    img2 = img_as_float(img2)
    score = np.mean(np.abs(img1 - img2))
    # Normalize: MAE in [0,1] for images in [0,1]
    score = np.clip(score, 0, 1)
    return float(score)

def rmse(img1, img2) -> float:
    """Root Mean Squared Error (19th century)
        output: float - RMSE in [0,1] for images in [0,1]
    """
    score = np.sqrt(mse(img1, img2))
    # RMSE in [0,1] for images in [0,1]
    score = np.clip(score, 0, 1)
    return float(score)

def psnr(img1, img2, data_range=1.0) -> float:
    """Peak Signal-to-Noise Ratio (mid 20th century)
        output: float - PSNR in [0,1] (normalized from dB scale)
    """
    mse_val = mse(img1, img2)
    if mse_val == 0:
        score = 1.0
    else:
        # PSNR can be very high, map to [0,1] using a reasonable max (e.g., 50dB)
        psnr_val = 20 * np.log10(data_range / np.sqrt(mse_val))
        score = np.clip(psnr_val / 50.0, 0, 1)
    return float(score)

# ============================================================================
# STRUCTURAL SIMILARITY METRICS (2002-2004)
# ============================================================================

def uiqi(img1, img2, ws=8) -> float:
    """Universal Image Quality Index (2002)
        output: float - UIQI in [0,1] (typically, can be outside for very dissimilar images, clipped)
    """
    img1 = img_as_float(img1)
    img2 = img_as_float(img2)
    if img1.ndim == 3:
        img1 = np.mean(img1, axis=2)
    if img2.ndim == 3:
        img2 = np.mean(img2, axis=2)
    ws = min(ws, img1.shape[0]//2, img1.shape[1]//2)
    if ws < 2:
        ws = 2
    mu1 = np.array(ndimage.uniform_filter(img1, size=ws))[ws//2::ws, ws//2::ws]
    mu2 = np.array(ndimage.uniform_filter(img2, size=ws))[ws//2::ws, ws//2::ws]
    mu1_sq = np.array(ndimage.uniform_filter(img1*img1, size=ws))[ws//2::ws, ws//2::ws]
    mu2_sq = np.array(ndimage.uniform_filter(img2*img2, size=ws))[ws//2::ws, ws//2::ws]
    mu1_mu2 = np.array(ndimage.uniform_filter(img1*img2, size=ws))[ws//2::ws, ws//2::ws]
    mu1, mu2 = np.array(mu1), np.array(mu2)
    mu1_sq, mu2_sq = np.array(mu1_sq), np.array(mu2_sq)
    mu1_mu2 = np.array(mu1_mu2)
    sigma1_sq = mu1_sq - mu1 * mu1
    sigma2_sq = mu2_sq - mu2 * mu2
    sigma12 = mu1_mu2 - mu1 * mu2
    numerator = 4 * sigma12 * mu1 * mu2
    denominator = (sigma1_sq + sigma2_sq) * (mu1 * mu1 + mu2 * mu2)
    index = np.ones(denominator.shape)
    valid_mask = denominator > 0
    if np.any(valid_mask):
        index[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
    score = float(np.mean(index))
    # UIQI can be outside [0,1], clip to [0,1]
    score = np.clip(score, 0, 1)
    return score

def ssim(img1, img2, data_range=1.0, multichannel=None) -> float:
    """Structural Similarity Index Measure (2003)
        output: float - SSIM in [0,1] (original SSIM is [-1,1], mapped to [0,1])
    """
    img1 = img_as_float(img1)
    img2 = img_as_float(img2)
    if multichannel is None:
        multichannel = img1.ndim == 3
    if multichannel and img1.ndim == 3:
        result = structural_similarity(img1, img2, data_range=data_range, 
                                     multichannel=True, channel_axis=-1)
    else:
        if img1.ndim == 3:
            img1 = np.mean(img1, axis=2)
        if img2.ndim == 3:
            img2 = np.mean(img2, axis=2)
        result = structural_similarity(img1, img2, data_range=data_range)
    if isinstance(result, tuple):
        score = float(result[0])
    else:
        score = float(result)
    # SSIM can be in [-1,1], map to [0,1]
    score = (score + 1) / 2
    score = np.clip(score, 0, 1)
    return score

def ms_ssim(img1, img2, weights=None, data_range=1.0) -> float:
    """
    Multi-Scale Structural Similarity Index (MS-SSIM)
    output: float - MS-SSIM in [0,1] (original MS-SSIM can be [-1,1], mapped to [0,1])
    
    Args:
        img1, img2: Input images (numpy arrays)
        weights: List of weights for each scale
        data_range: Range of the input images (default: 1.0 for [0,1] range)
    
    Returns:
        MS-SSIM value
    """
    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    img1 = np.array(img1, dtype=np.float64)
    img2 = np.array(img2, dtype=np.float64)
    if data_range != 1.0:
        img1 = img1 / data_range
        img2 = img2 / data_range
    if img1.ndim == 3:
        if CV2_AVAILABLE:
            img1_uint8 = np.clip(img1 * 255, 0, 255).astype(np.uint8)
            img2_uint8 = np.clip(img2 * 255, 0, 255).astype(np.uint8)
            img1 = cv2.cvtColor(img1_uint8, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
            img2 = cv2.cvtColor(img2_uint8, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
        else:
            img1 = np.mean(img1, axis=2)
            img2 = np.mean(img2, axis=2)
    scales = min(len(weights), 5)
    mssim_values = []
    current_img1 = img1.copy()
    current_img2 = img2.copy()
    for i in range(scales):
        ssim_val = ssim(current_img1, current_img2)
        mssim_values.append(ssim_val)
        if i < scales - 1:
            if CV2_AVAILABLE and current_img1.shape[0] > 32 and current_img1.shape[1] > 32:
                current_img1 = cv2.resize(current_img1, 
                                        (current_img1.shape[1]//2, current_img1.shape[0]//2), 
                                        interpolation=cv2.INTER_LINEAR)
                current_img2 = cv2.resize(current_img2, 
                                        (current_img2.shape[1]//2, current_img2.shape[0]//2), 
                                        interpolation=cv2.INTER_LINEAR)
            else:
                current_img1 = current_img1[::2, ::2]
                current_img2 = current_img2[::2, ::2]
            if current_img1.shape[0] < 11 or current_img1.shape[1] < 11:
                break
    if len(mssim_values) == 1:
        return mssim_values[0]
    actual_weights = weights[:len(mssim_values)]
    actual_weights = np.array(actual_weights) / np.sum(actual_weights)
    ms_ssim_val = np.prod([val**weight for val, weight in zip(mssim_values, actual_weights)])
    # MS-SSIM can be in [-1,1], map to [0,1]
    ms_ssim_val = (ms_ssim_val + 1) / 2
    ms_ssim_val = np.clip(ms_ssim_val, 0, 1)
    return float(ms_ssim_val)

def _ssim_single_scale(img1, img2, full=False):
    """Helper function for single-scale SSIM computation"""
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5) if CV2_AVAILABLE else img1
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5) if CV2_AVAILABLE else img2
    
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.GaussianBlur(img1 * img1, (11, 11), 1.5) - mu1_sq if CV2_AVAILABLE else np.var(img1)
    sigma2_sq = cv2.GaussianBlur(img2 * img2, (11, 11), 1.5) - mu2_sq if CV2_AVAILABLE else np.var(img2)
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2 if CV2_AVAILABLE else np.cov(img1.flatten(), img2.flatten())[0,1]
    
    c1 = (0.01 * 1.0) ** 2  # data_range = 1.0
    c2 = (0.03 * 1.0) ** 2
    
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    
    if full:
        return np.mean(ssim_map), np.mean(cs_map)
    else:
        return np.mean(ssim_map)

def vif(img1, img2) -> float:
    """Visual Information Fidelity (2006)
        output: float - VIF in [0,1] (original VIF can be >1, clipped)
    """
    img1 = img_as_float(img1)
    img2 = img_as_float(img2)
    
    if img1.ndim == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    if img2.ndim == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    
    sigma_nsq = 0.1
     
    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    
    g = sigma12 / (sigma1_sq + 1e-10)
    sv_sq = sigma2_sq - g * sigma12
    
    if sigma1_sq < 1e-10:
        return 0.0
    
    vif_val = np.log2(1 + g**2 * sigma1_sq / (sv_sq + sigma_nsq)) / np.log2(1 + sigma1_sq / sigma_nsq)
    vif_val = np.clip(vif_val, 0, 1)
    return float(vif_val)

def mad(img1, img2) -> float:
    """Most Apparent Distortion (2010) - Simplified implementation
        output: float - MAD in [0,1] (original MAD can be >1, clipped)
    """
    img1 = img_as_float(img1)
    img2 = img_as_float(img2)
    
    if img1.ndim == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    if img2.ndim == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
     
    kernel = np.ones((8, 8)) / 64
    local_var1 = ndimage.convolve(img1**2, kernel) - ndimage.convolve(img1, kernel)**2
    local_var2 = ndimage.convolve(img2**2, kernel) - ndimage.convolve(img2, kernel)**2
    
    mad_val = np.mean(np.abs(local_var1 - local_var2))
    mad_val = np.clip(mad_val, 0, 1)
    return float(mad_val)

def fsim(img1, img2) -> float:
    """Feature Similarity Index Method (2011)
        output: float - FSIM in [0,1]
    """
    img1 = img_as_float(img1)
    img2 = img_as_float(img2)
    
    if img1.ndim == 3:
        img1 = np.mean(img1, axis=2)
    if img2.ndim == 3:
        img2 = np.mean(img2, axis=2)
    
    pc1 = _phase_congruency(img1)
    pc2 = _phase_congruency(img2)
    
    gm1 = np.sqrt(filters.sobel_h(img1)**2 + filters.sobel_v(img1)**2)
    gm2 = np.sqrt(filters.sobel_h(img2)**2 + filters.sobel_v(img2)**2)
    
    T1, T2 = 0.85, 160
    
    PCm = np.maximum(pc1, pc2)
    S_pc = (2 * pc1 * pc2 + T1) / (pc1**2 + pc2**2 + T1)
    S_g = (2 * gm1 * gm2 + T2) / (gm1**2 + gm2**2 + T2)
    
    S_l = S_pc * S_g
    PCm_norm = PCm / np.max(PCm) if np.max(PCm) > 0 else PCm
    
    if np.sum(PCm_norm) > 0:
        score = np.sum(S_l * PCm_norm) / np.sum(PCm_norm)
    else:
        score = 0 
    score = np.clip(score, 0, 1)
    return float(score)

def _phase_congruency(img):
    """Simplified phase congruency calculation"""
    gx = filters.sobel_h(img)
    gy = filters.sobel_v(img)
    return np.sqrt(gx**2 + gy**2)

def gmsd(img1, img2) -> float:
    """Gradient Magnitude Similarity Deviation (2014)
        output: float - GMSD-based similarity in [0,1] (original GMSD is a deviation, lower is better, mapped to similarity)
    """
    img1 = img_as_float(img1)
    img2 = img_as_float(img2)
    
    if img1.ndim == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    if img2.ndim == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    
    gm1 = np.sqrt(filters.sobel_h(img1)**2 + filters.sobel_v(img1)**2)
    gm2 = np.sqrt(filters.sobel_h(img2)**2 + filters.sobel_v(img2)**2)
    
    c = 0.0026
    gms = (2 * gm1 * gm2 + c) / (gm1**2 + gm2**2 + c)
    
    gmsd_val = np.std(gms)
    # GMSD: lower is better, map to similarity [0,1] using exp decay (typical max ~0.5)
    gmsd_val = np.exp(-gmsd_val * 10)
    gmsd_val = np.clip(gmsd_val, 0, 1)
    return float(gmsd_val)

# ============================================================================
# DEEP LEARNING METRICS (2018-2020)
# ============================================================================

class LPIPS:
    """Learned Perceptual Image Patch Similarity (2018)
        output: float - LPIPS-based similarity in [0,1] (original LPIPS is a distance, lower is better, mapped to similarity)
    """
    
    def __init__(self, net='vgg', device='cpu'):
        self.device = device
        self.net = net
        if TORCH_AVAILABLE:
            self.model = self._load_model()
        else:
            self.model = None
    
    def _load_model(self):
        """Load pre-trained model for LPIPS"""
        try:
            # Simplified implementation using VGG features
            model = models.vgg16(pretrained=True)
            model = nn.Sequential(*list(model.features.children())[:16])
            model.eval()
            return model.to(self.device)
        except Exception:
            return None
    
    def __call__(self, img1, img2) -> float:
        """Calculate LPIPS distance
            output: float - LPIPS-based similarity in [0,1]
        """
        return self.compute(img1, img2)
    
    def compute(self, img1, img2) -> float:
        """Calculate LPIPS distance
            output: float - LPIPS-based similarity in [0,1]
        """
        if not TORCH_AVAILABLE or self.model is None:
            # Fallback to SSIM-based calculation
            ssim_val = ssim(img1, img2)
            return 1.0 - ssim_val
        
        try:
            img1_tensor = self._preprocess(img1)
            img2_tensor = self._preprocess(img2)
            if img1_tensor is None or img2_tensor is None:
                ssim_val = ssim(img1, img2)
                return 1.0 - ssim_val
            with torch.no_grad():
                feat1 = self.model(img1_tensor)
                feat2 = self.model(img2_tensor)
                diff = torch.mean((feat1 - feat2)**2)
                # LPIPS: lower is better, map to [0,1] using exp decay (typical max ~1)
                lpips_val = float(diff.item())
                lpips_val = np.exp(-lpips_val * 5)
                lpips_val = np.clip(lpips_val, 0, 1)
                return lpips_val
        except Exception:
            ssim_val = ssim(img1, img2)
            return 1.0 - ssim_val
    
    def _preprocess(self, img): # -> Optional[torch.Tensor] - Internal helper
        """Preprocess image for LPIPS"""
        if not TORCH_AVAILABLE:
            return None
            
        if isinstance(img, np.ndarray):
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            
            # Convert to PIL Image
            img_pil = Image.fromarray((img * 255).astype(np.uint8))
        else:
            img_pil = img
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        tensor_result = transform(img_pil)
        if hasattr(tensor_result, 'unsqueeze'):
            return tensor_result.unsqueeze(0).to(self.device)
        else:
            return None

def dists(img1, img2) -> float:
    """Deep Image Structure and Texture Similarity (2020) - Simplified
        output: float - DISTS in [0,1]
    """
    img1 = img_as_float(img1)
    img2 = img_as_float(img2)
    
    # Simplified implementation using structural and textural features
    # Structure: using SSIM-like calculation
    structure_sim = ssim(img1, img2)
    
    # Texture: using local binary patterns
    if CV2_AVAILABLE:
        if img1.ndim == 3:
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        else:
            img1_gray, img2_gray = img1, img2
        
        # Local Binary Pattern for texture
        lbp1 = feature.local_binary_pattern(img1_gray, 8, 1, method='uniform')
        lbp2 = feature.local_binary_pattern(img2_gray, 8, 1, method='uniform')
        
        texture_sim = 1.0 - np.mean(np.abs(lbp1 - lbp2)) / (lbp1.max() - lbp1.min() + 1e-10)
    else:
        # Fallback texture similarity using gradients
        if img1.ndim == 3:
            img1_gray = np.mean(img1, axis=2)
            img2_gray = np.mean(img2, axis=2)
        else:
            img1_gray, img2_gray = img1, img2
        
        grad1 = np.gradient(img1_gray)
        grad2 = np.gradient(img2_gray)
        texture_sim = 1.0 - np.mean(np.abs(grad1[0] - grad2[0]) + np.abs(grad1[1] - grad2[1]))
    
    # Combine structure and texture
    dists_val = 0.5 * structure_sim + 0.5 * texture_sim
    # DISTS: similarity, ensure in [0,1]
    dists_val = np.clip(dists_val, 0, 1)
    return float(dists_val)

# ============================================================================
# NO-REFERENCE METRICS (2011-2024)
# ============================================================================

def brisque(img) -> float:
    """Blind/Referenceless Image Spatial Quality Evaluator (2012)
        output: float - BRISQUE-based quality score in [0,1] (original BRISQUE is lower is better, mapped to higher is better)
    """
    img = img_as_float(img)
    
    if img.ndim == 3:
        if CV2_AVAILABLE:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img = np.mean(img, axis=2)
    
    # MSCN coefficients
    if CV2_AVAILABLE:
        mu = cv2.GaussianBlur(img, (7, 7), 7/6)
        mu_sq = cv2.GaussianBlur(img*img, (7, 7), 7/6)
    else:
        mu = gaussian_filter(img, sigma=7/6)
        mu_sq = gaussian_filter(img*img, sigma=7/6)
    
    sigma = np.sqrt(np.abs(mu_sq - mu*mu))
    mscn = (img - mu) / (sigma + 1e-10)
    
    # Calculate features
    features = []
    
    # MSCN distribution features
    alpha = _estimate_ggd_param(mscn.flatten())
    features.append(alpha)
    features.append(np.var(mscn))
    
    # Pairwise products
    shifts = [(0, 1), (1, 0), (1, 1), (-1, 1)]
    for shift in shifts:
        shifted = np.roll(np.roll(mscn, shift[0], axis=0), shift[1], axis=1)
        product = mscn * shifted
        alpha_prod = _estimate_ggd_param(product.flatten())
        features.extend([alpha_prod, np.var(product)])
    
    # Simple quality prediction (normally requires trained model)
    quality_score = 100 - np.mean(features) * 10
    # BRISQUE: lower is better, map to [0,1] (100=worst, 0=best)
    brisque_val = 1.0 - np.clip(quality_score, 0, 100) / 100.0
    return float(brisque_val)

def _estimate_ggd_param(data):
    """Estimate GGD parameter using method of moments"""
    sigma_sq = np.var(data)
    mean_abs = np.mean(np.abs(data))
    
    if mean_abs < 1e-10:
        return 1.0
    
    # Simplified estimation
    rho = sigma_sq / (mean_abs**2)
    alpha = 1 / (1 + rho)
    return np.clip(alpha, 0.1, 10.0)

def niqe(img) -> float:
    """Natural Image Quality Evaluator (2012) - Simplified
        output: float - NIQE-based quality score in [0,1] (original NIQE is lower is better, mapped to higher is better)
    """
    img = img_as_float(img)
    
    if img.ndim == 3:
        if CV2_AVAILABLE:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img = np.mean(img, axis=2)
    
    # MSCN coefficients
    mu = gaussian_filter(img, sigma=7/6)
    mu_sq = gaussian_filter(img**2, sigma=7/6)
    sigma = np.sqrt(np.abs(mu_sq - mu**2))
    mscn = (img - mu) / (sigma + 1e-10)
    
    # Extract features from patches
    patch_size = 96
    features = []
    
    for i in range(0, img.shape[0] - patch_size, patch_size//2):
        for j in range(0, img.shape[1] - patch_size, patch_size//2):
            patch = np.array(mscn)[i:i+patch_size, j:j+patch_size]
            if patch.size > 0:
                features.append([
                    np.mean(patch),
                    np.var(patch),
                    np.mean(np.abs(patch)),
                    _estimate_ggd_param(patch.flatten())
                ])
    
    if not features:
        return 50.0
    
    features = np.array(features)
    
    # Simplified quality calculation (normally uses MVG model)
    quality = 100 - np.mean(np.std(features, axis=0)) * 20
    # NIQE: lower is better, map to [0,1]
    niqe_val = 1.0 - np.clip(quality, 0, 100) / 100.0
    return float(niqe_val)

def pique(img) -> float:
    """Perception-Based Image Quality Evaluator (2011) - Simplified
        output: float - PIQUE-based quality score in [0,1] (original PIQUE is lower is better, mapped to higher is better)
    """
    img = img_as_float(img)
    
    if img.ndim == 3:
        if CV2_AVAILABLE:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img = np.mean(img, axis=2)
    
    # Block-wise distortion and noise estimation
    block_size = 16
    distortions = []
    
    for i in range(0, img.shape[0] - block_size, block_size):
        for j in range(0, img.shape[1] - block_size, block_size):
            block = np.array(img)[i:i+block_size, j:j+block_size]
            
            # Local variance as distortion measure
            local_var = float(np.var(block))
            distortions.append(local_var)
    
    # Quality score based on distortion distribution
    if distortions:
        pique_score = 100 - np.mean(distortions) * 1000
        score = np.clip(pique_score, 0, 100)
    else:
        score = 50.0
    # PIQUE: lower is better, map to [0,1]
    pique_val = 1.0 - score / 100.0
    return float(pique_val)

def blinds2(img) -> float:
    """BLINDS-II (2011) - Simplified implementation
        output: float - BLINDS-II-based quality score in [0,1] (original BLINDS-II is lower is better, mapped to higher is better)
    """
    img = img_as_float(img)
    
    if img.ndim == 3:
        if CV2_AVAILABLE:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img = np.mean(img, axis=2)
    
    # Multi-scale analysis
    scales = [1, 0.5, 0.25]
    features = []
    
    for scale in scales:
        if scale != 1:
            h, w = int(img.shape[0] * scale), int(img.shape[1] * scale)
            if h > 0 and w > 0:
                if CV2_AVAILABLE:
                    scaled_img = cv2.resize(img, (w, h))
                else:
                    # Fallback using scipy
                    from scipy.ndimage import zoom
                    scaled_img = zoom(img, scale)
            else:
                scaled_img = img
        else:
            scaled_img = img
        
        # Extract features at each scale
        gx = filters.sobel_h(scaled_img)
        gy = filters.sobel_v(scaled_img)
        gradient_mag = np.sqrt(gx**2 + gy**2)
        
        features.extend([
            np.mean(gradient_mag),
            np.var(gradient_mag),
            np.mean(scaled_img),
            np.var(scaled_img)
        ])
    
    # Simple quality estimation
    quality = 100 - np.std(features) * 50
    # BLINDS-II: lower is better, map to [0,1]
    blinds_val = 1.0 - np.clip(quality, 0, 100) / 100.0
    return float(blinds_val)

def inception_score(images, batch_size=32) -> float:
    """Inception Score (2016) - Simplified
        output: float - IS in [0,1] (original IS is higher is better, normalized)
    """
    # This is a simplified version - requires pre-trained Inception model
    # In practice, you would use a pre-trained Inception-v3 model
    
    if not isinstance(images, list):
        images = [images]
    
    # Simplified IS calculation
    scores = []
    for img in images:
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        
        # Simple feature extraction (placeholder for Inception features)
        features = np.mean(img.reshape(-1, img.shape[-1]), axis=0)
        score = entropy(features + 1e-10)
        scores.append(score)
    
    is_val = np.exp(np.mean(scores))
    # IS: higher is better, normalize to [0,1] using a reasonable max (e.g., 10)
    is_val = np.clip(is_val / 10.0, 0, 1)
    return is_val

def fid_score(real_images, generated_images) -> float:
    """Fréchet Inception Distance (2017) - Simplified
        output: float - FID-based similarity in [0,1] (original FID is lower is better, mapped to similarity)
    """
    # Simplified FID calculation
    # In practice, requires Inception-v3 features
    
    def extract_features(images):
        features = []
        for img in images:
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            
            # Simplified feature extraction
            feat = np.mean(img.reshape(-1, img.shape[-1]), axis=0)
            features.append(feat)
        
        return np.array(features)
    
    real_features = extract_features(real_images)
    gen_features = extract_features(generated_images)
    
    # Calculate FID
    mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)
    
    diff = mu1 - mu2
    covmean = np.sqrt(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2*covmean)
    # FID: lower is better, map to [0,1] using exp decay (typical max ~100)
    fid_val = np.exp(-fid / 50.0)
    fid_val = np.clip(fid_val, 0, 1)
    return fid_val

def cross_iqa(img1, img2) -> float:
    """Cross-IQA (2024) - Simplified implementation
        output: float - Cross-IQA score in [0,1]
    """
    # This represents a modern cross-modal IQA approach
    
    img1 = img_as_float(img1)
    img2 = img_as_float(img2)
    
    # Multi-scale, multi-metric combination
    metrics = {
        'ssim': ssim(img1, img2),
        'fsim': fsim(img1, img2),
        'gmsd': 1 / (1 + gmsd(img1, img2)),  # Convert to similarity
        'lpips': 1 / (1 + LPIPS()(img1, img2)),  # Convert to similarity
    }
    
    # Weighted combination (learned weights in practice)
    weights = [0.3, 0.25, 0.25, 0.2]
    cross_iqa_score = sum(w * score for w, score in zip(weights, metrics.values()))
    # Cross-IQA: similarity, ensure in [0,1]
    cross_iqa_score = np.clip(cross_iqa_score, 0, 1)
    return cross_iqa_score

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def evaluate_all_metrics(img1, img2=None, include_no_ref=True) -> dict:
    """
    Evaluate all implemented metrics on given image(s)
    output: dict - Dictionary of metric names to scores in [0,1]
    
    Parameters:
    -----------
    img1 : numpy.ndarray
        First image (reference for full-reference metrics, test image for no-reference)
    img2 : numpy.ndarray, optional
        Second image (distorted image for full-reference metrics)
    include_no_ref : bool
        Whether to include no-reference metrics
    
    Returns:
    --------
    dict : Dictionary containing all metric scores
    """
    
    results = {}
    
    # Full-reference metrics (require both images)
    if img2 is not None:
        try:
            results['MSE'] = mse(img1, img2)
            results['MAE'] = mae(img1, img2)
            results['RMSE'] = rmse(img1, img2)
            results['PSNR'] = psnr(img1, img2)
            results['UIQI'] = uiqi(img1, img2)
            results['SSIM'] = ssim(img1, img2)
            results['MS-SSIM'] = ms_ssim(img1, img2)
            results['VIF'] = vif(img1, img2)
            results['MAD'] = mad(img1, img2)
            results['FSIM'] = fsim(img1, img2)
            results['GMSD'] = gmsd(img1, img2)
            results['LPIPS'] = LPIPS()(img1, img2)
            results['DISTS'] = dists(img1, img2)
            results['Cross-IQA'] = cross_iqa(img1, img2)
        except Exception as e:
            print(f"Error calculating full-reference metrics: {e}")
    
    # No-reference metrics (require only one image)
    if include_no_ref:
        try:
            results['BRISQUE'] = brisque(img1)
            results['NIQE'] = niqe(img1)
            results['PIQUE'] = pique(img1)
            results['BLINDS-II'] = blinds2(img1)
        except Exception as e:
            print(f"Error calculating no-reference metrics: {e}")
    
    return results

def print_metrics_summary(): 
    """Print a summary of all implemented metrics"""
    
    print("=" * 80)
    print("IMAGE QUALITY ASSESSMENT METRICS SUMMARY")
    print("=" * 80)
    
    print("\n📊 FULL-REFERENCE METRICS (require reference and distorted images):")
    print("-" * 70)
    
    classical = [
        ("MSE", "Mean Squared Error", "18th-19th century"),
        ("MAE", "Mean Absolute Error", "18th-19th century"),
        ("RMSE", "Root Mean Squared Error", "19th century"),
        ("PSNR", "Peak Signal-to-Noise Ratio", "mid 20th century"),
    ]
    
    structural = [
        ("UIQI", "Universal Image Quality Index", "2002"),
        ("SSIM", "Structural Similarity Index Measure", "2003"),
        ("MS-SSIM", "Multi-Scale SSIM", "2004"),
    ]
    
    advanced = [
        ("VIF", "Visual Information Fidelity", "2006"),
        ("MAD", "Most Apparent Distortion", "2010"),
        ("FSIM", "Feature Similarity Index Method", "2011"),
        ("GMSD", "Gradient Magnitude Similarity Deviation", "2014"),
    ]
    
    deep_learning = [
        ("LPIPS", "Learned Perceptual Image Patch Similarity", "2018"),
        ("DISTS", "Deep Image Structure and Texture Similarity", "2020"),
    ]
    
    modern = [
        ("Cross-IQA", "Cross-modal Image Quality Assessment", "2024"),
    ]
    
    print("\n  🔹 Classical Metrics:")
    for abbr, name, year in classical:
        print(f"    {abbr:12} - {name:35} ({year})")
    
    print("\n  🔹 Structural Similarity Metrics:")
    for abbr, name, year in structural:
        print(f"    {abbr:12} - {name:35} ({year})")
    
    print("\n  🔹 Advanced Perceptual Metrics:")
    for abbr, name, year in advanced:
        print(f"    {abbr:12} - {name:35} ({year})")
    
    print("\n  🔹 Deep Learning Metrics:")
    for abbr, name, year in deep_learning:
        print(f"    {abbr:12} - {name:35} ({year})")
    
    print("\n  🔹 Modern Cross-modal Metrics:")
    for abbr, name, year in modern:
        print(f"    {abbr:12} - {name:35} ({year})")
    
    print("\n📈 NO-REFERENCE METRICS (require only test image):")
    print("-" * 70)
    
    no_ref = [
        ("BLINDS-II", "Blind Image Quality Index", "2011"),
        ("PIQUE", "Perception-Based Image Quality Evaluator", "2011"),
        ("BRISQUE", "Blind/Referenceless Image Spatial Quality Evaluator", "2012"),
        ("NIQE", "Natural Image Quality Evaluator", "2012"),
        ("IS", "Inception Score", "2016"),
        ("FID", "Fréchet Inception Distance", "2017"),
    ]
    
    for abbr, name, year in no_ref:
        print(f"  {abbr:12} - {name:40} ({year})")
    
    print("\n" + "=" * 80)
    print("Total: {} metrics implemented".format(len(classical) + len(structural) + 
                                                len(advanced) + len(deep_learning) + 
                                                len(modern) + len(no_ref)))
    print("=" * 80)

if __name__ == "__main__":
    print_metrics_summary()