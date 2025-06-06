#!/usr/bin/env python3
"""
IQA Metrics Library
==================
Centralized library for Image Quality Assessment metrics comparison.
Supports multiple metric types and provides unified interface.
"""
import torch
import numpy as np
import pyiqa
from PIL import Image
import sys
import os
from typing import Dict, List, Tuple, Optional
import warnings
import time
from metric import mse as imported_mse, mae as imported_mae, rmse as imported_rmse, \
                   uiqi as imported_uiqi, blinds2 as imported_blinds2, pique as imported_pique
sys.path.append('BAPPS')
import lpips
import cv2
warnings.filterwarnings('ignore')

class IQAMetrics:
    """
    Unified interface for Image Quality Assessment metrics.
    Supports both full-reference and no-reference metrics.
    """
    def __init__(self, device: str = 'auto', target_size: Tuple[int, int] = (256, 256)):
        self.device = self._get_device(device)
        self.target_size = target_size
        self.metrics = {}
        self.metric_info = {}
        print("🔧 IQA Metrics Evaluator initialized")
        print(f"   Device: {self.device}")
        print(f"   Target size: {target_size}")
        self._initialize_all_metrics()
    def _get_device(self, device: str) -> str:
        if device == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            # elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            #     return 'mps'
            else:
                return 'cpu'
        return device
    def _initialize_all_metrics(self):
        print("\n📊 Initializing IQA metrics...")
        def run_computation_iqa(metric_name, score_higher_better=True):
            def wrapper(ref, dis):
                start_time = time.time()
                metric_function = pyiqa.create_metric(metric_name, device=self.device)
                score_range: str = metric_function.score_range
                score = metric_function(dis, ref)
                if isinstance(score, torch.Tensor):
                    score = score.mean()
                if hasattr(score, 'item'):
                    score = score.item()
                else:
                    score = float(score)
                computation_time = time.time() - start_time
                min_val, max_val = [float(part.replace('~', '').strip()) for part in score_range.split(',')]
                if max_val == min_val:
                    norm_score = 0.0
                elif score_higher_better:
                    norm_score = (score - min_val) / (max_val - min_val)
                else:
                    norm_score = (max_val - score) / (max_val - min_val)
                norm_score = float(np.clip(norm_score, 0.0, 1.0))
                return norm_score, computation_time
            return wrapper
        def run_computation_iqa_nr(metric_name, score_higher_better=True):
            def wrapper(img):
                start_time = time.time()
                metric_function = pyiqa.create_metric(metric_name, device=self.device)
                score_range = metric_function.score_range
                score = metric_function(img)
                if isinstance(score, torch.Tensor):
                    score = score.mean()
                if hasattr(score, 'item'):
                    score = score.item()
                else:
                    score = float(score)
                computation_time = time.time() - start_time
                min_val, max_val = [float(part.replace('~', '').strip()) for part in score_range.split(',')]
                if max_val == min_val:
                    norm_score = 0.0
                elif score_higher_better:
                    norm_score = (score - min_val) / (max_val - min_val)
                else:
                    norm_score = (max_val - score) / (max_val - min_val)
                norm_score = float(np.clip(norm_score, 0.0, 1.0))
                return norm_score, computation_time
            return wrapper
        def run_computation_custom(metric_func, score_higher_better=True):
            def wrapper(*args):
                start_time = time.time()
                score = metric_func(*args)
                computation_time = time.time() - start_time
                if not score_higher_better:
                    score = 1.0 - score
                return float(score), computation_time
            return wrapper
        metric_configs = {
            # Structural Similarity Metrics
            'ssim': {
                'name': 'SSIM',
                'description': 'Structural Similarity Index (pyiqa)',
                'type': 'full_reference',
                'category': 'structural',
                'higher_better': True,
                'function': run_computation_iqa('ssim')
            },
            'ms_ssim': {
                'name': 'MS-SSIM',
                'description': 'Multi-Scale Structural Similarity (pyiqa)',
                'type': 'full_reference',
                'category': 'structural',
                'higher_better': True,
                'function': run_computation_iqa('ms_ssim')
            },
            'uiqi': { 
                'name': 'UIQI (Custom)',
                'description': 'Universal Image Quality Index (Custom Implementation)',
                'type': 'full_reference',
                'category': 'structural',
                'higher_better': True,
                'custom': True,
                'function': run_computation_custom(imported_uiqi)
            },
            
            # Signal-based Metrics
            'psnr': {
                'name': 'PSNR',
                'description': 'Peak Signal-to-Noise Ratio (pyiqa)',
                'type': 'full_reference',
                'category': 'signal',
                'higher_better': True,
                'function': run_computation_iqa('psnr')
            },
            'mse': { 
                'name': 'MSE (Custom)',
                'description': 'Mean Squared Error (Custom Implementation)',
                'type': 'full_reference',
                'category': 'signal',
                'higher_better': False,
                'custom': True,
                'function': run_computation_custom(imported_mse, score_higher_better=False)
            },
            'mae': { 
                'name': 'MAE (Custom)',
                'description': 'Mean Absolute Error (Custom Implementation)',
                'type': 'full_reference',
                'category': 'signal',
                'higher_better': False,
                'custom': True,
                'function': run_computation_custom(imported_mae, score_higher_better=False)
            },
            'rmse': { 
                'name': 'RMSE (Custom)',
                'description': 'Root Mean Squared Error (Custom Implementation)',
                'type': 'full_reference',
                'category': 'signal',
                'higher_better': False,
                'custom': True,
                'function': run_computation_custom(imported_rmse, score_higher_better=False)
            },
            
            # Perceptual Metrics
            'lpips': {  
                'name': 'LPIPS',
                'description': 'Learned Perceptual Image Patch Similarity',
                'type': 'full_reference',
                'category': 'perceptual',
                'higher_better': False,
                'custom': True
            },
            'fsim': {
                'name': 'FSIM',
                'description': 'Feature Similarity Index (pyiqa)',
                'type': 'full_reference',
                'category': 'feature',
                'higher_better': True,
                'function': run_computation_iqa('fsim')
            },
            'vsi': {
                'name': 'VSI',
                'description': 'Visual Saliency Index (pyiqa)',
                'type': 'full_reference',
                'category': 'feature',
                'higher_better': True,
                'function': run_computation_iqa('vsi')
            },
            'gmsd': {
                'name': 'GMSD',
                'description': 'Gradient Magnitude Similarity Deviation (pyiqa)',
                'type': 'full_reference',
                'category': 'feature',
                'higher_better': False,
                'function': run_computation_iqa('gmsd', score_higher_better=False)
            },
            
            # No-Reference Metrics
            'brisque': {
                'name': 'BRISQUE',
                'description': 'Blind/Referenceless Image Spatial Quality Evaluator (pyiqa)',
                'type': 'no_reference',
                'category': 'blind',
                'higher_better': False,
                'function': run_computation_iqa_nr('brisque', score_higher_better=False)
            },
            'niqe': {
                'name': 'NIQE',
                'description': 'Natural Image Quality Evaluator (pyiqa)',
                'type': 'no_reference',
                'category': 'blind',
                'higher_better': False,
                'function': run_computation_iqa_nr('niqe', score_higher_better=False)
            },
            'clipiqa': {
                'name': 'CLIP-IQA',
                'description': 'CLIP-based Image Quality Assessment (pyiqa)',
                'type': 'no_reference',
                'category': 'learning',
                'higher_better': True,
                'function': run_computation_iqa_nr('clipiqa')
            },
            'blinds2': { 
                'name': 'BLINDS-II (Custom)',
                'description': 'BLINDS-II (Custom Implementation)',
                'type': 'no_reference',
                'category': 'blind',
                'higher_better': True, 
                'custom': True,
                'function': run_computation_custom(imported_blinds2)
            },
            'pique': { 
                'name': 'PIQUE (Custom)',
                'description': 'Perception-Based Image Quality Evaluator (Custom Implementation)',
                'type': 'no_reference',
                'category': 'blind',
                'higher_better': True,
                'custom': True,
                'function': run_computation_custom(imported_pique)
            }
        }
        for metric_id, config in metric_configs.items():
            try:
                if 'function' in config:
                    metric_func = config['function']
                    self.metrics[metric_id] = metric_func
                else:
                    self.metrics[metric_id] = pyiqa.create_metric(metric_id, device=self.device)
                self.metric_info[metric_id] = {
                    'name': config['name'],
                    'description': config['description'],
                    'type': config['type'],
                    'category': config['category'],
                    'higher_better': config['higher_better']
                }
                print(f"   ✅ {config['name']}: {config['description']}")
            except Exception as e:
                print(f"   ⚠️  {config['name']}: {e}")
        self._initialize_lpips_variants()
        print(f"\n✅ Successfully initialized {len(self.metrics)} metrics")
    def _initialize_lpips_variants(self):
        lpips_variants = {
            'lpips_alex': {'backbone': 'alex', 'name': 'LPIPS-Alex'},
            'lpips_vgg': {'backbone': 'vgg', 'name': 'LPIPS-VGG'},
            'lpips_squeeze': {'backbone': 'squeeze', 'name': 'LPIPS-SqueezeNet'}
        }
        for variant_id, config in lpips_variants.items():
            try:
                metric = lpips.LPIPS(net=config['backbone']).to(self.device)
                self.metrics[variant_id] = metric
                self.metric_info[variant_id] = {
                    'name': config['name'],
                    'description': f"LPIPS with {config['backbone']} backbone",
                    'type': 'full_reference',
                    'category': 'perceptual',
                    'higher_better': False
                }
                print(f"   ✅ {config['name']}: LPIPS with {config['backbone']} backbone")
            except Exception as e:
                print(f"   ⚠️  {config['name']}: {e}")
    def preprocess_image(self, image_path: str) -> Optional[torch.Tensor]:
        try:
            if not os.path.exists(image_path):
                return None
            img = Image.open(image_path).convert('RGB')
            img = img.resize(self.target_size, Image.Resampling.LANCZOS)
            img_array = np.array(img, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
            tensor = tensor.float()
            tensor = tensor.to(self.device)
            return tensor
        except Exception as e:
            print(f"⚠️ Error loading {os.path.basename(image_path)}: {e}")
            return None
    def evaluate_pair(self, ref_image: str, dis_image: str, 
                     metrics_to_use: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        results = {}
        ref_tensor = self.preprocess_image(ref_image)
        dis_tensor = self.preprocess_image(dis_image)
        if ref_tensor is None or dis_tensor is None:
            print("⚠️ Could not load one or both images for pair evaluation.")
            if metrics_to_use is None:
                metrics_to_use = list(self.metrics.keys())
            for metric_name in metrics_to_use:
                if metric_name in self.metric_info and self.metric_info[metric_name]['type'] == 'full_reference':
                    results[metric_name] = {'score': np.nan, 'time': 0.0}
            return results
        active_metrics_to_use = metrics_to_use if metrics_to_use is not None else list(self.metrics.keys())
        with torch.no_grad():
            for metric_name in active_metrics_to_use:
                if metric_name not in self.metrics:
                    print(f"⚠️ Metric {metric_name} not initialized or available. Skipping.")
                    results[metric_name] = {'score': np.nan, 'time': 0.0}
                    continue
                try:
                    metric_func = self.metrics[metric_name]
                    info = self.metric_info[metric_name]
                    score = np.nan
                    computation_time = 0.0
                    if info['type'] != 'full_reference':
                        if metrics_to_use is not None and metric_name in metrics_to_use:
                            results[metric_name] = {'score': np.nan, 'time': 0.0}
                        continue
                    if info.get('custom'):
                        ref_np = ref_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                        dis_np = dis_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                        metric_result = metric_func(ref_np, dis_np)
                        if isinstance(metric_result, tuple):
                            score, computation_time = metric_result
                        else:
                            score = float(metric_result)
                            computation_time = 0.0
                    elif metric_name.startswith('lpips_'):
                        ref_norm = ref_tensor * 2.0 - 1.0
                        dis_norm = dis_tensor * 2.0 - 1.0
                        start_time = time.time()
                        metric_result = metric_func(ref_norm, dis_norm)
                        if isinstance(metric_result, tuple):
                            score, computation_time = metric_result
                        else:
                            score = float(metric_result.item()) if hasattr(metric_result, 'item') else float(metric_result)
                            computation_time = time.time() - start_time
                    else:
                        start_time = time.time()
                        metric_result = metric_func(dis_tensor, ref_tensor)
                        if isinstance(metric_result, tuple):
                            score, computation_time = metric_result
                        else:
                            score = float(metric_result.item()) if hasattr(metric_result, 'item') else float(metric_result)
                            computation_time = time.time() - start_time
                    results[metric_name] = {'score': score, 'time': computation_time}
                except Exception as e:
                    print(f"⚠️ Error evaluating FR metric {metric_name}: {e}")
                    results[metric_name] = {'score': np.nan, 'time': 0.0}
        return results
    def evaluate_no_reference(self, image_path: str, 
                            metrics_to_use: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        results = {}
        img_tensor = self.preprocess_image(image_path)
        if img_tensor is None:
            print(f"⚠️ Could not load image {image_path} for no-reference evaluation.")
            active_metrics_list = metrics_to_use
            if active_metrics_list is None:
                active_metrics_list = [name for name, info_val in self.metric_info.items() if info_val['type'] == 'no_reference']
            for metric_name in active_metrics_list:
                results[metric_name] = {'score': np.nan, 'time': 0.0}
            return results
        active_metrics_to_use = metrics_to_use
        if active_metrics_to_use is None:
            active_metrics_to_use = [name for name, info_val in self.metric_info.items() if info_val['type'] == 'no_reference']
        with torch.no_grad():
            for metric_name in active_metrics_to_use:
                if metric_name not in self.metrics:
                    print(f"⚠️ Metric {metric_name} not initialized or available. Skipping.")
                    results[metric_name] = {'score': np.nan, 'time': 0.0}
                    continue
                try:
                    metric_func = self.metrics[metric_name]
                    info = self.metric_info[metric_name]
                    score = np.nan
                    computation_time = 0.0
                    if info['type'] != 'no_reference':
                        if metrics_to_use is not None and metric_name in metrics_to_use:
                            results[metric_name] = {'score': np.nan, 'time': 0.0}
                        continue 
                    if info.get('custom'):
                        img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() 
                        metric_result = metric_func(img_np)
                        if isinstance(metric_result, tuple):
                            score, computation_time = metric_result
                        else:
                            score = float(metric_result)
                            computation_time = 0.0
                    else:
                        if metric_name == 'blinds2':
                            img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                            img_tensor = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                        start_time = time.time()
                        metric_value = metric_func(img_tensor)
                        if isinstance(metric_value, tuple):
                            score, computation_time = metric_value
                        else:
                            score = float(metric_value.item()) if hasattr(metric_value, 'item') else float(metric_value)
                            computation_time = time.time() - start_time
                    results[metric_name] = {'score': score, 'time': computation_time}
                except Exception as e:
                    print(f"⚠️ Error evaluating NR metric {metric_name}: {e}")
                    results[metric_name] = {'score': np.nan, 'time': 0.0}
        return results
    def get_metric_info(self) -> Dict[str, Dict]:
        return self.metric_info.copy()
    def get_available_metrics(self, metric_type: Optional[str] = None) -> List[str]:
        if metric_type is None:
            return list(self.metrics.keys())
        return [name for name, info in self.metric_info.items() 
                if info['type'] == metric_type]
    def get_metrics_by_category(self, category: str) -> List[str]:
        return [name for name, info in self.metric_info.items() 
                if info['category'] == category]
    def validate_image_path(self, image_path: str) -> bool:
        if not os.path.exists(image_path):
            return False
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except Exception:
            return False

def quick_evaluate(ref_image: str, dis_image: str, device: str = 'auto') -> Dict[str, float]:
    evaluator = IQAMetrics(device=device)
    common_metrics = evaluator.get_available_metrics('full_reference')
    return evaluator.evaluate_pair(ref_image, dis_image, common_metrics)

if __name__ == "__main__":
    print("🧪 Testing IQA Metrics Library")
    evaluator = IQAMetrics()
    print(f"\nAvailable metrics: {len(evaluator.get_available_metrics())}")
    print(f"Full-reference: {len(evaluator.get_available_metrics('full_reference'))}")
    print(f"No-reference: {len(evaluator.get_available_metrics('no_reference'))}")
    print("\nMetric categories:")
    for category in ['structural', 'signal', 'perceptual', 'feature', 'blind']:
        metrics = evaluator.get_metrics_by_category(category)
        if metrics:
            print(f"  {category.title()}: {metrics}")

