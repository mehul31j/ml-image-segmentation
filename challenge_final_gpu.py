# FINAL VERSION - 100 EPOCHS WITH GPU OPTIMIZATION
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os

from util import load_dataset, store_predictions, segment_with_knn, calculate_iou_metrics, print_iou_metrics
from dataset_utils import create_data_loader

def train_final_model_gpu(images, scribbles, ground_truth, epochs=100, batch_size=8, lr=3e-4):
    """GPU optimized training - 100 epochs"""
    # Force GPU usage and check availability
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        # Clear GPU cache
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
        print("⚠️  GPU not available, using CPU")
    
    # Use best model
    model = __import__('unet_model').ImprovedUNet(n_channels=4, n_classes=2).to(device)
    
    # GPU optimized settings
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.3, 0.7]).to(device))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Optimized scheduler for 100 epochs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr*3, epochs=epochs, 
        steps_per_epoch=len(images)//batch_size + 1,
        pct_start=0.2, div_factor=10, final_div_factor=100
    )
    
    # Enable GPU optimizations
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
        torch.backends.cudnn.enabled = True
    
    train_loader = create_data_loader(images, scribbles, ground_truth, batch_size, shuffle=True)
    
    model.train()
    best_iou = 0
    best_model_state = None
    
    print(f"🚀 GPU TRAINING: {epochs} epochs with {len(images)} images...")
    print(f"📊 Batch size: {batch_size}, Steps per epoch: {len(train_loader)}")
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        for inputs, targets in train_loader:
            # Move to GPU with non_blocking for speed
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / num_batches
            
            model.eval()
            with torch.no_grad():
                # Fast evaluation on 40 images
                eval_indices = np.random.choice(len(images), 40, replace=False)
                pred_list = []
                gt_list = []
                
                for idx in eval_indices:
                    image = torch.from_numpy(images[idx]).permute(2, 0, 1).float() / 255.0
                    scribble = scribbles[idx]
                    scribble_input = np.where(scribble == 255, 0, np.where(scribble == 0, -1, 1))
                    scribble_input = torch.from_numpy(scribble_input).unsqueeze(0).float()
                    input_tensor = torch.cat([image, scribble_input], dim=0).unsqueeze(0).to(device, non_blocking=True)
                    
                    output = model(input_tensor)
                    pred = torch.argmax(output, dim=1).cpu().numpy()[0]
                    pred_list.append(pred)
                    gt_list.append(ground_truth[idx])
                
                pred_array = np.array(pred_list)
                gt_array = np.array(gt_list)
                metrics = calculate_iou_metrics(pred_array, gt_array)
                current_iou = metrics["Mean IoU"]
                
                current_lr = scheduler.get_last_lr()[0]
                
                # GPU memory info
                if device.type == 'cuda':
                    memory_used = torch.cuda.memory_allocated() / 1024**3
                    memory_cached = torch.cuda.memory_reserved() / 1024**3
                    print(f'Epoch {epoch+1:3d}: Loss: {avg_loss:.4f}, IoU: {current_iou:.4f}, LR: {current_lr:.6f}, GPU: {memory_used:.1f}GB')
                else:
                    print(f'Epoch {epoch+1:3d}: Loss: {avg_loss:.4f}, IoU: {current_iou:.4f}, LR: {current_lr:.6f}')
                
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_model_state = model.state_dict().copy()
                    print(f"   🏆 NEW BEST IoU: {best_iou:.4f}")
                    
                    # Clear GPU cache after saving best model
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
            
            model.train()
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        torch.save(best_model_state, 'final_best_model_gpu.pth')
        print(f"💾 Best model saved: final_best_model_gpu.pth")
    
    print(f"🎯 TRAINING COMPLETE! Best IoU: {best_iou:.4f}")
    return model

def predict_gpu_optimized(model, images, scribbles, device, batch_size=12):
    """GPU optimized prediction with TTA"""
    model.eval()
    predictions = []
    
    print(f"🔮 Making GPU-optimized predictions on {len(images)} images...")
    
    with torch.no_grad():
        for i in range(len(images)):
            image = images[i]
            scribble = scribbles[i]
            
            # Original prediction
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            scribble_input = np.where(scribble == 255, 0, np.where(scribble == 0, -1, 1))
            scribble_tensor = torch.from_numpy(scribble_input).unsqueeze(0).float()
            input_tensor = torch.cat([image_tensor, scribble_tensor], dim=0).unsqueeze(0).to(device, non_blocking=True)
            
            output1 = model(input_tensor)
            prob1 = torch.softmax(output1, dim=1)[0, 1].cpu().numpy()
            
            # TTA - Flipped prediction (FIXED)
            flipped_image = np.fliplr(image).copy()  # FIX: Add .copy()
            flipped_scribble = np.fliplr(scribble).copy()  # FIX: Add .copy()
            
            flipped_image_tensor = torch.from_numpy(flipped_image).permute(2, 0, 1).float() / 255.0
            flipped_scribble_input = np.where(flipped_scribble == 255, 0, np.where(flipped_scribble == 0, -1, 1))
            flipped_scribble_tensor = torch.from_numpy(flipped_scribble_input).unsqueeze(0).float()
            flipped_input_tensor = torch.cat([flipped_image_tensor, flipped_scribble_tensor], dim=0).unsqueeze(0).to(device, non_blocking=True)
            
            output2 = model(flipped_input_tensor)
            prob2 = torch.softmax(output2, dim=1)[0, 1].cpu().numpy()
            prob2_flipped = np.fliplr(prob2).copy()  # FIX: Add .copy()
            
            # Ensemble prediction
            ensemble_prob = (prob1 + prob2_flipped) / 2
            final_pred = (ensemble_prob > 0.5).astype(np.uint8)
            predictions.append(final_pred)
            
            if (i + 1) % 25 == 0:
                print(f"   Processed {i + 1}/{len(images)} images")
                # Clear GPU cache periodically
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
    
    return np.array(predictions)

def predict_fast_gpu(model, images, scribbles, device, batch_size=16):
    """Fast GPU prediction without TTA"""
    model.eval()
    predictions = []
    
    print(f"🔮 Making FAST GPU predictions on {len(images)} images...")
    
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            end_idx = min(i + batch_size, len(images))
            
            batch_inputs = []
            for j in range(i, end_idx):
                image = torch.from_numpy(images[j]).permute(2, 0, 1).float() / 255.0
                scribble = scribbles[j]
                scribble_input = np.where(scribble == 255, 0, np.where(scribble == 0, -1, 1))
                scribble_input = torch.from_numpy(scribble_input).unsqueeze(0).float()
                input_tensor = torch.cat([image, scribble_input], dim=0)
                batch_inputs.append(input_tensor)
            
            batch_tensor = torch.stack(batch_inputs).to(device, non_blocking=True)
            outputs = model(batch_tensor)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            predictions.extend(preds)
            
            if (i // batch_size + 1) % 5 == 0:
                print(f"   Processed {min(i + batch_size, len(images))}/{len(images)} images")
    
    return np.array(predictions)

def main():
    print("🏆 FINAL CHAMPIONSHIP RUN - 100 EPOCHS GPU OPTIMIZED")
    print("="*75)
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("❌ No GPU available - will use CPU")
    
    # Load ALL training data
    print("\n📂 Loading ALL training data...")
    images_train, scrib_train, gt_train, fnames_train, palette = load_dataset(
        "dataset/training", "images", "scribbles", "ground_truth"
    )
    
    print(f"✅ Loaded {len(images_train)} training images")
    
    # Quick KNN baseline (k=5 only)
    print("\n🔍 Quick KNN baseline (k=5)...")
    pred_train_knn = np.stack([
        segment_with_knn(image, scribble, k=5)
        for image, scribble in zip(images_train, scrib_train)
    ])
    knn_metrics = calculate_iou_metrics(pred_train_knn, gt_train)
    print(f"KNN IoU: {knn_metrics['Mean IoU']:.4f}")
    
    # GPU U-Net training (100 epochs)
    print("\n🧠 GPU U-Net training (100 epochs)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = train_final_model_gpu(images_train, scrib_train, gt_train, epochs=100, batch_size=8, lr=3e-4)
    
    # Choose prediction method based on GPU availability
    if device.type == 'cuda':
        print("\n🚀 Using TTA prediction (GPU available)")
        pred_train_unet = predict_gpu_optimized(model, images_train, scrib_train, device)
    else:
        print("\n⚡ Using FAST prediction (CPU fallback)")
        pred_train_unet = predict_fast_gpu(model, images_train, scrib_train, device, batch_size=8)
    
    unet_metrics = calculate_iou_metrics(pred_train_unet, gt_train)
    print(f"\n🎯 FINAL U-Net IoU: {unet_metrics['Mean IoU']:.4f}")
    
    # Choose best method
    if unet_metrics['Mean IoU'] > knn_metrics['Mean IoU']:
        best_method = "U-Net"
        best_predictions = pred_train_unet
        improvement = unet_metrics['Mean IoU'] - knn_metrics['Mean IoU']
        print(f"🏆 U-Net CHAMPION! IoU: {unet_metrics['Mean IoU']:.4f} (+{improvement:.4f} vs KNN)")
    else:
        best_method = "KNN"
        best_predictions = pred_train_knn
        print(f"🏆 KNN wins! IoU: {knn_metrics['Mean IoU']:.4f}")
    
    # Store FINAL training predictions
    print(f"\n💾 Storing FINAL results...")
    store_predictions(best_predictions, "dataset/training", "final_predictions", fnames_train, palette)
    
    # Process ALL test data
    print(f"\n📊 Processing ALL test data...")
    images_test, scrib_test, fnames_test = load_dataset("dataset/test", "images", "scribbles")
    
    if best_method == "U-Net":
        if device.type == 'cuda':
            pred_test = predict_gpu_optimized(model, images_test, scrib_test, device)
        else:
            pred_test = predict_fast_gpu(model, images_test, scrib_test, device, batch_size=8)
    else:
        print("Using KNN for test predictions...")
        pred_test = np.stack([
            segment_with_knn(image, scribble, k=5)
            for image, scribble in zip(images_test, scrib_test)
        ])
    
    # Store FINAL test predictions
    store_predictions(pred_test, "dataset/test", "final_predictions", fnames_test, palette)
    
    print(f"\n🎯 FINAL CHAMPIONSHIP RESULTS:")
    final_metrics = calculate_iou_metrics(best_predictions, gt_train)
    print(f"🏆 Method: {best_method}")
    print(f"📈 Mean IoU: {final_metrics['Mean IoU']:.4f}")
    print(f"📈 Background IoU: {final_metrics['Background IoU']:.4f}")
    print(f"📈 Object IoU: {final_metrics['Object IoU']:.4f}")
    print(f"📁 FINAL results saved in: final_predictions/")
    
    # Championship assessment
    if final_metrics['Mean IoU'] > 0.8187:
        print("🥇 WORLD CHAMPION! Beat Team 44!")
    elif final_metrics['Mean IoU'] > 0.8149:
        print("🥈 SILVER MEDAL! Beat Team 4!")
    elif final_metrics['Mean IoU'] > 0.7763:
        print("🥉 BRONZE MEDAL! Beat Team 46!")
    else:
        print("💪 Strong performance!")
    
    print(f"\n🚀 CHAMPIONSHIP RUN COMPLETE!")
    print(f"🏆 Final Score: {final_metrics['Mean IoU']:.4f}")
    
    # Final GPU cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("🧹 GPU cache cleared")

if __name__ == "__main__":
    main()