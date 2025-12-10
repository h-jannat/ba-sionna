import tensorflow as tf
import os
from config import Config
from train import create_model

def inspect_checkpoint():
    checkpoint_dir = './checkpoints'
    latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    
    if not latest_ckpt:
        print("No checkpoint found!")
        return

    print(f"Inspecting checkpoint: {latest_ckpt}")
    print("=" * 60)
    
    # List variables in checkpoint
    ckpt_vars = tf.train.list_variables(latest_ckpt)
    ckpt_var_names = set(name for name, shape in ckpt_vars)
    
    print(f"Found {len(ckpt_vars)} variables in checkpoint.")
    for name, shape in ckpt_vars:
        print(f"  {name}: {shape}")
        
    print("\n" + "=" * 60)
    print("Current Model Variables")
    print("=" * 60)
    
    # Create model and build it
    model = create_model(Config)
    # Run dummy pass to create variables
    _ = model(batch_size=1, snr_db=10.0, training=False)
    
    model_vars = model.trainable_variables
    model_var_names = set(var.name.split(':')[0] for var in model_vars)
    
    print(f"Found {len(model_vars)} trainable variables in current model.")
    for var in model_vars:
        print(f"  {var.name}: {var.shape}")

    print("\n" + "=" * 60)
    print("Mismatch Analysis")
    print("=" * 60)
    
    # Check for missing variables
    # Note: Checkpoint names are like 'model/layer/kernel', Model names are 'layer/kernel:0'
    # We need to map them.
    
    # Simple check: Look for "unmatched"
    print("Variables in Checkpoint but NOT in Model (Potential Orphans):")
    # This is hard to match exactly due to prefixing, but let's look for obvious ones
    
    print("\nVariables in Model but NOT in Checkpoint (Not Restored):")
    # This is the critical part. If model expects 'gru_cell/kernel' but checkpoint has 'gru_cell_1/kernel'
    
if __name__ == "__main__":
    inspect_checkpoint()
