import torch
import sys
import os

# Add gsegmenter to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from gsegmenter.editor.transform import apply_object_transform
except ImportError:
    # Handle the fact that we might run this directly without package setup
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from gsegmenter.editor.transform import apply_object_transform

def test_apply_object_transform():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running test on device: {device}")
    
    # Generate dummy data
    N = 100
    means = torch.randn(N, 3, device=device)
    
    # Dummy valid quaternions (w, x, y, z)
    rotations = torch.randn(N, 4, device=device)
    rotations = torch.nn.functional.normalize(rotations, p=2, dim=1)
    
    # Dummy Object IDs
    object_ids = torch.randint(0, 3, (N,), device=device) # objects 0, 1, 2
    
    target_id = 1
    
    # Transformation parameters
    translation = torch.tensor([1.0, 2.0, 3.0], device=device)
    # 90 degrees rotation around Z-axis
    import math
    theta = math.pi / 2
    rotation_matrix = torch.tensor([
        [math.cos(theta), -math.sin(theta), 0.0],
        [math.sin(theta),  math.cos(theta), 0.0],
        [0.0,             0.0,            1.0]
    ], device=device)
    
    # Save original values for target objects to verify non-destructive edit on others
    original_means = means.clone()
    original_rotations = rotations.clone()
    
    new_means, new_rotations = apply_object_transform(
        means, rotations, object_ids, target_id, translation, rotation_matrix
    )
    
    # Verification
    mask = (object_ids == target_id)
    not_mask = ~mask
    
    # 1. Unaffected objects should remain the same
    assert torch.allclose(new_means[not_mask], original_means[not_mask]), "Unaffected means changed!"
    assert torch.allclose(new_rotations[not_mask], original_rotations[not_mask]), "Unaffected rotations changed!"
    print("✓ Unaffected objects remained unchanged.")
    
    # 2. Affected objects should be transformed
    if mask.any():
        expected_target_means = torch.matmul(original_means[mask], rotation_matrix.transpose(0, 1)) + translation
        assert torch.allclose(new_means[mask], expected_target_means, atol=1e-5), "Affected means transform incorrect!"
        print("✓ Affected means properly translated and rotated.")
        
        # Quaternion norms should still be 1
        norms = torch.norm(new_rotations[mask], p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms)), "Affected rotation quaternions are not normalized!"
        print("✓ Affected rotations are valid quaternions.")

if __name__ == "__main__":
    test_apply_object_transform()
    print("All tests passed successfully!")
