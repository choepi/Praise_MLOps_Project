import pytest
import numpy as np
import math
from unittest.mock import Mock

# Import the functions we want to test
# Adjust the import path as needed for your project structure
try:
    from train import distance, calculate_angle, extract_hand_features
except ImportError:
    # If running from different directory, try relative import
    import sys
    import os
    
    # Get parent directory
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Add parent directory to path
    sys.path.append(parent_dir)
    
    # Try to find the correct module location by checking sister directories
    sister_dirs = ['src', 'app', 'models']  # Common directory names
    for dirname in sister_dirs:
        sister_path = os.path.join(parent_dir, dirname)
        if os.path.exists(sister_path):
            sys.path.append(sister_path)
    
    # Try the import again
    from train import distance, calculate_angle, extract_hand_features



class TestDistanceCalculation:
    """Test the 3D distance calculation function."""
    
    def test_distance_zero(self):
        """Distance between same point should be 0."""
        p1 = Mock()
        p1.x, p1.y, p1.z = 1.0, 2.0, 3.0
        
        result = distance(p1, p1)
        assert result == 0.0
    
    def test_distance_unit_axis(self):
        """Distance along unit axis should be 1."""
        p1 = Mock()
        p1.x, p1.y, p1.z = 0.0, 0.0, 0.0
        
        p2 = Mock()
        p2.x, p2.y, p2.z = 1.0, 0.0, 0.0
        
        result = distance(p1, p2)
        assert abs(result - 1.0) < 1e-6
    
    def test_distance_3d_pythagorean(self):
        """Test 3D Pythagorean theorem: 3-4-5 triangle in 3D."""
        p1 = Mock()
        p1.x, p1.y, p1.z = 0.0, 0.0, 0.0
        
        p2 = Mock()
        p2.x, p2.y, p2.z = 3.0, 4.0, 0.0  # Should be distance 5
        
        result = distance(p1, p2)
        assert abs(result - 5.0) < 1e-6
    
    def test_distance_negative_coords(self):
        """Distance calculation with negative coordinates."""
        p1 = Mock()
        p1.x, p1.y, p1.z = -1.0, -1.0, -1.0
        
        p2 = Mock()
        p2.x, p2.y, p2.z = 1.0, 1.0, 1.0
        
        # Distance should be sqrt(4 + 4 + 4) = sqrt(12) = 2*sqrt(3)
        expected = 2 * math.sqrt(3)
        result = distance(p1, p2)
        assert abs(result - expected) < 1e-6


class TestAngleCalculation:
    """Test the angle calculation between three points."""
    
    def test_angle_straight_line_180(self):
        """Three collinear points should give 180 degrees."""
        p1 = Mock()
        p1.x, p1.y, p1.z = 0.0, 0.0, 0.0
        
        p2 = Mock()  # Middle point
        p2.x, p2.y, p2.z = 1.0, 0.0, 0.0
        
        p3 = Mock()
        p3.x, p3.y, p3.z = 2.0, 0.0, 0.0
        
        result = calculate_angle(p1, p2, p3)
        assert abs(result - 180.0) < 1e-6
    
    def test_angle_right_angle_90(self):
        """Perpendicular lines should give 90 degrees."""
        p1 = Mock()
        p1.x, p1.y, p1.z = 0.0, 0.0, 0.0
        
        p2 = Mock()  # Corner point
        p2.x, p2.y, p2.z = 1.0, 0.0, 0.0
        
        p3 = Mock()
        p3.x, p3.y, p3.z = 1.0, 1.0, 0.0
        
        result = calculate_angle(p1, p2, p3)
        assert abs(result - 90.0) < 1e-6
    
    def test_angle_60_degrees(self):
        """Test known 60-degree angle."""
        p1 = Mock()
        p1.x, p1.y, p1.z = 0.0, 0.0, 0.0
        
        p2 = Mock()  # Corner point
        p2.x, p2.y, p2.z = 1.0, 0.0, 0.0
        
        p3 = Mock()  # 60 degrees from p1-p2 line
        p3.x, p3.y, p3.z = 0.5, math.sqrt(3)/2, 0.0
        
        result = calculate_angle(p1, p2, p3)
        assert abs(result - 60.0) < 1e-6  # More reasonable tolerance
    
    def test_angle_120_degrees(self):
        """Test known 120-degree angle."""
        p1 = Mock()
        p1.x, p1.y, p1.z = 0.0, 0.0, 0.0
        
        p2 = Mock()  # Corner point
        p2.x, p2.y, p2.z = 1.0, 0.0, 0.0
        
        p3 = Mock()  # 120 degrees from p1-p2 line
        p3.x, p3.y, p3.z = 1.5, math.sqrt(3)/2, 0.0
        
        result = calculate_angle(p1, p2, p3)
        assert abs(result - 120.0) < 1e-6
    
    def test_angle_zero_vector_handling(self):
        """Handle zero-length vectors gracefully."""
        p1 = Mock()
        p1.x, p1.y, p1.z = 1.0, 1.0, 1.0
        
        p2 = Mock()  # Same as p1 - creates zero vector
        p2.x, p2.y, p2.z = 1.0, 1.0, 1.0
        
        p3 = Mock()
        p3.x, p3.y, p3.z = 2.0, 2.0, 2.0
        
        result = calculate_angle(p1, p2, p3)
        assert result == 0  # Should handle division by zero


class TestHandFeatureExtraction:
    """Test the main hand feature extraction function."""
    
    def create_simple_hand_landmarks(self):
        """Create a simple, predictable set of hand landmarks for testing."""
        landmarks = []
        
        # Define landmarks in a predictable pattern
        # We'll create an "open hand" with fingers extended
        positions = [
            # Wrist (0)
            (0.5, 0.9, 0.0),
            
            # Thumb (1-4) - pointing up and left
            (0.3, 0.8, 0.0), (0.25, 0.7, 0.0), (0.2, 0.6, 0.0), (0.15, 0.5, 0.0),
            
            # Index finger (5-8) - pointing straight up
            (0.4, 0.7, 0.0), (0.4, 0.6, 0.0), (0.4, 0.5, 0.0), (0.4, 0.4, 0.0),
            
            # Middle finger (9-12) - pointing straight up (longest)
            (0.5, 0.7, 0.0), (0.5, 0.6, 0.0), (0.5, 0.5, 0.0), (0.5, 0.3, 0.0),
            
            # Ring finger (13-16) - pointing straight up
            (0.6, 0.7, 0.0), (0.6, 0.6, 0.0), (0.6, 0.5, 0.0), (0.6, 0.4, 0.0),
            
            # Pinky (17-20) - pointing up and right
            (0.7, 0.7, 0.0), (0.7, 0.6, 0.0), (0.7, 0.5, 0.0), (0.7, 0.45, 0.0)
        ]
        
        for x, y, z in positions:
            landmark = Mock()
            landmark.x, landmark.y, landmark.z = x, y, z
            landmarks.append(landmark)
        
        return landmarks
    
    def test_feature_extraction_basic_properties(self):
        """Test that feature extraction returns expected structure."""
        landmarks = self.create_simple_hand_landmarks()
        
        features = extract_hand_features(landmarks)
        
        # Should return exactly 17 features
        assert len(features) == 17
        assert isinstance(features, np.ndarray)
        assert features.dtype == np.float32
        
        # All features should be finite numbers
        assert np.all(np.isfinite(features))
        
        # All features should be non-negative (distances and ratios)
        assert np.all(features >= 0)
    
    def test_feature_groups_structure(self):
        """Test that features are grouped correctly."""
        landmarks = self.create_simple_hand_landmarks()
        features = extract_hand_features(landmarks)
        
        # Verify the 4 feature groups
        extension_ratios = features[0:5]    # Finger extension ratios
        bending_angles = features[5:10]     # Finger bending angles  
        inter_distances = features[10:14]   # Inter-fingertip distances
        thumb_opposition = features[14:17]  # Thumb opposition distances
        
        # Extension ratios should be positive
        assert np.all(extension_ratios > 0)
        
        # Angles should be between 0 and 180 degrees
        assert np.all(bending_angles >= 0)
        assert np.all(bending_angles <= 180)
        
        # Distances should be positive
        assert np.all(inter_distances > 0)
        assert np.all(thumb_opposition > 0)
    
    def test_palm_width_calculation(self):
        """Test that palm width calculation works correctly."""
        landmarks = self.create_simple_hand_landmarks()
        
        # Calculate expected palm width manually
        # Index MCP (5): (0.4, 0.7, 0.0)
        # Pinky MCP (17): (0.7, 0.7, 0.0)
        expected_palm_width = 0.3  # |0.7 - 0.4| = 0.3
        
        # Extract features and check that ratios make sense
        features = extract_hand_features(landmarks)
        
        # Extension ratios should be normalized by palm width
        # So they should be reasonable values (not too large)
        extension_ratios = features[0:5]
        assert np.all(extension_ratios < 10)  # Sanity check
        assert np.all(extension_ratios > 0.1)  # Should be reasonable
    
    def test_angle_calculations_realistic(self):
        """Test that calculated angles are in realistic ranges."""
        landmarks = self.create_simple_hand_landmarks()
        features = extract_hand_features(landmarks)
        
        # Finger bending angles (features 5-9)
        angles = features[5:10]
        
        # For our "open hand", angles should be relatively large
        # (close to 180 degrees for extended fingers)
        assert np.all(angles > 90)  # Should be extended
        assert np.all(angles <= 180)  # Can't be more than straight
    
    def test_reproducibility(self):
        """Test that same input always gives same output."""
        landmarks = self.create_simple_hand_landmarks()
        
        features1 = extract_hand_features(landmarks)
        features2 = extract_hand_features(landmarks)
        
        # Should be exactly the same
        np.testing.assert_array_equal(features1, features2)
    
    def test_different_hand_poses_give_different_features(self):
        """Test that different hand poses produce different features."""
        open_hand = self.create_simple_hand_landmarks()
        
        # Create a "closed fist" by moving fingertips closer to palm
        closed_hand = []
        for i, landmark in enumerate(open_hand):
            new_landmark = Mock()
            if i in [4, 8, 12, 16, 20]:  # Fingertips
                # Move fingertips closer to palm center
                new_landmark.x = landmark.x * 0.9 + 0.5 * 0.1
                new_landmark.y = landmark.y * 0.9 + 0.7 * 0.1
                new_landmark.z = landmark.z
            else:
                new_landmark.x = landmark.x
                new_landmark.y = landmark.y
                new_landmark.z = landmark.z
            closed_hand.append(new_landmark)
        
        open_features = extract_hand_features(open_hand)
        closed_features = extract_hand_features(closed_hand)
        
        # Features should be different
        assert not np.array_equal(open_features, closed_features)
        
        # Extension ratios should be smaller for closed hand
        assert np.all(closed_features[0:5] <= open_features[0:5])


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_zero_palm_width_protection(self):
        """Test behavior when palm width would be zero."""
        landmarks = []
        
        # Create landmarks where index and pinky MCP are at same position
        for i in range(21):
            landmark = Mock()
            if i in [5, 17]:  # Index MCP and Pinky MCP - same position
                landmark.x, landmark.y, landmark.z = 0.5, 0.7, 0.0
            else:
                landmark.x, landmark.y, landmark.z = 0.5, 0.6, 0.0
            landmarks.append(landmark)
        
        # This should either handle gracefully or raise a clear error
        try:
            features = extract_hand_features(landmarks)
            # If it doesn't crash, check that we don't get invalid values
            assert not np.any(np.isinf(features))
            assert not np.any(np.isnan(features))
        except ZeroDivisionError:
            # This is also acceptable behavior
            pass
    
    def test_extreme_coordinate_values(self):
        """Test with extreme coordinate values."""
        landmarks = []
        
        # Create landmarks with very large coordinates
        base_positions = [
            (0.5, 0.9, 0.0), (0.3, 0.8, 0.0), (0.25, 0.7, 0.0), (0.2, 0.6, 0.0), 
            (0.15, 0.5, 0.0), (0.4, 0.7, 0.0), (0.4, 0.6, 0.0), (0.4, 0.5, 0.0), 
            (0.4, 0.4, 0.0), (0.5, 0.7, 0.0), (0.5, 0.6, 0.0), (0.5, 0.5, 0.0), 
            (0.5, 0.3, 0.0), (0.6, 0.7, 0.0), (0.6, 0.6, 0.0), (0.6, 0.5, 0.0), 
            (0.6, 0.4, 0.0), (0.7, 0.7, 0.0), (0.7, 0.6, 0.0), (0.7, 0.5, 0.0), 
            (0.7, 0.45, 0.0)
        ]
        
        for x, y, z in base_positions:
            landmark = Mock()
            # Scale coordinates by large factor
            landmark.x, landmark.y, landmark.z = x * 1000, y * 1000, z
            landmarks.append(landmark)
        
        features = extract_hand_features(landmarks)
        
        # Features should still be finite and reasonable
        assert np.all(np.isfinite(features))
        assert np.all(features >= 0)


if __name__ == "__main__":
    # Simple test runner if pytest not available
    import sys
    
    try:
        import pytest
        pytest.main([__file__, "-v"])
    except ImportError:
        print("pytest not available, running basic tests...")
        
        # Run a few basic tests manually
        test_distance = TestDistanceCalculation()
        test_distance.test_distance_zero()
        test_distance.test_distance_unit_axis()
        print("✅ Distance tests passed")
        
        test_angle = TestAngleCalculation()
        test_angle.test_angle_straight_line_180()
        test_angle.test_angle_right_angle_90()
        print("✅ Angle tests passed")
        
        test_features = TestHandFeatureExtraction()
        test_features.test_feature_extraction_basic_properties()
        test_features.test_reproducibility()
        print("✅ Feature extraction tests passed")
        
        print("🎉 All basic tests completed successfully!")