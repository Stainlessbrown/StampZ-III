#!/usr/bin/env python3
"""
Machine Learning Integration Concept for StampZ
Shows different approaches to adding ML capabilities to the application.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json

# Classical ML approach using existing StampZ data
class StampZClassicalML:
    """Classical ML using StampZ's L*a*b* color measurements."""
    
    def __init__(self):
        self.models = {}
        self.training_data = []
        
    def prepare_training_data_from_stampz(self, database_measurements: List[Dict]) -> np.ndarray:
        """Convert StampZ measurements to ML training format."""
        features = []
        
        for measurement in database_measurements:
            # Extract color features from StampZ data
            feature_vector = [
                measurement.get('l_value', 0),      # Lightness
                measurement.get('a_value', 0),      # Green-Red axis
                measurement.get('b_value', 0),      # Blue-Yellow axis
                measurement.get('rgb_r', 0) / 255,  # Normalized RGB
                measurement.get('rgb_g', 0) / 255,
                measurement.get('rgb_b', 0) / 255,
                # Derived features
                self._calculate_chroma(measurement.get('a_value', 0), measurement.get('b_value', 0)),
                self._calculate_hue_angle(measurement.get('a_value', 0), measurement.get('b_value', 0)),
                measurement.get('l_value', 0) / 100,  # Normalized lightness
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    def _calculate_chroma(self, a: float, b: float) -> float:
        """Calculate chroma (color intensity) from a* and b* values."""
        return np.sqrt(a**2 + b**2)
    
    def _calculate_hue_angle(self, a: float, b: float) -> float:
        """Calculate hue angle from a* and b* values."""
        return np.arctan2(b, a) * 180 / np.pi
    
    def train_color_classifier(self, measurements_by_class: Dict[str, List[Dict]]):
        """Train a classifier to recognize different color varieties."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score
            
            # Prepare training data
            X = []  # Features
            y = []  # Labels
            
            for color_class, measurements in measurements_by_class.items():
                features = self.prepare_training_data_from_stampz(measurements)
                X.extend(features)
                y.extend([color_class] * len(features))
            
            X = np.array(X)
            y = np.array(y)
            
            print(f"Training classifier with {len(X)} samples, {len(set(y))} classes")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            self.models['color_classifier'] = RandomForestClassifier(n_estimators=100, random_state=42)
            self.models['color_classifier'].fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.models['color_classifier'].predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"Classifier accuracy: {accuracy:.2%}")
            return True
            
        except ImportError:
            print("scikit-learn not available. Install with: pip install scikit-learn")
            return False
    
    def predict_color_variety(self, measurement: Dict) -> Dict:
        """Predict color variety from a single measurement."""
        if 'color_classifier' not in self.models:
            return {"error": "Classifier not trained"}
        
        # Prepare feature vector
        features = self.prepare_training_data_from_stampz([measurement])
        
        # Make prediction
        prediction = self.models['color_classifier'].predict(features[0].reshape(1, -1))[0]
        probabilities = self.models['color_classifier'].predict_proba(features[0].reshape(1, -1))[0]
        
        # Get class names and probabilities
        classes = self.models['color_classifier'].classes_
        prob_dict = dict(zip(classes, probabilities))
        
        return {
            'predicted_variety': prediction,
            'confidence': max(probabilities),
            'all_probabilities': prob_dict
        }

# Computer Vision ML approach
class StampZComputerVisionML:
    """Computer vision ML for image-based analysis."""
    
    def __init__(self):
        self.models = {}
        
    def prepare_image_features(self, image_array: np.ndarray) -> np.ndarray:
        """Extract features from stamp images for ML."""
        features = []
        
        # Color histogram features
        for channel in range(3):  # RGB channels
            hist, _ = np.histogram(image_array[:, :, channel], bins=32, range=(0, 256))
            features.extend(hist / hist.sum())  # Normalize
        
        # Statistical features per channel
        for channel in range(3):
            ch = image_array[:, :, channel]
            features.extend([
                ch.mean(),      # Average color
                ch.std(),       # Color variation
                np.median(ch),  # Median color
                ch.min(),       # Darkest value
                ch.max()        # Brightest value
            ])
        
        # Overall image features
        gray = np.mean(image_array, axis=2)
        features.extend([
            gray.mean(),                    # Overall brightness
            gray.std(),                     # Overall contrast
            np.sum(gray < 50) / gray.size,  # Dark pixel ratio (ink coverage)
        ])
        
        return np.array(features)
    
    def train_quality_assessor(self, image_quality_dataset: List[Tuple[np.ndarray, str]]):
        """Train a model to assess stamp condition/quality."""
        try:
            from sklearn.svm import SVC
            from sklearn.preprocessing import StandardScaler
            
            # Prepare features
            X = []
            y = []
            
            for image, quality_label in image_quality_dataset:
                features = self.prepare_image_features(image)
                X.append(features)
                y.append(quality_label)
            
            X = np.array(X)
            y = np.array(y)
            
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Train SVM
            self.models['quality_assessor'] = SVC(probability=True)
            self.models['quality_assessor'].fit(X_scaled, y)
            
            print(f"Quality assessor trained with {len(X)} samples")
            return True
            
        except ImportError:
            print("scikit-learn not available")
            return False

# Deep Learning approach (advanced)
class StampZDeepLearningML:
    """Advanced deep learning for stamp analysis."""
    
    def __init__(self):
        self.models = {}
    
    def create_cnn_classifier(self, image_shape: Tuple[int, int, int], num_classes: int):
        """Create a CNN for stamp variety classification."""
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, models
            
            model = models.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=image_shape),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.Flatten(),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation='softmax')
            ])
            
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self.models['cnn_classifier'] = model
            print(f"CNN classifier created for {num_classes} classes")
            return True
            
        except ImportError:
            print("TensorFlow not available. Install with: pip install tensorflow")
            return False

# Integration with StampZ workflow
class StampZMLIntegration:
    """Main class to integrate ML capabilities with StampZ."""
    
    def __init__(self):
        self.classical_ml = StampZClassicalML()
        self.cv_ml = StampZComputerVisionML()
        self.deep_ml = StampZDeepLearningML()
        
    def analyze_with_ml(self, image_path: str, measurements: List[Dict]) -> Dict:
        """Complete ML analysis of a stamp image."""
        results = {
            'image_path': image_path,
            'ml_analysis': {},
            'recommendations': []
        }
        
        # Classical ML analysis using color measurements
        if measurements and 'color_classifier' in self.classical_ml.models:
            for measurement in measurements:
                color_prediction = self.classical_ml.predict_color_variety(measurement)
                results['ml_analysis']['color_classification'] = color_prediction
        
        # Add recommendations based on analysis
        if 'color_classification' in results['ml_analysis']:
            confidence = results['ml_analysis']['color_classification'].get('confidence', 0)
            if confidence > 0.8:
                results['recommendations'].append("High confidence color classification")
            elif confidence > 0.6:
                results['recommendations'].append("Moderate confidence - consider additional samples")
            else:
                results['recommendations'].append("Low confidence - may be unusual variety")
        
        return results

# Demo function
def demo_ml_integration():
    """Demonstrate ML integration concepts."""
    print("=== StampZ Machine Learning Integration Demo ===\n")
    
    # Simulate StampZ measurement data
    sample_measurements = {
        'light_blue': [
            {'l_value': 75, 'a_value': -5, 'b_value': -15, 'rgb_r': 180, 'rgb_g': 190, 'rgb_b': 220},
            {'l_value': 78, 'a_value': -3, 'b_value': -18, 'rgb_r': 185, 'rgb_g': 195, 'rgb_b': 225},
        ],
        'dark_blue': [
            {'l_value': 45, 'a_value': -8, 'b_value': -25, 'rgb_r': 80, 'rgb_g': 90, 'rgb_b': 140},
            {'l_value': 42, 'a_value': -6, 'b_value': -28, 'rgb_r': 75, 'rgb_g': 85, 'rgb_b': 135},
        ],
        'red': [
            {'l_value': 55, 'a_value': 35, 'b_value': 15, 'rgb_r': 180, 'rgb_g': 80, 'rgb_b': 90},
            {'l_value': 58, 'a_value': 32, 'b_value': 18, 'rgb_r': 185, 'rgb_g': 85, 'rgb_b': 95},
        ]
    }
    
    # Test classical ML
    print("1. Testing Classical ML with StampZ Color Data...")
    ml_integration = StampZMLIntegration()
    
    # Train classifier
    success = ml_integration.classical_ml.train_color_classifier(sample_measurements)
    
    if success:
        # Test prediction
        test_measurement = {'l_value': 76, 'a_value': -4, 'b_value': -16, 'rgb_r': 182, 'rgb_g': 192, 'rgb_b': 222}
        prediction = ml_integration.classical_ml.predict_color_variety(test_measurement)
        
        print(f"✅ Prediction for test sample:")
        print(f"   Variety: {prediction['predicted_variety']}")
        print(f"   Confidence: {prediction['confidence']:.2%}")
        
        # Show all probabilities
        print("   All probabilities:")
        for variety, prob in prediction['all_probabilities'].items():
            print(f"     {variety}: {prob:.2%}")
    
    print(f"\n2. ML Integration Features Available:")
    print(f"   ✅ Classical ML (color classification)")
    print(f"   ✅ Computer Vision ML (image features)")
    print(f"   ✅ Deep Learning setup (CNN architectures)")
    
    print(f"\n3. Integration Points with StampZ:")
    print(f"   • Use existing L*a*b* measurements for training")
    print(f"   • Integrate with color analysis workflow")
    print(f"   • Add ML predictions to export data")
    print(f"   • Provide confidence scores for classifications")
    
    print(f"\n4. Potential Applications:")
    print(f"   • Automatic color variety identification")
    print(f"   • Quality/condition assessment")
    print(f"   • Outlier detection in color measurements")
    print(f"   • Sorting/ranking by visual characteristics")

if __name__ == "__main__":
    demo_ml_integration()