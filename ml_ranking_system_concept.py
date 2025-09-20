#!/usr/bin/env python3
"""
ML-Powered Image Ranking System for StampZ
Concept implementation for automated stamp variety ordering and analysis.

This system would rank stamp images based on various criteria:
- Lightness/darkness progression
- Color intensity variations
- Shade classifications
- Quality assessments
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class RankingCriterion(Enum):
    LIGHTNESS = "lightness"
    COLOR_INTENSITY = "color_intensity" 
    HUE_PROGRESSION = "hue_progression"
    SHADE_VARIETY = "shade_variety"
    OVERALL_DARKNESS = "overall_darkness"
    INK_DENSITY = "ink_density"
    COLOR_PURITY = "color_purity"

@dataclass
class ImageAnalysisData:
    """Container for image analysis data from StampZ."""
    image_path: str
    filename: str
    lab_values: List[Tuple[float, float, float]]  # L*a*b* measurements
    rgb_values: List[Tuple[int, int, int]]        # RGB measurements
    avg_lab: Tuple[float, float, float]           # Quality-controlled average
    avg_rgb: Tuple[int, int, int]                 # Average RGB
    sample_count: int
    measurement_quality: float                    # ΔE max from averaging

@dataclass
class RankingResult:
    """Result from ML ranking analysis."""
    image_path: str
    filename: str
    rank_score: float
    rank_position: int
    criterion_scores: Dict[str, float]
    confidence: float
    notes: str

class MLImageRanker:
    """Machine Learning-based image ranking system for stamp analysis."""
    
    def __init__(self):
        self.ranking_models = {}
        self.feature_extractors = {}
        self._initialize_feature_extractors()
    
    def _initialize_feature_extractors(self):
        """Initialize feature extraction functions."""
        
        def lightness_features(data: ImageAnalysisData) -> Dict[str, float]:
            """Extract lightness-related features."""
            l_values = [lab[0] for lab in data.lab_values]
            return {
                'avg_lightness': data.avg_lab[0],
                'lightness_std': np.std(l_values),
                'lightness_range': max(l_values) - min(l_values),
                'lightness_consistency': 1.0 - (np.std(l_values) / 100.0),
                'darkness_score': 100.0 - data.avg_lab[0]  # Inverse lightness
            }
        
        def color_intensity_features(data: ImageAnalysisData) -> Dict[str, float]:
            """Extract color intensity/saturation features."""
            a_values = [lab[1] for lab in data.lab_values]
            b_values = [lab[2] for lab in data.lab_values]
            
            # Calculate chroma (color intensity)
            chroma_values = [np.sqrt(a**2 + b**2) for a, b in zip(a_values, b_values)]
            avg_chroma = np.sqrt(data.avg_lab[1]**2 + data.avg_lab[2]**2)
            
            return {
                'avg_chroma': avg_chroma,
                'chroma_std': np.std(chroma_values),
                'color_purity': avg_chroma / 100.0,  # Normalized intensity
                'saturation_consistency': 1.0 - (np.std(chroma_values) / max(chroma_values) if chroma_values else 0)
            }
        
        def ink_density_features(data: ImageAnalysisData) -> Dict[str, float]:
            """Extract ink density and coverage features."""
            # This would analyze the distribution of color vs background
            # For now, using simplified metrics based on color data
            
            return {
                'estimated_ink_density': (100.0 - data.avg_lab[0]) / 100.0,
                'color_uniformity': 1.0 / (data.measurement_quality + 1.0),
                'coverage_quality': min(1.0, data.sample_count / 10.0)
            }
        
        self.feature_extractors = {
            RankingCriterion.LIGHTNESS: lightness_features,
            RankingCriterion.COLOR_INTENSITY: color_intensity_features,
            RankingCriterion.INK_DENSITY: ink_density_features
        }
    
    def rank_images(self, 
                   image_data_list: List[ImageAnalysisData], 
                   criterion: RankingCriterion = RankingCriterion.LIGHTNESS,
                   reverse: bool = False) -> List[RankingResult]:
        """
        Rank images based on specified criterion.
        
        Args:
            image_data_list: List of image analysis data
            criterion: Ranking criterion to use
            reverse: If True, rank from darkest to lightest (or opposite of default)
            
        Returns:
            List of ranking results, ordered by rank
        """
        
        if criterion not in self.feature_extractors:
            raise ValueError(f"Unsupported ranking criterion: {criterion}")
        
        extractor = self.feature_extractors[criterion]
        scored_images = []
        
        for data in image_data_list:
            features = extractor(data)
            score = self._calculate_ranking_score(features, criterion)
            
            scored_images.append({
                'data': data,
                'features': features,
                'score': score
            })
        
        # Sort by score
        scored_images.sort(key=lambda x: x['score'], reverse=reverse)
        
        # Create ranking results
        results = []
        for i, item in enumerate(scored_images):
            result = RankingResult(
                image_path=item['data'].image_path,
                filename=item['data'].filename,
                rank_score=item['score'],
                rank_position=i + 1,
                criterion_scores=item['features'],
                confidence=self._calculate_confidence(item['features'], item['data']),
                notes=self._generate_ranking_notes(item['features'], criterion)
            )
            results.append(result)
        
        return results
    
    def _calculate_ranking_score(self, features: Dict[str, float], criterion: RankingCriterion) -> float:
        """Calculate overall ranking score based on features."""
        
        if criterion == RankingCriterion.LIGHTNESS:
            # Primary: average lightness, Secondary: consistency
            return features['avg_lightness'] * 0.8 + features['lightness_consistency'] * 20.0
        
        elif criterion == RankingCriterion.COLOR_INTENSITY:
            # Primary: chroma intensity, Secondary: purity
            return features['avg_chroma'] * 0.7 + features['color_purity'] * 30.0
        
        elif criterion == RankingCriterion.INK_DENSITY:
            # Combination of density and uniformity
            return (features['estimated_ink_density'] * 70.0 + 
                   features['color_uniformity'] * 20.0 + 
                   features['coverage_quality'] * 10.0)
        
        return 0.0
    
    def _calculate_confidence(self, features: Dict[str, float], data: ImageAnalysisData) -> float:
        """Calculate confidence in ranking based on data quality."""
        # Higher sample count = higher confidence
        sample_confidence = min(1.0, data.sample_count / 20.0)
        
        # Lower measurement error = higher confidence  
        quality_confidence = 1.0 / (data.measurement_quality + 1.0)
        
        # Combine factors
        return (sample_confidence * 0.6 + quality_confidence * 0.4)
    
    def _generate_ranking_notes(self, features: Dict[str, float], criterion: RankingCriterion) -> str:
        """Generate descriptive notes for ranking."""
        if criterion == RankingCriterion.LIGHTNESS:
            lightness = features.get('avg_lightness', 0)
            if lightness > 80:
                return "Very light shade"
            elif lightness > 60:
                return "Light shade" 
            elif lightness > 40:
                return "Medium shade"
            elif lightness > 20:
                return "Dark shade"
            else:
                return "Very dark shade"
        
        elif criterion == RankingCriterion.COLOR_INTENSITY:
            intensity = features.get('avg_chroma', 0)
            if intensity > 50:
                return "High color intensity"
            elif intensity > 30:
                return "Medium color intensity"
            else:
                return "Low color intensity"
        
        return "Ranking analysis complete"

class PhilatelicMLAnalyzer:
    """High-level analyzer for philatelic ML applications."""
    
    def __init__(self):
        self.ranker = MLImageRanker()
    
    def analyze_shade_progression(self, stamps_data: List[ImageAnalysisData]) -> Dict:
        """Analyze a group of stamps to determine shade progression."""
        
        # Rank by lightness (lightest to darkest)
        lightness_ranking = self.ranker.rank_images(stamps_data, RankingCriterion.LIGHTNESS)
        
        # Rank by color intensity 
        intensity_ranking = self.ranker.rank_images(stamps_data, RankingCriterion.COLOR_INTENSITY)
        
        # Analyze progression patterns
        progression_analysis = self._analyze_progression_patterns(lightness_ranking)
        
        return {
            'lightness_progression': lightness_ranking,
            'intensity_progression': intensity_ranking,
            'progression_analysis': progression_analysis,
            'variety_count': len(stamps_data),
            'recommended_ordering': self._recommend_catalog_ordering(lightness_ranking, intensity_ranking)
        }
    
    def _analyze_progression_patterns(self, rankings: List[RankingResult]) -> Dict:
        """Analyze patterns in the ranking progression."""
        scores = [r.rank_score for r in rankings]
        
        # Calculate progression metrics
        score_gaps = [scores[i+1] - scores[i] for i in range(len(scores)-1)]
        
        return {
            'total_range': max(scores) - min(scores) if scores else 0,
            'average_gap': np.mean(score_gaps) if score_gaps else 0,
            'largest_gap': max(score_gaps) if score_gaps else 0,
            'progression_evenness': 1.0 - (np.std(score_gaps) / np.mean(score_gaps)) if score_gaps else 0,
            'distinct_groups': self._identify_distinct_groups(scores)
        }
    
    def _identify_distinct_groups(self, scores: List[float]) -> int:
        """Identify distinct groupings in the score progression."""
        if len(scores) < 2:
            return len(scores)
        
        # Simple gap analysis to identify natural groupings
        score_gaps = [scores[i+1] - scores[i] for i in range(len(scores)-1)]
        avg_gap = np.mean(score_gaps)
        
        # Count gaps significantly larger than average (potential group boundaries)
        large_gaps = sum(1 for gap in score_gaps if gap > avg_gap * 1.5)
        
        return large_gaps + 1  # Groups = large gaps + 1
    
    def _recommend_catalog_ordering(self, lightness_ranking: List[RankingResult], 
                                  intensity_ranking: List[RankingResult]) -> List[Dict]:
        """Recommend catalog ordering based on multiple criteria."""
        
        recommendations = []
        for i, light_result in enumerate(lightness_ranking):
            # Find corresponding intensity ranking
            intensity_pos = None
            for j, int_result in enumerate(intensity_ranking):
                if int_result.image_path == light_result.image_path:
                    intensity_pos = j + 1
                    break
            
            recommendations.append({
                'filename': light_result.filename,
                'lightness_rank': i + 1,
                'intensity_rank': intensity_pos,
                'primary_classification': light_result.notes,
                'confidence': light_result.confidence,
                'recommended_position': i + 1  # Could be more sophisticated
            })
        
        return recommendations

def demo_ml_ranking():
    """Demonstration of ML ranking functionality."""
    print("=== StampZ ML Image Ranking Demo ===\n")
    
    # Create sample data (this would come from StampZ analysis)
    sample_data = [
        ImageAnalysisData(
            image_path="/path/to/stamp1.jpg",
            filename="1920_george_v_light.jpg", 
            lab_values=[(75, -5, 15), (73, -4, 16), (76, -6, 14)],
            rgb_values=[(180, 190, 165), (175, 185, 160), (185, 195, 170)],
            avg_lab=(74.7, -5.0, 15.0),
            avg_rgb=(180, 190, 165),
            sample_count=3,
            measurement_quality=2.1
        ),
        ImageAnalysisData(
            image_path="/path/to/stamp2.jpg", 
            filename="1920_george_v_medium.jpg",
            lab_values=[(45, -8, 20), (43, -7, 22), (47, -9, 18)],
            rgb_values=[(110, 125, 95), (105, 120, 90), (115, 130, 100)],
            avg_lab=(45.0, -8.0, 20.0),
            avg_rgb=(110, 125, 95),
            sample_count=3,
            measurement_quality=1.8
        ),
        ImageAnalysisData(
            image_path="/path/to/stamp3.jpg",
            filename="1920_george_v_dark.jpg", 
            lab_values=[(25, -12, 25), (23, -11, 27), (27, -13, 23)],
            rgb_values=[(60, 75, 45), (55, 70, 40), (65, 80, 50)],
            avg_lab=(25.0, -12.0, 25.0),
            avg_rgb=(60, 75, 45),
            sample_count=3,
            measurement_quality=2.3
        )
    ]
    
    # Create analyzer
    analyzer = PhilatelicMLAnalyzer()
    
    # Analyze shade progression
    results = analyzer.analyze_shade_progression(sample_data)
    
    print("LIGHTNESS PROGRESSION (Lightest to Darkest):")
    print("-" * 50)
    for result in results['lightness_progression']:
        print(f"{result.rank_position}. {result.filename}")
        print(f"   Score: {result.rank_score:.1f} | {result.notes} | Confidence: {result.confidence:.2f}")
        print()
    
    print("PROGRESSION ANALYSIS:")
    print("-" * 20)
    analysis = results['progression_analysis']
    print(f"Total lightness range: {analysis['total_range']:.1f}")
    print(f"Average gap between varieties: {analysis['average_gap']:.1f}")
    print(f"Distinct groups identified: {analysis['distinct_groups']}")
    print(f"Progression evenness: {analysis['progression_evenness']:.2f}")
    
    print("\nRECOMMENDED CATALOG ORDERING:")
    print("-" * 30)
    for rec in results['recommended_ordering']:
        print(f"{rec['recommended_position']}. {rec['filename']} - {rec['primary_classification']}")

if __name__ == "__main__":
    demo_ml_ranking()