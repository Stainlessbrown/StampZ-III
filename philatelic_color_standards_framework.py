#!/usr/bin/env python3
"""
Philatelic Color Standards Framework for StampZ
Establishing objective, reproducible color standards for stamp analysis.

This framework addresses the challenge of inconsistent color terminology
across major catalogs (Scott, Stanley Gibbons, Michel, Yvert & Tellier)
by creating precise, measurable color definitions.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime

class CatalogPublisher(Enum):
    SCOTT = "scott"
    STANLEY_GIBBONS = "sg"
    MICHEL = "michel"
    YVERT_TELLIER = "yt"
    UNKNOWN = "unknown"

@dataclass
class ColorMeasurement:
    """Precise color measurement with metadata."""
    lab_values: Tuple[float, float, float]  # L*a*b* coordinates
    delta_e_tolerance: float                # Acceptable variation (ΔE)
    sample_count: int                       # Number of measurements averaged
    confidence: float                       # Measurement confidence (0-1)
    measurement_date: str
    equipment_info: str = "StampZ-III"
    notes: Optional[str] = None

@dataclass
class CatalogColorDefinition:
    """Color definition from a specific catalog."""
    catalog: CatalogPublisher
    color_term: str                         # e.g., "rose", "deep blue"
    catalog_number: str                     # e.g., "123", "SG45"
    issue_description: str                  # e.g., "1920 King George V"
    measurement: ColorMeasurement
    alternative_terms: List[str] = field(default_factory=list)  # Synonyms used

@dataclass
class StandardizedColorDefinition:
    """Objective color standard combining multiple catalog sources."""
    standard_name: str                      # e.g., "Philatelic Rose Type 1"
    primary_lab: Tuple[float, float, float] # Consensus L*a*b* values
    tolerance_range: float                  # ΔE range for this standard
    source_definitions: List[CatalogColorDefinition] = field(default_factory=list)
    sample_size: int = 0                    # Total measurements used
    consensus_confidence: float = 0.0       # Agreement between sources
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())

class PhilatelicColorStandardsDB:
    """Database for managing objective color standards."""
    
    def __init__(self):
        self.catalog_definitions: Dict[str, CatalogColorDefinition] = {}
        self.standardized_colors: Dict[str, StandardizedColorDefinition] = {}
        self.color_families: Dict[str, List[str]] = {}  # e.g., "red" -> ["rose", "carmine", "scarlet"]
        
    def add_catalog_definition(self, definition: CatalogColorDefinition):
        """Add a color definition from a specific catalog."""
        key = f"{definition.catalog.value}_{definition.catalog_number}_{definition.color_term}"
        self.catalog_definitions[key] = definition
        
    def create_standardized_color(self, 
                                color_name: str,
                                source_definitions: List[CatalogColorDefinition]) -> StandardizedColorDefinition:
        """Create standardized color from multiple catalog sources."""
        
        if not source_definitions:
            raise ValueError("At least one source definition required")
        
        # Calculate consensus L*a*b* values
        lab_measurements = [defn.measurement.lab_values for defn in source_definitions]
        weights = [defn.measurement.confidence for defn in source_definitions]
        
        consensus_lab = self._calculate_weighted_average(lab_measurements, weights)
        
        # Calculate tolerance based on variation between sources
        tolerance = self._calculate_consensus_tolerance(lab_measurements)
        
        # Assess consensus confidence
        confidence = self._assess_consensus_confidence(lab_measurements, weights)
        
        standard = StandardizedColorDefinition(
            standard_name=color_name,
            primary_lab=consensus_lab,
            tolerance_range=tolerance,
            source_definitions=source_definitions,
            sample_size=sum(defn.measurement.sample_count for defn in source_definitions),
            consensus_confidence=confidence
        )
        
        self.standardized_colors[color_name] = standard
        return standard
    
    def find_color_matches(self, 
                          lab_measurement: Tuple[float, float, float], 
                          max_delta_e: float = 5.0) -> List[Dict]:
        """Find standardized colors matching a given measurement."""
        
        matches = []
        for name, standard in self.standardized_colors.items():
            delta_e = self._calculate_delta_e(lab_measurement, standard.primary_lab)
            
            if delta_e <= max_delta_e:
                matches.append({
                    'standard_name': name,
                    'delta_e': delta_e,
                    'confidence': standard.consensus_confidence,
                    'primary_lab': standard.primary_lab,
                    'catalog_sources': [defn.catalog.value for defn in standard.source_definitions]
                })
        
        # Sort by delta E (best matches first)
        matches.sort(key=lambda x: x['delta_e'])
        return matches
    
    def analyze_catalog_disagreements(self, color_family: str = None) -> Dict:
        """Analyze disagreements between catalogs on color terminology."""
        
        disagreements = []
        
        # Group definitions by similar color terms
        color_groups = self._group_similar_color_terms(color_family)
        
        for group_name, definitions in color_groups.items():
            if len(definitions) < 2:
                continue
                
            # Calculate variation within this color group
            lab_values = [defn.measurement.lab_values for defn in definitions]
            variation = self._calculate_color_variation(lab_values)
            
            # Identify specific catalog disagreements
            catalog_differences = []
            for i, defn1 in enumerate(definitions):
                for j, defn2 in enumerate(definitions[i+1:], i+1):
                    delta_e = self._calculate_delta_e(
                        defn1.measurement.lab_values,
                        defn2.measurement.lab_values
                    )
                    
                    if delta_e > 3.0:  # Significant perceptual difference
                        catalog_differences.append({
                            'catalog1': defn1.catalog.value,
                            'catalog2': defn2.catalog.value,
                            'term1': defn1.color_term,
                            'term2': defn2.color_term,
                            'delta_e': delta_e,
                            'lab1': defn1.measurement.lab_values,
                            'lab2': defn2.measurement.lab_values
                        })
            
            if catalog_differences:
                disagreements.append({
                    'color_group': group_name,
                    'variation_range': variation,
                    'catalog_differences': catalog_differences
                })
        
        return {
            'total_disagreements': len(disagreements),
            'disagreement_details': disagreements,
            'analysis_date': datetime.now().isoformat()
        }
    
    def _calculate_weighted_average(self, 
                                  lab_measurements: List[Tuple[float, float, float]], 
                                  weights: List[float]) -> Tuple[float, float, float]:
        """Calculate weighted average of L*a*b* measurements."""
        
        if len(lab_measurements) != len(weights):
            weights = [1.0] * len(lab_measurements)  # Equal weights if mismatch
        
        total_weight = sum(weights)
        if total_weight == 0:
            total_weight = 1.0
            
        weighted_l = sum(lab[0] * weight for lab, weight in zip(lab_measurements, weights)) / total_weight
        weighted_a = sum(lab[1] * weight for lab, weight in zip(lab_measurements, weights)) / total_weight
        weighted_b = sum(lab[2] * weight for lab, weight in zip(lab_measurements, weights)) / total_weight
        
        return (weighted_l, weighted_a, weighted_b)
    
    def _calculate_consensus_tolerance(self, lab_measurements: List[Tuple[float, float, float]]) -> float:
        """Calculate appropriate tolerance range based on measurement variation."""
        
        if len(lab_measurements) < 2:
            return 2.0  # Default tolerance
        
        # Calculate pairwise ΔE values
        delta_e_values = []
        for i in range(len(lab_measurements)):
            for j in range(i+1, len(lab_measurements)):
                delta_e = self._calculate_delta_e(lab_measurements[i], lab_measurements[j])
                delta_e_values.append(delta_e)
        
        # Use 95th percentile as tolerance to capture most variation
        return np.percentile(delta_e_values, 95) if delta_e_values else 2.0
    
    def _assess_consensus_confidence(self, 
                                   lab_measurements: List[Tuple[float, float, float]], 
                                   weights: List[float]) -> float:
        """Assess confidence in consensus based on agreement between sources."""
        
        if len(lab_measurements) < 2:
            return max(weights) if weights else 0.5
        
        # Calculate average pairwise agreement
        agreements = []
        for i in range(len(lab_measurements)):
            for j in range(i+1, len(lab_measurements)):
                delta_e = self._calculate_delta_e(lab_measurements[i], lab_measurements[j])
                # Convert ΔE to agreement score (closer = higher agreement)
                agreement = max(0.0, 1.0 - delta_e / 10.0)
                agreements.append(agreement)
        
        avg_agreement = np.mean(agreements) if agreements else 0.0
        
        # Weight by individual measurement confidence
        weighted_confidence = np.mean(weights) if weights else 0.5
        
        return (avg_agreement + weighted_confidence) / 2.0
    
    def _group_similar_color_terms(self, color_family: str = None) -> Dict[str, List[CatalogColorDefinition]]:
        """Group catalog definitions by similar color terminology."""
        
        groups = {}
        
        for defn in self.catalog_definitions.values():
            if color_family and color_family.lower() not in defn.color_term.lower():
                continue
            
            # Normalize color term for grouping
            normalized_term = self._normalize_color_term(defn.color_term)
            
            if normalized_term not in groups:
                groups[normalized_term] = []
            
            groups[normalized_term].append(defn)
        
        return groups
    
    def _normalize_color_term(self, term: str) -> str:
        """Normalize color terms for comparison."""
        # Remove modifiers and standardize
        normalized = term.lower().strip()
        
        # Remove common modifiers
        modifiers = ['deep', 'light', 'pale', 'dark', 'bright', 'dull', 'vivid']
        for modifier in modifiers:
            normalized = normalized.replace(modifier, '').strip()
        
        return normalized
    
    def _calculate_color_variation(self, lab_measurements: List[Tuple[float, float, float]]) -> float:
        """Calculate overall color variation within a group."""
        
        if len(lab_measurements) < 2:
            return 0.0
        
        # Calculate maximum ΔE within the group
        max_delta_e = 0.0
        for i in range(len(lab_measurements)):
            for j in range(i+1, len(lab_measurements)):
                delta_e = self._calculate_delta_e(lab_measurements[i], lab_measurements[j])
                max_delta_e = max(max_delta_e, delta_e)
        
        return max_delta_e
    
    def _calculate_delta_e(self, lab1: Tuple[float, float, float], lab2: Tuple[float, float, float]) -> float:
        """Calculate ΔE CIE76 between two L*a*b* colors."""
        dl = lab1[0] - lab2[0]
        da = lab1[1] - lab2[1]
        db = lab1[2] - lab2[2]
        
        return np.sqrt(dl**2 + da**2 + db**2)
    
    def export_standards(self, filename: str):
        """Export color standards to JSON file."""
        
        export_data = {
            'created_date': datetime.now().isoformat(),
            'standards_count': len(self.standardized_colors),
            'catalog_definitions_count': len(self.catalog_definitions),
            'standards': {}
        }
        
        for name, standard in self.standardized_colors.items():
            export_data['standards'][name] = {
                'primary_lab': standard.primary_lab,
                'tolerance_range': standard.tolerance_range,
                'sample_size': standard.sample_size,
                'consensus_confidence': standard.consensus_confidence,
                'source_catalogs': list(set(defn.catalog.value for defn in standard.source_definitions)),
                'created_date': standard.created_date
            }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)

def demo_standards_framework():
    """Demonstrate the color standards framework."""
    print("=== Philatelic Color Standards Framework Demo ===\n")
    
    # Create standards database
    standards_db = PhilatelicColorStandardsDB()
    
    # Add catalog definitions for "rose" color from different catalogs
    scott_rose = CatalogColorDefinition(
        catalog=CatalogPublisher.SCOTT,
        color_term="rose",
        catalog_number="123",
        issue_description="1920 King George V",
        measurement=ColorMeasurement(
            lab_values=(65.2, 24.8, 7.9),
            delta_e_tolerance=2.1,
            sample_count=15,
            confidence=0.85,
            measurement_date="2024-01-15",
            notes="Measured from VF examples"
        )
    )
    
    sg_rose = CatalogColorDefinition(
        catalog=CatalogPublisher.STANLEY_GIBBONS,
        color_term="deep rose",
        catalog_number="SG45",
        issue_description="1920 King George V",
        measurement=ColorMeasurement(
            lab_values=(58.1, 28.3, 12.4),
            delta_e_tolerance=2.5,
            sample_count=12,
            confidence=0.78,
            measurement_date="2024-01-16",
            notes="SG color classification"
        )
    )
    
    michel_rosa = CatalogColorDefinition(
        catalog=CatalogPublisher.MICHEL,
        color_term="rosa",
        catalog_number="M67",
        issue_description="1920 König Georg V",
        measurement=ColorMeasurement(
            lab_values=(62.7, 22.1, 15.2),
            delta_e_tolerance=2.8,
            sample_count=10,
            confidence=0.72,
            measurement_date="2024-01-17",
            notes="Michel Katalog definition"
        )
    )
    
    # Add definitions to database
    standards_db.add_catalog_definition(scott_rose)
    standards_db.add_catalog_definition(sg_rose)
    standards_db.add_catalog_definition(michel_rosa)
    
    # Create standardized color
    rose_standard = standards_db.create_standardized_color(
        "Philatelic Rose Standard 1920",
        [scott_rose, sg_rose, michel_rosa]
    )
    
    print("STANDARDIZED COLOR CREATED:")
    print("-" * 30)
    print(f"Name: {rose_standard.standard_name}")
    print(f"Standard L*a*b*: {rose_standard.primary_lab}")
    print(f"Tolerance Range: ΔE ≤ {rose_standard.tolerance_range:.1f}")
    print(f"Consensus Confidence: {rose_standard.consensus_confidence:.2f}")
    print(f"Total Sample Size: {rose_standard.sample_size}")
    print(f"Source Catalogs: {[defn.catalog.value for defn in rose_standard.source_definitions]}")
    
    print("\nCATALOG DISAGREEMENT ANALYSIS:")
    print("-" * 30)
    disagreements = standards_db.analyze_catalog_disagreements()
    
    for disagreement in disagreements['disagreement_details']:
        print(f"\nColor Group: {disagreement['color_group']}")
        print(f"Variation Range: ΔE {disagreement['variation_range']:.1f}")
        
        for diff in disagreement['catalog_differences']:
            print(f"  {diff['catalog1']} '{diff['term1']}' vs {diff['catalog2']} '{diff['term2']}':")
            print(f"    ΔE = {diff['delta_e']:.1f} (perceptually {'different' if diff['delta_e'] > 3 else 'similar'})")
    
    # Test color matching
    print("\nCOLOR MATCHING TEST:")
    print("-" * 20)
    test_color = (60.5, 26.0, 10.2)  # Unknown rose sample
    matches = standards_db.find_color_matches(test_color)
    
    print(f"Test Color L*a*b*: {test_color}")
    print("Matches found:")
    for match in matches:
        print(f"  {match['standard_name']}: ΔE = {match['delta_e']:.1f}, Confidence = {match['confidence']:.2f}")
    
if __name__ == "__main__":
    demo_standards_framework()