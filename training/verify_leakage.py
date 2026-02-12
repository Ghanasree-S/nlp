"""
Data Leakage Verification Script
Checks for common data leakage issues before training ML models.
"""

import os
import json
import pickle
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import hashlib


class LeakageDetector:
    """Detects various types of data leakage in ML datasets."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.issues_found = []
        self.warnings = []
        
    def hash_text(self, text: str) -> str:
        """Create hash of text for duplicate detection."""
        # Normalize: lowercase, strip whitespace
        normalized = " ".join(text.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def check_exact_duplicates(self, splits: Dict[str, List[dict]]) -> bool:
        """Check for exact duplicate texts across train/val/test splits."""
        print("\n🔍 Checking for exact duplicates across splits...")
        
        split_hashes = {}
        all_hashes = set()
        duplicates_found = False
        
        for split_name, data in splits.items():
            hashes = set()
            for item in data:
                text = item.get('text', '') or item.get('sentence', '') or item.get('content', '')
                text_hash = self.hash_text(text)
                hashes.add(text_hash)
                
                if text_hash in all_hashes:
                    duplicates_found = True
                    self.issues_found.append(
                        f"❌ DUPLICATE found: Same text appears in multiple splits"
                    )
                    break
                    
            split_hashes[split_name] = hashes
            all_hashes.update(hashes)
            print(f"  {split_name}: {len(hashes)} unique samples")
        
        # Check for overlap between splits
        splits_list = list(split_hashes.keys())
        for i in range(len(splits_list)):
            for j in range(i + 1, len(splits_list)):
                split_a = splits_list[i]
                split_b = splits_list[j]
                overlap = split_hashes[split_a] & split_hashes[split_b]
                
                if overlap:
                    duplicates_found = True
                    self.issues_found.append(
                        f"❌ LEAKAGE: {len(overlap)} duplicates between {split_a} and {split_b}"
                    )
        
        if not duplicates_found:
            print("  ✅ No exact duplicates found across splits")
        
        return not duplicates_found
    
    def check_document_level_leakage(self, splits: Dict[str, List[dict]]) -> bool:
        """Check if sentences from the same document appear in different splits."""
        print("\n🔍 Checking for document-level leakage...")
        
        doc_to_splits = defaultdict(set)
        leakage_found = False
        
        for split_name, data in splits.items():
            for item in data:
                doc_id = item.get('doc_id') or item.get('document_id') or item.get('source')
                if doc_id:
                    doc_to_splits[doc_id].add(split_name)
        
        # Find documents appearing in multiple splits
        leaked_docs = {doc_id: splits for doc_id, splits in doc_to_splits.items() 
                       if len(splits) > 1}
        
        if leaked_docs:
            leakage_found = True
            print(f"  ❌ LEAKAGE: {len(leaked_docs)} documents span multiple splits")
            for doc_id, splits in list(leaked_docs.items())[:5]:  # Show first 5
                self.issues_found.append(
                    f"❌ Document '{doc_id}' appears in: {', '.join(splits)}"
                )
            if len(leaked_docs) > 5:
                print(f"     ... and {len(leaked_docs) - 5} more")
        else:
            print("  ✅ No document-level leakage found")
        
        return not leakage_found
    
    def check_temporal_leakage(self, splits: Dict[str, List[dict]]) -> bool:
        """Check if test data is older than training data (temporal leakage)."""
        print("\n🔍 Checking for temporal leakage...")
        
        has_timestamps = any(
            item.get('timestamp') or item.get('date') 
            for data in splits.values() 
            for item in data
        )
        
        if not has_timestamps:
            self.warnings.append(
                "⚠️  No timestamps found - cannot verify temporal ordering"
            )
            return True
        
        # Extract date ranges for each split
        split_dates = {}
        for split_name, data in splits.items():
            dates = [
                item.get('timestamp') or item.get('date') 
                for item in data 
                if item.get('timestamp') or item.get('date')
            ]
            if dates:
                split_dates[split_name] = (min(dates), max(dates))
        
        # Check if test dates are after train dates
        if 'train' in split_dates and 'test' in split_dates:
            train_max = split_dates['train'][1]
            test_min = split_dates['test'][0]
            
            if test_min < train_max:
                self.issues_found.append(
                    f"❌ TEMPORAL LEAKAGE: Test data ({test_min}) is older than training data ({train_max})"
                )
                return False
        
        print("  ✅ No temporal leakage found")
        return True
    
    def check_label_leakage(self, splits: Dict[str, List[dict]]) -> bool:
        """Check for features that are too predictive (possible label leakage)."""
        print("\n🔍 Checking for label leakage patterns...")
        
        # This is a simple heuristic check
        for split_name, data in splits.items():
            if split_name == 'train':
                # Check if any feature has perfect correlation with label
                feature_label_pairs = defaultdict(set)
                
                for item in data:
                    label = item.get('label') or item.get('category')
                    text = item.get('text', '')
                    
                    # Check if text starts/ends with label (common mistake)
                    if label and text:
                        if text.lower().startswith(label.lower()):
                            self.issues_found.append(
                                f"❌ LABEL LEAKAGE: Text starts with label '{label}'"
                            )
                            return False
                        
                        # Check for label in text
                        if f"[{label}]" in text or f"Label: {label}" in text:
                            self.issues_found.append(
                                f"❌ LABEL LEAKAGE: Label '{label}' found in text"
                            )
                            return False
        
        print("  ✅ No obvious label leakage patterns found")
        return True
    
    def check_group_structure(self, data_with_groups: List[dict]) -> bool:
        """Verify that group IDs are properly structured for GroupKFold."""
        print("\n🔍 Checking group structure for K-Fold...")
        
        groups = [item.get('group_id') or item.get('doc_id') for item in data_with_groups]
        
        if None in groups:
            self.issues_found.append(
                "❌ Missing group IDs - GroupKFold requires group_id for each sample"
            )
            return False
        
        unique_groups = len(set(groups))
        total_samples = len(groups)
        
        print(f"  Total samples: {total_samples}")
        print(f"  Unique groups: {unique_groups}")
        print(f"  Avg samples per group: {total_samples / unique_groups:.1f}")
        
        if unique_groups < 5:
            self.warnings.append(
                f"⚠️  Only {unique_groups} groups - need at least 5 for 5-fold CV"
            )
            return False
        
        print("  ✅ Group structure is valid for K-Fold")
        return True
    
    def run_all_checks(self, dataset_config: Dict) -> bool:
        """Run all leakage checks on the provided dataset configuration."""
        print("="*60)
        print("🛡️  DATA LEAKAGE VERIFICATION")
        print("="*60)
        
        all_passed = True
        
        # Load splits (this is a placeholder - adapt to your data structure)
        splits = dataset_config.get('splits', {})
        
        if not splits:
            print("❌ No splits found in dataset configuration")
            return False
        
        # Run all checks
        checks = [
            ('Exact Duplicates', lambda: self.check_exact_duplicates(splits)),
            ('Document-Level Leakage', lambda: self.check_document_level_leakage(splits)),
            ('Temporal Leakage', lambda: self.check_temporal_leakage(splits)),
            ('Label Leakage', lambda: self.check_label_leakage(splits)),
        ]
        
        for check_name, check_func in checks:
            try:
                passed = check_func()
                all_passed = all_passed and passed
            except Exception as e:
                print(f"  ❌ Error running {check_name}: {e}")
                all_passed = False
        
        # Check group structure if available
        if 'full_dataset' in dataset_config:
            all_passed = all_passed and self.check_group_structure(dataset_config['full_dataset'])
        
        # Print summary
        print("\n" + "="*60)
        print("📊 VERIFICATION SUMMARY")
        print("="*60)
        
        if self.issues_found:
            print(f"\n❌ {len(self.issues_found)} CRITICAL ISSUES FOUND:\n")
            for issue in self.issues_found:
                print(f"  {issue}")
        
        if self.warnings:
            print(f"\n⚠️  {len(self.warnings)} WARNINGS:\n")
            for warning in self.warnings:
                print(f"  {warning}")
        
        if all_passed and not self.issues_found:
            print("\n✅ ALL CHECKS PASSED - Safe to proceed with training!")
        else:
            print("\n❌ ISSUES DETECTED - Fix these before training!")
        
        print("="*60)
        
        return all_passed


def load_dataset_for_verification(dataset_name: str) -> Dict:
    """
    Load your dataset in a format suitable for verification.
    Adapt this function to your specific data structure.
    """
    # This is a placeholder - implement based on your data format
    # Expected structure:
    # {
    #     'splits': {
    #         'train': [{'text': '...', 'label': '...', 'doc_id': '...'}],
    #         'val': [...],
    #         'test': [...]
    #     },
    #     'full_dataset': [...]  # Optional, for group structure check
    # }
    
    base_path = Path("data") / dataset_name
    
    dataset_config = {
        'splits': {},
        'full_dataset': []
    }
    
    # Try to load common formats
    for split in ['train', 'val', 'test']:
        json_path = base_path / f"{split}.json"
        pkl_path = base_path / f"{split}.pkl"
        
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                dataset_config['splits'][split] = json.load(f)
        elif pkl_path.exists():
            with open(pkl_path, 'rb') as f:
                dataset_config['splits'][split] = pickle.load(f)
    
    return dataset_config


if __name__ == "__main__":
    import sys
    
    # Example usage
    print("\n🔍 Starting Data Leakage Verification...\n")
    
    # List of datasets to check
    datasets_to_check = [
        'keyphrase_data',
        'text_classification_data',
        'topic_model_data',
        'relation_extraction_data'
    ]
    
    all_datasets_passed = True
    
    for dataset_name in datasets_to_check:
        print(f"\n{'='*60}")
        print(f"Checking dataset: {dataset_name}")
        print(f"{'='*60}\n")
        
        try:
            dataset_config = load_dataset_for_verification(dataset_name)
            
            if not dataset_config['splits']:
                print(f"⚠️  Skipping {dataset_name} - no data found")
                continue
            
            detector = LeakageDetector()
            passed = detector.run_all_checks(dataset_config)
            
            if not passed:
                all_datasets_passed = False
                
        except Exception as e:
            print(f"❌ Error checking {dataset_name}: {e}")
            all_datasets_passed = False
    
    print(f"\n\n{'='*60}")
    print("🏁 FINAL VERIFICATION RESULT")
    print(f"{'='*60}\n")
    
    if all_datasets_passed:
        print("✅ ALL DATASETS PASSED - Ready for training!")
        sys.exit(0)
    else:
        print("❌ SOME DATASETS FAILED - Fix issues before training!")
        sys.exit(1)
