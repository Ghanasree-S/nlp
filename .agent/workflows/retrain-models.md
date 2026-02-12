---
description: Complete ML Model Retraining Workflow
---

# Complete ML Model Retraining Workflow

## 🎯 Purpose
Retrain all VisualVerse NLP models with proper data splitting, no leakage, and realistic evaluation.

---

## Phase 1: Pre-Training Verification (Day 1 Morning)

### Step 1: Run Leakage Detection Script
```bash
python training/verify_leakage.py
```
**Expected Output:** 
- ✅ No duplicate documents across splits
- ✅ No document-level leakage (same doc in train/val/test)
- ✅ Proper grouping structures verified

**If fails:** Fix issues before proceeding

---

### Step 2: Verify Dataset Preparation
```bash
python training/prepare_datasets.py --verify
```
**Checks:**
- ✅ All datasets downloaded
- ✅ Deduplication completed
- ✅ Domain labels assigned
- ✅ Group IDs created

---

### Step 3: Review Data Statistics
```bash
python training/analyze_datasets.py
```
**Review:**
- Class distributions
- Domain distributions  
- Document length statistics
- Training/validation/test split sizes

---

## Phase 2: Keyphrase Extractor Training (Day 1 Afternoon - Day 2)

### Step 4: Train Keyphrase Extractor (Priority 1)
// turbo
```bash
python training/train_keyphrase.py --use-multi-dataset --use-group-kfold --k=5
```
**Duration:** 2-4 hours  
**Expected Metrics:**
- Precision: 75-85%
- Recall: 70-80%
- F1: 72-82%

---

### Step 5: Validate Keyphrase Results
```bash
python training/validate_model.py --model keyphrase --visualize
```
**Check:**
- Per-fold metrics consistency
- No overfitting (train vs val gap < 10%)
- Confusion analysis

---

## Phase 3: Text Classifier Training (Day 2 Morning)

### Step 6: Fix Text Classifier Dataset
First ensure proper split:
```bash
python training/prepare_text_classifier_data.py --remove-easy-examples --balance-domains
```

### Step 7: Train Text Classifier (Priority 2)
// turbo
```bash
python training/train_text_classifier.py --stratified --k=5
```
**Duration:** 30-60 minutes  
**Expected Metrics:**
- Accuracy: 85-92% (realistic, not 99.9%)
- Per-class F1: > 80%

---

### Step 8: Validate Text Classifier
```bash
python training/validate_model.py --model text_classifier --confusion-matrix
```

---

## Phase 4: Topic Model Training (Day 2 Afternoon)

### Step 9: Prepare Topic Model Data
```bash
python training/prepare_topic_data.py --separate-domains --tune-topics
```

### Step 10: Train Topic Model (Priority 3 - Optional)
// turbo
```bash
python training/train_topic_model.py --num-topics=10 --domain-aware
```
**Duration:** 1-2 hours  
**Expected Metrics:**
- Coherence Score: 0.40-0.60 (realistic)

---

## Phase 5: Final Validation & Documentation (Day 3)

### Step 11: Run Full Pipeline Test
```bash
python training/test_full_pipeline.py --sample-size=100
```
**Verify:**
- All models load correctly
- End-to-end inference works
- No runtime errors

---

### Step 12: Generate Final Report
```bash
python training/generate_training_report.py --output=training_report.html
```
**Includes:**
- All metrics per model
- Training curves
- Validation results
- Sample predictions

---

### Step 13: Update Project Documentation
Manually update:
- `PROJECT_DOCUMENTATION.html` with new metrics
- `README.md` with training details
- Model cards for each trained model

---

## 🚨 Critical Rules

1. **Never skip leakage verification** (Step 1)
2. **Always check train/val gap** - if > 15%, you're overfitting
3. **Save all training logs** - you'll need them for the viva
4. **Don't chase 99% accuracy** - realistic scores are better

---

## 📊 Success Criteria

| Model | Metric | Realistic Target |
|-------|--------|------------------|
| Keyphrase | F1 | 72-82% |
| Text Classifier | Accuracy | 85-92% |
| Topic Model | Coherence | 0.40-0.60 |
| Relation Extractor | F1 | 70-80% (after new data) |

---

## 🎓 For Viva/Interview

When asked "Why did you retrain?":

> "We detected dataset bias and document-level leakage in our initial training. We redesigned our evaluation strategy using GroupKFold to prevent leakage, incorporated multiple datasets to reduce bias, and retrained all models to obtain realistic performance metrics that better represent real-world usage."

---

## Notes
- Relation Extractor (Priority 4) should be trained AFTER getting proper dataset
- Keep all old models in `models/old/` for comparison
- Document all metric changes in `training/CHANGELOG.md`
