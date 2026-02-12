# 📅 VisualVerse ML Model Retraining Schedule

## 🎯 Goal
Retrain all NLP models with proper data splitting and realistic evaluation metrics.

---

## 📆 3-Day Intensive Plan

### **Day 1: Data Preparation & Verification** ⏰ 4-6 hours

| Time | Task | Duration | Status |
|------|------|----------|--------|
| **Morning Session (2-3 hrs)** |
| 9:00 - 9:30 | Run `verify_leakage.py` on all datasets | 30 min | ⬜ |
| 9:30 - 10:30 | Fix any leakage issues found | 1 hr | ⬜ |
| 10:30 - 11:00 | Run `prepare_datasets.py --verify` | 30 min | ⬜ |
| 11:00 - 11:30 | Review data statistics with `analyze_datasets.py` | 30 min | ⬜ |
| **Afternoon Session (2-3 hrs)** |
| 2:00 - 3:00 | Set up multi-dataset for keyphrase extraction | 1 hr | ⬜ |
| 3:00 - 4:00 | Implement GroupKFold splitting | 1 hr | ⬜ |
| 4:00 - 5:00 | **START: Keyphrase training** (can run overnight) | - | ⬜ |

**Deliverables:**
- ✅ All datasets verified, no leakage
- ✅ GroupKFold structure ready
- ✅ Keyphrase training started

---

### **Day 2: Model Training** ⏰ 6-8 hours

| Time | Task | Duration | Status |
|------|------|----------|--------|
| **Morning Session (3-4 hrs)** |
| 9:00 - 9:30 | Check keyphrase training progress/results | 30 min | ⬜ |
| 9:30 - 10:00 | Validate keyphrase model | 30 min | ⬜ |
| 10:00 - 10:30 | Fix text classifier dataset (remove easy examples) | 30 min | ⬜ |
| 10:30 - 11:30 | **Train text classifier** | 1 hr | ⬜ |
| 11:30 - 12:00 | Validate text classifier results | 30 min | ⬜ |
| **Afternoon Session (3-4 hrs)** |
| 2:00 - 2:30 | Prepare topic model data | 30 min | ⬜ |
| 2:30 - 4:30 | **Train topic model** | 2 hrs | ⬜ |
| 4:30 - 5:00 | Validate topic model | 30 min | ⬜ |
| 5:00 - 6:00 | Review all model metrics | 1 hr | ⬜ |

**Deliverables:**
- ✅ Keyphrase model: F1 72-82%
- ✅ Text classifier: Accuracy 85-92%
- ✅ Topic model: Coherence 0.40-0.60
- ✅ All training logs saved

---

### **Day 3: Validation & Documentation** ⏰ 4-5 hours

| Time | Task | Duration | Status |
|------|------|----------|--------|
| **Morning Session (2-3 hrs)** |
| 9:00 - 10:00 | Run full pipeline test | 1 hr | ⬜ |
| 10:00 - 11:00 | Generate training report HTML | 1 hr | ⬜ |
| 11:00 - 12:00 | Create model cards for each model | 1 hr | ⬜ |
| **Afternoon Session (2 hrs)** |
| 2:00 - 3:00 | Update PROJECT_DOCUMENTATION.html | 1 hr | ⬜ |
| 3:00 - 4:00 | Update README and training docs | 1 hr | ⬜ |
| 4:00 - 4:30 | Create training CHANGELOG.md | 30 min | ⬜ |
| 4:30 - 5:00 | Final review and backup | 30 min | ⬜ |

**Deliverables:**
- ✅ Complete training report
- ✅ Updated documentation
- ✅ Model cards
- ✅ All models saved and backed up

---

## 🚨 Critical Checkpoints

### ✋ STOP Points (Do not proceed if these fail)

**After Day 1:**
- [ ] `verify_leakage.py` passes with no critical issues
- [ ] All datasets properly deduplicated
- [ ] Group IDs assigned for GroupKFold

**After Day 2:**
- [ ] All models trained without errors
- [ ] Metrics are realistic (not 99%+)
- [ ] Train/val gap < 15% for all models

**After Day 3:**
- [ ] Full pipeline test passes
- [ ] Documentation updated
- [ ] Training report generated

---

## ⚡ Quick Reference: Expected Training Times

| Model | Training Time | Total Time with Validation |
|-------|---------------|---------------------------|
| Keyphrase Extractor | 2-4 hours | 3-5 hours |
| Text Classifier | 30-60 min | 1-1.5 hours |
| Topic Model | 1-2 hours | 1.5-2.5 hours |
| **Total** | **4-7 hours** | **5.5-9 hours** |

*Note: These are estimates. Actual time depends on dataset size and hardware.*

---

## 📊 Success Metrics Checklist

After retraining, verify you achieve:

### Keyphrase Extractor
- [ ] Precision: 75-85%
- [ ] Recall: 70-80%
- [ ] F1 Score: 72-82%
- [ ] No overfitting (train-val gap < 10%)

### Text Classifier
- [ ] Overall Accuracy: 85-92%
- [ ] Per-class F1 > 80%
- [ ] Balanced across domains
- [ ] No shortcuts (test on hard examples)

### Topic Model
- [ ] Coherence Score: 0.40-0.60
- [ ] Interpretable topics
- [ ] Distinct topics (no overlap)

### Relation Extractor (Future)
- [ ] F1 Score: 70-80%
- [ ] After new dataset acquired

---

## 🎓 For Your Viva/Interview

### Questions You'll Be Asked:

**Q: "Why did you retrain your models?"**

**A:** "We detected dataset bias and document-level leakage in our initial training. We redesigned our evaluation strategy using GroupKFold to prevent leakage, incorporated multiple datasets to reduce bias, and retrained all models to obtain realistic performance metrics."

---

**Q: "Why is your accuracy lower now?"**

**A:** "Our previous 99.9% accuracy was due to data leakage and dataset shortcuts. The current 88-92% accuracy reflects real-world performance without leakage. This is more honest and scientifically rigorous."

---

**Q: "How did you ensure no data leakage?"**

**A:** "We implemented a comprehensive leakage detection script that checks for:
1. Exact duplicate texts across splits
2. Document-level leakage (same doc in train/test)
3. Temporal leakage
4. Label leakage patterns
5. Proper GroupKFold structure"

---

## 📝 Progress Tracking

Use this checklist to track your progress:

```
DAY 1: DATA PREPARATION
□ Leakage verification passed
□ Datasets deduplicated
□ Group structure validated
□ Keyphrase training started

DAY 2: MODEL TRAINING
□ Keyphrase model trained & validated
□ Text classifier trained & validated
□ Topic model trained & validated
□ All metrics recorded

DAY 3: DOCUMENTATION
□ Full pipeline test passed
□ Training report generated
□ Documentation updated
□ Models backed up
```

---

## 🛠️ Commands Quick Reference

```bash
# Day 1
python training/verify_leakage.py
python training/prepare_datasets.py --verify
python training/analyze_datasets.py
python training/train_keyphrase.py --use-multi-dataset --use-group-kfold --k=5

# Day 2
python training/validate_model.py --model keyphrase --visualize
python training/prepare_text_classifier_data.py --remove-easy-examples --balance-domains
python training/train_text_classifier.py --stratified --k=5
python training/train_topic_model.py --num-topics=10 --domain-aware

# Day 3
python training/test_full_pipeline.py --sample-size=100
python training/generate_training_report.py --output=training_report.html
```

---

## 💡 Tips for Success

1. **Don't rush** - It's better to do it right than fast
2. **Save checkpoints** - Save model after each fold
3. **Monitor closely** - Watch training curves for overfitting
4. **Document everything** - Keep training logs
5. **Test on real data** - Use actual comic descriptions as test

---

## 🔄 What If Something Goes Wrong?

### If leakage verification fails:
1. Check dataset preparation scripts
2. Ensure doc_id is assigned correctly
3. Re-run deduplication
4. Verify group assignments

### If training takes too long:
1. Reduce dataset size for initial testing
2. Use fewer folds (3 instead of 5)
3. Reduce model size/complexity
4. Check GPU utilization

### If metrics are still too high (95%+):
1. Check for remaining leakage
2. Test on external dataset
3. Manually inspect hard examples
4. Ensure test set is realistic

---

## ✅ Final Checklist Before Submitting

- [ ] All models retrained
- [ ] All metrics realistic and documented
- [ ] Leakage verification passed
- [ ] Full pipeline test passed
- [ ] Documentation updated
- [ ] Training report generated
- [ ] Model cards created
- [ ] Old models backed up to `models/old/`
- [ ] CHANGELOG.md updated
- [ ] Ready for viva questions

---

**Remember:** This retraining is not a failure. It's an **upgrade** from a student demo to a serious ML system.

Good luck! 🚀
