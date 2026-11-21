# Ship a Model: From a Trained PyTorch Network to a Monitored Service

Take a delivery-delay model from training to a service you can trust. Train it so two runs give identical weights, put the preprocessing inside the model, and reject bad requests before they reach it. Save, reload and export it to TorchScript, proving each time that the predictions did not change. Batch requests, shrink the weights to 8 bits and measure the cost. Then watch winter traffic drift away from the training data and decide from a shadow run whether a retrained model should replace the one in production.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** set_seed
- [x] **2.** make_loader
- [x] **3.** DeliveryNet
- [x] **4.** train_model
- [ ] **5.** predict_proba
- [ ] **6.** validate_request
- [ ] **7.** save_artifact
- [ ] **8.** load_artifact
- [ ] **9.** export_model
- [ ] **10.** batch_predict
- [ ] **11.** quantize_weights
- [ ] **12.** drift_report
- [ ] **13.** shadow_compare

---

Built on Deep-ML.
