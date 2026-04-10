# Technical Choices — Conser-vision: Wildlife Image Classification

> Competition: https://www.drivendata.org/competitions/87/competition-image-classification-wildlife-conservation/

---

## Changelog

### 2026-04-10 — Fixed site leakage in CV split
**Impact**: expected to close most of the 0.62 gap between local val (~0.85) and public LB (1.4722).

**Bug**: the previous `train_test_split(stratify=class)` split ignored the `site` column. 100% of val images came from sites also present in the train fold (median 168 train images per val-image site), while the public test set uses 51 sites completely disjoint from train. Local log-loss measured in-site memorisation, not out-of-site generalisation — exactly the opposite of the LB. Full audit: `reports/diagnostic_audit.md`.

**Fix**: replaced with `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)` using `groups = train_df["site"].values`. When `n_splits=1` in config, the code now takes the first fold of the 5-way split, giving a ~20% site-disjoint hold-out. A runtime `assert len(train_groups & val_groups) == 0` stays in the code as permanent guardrail. Temperature scaling is now automatically fit on site-disjoint logits, because `val_logits`/`val_labels` are saved from the same site-disjoint val fold.

**Expected local metric change**: local val log-loss will rise from ~0.85 to ~1.3–1.5 on first retrain. **This is not a regression** — it is the honest number, and the new local ↔ LB gap should be < 0.10.

**Consequence**: every prior tuning decision (LR, dropout, MixUp probability, early stopping patience, augmentation intensity) was made against a broken metric and must be re-validated before being trusted.

---

## 1. Cross-Validation Strategy

**Choice**: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)` with `groups = train_df["site"].values`.

- `n_splits=1` in `configs/config.yaml` → use only the **first fold** of the 5-way split (≈20% site-disjoint hold-out, fast iteration mode)
- `n_splits=5` → iterate all 5 folds (final ensemble mode)

**Rationale**:
Il public test set usa 51 camera-trap sites che sono **completamente disgiunti** dai 148 sites del train set (verificato in `reports/diagnostic_audit.md` / `reports/check1_data_inventory.txt`). La LB misura quindi generalizzazione out-of-site. Qualsiasi splitter che non raggruppi per `site` produce val folds in cui la rete può sfruttare lo sfondo fisso della camera (stessa vegetazione, stessi animali che passano ripetutamente) — esattamente la scorciatoia che non esiste sul test reale. `StratifiedGroupKFold` è l'unico splitter di sklearn che combina group-disjoint e stratificazione di classe, entrambi necessari qui (il dataset ha `hog` al 5.9%, sotto-rappresentata).

**Stratification key**: la classe con probabilità più alta nel vettore one-hot (argmax dei label).

**Grouping key**: `site` (colonna presente in `train_features.csv`, esempio S0120, S0069, ...).

**Runtime guardrail**: `assert len(set(groups[train_idx]) & set(groups[val_idx])) == 0` viene eseguito ad ogni fold in `src/training/train.py::run_cv`. La linea di log per fold riporta: train rows, val rows, train sites, val sites, per-class distribution sul val fold.

**Alternatives considered**:
- `StratifiedKFold k=5` senza group: **rejected 2026-04-10** — stratifica correttamente le classi ma ignora `site`, quindi condivide sites tra train e val (100% overlap misurato). Causa diretta del gap tra local CV e LB.
- `KFold` (anche con group): scartato perché non tiene conto dello sbilanciamento di classe.
- `train_test_split(stratify=class)`: la scelta originale. **Rejected 2026-04-10**: ignorava `site`, produceva un val set in-site, stima di log-loss ottimistica di ≈0.4–0.6. Vedi changelog.

---

## 2. Model Architecture

**Choice**: EfficientNetV2-M (`tf_efficientnetv2_m`) come modello corrente

**Rationale**:

| Model | Params | ImageNet Top-1 | Stato |
|-------|--------|----------------|-------|
| EfficientNet-B3 | 12M | 81.6% | v1 — baseline iniziale |
| EfficientNetV2-M | 54M | 85.2% | **Corrente** — miglior rapporto accuracy/params tra i testati |
| ConvNeXt-Base | 89M | 83.8% | Pianificato per ensemble avanzato |

EfficientNetV2-M è stato scelto come upgrade naturale da B3: stessa famiglia (compound scaling), training più stabile grazie al training-aware NAS usato in V2, e top-1 significativamente superiore con soli 4× più parametri.

**Head**: BatchNorm → Dropout(0.3) → Linear(num_features, 8)
La BatchNorm sulla feature del backbone normalizza la distribuzione prima del classificatore, riducendo la sensibilità al learning rate.

**Differential learning rate**: backbone usa lr×0.1, head usa lr piena. Il backbone è già pretrained → aggiornamenti piccoli per non distruggere le feature; la head è random → aggiornamenti grandi.

**Alternatives considered**:
- `ResNet50`: più vecchio, peggiori feature per immagini naturali wild
- `ViT-B/16`: eccellente ma richiede più dati o più epoch per convergere; da testare in ensemble
- `CLIP-ViT`: pretrained su image-text pairs, potenzialmente forte su animali rari, ma external data → non permesso

---

## 3. Loss Function

**Choice**: CrossEntropyLoss con label_smoothing=0.1

**Rationale**:
Label smoothing impedisce alla rete di diventare overconfident sulle predizioni (probabilità → 1.0 sulle immagini easy), il che migliora direttamente il log-loss che penalizza pesantemente le predizioni confident ma errate. Smoothing=0.1 è un valore standard empiricamente validato in letteratura (Szegedy et al., 2016).

**Alternatives considered**:
- `FocalLoss`: utile per class imbalance estremo in object detection; meno indicato per classificazione multi-classe bilanciata da CV
- `CrossEntropyLoss` senza smoothing: porta spesso a overconfidence e peggiori log-loss

---

## 4. Augmentation Strategy

**Choice**: ColorJitter + RandomGrayscale + flips + rotation

**Rationale domain-specific**:
- **ColorJitter** (brightness/contrast forti): le camera trap scattano in condizioni di luce molto variabili (pieno giorno, tramonto, notte con IR)
- **RandomGrayscale(p=0.1)**: simula le immagini IR notturne che sono praticamente in bianco/nero
- **HorizontalFlip**: valido, gli animali possono attraversare il frame in entrambe le direzioni
- **VerticalFlip(p=0.2)**: più raro, ma alcune trap sono incassate ad angoli inusuali
- **GaussianBlur**: simula motion blur di animali in movimento
- **RandomResizedCrop(scale=0.6-1.0)**: gestisce animali vicini vs. distanti; scale non scende sotto 0.6 per non ritagliare completamente l'animale

**MixUp (alpha=0.4, p=0.5)**:
Aggiunto nella versione corrente. Mescola coppie di immagini e label (lam ~ Beta(0.4, 0.4)) con probabilità 0.5 per batch. Riduce l'overconfidence del modello durante il training e migliora la generalizzazione. I label one-hot sono già supportati da `nn.CrossEntropyLoss` in PyTorch ≥ 1.10 (interpreta float labels come distribuzioni soft).

**Evitato**:
- `RandomErasing`: potrebbe eliminare il solo animale nell'immagine
- `CutMix`: simile a MixUp ma richiede regioni spaziali coerenti; potenzialmente utile ma più complesso; da testare in futuro

---

## 5. Post-hoc Calibration: Temperature Scaling

**Choice**: Temperature scaling con T ottimizzato su val set

**Rationale**:
Il gap tra local CV (0.85) e LB (1.83) è dovuto principalmente all'overconfidence del modello sul test set. Temperature scaling è la tecnica di calibrazione più semplice ed efficace: un singolo scalare T viene ottimizzato minimizzando la NLL sul validation set.

```
p_calibrated = softmax(logits / T)
```

T > 1 appiattisce la distribuzione (meno confidenza), T < 1 la acuisce. In pratica per reti neurali overconfident T ∈ [1.5, 2.5].

**Implementazione**: `scipy.optimize.minimize_scalar(method='bounded', bounds=(0.5, 5.0))` su `src/evaluation/eval.py`. I logits del validation set vengono salvati nel checkpoint migliore e caricati automaticamente da `predict.py`.

**Alternatives considered**:
- `Platt scaling`: regressione logistica sui logits — più parametri, poco guadagno rispetto a T scaling su 8 classi
- `Isotonic regression / histogram binning`: non parametrici, utili per calibrazione per classe; da valutare se T globale è insufficiente
- `Label smoothing`: già usato (0.1) — riduce l'overconfidence durante il training ma non la elimina

---

## 6. Optimizer & Scheduler

**Choice**: AdamW + CosineAnnealingLR

**Rationale**:
- AdamW separa il weight decay dalla gradient update (Loshchilov & Hutter, 2019), dando migliore regolarizzazione rispetto ad Adam con L2
- CosineAnnealingLR converge in modo smooth senza richiedere tuning manuale degli step; η_min=1e-6 evita learning rate zero

**Alternatives considered**:
- `SGD + momentum`: richiede più tuning, generalmente più lento a convergere su pretrained models
- `OneCycleLR`: ottima per training rapido in pochi epoch; da testare se il budget GPU è limitato

---

## 6. Test Time Augmentation (TTA)

**Choice**: 5 viste = {center crop, random crop} × {originale, horizontal flip} + original val transform

**Rationale**:
TTA riduce la varianza della predizione mediando le probabilità su più viste della stessa immagine. Per camera trap, il flip orizzontale è la trasformazione più naturale. Costo: 5× il tempo di inference, accettabile dato il piccolo test set.

**Formula**: p_final = (1/5) × Σ softmax(f(aug_i(x)))

---

## 7. Ensemble (fase avanzata)

**Choice**: Average delle probabilità OOF tra EfficientNet-B3 e ConvNeXt-Base

**Rationale**:
I due modelli hanno architetture diverse (depthwise conv vs. standard conv) → diversità nelle feature → errori non correlati → il loro average riduce la varianza del classificatore finale.

**Weights**: null (equal weights come baseline); ottimizzare con scipy.optimize.minimize sul log-loss OOF se il guadagno giustifica la complessità.

---

## 9. Gestione Class Imbalance

**Da verificare dopo EDA**. Le possibili strategie sono:
- **Oversampling** delle classi rare nel DataLoader (WeightedRandomSampler)
- **Class weights** nella CrossEntropyLoss (weight=inverse_freq)
- **Nessuna correzione** se la distribuzione del train rispecchia il test → log-loss già tiene conto della calibrazione

---

## 10. Esperimenti Falliti

| Esperimento | Ipotesi | Risultato | Causa probabile |
|-------------|---------|-----------|-----------------|
| — | — | — | Da riempire durante la competizione |

---

## References

- EfficientNet: Tan & Le, 2019 — https://arxiv.org/abs/1905.11946
- EfficientNetV2: Tan & Le, 2021 — https://arxiv.org/abs/2104.00298
- ConvNeXt: Liu et al., 2022 — https://arxiv.org/abs/2201.03545
- Label Smoothing: Szegedy et al., 2016 — https://arxiv.org/abs/1512.00567
- AdamW: Loshchilov & Hutter, 2019 — https://arxiv.org/abs/1711.05101
- Temperature Scaling: Guo et al., 2017 — https://arxiv.org/abs/1706.04599
- MixUp: Zhang et al., 2018 — https://arxiv.org/abs/1710.09412
- timm library: https://github.com/huggingface/pytorch-image-models
