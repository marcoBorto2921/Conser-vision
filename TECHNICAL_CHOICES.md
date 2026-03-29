# Technical Choices — Conser-vision: Wildlife Image Classification

> Competition: https://www.drivendata.org/competitions/87/competition-image-classification-wildlife-conservation/

---

## 1. Cross-Validation Strategy

**Choice**: StratifiedKFold with k=5

**Rationale**:
Il dataset presenta class imbalance (le immagini "blank" sono tipicamente la classe più frequente nelle camera trap). StratifiedKFold garantisce che ogni fold abbia la stessa distribuzione di classi del dataset completo, riducendo la varianza nella stima del log-loss locale. k=5 offre un buon trade-off tra bias/varianza della stima e costo computazionale.

**Stratification key**: la classe con probabilità più alta nel vettore one-hot (argmax dei label).

**Alternatives considered**:
- `KFold`: scartato perché non tiene conto dello sbilanciamento → CV score meno correlato con LB
- `k=10`: più accurato ma computazionalmente costoso su GPU; riservo questa scelta per la fase finale

---

## 2. Model Architecture

**Choice**: EfficientNet-B3 (baseline) → ConvNeXt-Base (advanced)

**Rationale**:

| Model | Params | ImageNet Top-1 | Note |
|-------|--------|----------------|------|
| EfficientNet-B3 | 12M | 81.6% | Ottimo rapporto params/accuracy per baseline rapido |
| ConvNeXt-Base | 89M | 83.8% | State-of-the-art su dataset medi, stabile da trainare |
| EfficientNetV2-M | 54M | 85.2% | Possibile upgrade da testare |

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

**Evitato**:
- `RandomErasing`: potrebbe eliminare il solo animale nell'immagine
- `MixUp/CutMix`: merita sperimentazione; potenzialmente utile ma può rendere il training instabile con label one-hot

---

## 5. Optimizer & Scheduler

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

## 8. Gestione Class Imbalance

**Da verificare dopo EDA**. Le possibili strategie sono:
- **Oversampling** delle classi rare nel DataLoader (WeightedRandomSampler)
- **Class weights** nella CrossEntropyLoss (weight=inverse_freq)
- **Nessuna correzione** se la distribuzione del train rispecchia il test → log-loss già tiene conto della calibrazione

---

## 9. Esperimenti Falliti

| Esperimento | Ipotesi | Risultato | Causa probabile |
|-------------|---------|-----------|-----------------|
| — | — | — | Da riempire durante la competizione |

---

## References

- EfficientNet: Tan & Le, 2019 — https://arxiv.org/abs/1905.11946
- ConvNeXt: Liu et al., 2022 — https://arxiv.org/abs/2201.03545
- Label Smoothing: Szegedy et al., 2016 — https://arxiv.org/abs/1512.00567
- AdamW: Loshchilov & Hutter, 2019 — https://arxiv.org/abs/1711.05101
- timm library: https://github.com/huggingface/pytorch-image-models
