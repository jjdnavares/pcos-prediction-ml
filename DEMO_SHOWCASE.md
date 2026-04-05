# PCOS Prediction API -- Demo Showcase

Live demonstrations of the PCOS Prediction API in action.

---

## Swagger UI & Prediction Demo

The API provides an interactive Swagger UI at `/docs` where community health workers can test predictions directly in the browser.

### API Overview

![Swagger UI Overview](https://github.com/user-attachments/assets/c8688f80-0aea-4ecc-8d1b-ce7d917c24e9)

> The Swagger UI exposes all endpoints: health checks, single prediction, and batch prediction.

### High-Risk Prediction

![High-Risk Prediction](https://github.com/user-attachments/assets/e9e8fff4-d35f-46ec-9674-4d7818a29d43)

> A 28-year-old patient with irregular cycles, elevated follicle counts, and multiple symptoms. The model predicts PCOS with ~91% probability and recommends **immediate specialist referral**.

<details>
<summary>Request payload used</summary>

```json
{
  "age": 28,
  "weight": 65.0,
  "waist": 34,
  "marriage_status": 3,
  "tsh": 2.5,
  "follicle_no_l": 13,
  "follicle_no_r": 14,
  "skin_darkening": 1,
  "hair_growth": 1,
  "weight_gain": 1,
  "hair_loss": 0,
  "fast_food": 1,
  "cycle_regularity": 5,
  "cycle_length": 5
}
```

</details>

### Low-Risk Prediction

![Low-Risk Prediction](https://github.com/user-attachments/assets/bdeb08ab-12fa-4e8b-9c56-66ef948648ea)

> A 25-year-old patient with regular cycles, normal follicle counts, and no symptoms. The model predicts no PCOS and recommends **routine annual screening**.

<details>
<summary>Request payload used</summary>

```json
{
  "age": 25,
  "weight": 55.0,
  "waist": 30,
  "marriage_status": 0,
  "tsh": 3.0,
  "follicle_no_l": 4,
  "follicle_no_r": 5,
  "skin_darkening": 0,
  "hair_growth": 0,
  "weight_gain": 0,
  "hair_loss": 0,
  "fast_food": 0,
  "cycle_regularity": 2,
  "cycle_length": 4
}
```

</details>

---

## See Also

- [README](../../README.md) -- Project Overview
