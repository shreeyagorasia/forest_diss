# Forest LiDAR & Inventory – Key Terms Cheat Sheet
*Sitka Spruce (Picea sitchensis) — Aberfoyle Forest*

---

## 1. LiDAR Acquisition

Active remote sensing system emitting laser pulses and measuring return time to estimate 3D positions. Data processed to CSV by Forest Research; Top Height derived from elevation percentiles.

| Term | Meaning |
|---|---|
| ALS | Airborne Laser Scanning — LiDAR from aircraft. Main technology for large-scale forest inventory. |
| Pulse | Laser beam emitted by the sensor. |
| First return | Usually canopy top. |
| Intermediate returns | Branches / understorey. |
| Last return | Usually ground (not always). |
| Intensity | Strength of returned signal. Mainly useful for visualisation, not analysis. |
| PRF | Pulse Repetition Frequency — pulses per second (Hz). |
| Point density | National surveys: 1–4 pts/m². Local: 10–30 pts/m². Drone: 100+ pts/m². Higher = better canopy penetration. |
| AOI | Area of Interest label — parsed from CSV filename prefix. |
| LiDAR_year | Year of the LiDAR acquisition flight. |
| identification | Unique grid cell identifier (computed). |

---

## 2. Terrain & Canopy Models

| Acronym | Full name | Definition |
|---|---|---|
| DTM / DEM | Digital Terrain Model | Raster of bare ground elevation. |
| DSM | Digital Surface Model | Raster of highest surface (trees, buildings). |
| CHM | Canopy Height Model | `CHM = DSM − DTM` |
| nDSM | Normalised DSM | Point cloud heights referenced to ground level, not sea level. |

---

## 3. Forest Structure Metrics

### 3.1 Top Height (TH)

Average height of the 100 largest-diameter trees per hectare. The primary LiDAR-derived variable — used as the response variable in the CR growth model.

For Sitka spruce, two estimates are derived from LiDAR percentiles:

```
Top_Height95 = elev_percentile_95th × 1.1
```
> Correction factor of 1.1 compensates for known underestimation of top height at P95 due to dense Sitka foliage.

```
Top_Height99 = elev_percentile_99th
```
> No correction applied — P99 is closer to the actual canopy top.

| Term | Definition |
|---|---|
| P95 | 95th percentile of LiDAR return heights (m above ground). |
| P99 | 99th percentile of LiDAR return heights (m above ground). |
| DBH | Diameter at Breast Height — stem diameter at 1.3m above ground. |
| Mean DBH | Average DBH across all trees in a stand. |
| Basal Area (BA) | Total cross-sectional stem area at breast height. Units: m²/ha. Indicator of stand density, biomass, timber stocking. |
| TPH | Trees per Hectare — typically DBH > 7cm and height > 3m. |

---

### 3.2 Canopy Cover (Fractional Cover)

Proportion of ground covered by tree canopy (0 = no canopy, 1 = complete cover). Derived from LiDAR point cloud. Used in volume calculations to account for stocking density.

> CanopyCover is a **downstream consequence** of the growth process — do not use as a predictor of top height or CR residuals (circular reasoning).

---

### 3.3 Gap Fraction

```
GapFraction = ground returns / total returns
```

Ratio of ground returns to total returns. Values near 0 indicate dense cover; values near 1 indicate exposed ground. It is related to CanopyCover but is not necessarily its exact complement.

---

### 3.4 Leaf Area Index (LAI)

Total one-sided leaf area per unit ground area. Derived from LiDAR. Higher LAI = denser canopy, greater photosynthetic capacity.

> LAI, CanopyCover, and GapFraction are all **downstream consequences** of the growth process — do not use as predictors of top height or CR residuals (circular).

---

### 3.5 Stand Age

```
Age = LiDAR_year − plyr
```

Stand age at time of LiDAR acquisition (years).

> ⚠️ Data quality notes:
> - `plyr = 0` → Age = LiDAR_year (nonsensical — flag as unknown_age)
> - Negative Age (`plyr > LiDAR_year`) → likely a newer forestry record overwriting an older scan row — flag and retain, do not delete
> - Records outside 0–150 years filtered as likely errors before writing

---

## 4. Volume Models (Sitka Spruce)

Stand volume in m³. Three formulations available.

### 4.1 Thinning Flag

```
Thin = ifelse(last_thinn ≠ 0, 1, 0)
```

Binary flag: 1 = stand has been thinned at least once; 0 = never thinned. Determines which g1/g3 coefficients to apply.

---

### 4.2 Species/Thin-Specific Volume (g1, g3 model)

Sitka spruce coefficients:

| Condition | g1 | g3 |
|---|---|---|
| Thinned (Thin = 1) | 3.509144 | 1.478136 |
| Unthinned (Thin = 0) | 2.922183 | 1.694732 |

```
Vol95 = (g1 × Top_Height95^g3) × (area / 10000) × CanopyCover
Vol99 = (g1 × Top_Height99^g3) × (area / 10000) × CanopyCover
```

> Units: m³. Scaled by stand area (ha) and CanopyCover to account for stocking density.

---

### 4.3 Robertson & Miller Generic Volume Equation (Sitka spruce)

Does not depend on thinning flag:

```
Vol_RM95 = (−35.8733 + 5.3486 × Top_Height95^1.5424) × (area / 10000) × CanopyCover
```

> Source: Robertson & Miller equation for Sitka spruce. Units: m³. Use as cross-check against the g1/g3 model.

---

## 5. Yield Class & Productivity

### 5.1 General Yield Class (GYC)

British productivity index. Represents **maximum mean annual timber volume increment (m³/ha/year)**. Higher GYC = more productive site.

Typical Sitka spruce range at Aberfoyle: **GYC 8–20**. Corresponding implied y_max for CR model: ~28–50m.

---

### 5.2 GYC from Chapman-Richards Site Index Model

GYC is derived by inverting the CR site index model using species-specific parameters (p1–p5):

```
GYCspec95 = (Top_Height95 / (1 − exp(−p4 × Age))^p5 − p1 − p3 × 2) / p2
GYCspec99 = (Top_Height99 / (1 − exp(−p4 × Age))^p5 − p1 − p3 × 2) / p2
```

Parameters p1–p5 are species-specific, sourced from: *'Use of Airborne Laser Scanning (ALS) for Forest Inventory'* (Forest Research PDF).

| Field | Role in GYC formula |
|---|---|
| p1 | CR site index intercept |
| p2 | GYC scaling coefficient |
| p3 | Age correction term |
| p4 | CR growth rate parameter (k equivalent) |
| p5 | CR shape parameter (p equivalent) |

---

### 5.3 Site Index (SI)

Height of dominant trees at a reference age (typically **age 50** for UK conifers). Internationally comparable alternative to GYC. Higher SI = better growing site.

> Site Index and the Chapman–Richards asymptote `y_max` are related measures of site productivity, but they are not identical. SI is height at a fixed reference age (50 years for Sitka spruce), whereas `y_max` is the modelled asymptotic height.

---

---

## 5. Forest Research Hybrid Method — Sitka Spruce Formula Reference

The Forest Research **Hybrid Method** uses LiDAR to estimate **Top Height directly**. Other inventory variables are then derived from Top Height together with species, age and management information. Therefore, DBH, basal area, volume, GYC, Site Index and tree number are **model-derived estimates**, not direct LiDAR measurements.

### 5.1 Top Height — directly estimated from LiDAR

Top Height is the average height of the **100 largest-diameter trees per hectare**. For a 30 × 30 m field plot, this is approximately equivalent to measuring the 9 largest-diameter trees.

For Sitka spruce:

```text
Top Height = P95 × 1.10
```

where:

- `P95` = 95th percentile of LiDAR heights above the ground;
- `1.10` = empirical correction for the tendency of LiDAR to miss the true treetop and underestimate Sitka spruce height.

Important:

- The manual states that this Top Height relationship has been tested for **Sitka spruce**.
- The correction is particularly relevant at low point densities because laser returns may miss the narrow treetop.
- The manual suggests that no correction may be needed for very dense data, such as approximately 25 points/m², but this should be validated for the dataset being used.
- `Top_Height99 = P99` exists in the supplied dataset, but it is **not the validated Sitka spruce equation given in the manual**. Treat it as an alternative metric rather than the official Hybrid Method estimate.

### 5.2 General Yield Class (GYC)

GYC is the maximum mean annual increment of cumulative timber volume, expressed in:

```text
m³ ha⁻¹ year⁻¹
```

Forest yield tables normally report GYC in increments of 2, such as YC14, YC16 and YC18. The equation below produces a **continuous** estimate.

General formula:

```text
GYC = [TH / (1 − exp(−p4 × Age))^p5 − p1 − (2 × p3)] / p2
```

Sitka spruce coefficients:

| Parameter | Value |
|---|---:|
| `p1` | 14.420680 |
| `p2` | 1.473882 |
| `p3` | −1.142290 |
| `p4` | 0.035067 |
| `p5` | 1.923684 |

Therefore, for Sitka spruce:

```text
GYC_SS =
[TH / (1 − exp(−0.035067 × Age))^1.923684
 − 14.420680
 − 2(−1.142290)]
/ 1.473882
```

Equivalent simplified form:

```text
GYC_SS =
[TH / (1 − exp(−0.035067 × Age))^1.923684
 − 12.136100]
/ 1.473882
```

where:

- `TH` = Sitka spruce Top Height in metres;
- `Age = LiDAR_year − planting_year`.

Interpretation: GYC estimates site productivity from the height achieved at a given age. A taller stand at the same age receives a higher estimated GYC.

### 5.3 GYC from Site Index

The manual also gives a conversion from Site Index to GYC:

```text
GYC = a1 + a2 × SI50
```

For Sitka spruce:

```text
GYC = −7.59718 + 0.95728 × SI50
```

where `SI50` is Site Index at age 50.

### 5.4 Site Index (SI)

Site Index is the expected dominant/top height at a reference age. The reference age for Sitka spruce is **50 years**.

The manual provides two routes.

#### Route A — convert GYC to SI50

```text
SI50 = −a1 + GYC / a2
```

For Sitka spruce:

```text
SI50 = 13.7952 + GYC / 1.21946
```

This is the equation printed in the manual. However, it does **not appear to be the exact algebraic inverse** of the separate Sitka `GYC = −7.59718 + 0.95728 × SI50` equation. These coefficient sets should therefore be checked against the source spreadsheet/code before treating the two routes as interchangeable.

#### Route B — calculate SI directly from Top Height and age

This direct equation is given specifically for Sitka spruce:

```text
SI50 =
0.621148 × TH
────────────────────────────────────────
(1 − exp(−0.025461 × Age))^1.449701
```

This is useful because it estimates Site Index directly from LiDAR-derived Top Height and stand age.

### 5.5 Mean DBH

Mean DBH is the average stem diameter measured at **1.3 m above ground**. It is required for estimating the number of trees per hectare.

General formula:

```text
MeanDBH = (b1 + b2 × Spacing) × TH^b4
```

Sitka spruce coefficients:

| Management | `b1` | `b2` | `b4` |
|---|---:|---:|---:|
| Thinned | 0.714157 | 0.239399 | 1.079545 |
| Unthinned | 1.212457 | 0.609335 | 0.754978 |

Therefore:

```text
MeanDBH_thinned =
(0.714157 + 0.239399 × Spacing)
× TH^1.079545
```

```text
MeanDBH_unthinned =
(1.212457 + 0.609335 × Spacing)
× TH^0.754978
```

where:

- `MeanDBH` is in cm;
- `Spacing` is the initial planting spacing recorded in the inventory;
- `TH` is Top Height in m.

The coefficients differ by thinning status because thinning changes competition and allows the remaining trees to increase diameter growth.

### 5.6 Basal Area (BA)

Basal area is the combined cross-sectional stem area at breast height and is expressed in:

```text
m² ha⁻¹
```

For one tree:

```text
BA_tree = π × (DBH / 2)²
```

If DBH is in centimetres, divide the result by `10,000` to convert from cm² to m².

The Hybrid Method estimates stand basal area directly from Top Height:

```text
BA = h1 × TH^h4
```

Sitka spruce coefficients:

| Management | `h1` | `h4` |
|---|---:|---:|
| Thinned | 7.499437 | 0.497925 |
| Unthinned | 5.517392 | 0.771801 |

Therefore:

```text
BA_thinned = 7.499437 × TH^0.497925
```

```text
BA_unthinned = 5.517392 × TH^0.771801
```

These equations estimate basal area per hectare. For a partially forested grid cell or polygon, the project dataset may additionally scale the estimate by polygon area and Canopy Cover.

### 5.7 Stand Volume

The manual expresses stand volume as:

```text
m³ over bark ha⁻¹
```

General Hybrid Method equation:

```text
Volume_ha = g1 × TH^g3
```

Sitka spruce coefficients:

| Management | `g1` | `g3` |
|---|---:|---:|
| Thinned | 3.509144 | 1.478136 |
| Unthinned | 2.922183 | 1.694732 |

Therefore:

```text
Volume_ha_thinned =
3.509144 × TH^1.478136
```

```text
Volume_ha_unthinned =
2.922183 × TH^1.694732
```

The dataset then converts the per-hectare estimate to the timber volume represented by a polygon/grid cell:

```text
Volume_polygon =
Volume_ha
× (area_m² / 10,000)
× CanopyCover
```

This distinction is important:

- `g1 × TH^g3` gives the modelled volume **per hectare**;
- multiplying by `area / 10,000` converts polygon area from m² to hectares;
- multiplying by `CanopyCover` adjusts for the proportion of the polygon occupied by canopy.

### 5.8 Robertson–Miller Sitka spruce volume equation

An alternative Sitka spruce equation in the dataset is:

```text
Volume_RM_ha =
−35.8733 + 5.3486 × TH^1.5424
```

For a polygon:

```text
Volume_RM_polygon =
(−35.8733 + 5.3486 × TH^1.5424)
× (area_m² / 10,000)
× CanopyCover
```

This model does not use thinning-specific coefficients.

A negative predicted value is mathematically possible at very low Top Heights because the equation has a negative intercept. This is an extrapolation/low-height behaviour of the equation, not a physically meaningful negative timber volume.

### 5.9 Number of living trees per hectare

The manual defines this as living trees with:

- DBH greater than 7 cm; and
- height greater than 3 m.

The model first calculates:

```text
N_model =
exp(c1 − c3 × Spacing − c4 × ln(MeanDBH))
```

It then caps the estimate at the initial stocking density:

```text
TreesPerHa = min(StockingDensity, N_model)
```

Sitka spruce coefficients:

| Management | `c1` | `c3` | `c4` |
|---|---:|---:|---:|
| Thinned | 12.02443 | −0.27860 | 1.527624 |
| Unthinned | 10.76705 | −0.43040 | 0.848349 |

Expanded Sitka equations:

```text
N_thinned =
exp(12.02443
    − (−0.27860 × Spacing)
    − 1.527624 × ln(MeanDBH))
```

```text
N_unthinned =
exp(10.76705
    − (−0.43040 × Spacing)
    − 0.848349 × ln(MeanDBH))
```

Because the published `c3` values are negative, the spacing term becomes positive after substitution. Keep the original general equation and coefficient signs when implementing it, rather than manually changing signs.

The final estimate is:

```text
TreesPerHa =
min(StockingDensity, N_thinned_or_unthinned)
```

The initial stocking density should come from the inventory record. If it must be reconstructed from regular square spacing, a common approximation is:

```text
StockingDensity ≈ 10,000 / Spacing²
```

but this approximation is not the tree-number equation itself and should only be used if the inventory stocking field is unavailable.

### 5.10 Calculation chain

```text
LiDAR normalised heights
        ↓
P95
        ↓
Top Height = P95 × 1.10
        ↓
        ├── + Age → GYC
        │              ↓
        │             SI50
        │
        ├── + Spacing + thinning status → Mean DBH
        │                                      ↓
        │                              Trees per hectare
        │
        ├── + thinning status → Basal Area
        │
        └── + thinning status → Volume per hectare
                                       ↓
                    × area in hectares × Canopy Cover
                                       ↓
                              Polygon/grid-cell volume
```

---

## 6. Sitka Spruce LiDAR Behaviour and Data Limitations

### Dense canopy and poor ground penetration

Sitka spruce has very dense, closely packed foliage. In young plantations, the canopy may be sufficiently closed that very few laser pulses reach the ground. This matters because canopy height depends on an accurate DTM:

```text
Canopy height = surface elevation − ground elevation
```

If the ground surface is poorly sampled or interpolated, normalised canopy heights can be biased.

Mature stands may have more canopy gaps because of thinning or local mortality, allowing more returns to reach the ground.

### LiDAR is effectively 2.5-D, not a full internal scan

ALS mainly samples the visible canopy surface and the ground where gaps allow penetration. Stems are generally not visible. Therefore:

- Top Height is estimated from canopy returns;
- DBH, basal area, volume and tree number are inferred through empirical equations;
- these derived variables should not be described as directly measured by LiDAR.

### Point density and treetop underestimation

At low point density, laser pulses can miss narrow treetops and return from lower parts of the crown. The manual notes that:

- approximately 1 point/m² can be sufficient for area-based stand estimates;
- individual-tree analysis requires roughly 6–8 returns per canopy;
- individual-tree analysis is difficult for dense Sitka spruce canopies;
- the `P95 × 1.10` correction is intended to compensate for systematic Top Height underestimation.

### Canopy Cover, Gap Fraction and LAI

```text
CanopyCover = proportion of land vertically occupied by canopy
```

Values range from `0` to `1`.

```text
GapFraction = ground returns / total returns
```

Values near `1` indicate exposed ground; values near `0` indicate dense vegetation cover.

LAI is defined as one-sided leaf area per unit ground area. Higher values generally indicate more foliar material, although LAI is still a LiDAR-derived structural estimate rather than a direct measurement of photosynthesis or forest health.

Do not assume:

```text
GapFraction = ground returns / total returns
```

unless the processing workflow explicitly defines the two fields that way. The manual defines Gap Fraction from the ratio of ground to total returns, while Canopy Cover is the vertical canopy projection. They may be strongly related without being exact complements.

---

## 7. Chapman-Richards (CR) Growth Equation

The biological baseline growth model. Predicts top height as a sigmoidal function of stand age:

```
y(t) = y_max × [1 − exp(−k × t)]^p
```

| Parameter | Meaning | Sitka spruce (Aberfoyle) |
|---|---|---|
| y(t) | Predicted Top Height at age t (m) | Response variable |
| y_max | Asymptotic maximum height (m) | ~46m global; 28–50m per site by GYC |
| k | Growth rate — how quickly stand approaches y_max | ~0.019 (to be refitted from new dataset) |
| p | Shape parameter — controls inflection point | ~1.0 (to be refitted from new dataset) |
| t | Stand age (years) | Age = LiDAR_year − plyr |

The CR residual — the target variable for environmental attribution:

```
ε(i,t) = y_obs(i,t) − y_CR(t | y_max_global, k, p)
```

- ε > 0: plot growing **faster** than the biological average
- ε < 0: plot **suppressed** relative to age-matched expectation
- Spatial clustering of ε motivates terrain and wind attribution

> ⚠️ CR limitation: a single global y_max cannot represent plot-level variation. An exposed ridge plot (GYC 8, y_max ≈ 28m) and a sheltered valley plot (GYC 18, y_max ≈ 46m) are forced to share the same ceiling. The Env-PINN addresses this by learning `y_max(e)` as a function of terrain and wind features.

---

## 8. Terrain & Wind Features

| Feature | Formula / Source | Ecological role |
|---|---|---|
| Elevation | OS Terrain 50 DTM (50m) | Temperature lapse rate, frost frequency, wind exposure |
| Slope | Derived from DTM in QGIS | Drainage, soil depth, radiation receipt |
| Northness | cos(aspect) | Radiation receipt — negative = north-facing, shaded |
| Eastness | sin(aspect) | Radiation receipt directionality |
| TWI | ln(A / tan(β)) via GRASS r.watershed | Terrain-driven soil moisture accumulation. Proxy for drainage without soil data. |
| TOPEX | Sum of max upward horizon angles in 8 compass directions (DTM) | Wind shelter index. Worrell (1987): strongest predictor of Sitka GYC in Scotland. |
| WASP wind speed | Mean wind speed at canopy height (m/s) — via Dr. Suárez-Minguez | Chronic wind exposure driving thigmomorphogenesis and growth suppression. |

> **Why no soil data:** CEH 50m soil dataset is interpolated from a 1:250,000 survey — only a handful of distinct values within Aberfoyle. TWI from DTM captures terrain-driven drainage at genuine 50m resolution and subsumes the role soil data would play.

> **Why no spatial climate data:** HadUK-Grid at 1km gives ~2–3 distinct values across Aberfoyle — neighbouring plots receive identical values and it cannot explain plot-level variation. Retained for temporal analysis only.

---

## 9. Dataset Field Reference (LiDAR_Years_All.gpkg)

### 8.1 LiDAR Acquisition Fields

| Field | Type | Source | Description |
|---|---|---|---|
| LiDAR_year | integer | LiDAR CSV | Year of acquisition flight |
| identification | integer | Grid (computed) | Unique grid cell identifier |
| CanopyCover | numeric | LiDAR CSV | Canopy cover fraction (0–1) |
| GapFraction | numeric | LiDAR CSV | = 1 − CanopyCover |
| LAI | numeric | LiDAR CSV | Leaf Area Index from LiDAR |
| elev_percentile_95th | numeric | LiDAR CSV | P95 of LiDAR return elevations (m) |
| elev_percentile_99th | numeric | LiDAR CSV | P99 of LiDAR return elevations (m) |
| AOI | character | CSV filename | Area of Interest label |
| block | integer | CSV filename | Survey block number |

---

### 8.2 Grid Polygon / Inventory Fields

| Field | Type | Description |
|---|---|---|
| blk | integer | Block identifier from grid polygon layer |
| cpmt | integer | Compartment number |
| scpt | character | Sub-compartment code |
| spis | character | Species code (SS = Sitka spruce) |
| plyr | integer | Year of planting |
| yldc | integer | FC Yield Class |
| whcl | integer | Windthrow hazard class |
| next_thin_ | integer | Year of next planned thinning (0 = not scheduled) |
| last_thinn | integer | Year of last thinning (0 = never thinned) |
| flyr | integer | Final year / rotation end year |
| area | numeric | Polygon area (m²) |

---

### 8.3 Species Lookup Fields (p1–p5, g1, g3)

Source: *'Use of Airborne Laser Scanning (ALS) for Forest Inventory'* (Forest Research PDF).

| Field | Role |
|---|---|
| SPIS_RF | Species code from Random Forest prediction raster |
| Species | Species name (resolved from spis_key.csv) |
| p1 | CR site index intercept |
| p2 | GYC scaling coefficient |
| p3 | Age correction term |
| p4 | CR growth rate parameter (k equivalent) |
| p5 | CR shape parameter (p equivalent) |
| g1 | Volume model coefficient (species + thinning specific) |
| g3 | Volume model power exponent (species + thinning specific) |

---

### 8.4 Derived Fields

| Field | Formula | Units |
|---|---|---|
| Age | LiDAR_year − plyr | years |
| Thin | ifelse(last_thinn ≠ 0, 1, 0) | binary |
| Top_Height95 | elev_percentile_95th × 1.1 | m |
| Top_Height99 | elev_percentile_99th | m |
| Vol95 | (g1 × Top_Height95^g3) × (area/10000) × CanopyCover | m³ |
| Vol99 | (g1 × Top_Height99^g3) × (area/10000) × CanopyCover | m³ |
| Vol_RM95 | (−35.8733 + 5.3486 × Top_Height95^1.5424) × (area/10000) × CanopyCover | m³ |
| GYCspec95 | (Top_Height95 / (1−exp(−p4×Age))^p5 − p1 − p3×2) / p2 | m³/ha/yr |
| GYCspec99 | (Top_Height99 / (1−exp(−p4×Age))^p5 − p1 − p3×2) / p2 | m³/ha/yr |

---

## 10. Analysis Methods

| Method | Description | Relevance to this study |
|---|---|---|
| Area-Based Analysis (ABA) | Statistics from groups of LiDAR points over grid cells. Predicts height, volume, BA, DBH, biomass. | How the Forest Research CSV was produced from raw point clouds. |
| Hybrid Method | Combines LiDAR-derived Top Height + species + age + growth models to estimate GYC, SI, DBH, BA, volume. | The pipeline this dissertation extends — adding terrain and wind as environmental conditioning. |
| Individual Tree Analysis (ITA) | Detects and measures individual trees. Requires 6–8+ points per canopy. | Not used — plot-level aggregates only. Note: Sitka is particularly difficult for ITA due to dense foliage. |

---

## 11. Data Quality Flags

Apply a flag-and-retain strategy. Only confirmed deletions: 9 rows with Top_Height > 150m (artefacts).

| Issue | Scale | Flag / Action |
|---|---|---|
| Duplicate plot/year pairs | 3,849 | Keep row with higher CanopyCover (confirm with Forest Research) |
| plyr = 0 → Age = LiDAR_year | Many | `unknown_age = 1` |
| Negative Age (plyr > LiDAR_year) | 72,633 | `negative_age = 1` — likely pre-planting open ground, keep |
| Negative Vol | Subset | 99.35% also have low vegetation signal — `negative_vol = 1` |
| Top_Height > 150m | 9 rows | DELETE — confirmed artefacts |
| Non-standard area (≠ 400m²) | 345,391 | `nonstandard_area = 1` — consider area-weighting |
| block vs blk disagreement | 16 rows | Confirm authoritative field with Forest Research |
| CanopyCover < 0.10, Top_Height < 5m | Subset | `low_veg = 1` — may be felled / replanted / storm damage |

---

## 12. Spatial Analysis Methods

| Method | What it tests | Tool |
|---|---|---|
| Global Moran's I | Whether CR residuals cluster spatially across the forest | PySAL `esda.Moran` (k=8 neighbours) |
| LISA (Local Moran's I) | Where specifically residuals cluster — hotspot and coldspot maps | PySAL `esda.Moran_Local` |
| Geary's C | Alternative to Moran's I — more sensitive to local, non-linear clustering | PySAL (robustness check) |
| Geographically Weighted Regression (GWR) | Whether the relationship between terrain features and growth varies spatially | `mgwr` Python package |
| Variogram | Spatial range of dependence — how far apart plots need to be to be independent | Used to set k-neighbours parameter |

---

## 13. Important Practical Rules (Sitka Spruce)

- Top Height is the key LiDAR-derived variable from which GYC, SI, and volume are all estimated
- `Top_Height ≈ P95 × 1.10` for Sitka spruce (dense foliage causes P95 underestimation)
- Closed canopies reduce ground penetration — Sitka is particularly difficult due to dense foliage
- LAI, CanopyCover, GapFraction, and Volume are downstream consequences of height — not predictors of it
- Age vs Top Height correlation is weak (~0.2 Spearman) because site productivity (y_max) dominates; age only explains variation within a cohort, not across sites
- Thinning events produce abrupt changes in stand metrics that can resemble environmental suppression — unobserved without management records

---

## 14. Acronym Reference

| Acronym | Meaning |
|---|---|
| ALS | Airborne Laser Scanning |
| ABA | Area-Based Analysis |
| BA | Basal Area (m²/ha) |
| CHM | Canopy Height Model |
| CR | Chapman-Richards (growth equation) |
| DBH | Diameter at Breast Height |
| DEM / DTM | Digital Elevation / Terrain Model |
| DSM | Digital Surface Model |
| Env-PINN | Environmentally-conditioned Physics-Informed Neural Network |
| FOV | Field of View |
| GDD | Growing Degree Days |
| GYC / GYCspec | General Yield Class / species-specific GYC |
| ITA | Individual Tree Analysis |
| LAI | Leaf Area Index |
| LiDAR | Light Detection and Ranging |
| PINN | Physics-Informed Neural Network |
| PRF | Pulse Repetition Frequency |
| SHAP | SHapley Additive exPlanations |
| SI | Site Index |
| SMD | Soil Moisture Deficit |
| SS | Sitka Spruce (Picea sitchensis) |
| TH / Top_Height | Top Height (m) |
| TOPEX | Topographic Exposure index (wind shelter) |
| TPH | Trees per Hectare |
| TWI | Topographic Wetness Index |
| WASP | Wind Atlas Analysis and Application Program |
| yldc | Yield Class (from Forest Research inventory) |

---

*Last updated: 12th July 2026. Sitka spruce only. Formulas from `LiDAR_Years_All.gpkg` field descriptions and Forest Research ALS inventory documentation.*
