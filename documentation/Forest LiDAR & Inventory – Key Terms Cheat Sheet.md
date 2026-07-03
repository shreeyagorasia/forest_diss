# Forest LiDAR & Inventory – Key Terms Cheat Sheet

## Core LiDAR Terms

### LiDAR (Light Detection and Ranging)

Active remote sensing system that emits laser pulses and measures return time to estimate 3D positions.

### ALS (Airborne Laser Scanning)

LiDAR collected from an aircraft. Main technology used for large-scale forest inventory.

### Pulse

Laser beam emitted by the sensor.

### Return / Echo

Reflection of a pulse from an object.

*   **First return:** Usually canopy top.
*   **Intermediate returns:** Branches/understorey.
*   **Last return:** Usually ground (not always).

### Intensity

Strength of the returned laser signal.

*   Influenced by surface type, scan angle, moisture.
*   Mainly useful for visualisation rather than analysis.

### Pulse Repetition Frequency (PRF)

Number of laser pulses emitted per second (Hz).

### Scan Rate

Number of scan lines produced per second.

### Field of View (FOV) / Scan Angle

Maximum angle covered by the scanner.

*   Larger angle → wider coverage, lower point density.
*   Smaller angle → better canopy penetration.

### Beam Divergence

Spread of the laser beam with distance.

*   Smaller divergence = smaller footprint = higher accuracy.

### Point Density

Number of LiDAR points per square metre (points/m²). Typical values:

*   National surveys: **1–4 pts/m²**
*   Local surveys: **10–30 pts/m²**
*   Drone surveys: **100+ pts/m²**

***

# Point Cloud Classes

### Ground Points

Returns classified as terrain.

### Vegetation Points

Returns above the terrain (low, medium, high vegetation).

### Unclassified Points

Points not yet assigned to any class.

***

# Terrain & Canopy Models

### DTM (Digital Terrain Model)

Raster representing bare ground elevation.

### DEM (Digital Elevation Model)

Used interchangeably with DTM in LiDAR360.

### DSM (Digital Surface Model)

Raster representing the highest surface (trees, buildings, etc.).

### CHM (Canopy Height Model)

Canopy height above ground.

**CHM = DSM − DTM**

### Normalised Point Cloud

Point cloud where heights are referenced to ground level instead of sea level.

***

# Forest Structure Metrics

### Canopy Cover (Fractional Cover)

Proportion of ground covered by tree canopy.

Range:

*   0 = no canopy
*   1 = complete canopy cover

***

### Leaf Area Index (LAI)

Total one-sided leaf area per unit ground area.

Higher LAI indicates:

*   Denser canopy
*   Greater photosynthetic capacity
*   Healthier forest

***

### Gap Fraction

Fraction of laser pulses reaching the ground.

Range:

*   0 = closed canopy
*   1 = open ground

Higher gap fraction → more light reaches the forest floor.

***

# Forest Mensuration Terms

### Top Height (TH)

Average height of the **100 largest-diameter trees per hectare**.

For 30×30 m plots:

*   Approximate using tallest **9 trees**

For Sitka spruce:

**Top Height ≈ P95 × 1.10**

***

### Height Percentile (P95, P99, etc.)

Height below which a given percentage of LiDAR returns fall.

Examples:

*   **P95** = 95% of returns are below this height.
*   Often used as a robust estimate of canopy height.

***

### DBH (Diameter at Breast Height)

Stem diameter measured at **1.3 m above ground**.

Standard forestry measurement.

***

### Mean DBH

Average DBH across all trees in a stand.

***

### Basal Area (BA)

Total cross-sectional stem area at breast height.

Units:

**m²/ha**

Indicator of:

*   Stand density
*   Biomass
*   Timber stocking

***

### Volume

Standing timber volume.

Units:

**m³/ha**

***

### Trees per Hectare (TPH)

Number of living trees per hectare.

Typically includes:

*   DBH > 7 cm
*   Height > 3 m

***

# Productivity Measures

### Yield Class (GYC)

British productivity index.

Represents:

**Maximum mean annual timber volume increment (m³/ha/year).**

Higher GYC = more productive site.

Examples:

*   YC14
*   YC18
*   YC24

***

### Site Index (SI)

Height of dominant trees at a reference age.

Usually:

*   **Age 50** (most UK species)
*   **Age 30** (poplar)

Unlike GYC, SI is internationally used.

Higher SI = better growing site.

***

# Spatial Products

### Slope

Terrain steepness.

***

### Aspect

Direction the slope faces.

Measured in degrees:

*   0° = North
*   90° = East
*   180° = South
*   270° = West

***

### Surface Roughness

Measure of terrain irregularity.

Useful for:

*   Ground preparation
*   Erosion assessment

***

# Analysis Methods

### Area-Based Analysis (ABA)

Uses statistics from groups of LiDAR points over grid cells or forest stands.

Predicts:

*   Height
*   Volume
*   Basal area
*   DBH
*   Biomass

Best for moderate point densities.

***

### Hybrid Method

Combines:

*   LiDAR-derived Top Height
*   Species
*   Age
*   Growth models

to estimate:

*   Yield Class
*   Site Index
*   DBH
*   Basal Area
*   Volume

Requires much less field data than full regression methods.

***

### Individual Tree Analysis (ITA)

Detects and measures individual trees from LiDAR.

Requires high point density.

Generally needs: **6–8+ points per canopy**

Outputs:

*   Tree height
*   Crown diameter
*   Crown area
*   Tree location

***

# Interpolation Methods

### TIN (Triangulated Irregular Network)

Exact interpolation using Delaunay triangles.

Best when points are evenly distributed.

***

### Kriging

Geostatistical interpolation based on spatial autocorrelation.

Best when data contain gaps.

***

### IDW (Inverse Distance Weighting)

Nearby points have greater influence than distant points.

Simple and computationally efficient.

***

# Important Practical Rules

*   **First return ≈ canopy**
*   **Last return ≈ ground**
*   **CHM = DSM − DTM**
*   **Higher point density = better canopy representation**
*   **Closed canopies reduce ground penetration**
*   **Sitka spruce is particularly difficult because of dense foliage**
*   **Individual tree detection requires much higher point density than stand-level inventory**
*   **Top Height is the key LiDAR-derived variable from which many other forest metrics are estimated**

***

# Common Acronyms

| Acronym | Meaning                     |
| ------- | --------------------------- |
| ALS     | Airborne Laser Scanning     |
| LiDAR   | Light Detection and Ranging |
| PRF     | Pulse Repetition Frequency  |
| FOV     | Field of View               |
| DTM     | Digital Terrain Model       |
| DEM     | Digital Elevation Model     |
| DSM     | Digital Surface Model       |
| CHM     | Canopy Height Model         |
| LAI     | Leaf Area Index             |
| DBH     | Diameter at Breast Height   |
| BA      | Basal Area                  |
| TH      | Top Height                  |
| GYC     | General Yield Class         |
| SI      | Site Index                  |
| ABA     | Area-Based Analysis         |
| ITA     | Individual Tree Analysis    |
| TPH     | Trees per Hectare           |
