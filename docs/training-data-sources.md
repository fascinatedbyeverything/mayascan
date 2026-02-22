# MayaScan Training Data Sources

Comprehensive catalogue of publicly available LiDAR datasets for training archaeological feature detection models focused on Maya archaeology.

## Immediately Usable (Open Access, ML-Ready)

### 1. Chactun ML-Ready Dataset (PRIMARY)
- **Source:** [Figshare](https://figshare.com/articles/dataset/22202395)
- **Paper:** [Kokalj et al., 2023 (Nature Scientific Data)](https://doi.org/10.1038/s41597-023-02455-x)
- **Region:** Chactun, central Yucatan, Campeche, Mexico
- **Area:** ~240 km², 2,094 annotated tiles
- **Annotations:** 10,000+ objects — buildings (9,303), platforms (2,110), aguadas (95)
- **Format:** GeoTIFF rasters (SVF, openness, slope @ 0.5m) + binary segmentation masks
- **Access:** Open, CC BY 4.0
- **Status:** START HERE — purpose-built for CNN training

### 2. NASA G-LiHT Mexico (LARGEST OPEN COVERAGE)
- **Source:** [G-LiHT Data Center](https://glihtdata.gsfc.nasa.gov/)
- **Region:** Southern Mexico, entire Yucatan Peninsula
- **Area:** ~36,584 km² (458 narrow transect tiles)
- **Resolution:** 8-12 pts/m², 1m raster products
- **Annotations:** Published annotation workflows (Schroder 2020, Character 2024)
- **Format:** LAS 1.1, GeoTIFF
- **Access:** Fully open, no registration
- **Note:** F1=0.89 achieved by Character et al. (2025) for settlement detection

### 3. Caracol/Chiquibul, Belize
- **Source:** [OpenTopography](https://portal.opentopography.org/dataspace/dataset?opentopoID=OTDS.022019.32616.2)
- **Region:** Caracol archaeological site, Belize
- **Area:** 64 km²
- **Resolution:** 23.91 pts/m² (1.53 billion points)
- **Format:** LAZ (60 tiles)
- **Access:** Open (OpenTopography registration)
- **Note:** Iconic Maya city — pyramids, terraces, causeways

### 4. Middle Usumacinta / Aguada Fenix, Mexico
- **Source:** [OpenTopography](https://portal.opentopography.org/raster?opentopoID=OTSDEM.042022.32615.1)
- **Region:** Eastern Tabasco, Mexico
- **Area:** 943 km²
- **Resolution:** 0.5m DEM
- **Format:** GeoTIFF, point cloud
- **Access:** Open, CC BY 4.0
- **Note:** Contains oldest/largest Maya monumental architecture

### 5. Puuc Region, Yucatan, Mexico
- **Source:** [OpenTopography](https://portal.opentopography.org/raster?opentopoID=OTSDEM.082019.32616.1)
- **Paper:** [Ringle et al., 2021 (PLOS ONE)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0249314)
- **Region:** Puuc hills, Campeche/Yucatan border
- **Area:** 238 km²
- **Resolution:** 38.90 pts/m² (9.25 billion points!)
- **Format:** Point cloud + GeoTIFF
- **Access:** Open (registration)
- **Note:** Zhang et al. achieved IoU 0.842 (platforms) with YOLOv8

### 6. Archaeoscape, Cambodia (Transfer Learning)
- **Source:** [archaeoscape.ai](https://archaeoscape.ai/data/2024/)
- **Paper:** [arXiv:2412.05203 (NeurIPS 2024)](https://arxiv.org/abs/2412.05203)
- **Region:** Cambodia (Angkorian sites — NOT Maya, but analogous tropical archaeology)
- **Area:** 888 km²
- **Annotations:** 31,141 features, 5-class semantic segmentation
- **Format:** GeoTIFF (nDTM + RGB orthophoto)
- **Access:** Credentialized open access
- **Note:** 4x larger than any comparable dataset, useful for pre-training

## Requires Registration or PI Contact

### 7. Copan, Honduras (MayaArch3D)
- **Source:** [mayaarch3d.org](https://mayaarch3d.org/)
- **Area:** 25 km², 21.57 pts/m²
- **Annotations:** 5-class 3D point cloud annotations (142 objects)
- **Note:** Only dataset with 3D archaeological point cloud annotations

### 8. PACUNAM LiDAR Initiative, Guatemala
- **Source:** [Science (2018)](https://www.science.org/doi/10.1126/science.aau0137)
- **Area:** 2,144 km² — largest archaeological LiDAR survey ever
- **Annotations:** 60,000+ structures identified
- **Access:** Restricted — Fundacion PACUNAM collaboration required
- **Note:** Used in Bundzel 2020 U-Net study (66% detection rate)

### 9. Mirador-Calakmul Basin, Guatemala/Mexico
- **Source:** [Hansen et al., 2022](https://doi.org/10.1017/S0956536121000195)
- **Area:** 1,703 km²
- **Features:** 964 settlements, 30 ball courts, 195 reservoirs, 177 km of causeways
- **Access:** Restricted — contact Richard Hansen (Idaho State)

## ML Competition Datasets

### 10. ECML PKDD 2021 "Discover the Mysteries of the Maya"
- **Source:** [Challenge site](https://biasvariancelabs.github.io/maya_challenge/)
- **Data:** Same as Chactun (#1) — published baselines and winning solutions
- **Note:** Evaluation scripts and competition code available on GitHub

## Government / National Data

### 11. INEGI Mexico National LiDAR
- **Source:** [inegi.org.mx](https://www.inegi.org.mx/)
- **Coverage:** Nationwide Mexico including Yucatan states
- **Resolution:** 5m DEM/DSM (low — not ideal for structure detection)
- **Access:** Free at INEGI centers

## Recommended Training Strategy

1. **Phase 1 (Now):** Train on Chactun ML-ready Figshare dataset — it's pre-formatted for CNN training
2. **Phase 2:** Evaluate on NASA G-LiHT transects using published annotation workflows
3. **Phase 3:** Fine-tune with Caracol and Middle Usumacinta open point clouds (need to generate visualizations + manual annotation)
4. **Phase 4:** Pre-train on Archaeoscape (Cambodia) for cross-cultural transfer learning
5. **Phase 5:** Pursue PACUNAM collaboration for the 2,144 km² Guatemala dataset

## Total Open Data Available

| Tier | Datasets | Total Area |
|------|----------|------------|
| Immediately usable (annotated) | 2 | ~1,128 km² |
| Open point cloud (needs annotation) | 4 | ~38,829 km² |
| Registration/contact required | 3 | ~3,872 km² |
| **Total accessible** | **9** | **~43,829 km²** |
