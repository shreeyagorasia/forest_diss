# Citation Reference List

---

## Physics-Informed / Process-Informed Neural Networks (PINN methodology)

---

### AgriPINN (2026) — Process-informed neural network for crop biomass under water stress
**Full ref:** Shi Y, Han L, Zhang X, Sobeih T, Srivastava AK, et al. *AgriPINN: A Process-Informed Neural Network for Interpretable and Scalable Crop Biomass Prediction under Water Stress.* arXiv:2601.16045, 2026.
**URL:** https://arxiv.org/pdf/2601.16045

**Why cite:** The closest architectural parallel to your Env-PINN. AgriPINN embeds a crop-growth ODE (LINTUL5 biomass dynamics) as a differentiable soft constraint in the loss function — the same design philosophy as your CR physics loss. Key lessons: (1) the process-informed loss acts as regularisation, reducing overfitting with sparse data; (2) latent physiological variables (their LAI/RUE/water-stress factor = your plot-specific $\hat{y}_{\max}$) are recovered without direct supervision; (3) ablation across CNN, LSTM, ResNet, Transformer backbones shows the process loss improves all architectures. Directly citable as a PINN-in-agriculture precedent and justification for your loss function design. Also confirms 8× faster inference than process-based models — relevant to your computational performance section.

**Sections:** Ch. 2.4 (PINN framework), Ch. 4 (Env-PINN architecture justification), Ch. 5 (ablation comparison)

---

### Raissi et al. (2019) — Original PINN paper
**Full ref:** Raissi M, Perdikaris P, Karniadakis GE. Physics-informed neural networks: a deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378:686–707, 2019.

**Why cite:** The foundational paper introducing PINNs. Required citation whenever you claim to be using a PINN architecture. Establishes that embedding PDE constraints in the loss function enables solving forward and inverse problems with sparse data.

**Sections:** Ch. 2.4, Ch. 4 (PINN Version 1 architecture)

---

### Karniadakis et al. (2021) — PINNs improve data efficiency in small-data regimes
**Full ref:** Karniadakis GE, Kevrekidis IG, Lu L, Perdikaris P, Wang S, Yang L. Physics-informed machine learning. *Nature Reviews Physics*, 3:422–440, 2021.

**Why cite:** Establishes that physics-informed approaches specifically improve generalisation in small-data regimes — directly applicable to Aberfoyle's six-timestamp dataset. Your key citation for *why* a PINN is appropriate for sparse LiDAR data. Also establishes the convergent validity principle: if two independent methods (physics-constrained network + interpretable ML) agree on feature importance, that constitutes stronger evidence than either alone.

**Sections:** Ch. 2.4, Ch. 5.4.4 (convergent validity justification)

---

## Hybrid ML + Process Models (Knowledge-Guided ML)

---

### Zhang et al. (2023) — Machine learning vs crop growth models: an ally, not a rival
**Full ref:** Zhang N, Zhou X, Kang M, Hu BG, Heuvelink E, Marcelis LFM. Machine learning versus crop growth models: an ally, not a rival. *AoB PLANTS*, 15(2):plac061, 2023. DOI: 10.1093/aobpla/plac061
**URL:** https://academic.oup.com/aobpla/article/15/2/plac061/6855705

**Why cite:** Review paper establishing the three structural modes for combining process-based models (PBMs) with ML — parallel, serial, and modular. Your Env-PINN uses the modular structure (sub-network replaces the fixed $y_{\max}$ parameter) and the serial structure (CR equation constrains the loss). Provides the taxonomic framework to position your contribution within the broader hybrid modelling literature. Good for the Background chapter as a general framing reference.

**Sections:** Ch. 2.4 (KDDM framework), Ch. 4 (architecture justification)

---

## Growth Modelling — Forestry

---

### Socha et al. (2021) — Regional height growth models for Scots pine using CR and mixed-effects
**Full ref:** Socha J, Tymińska-Czabańska L, Bronisz K, Zięba S, Hawryło P. Regional height growth models for Scots pine in Poland. *Scientific Reports*, 11:10330, 2021. DOI: 10.1038/s41598-021-89826-9
**URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC8121862/

**Why cite:** Demonstrates that the CR equation's asymptote parameter ($a_1$ = maximum height, equivalent to your $y_{\max}$) varies significantly by region — and that allowing it to vary by site substantially improves prediction accuracy. Directly supports your central claim that a single global $y_{\max}$ is insufficient. Also shows that climate and soil conditions explain between-region height growth differences. Their ADA/GADA framework (Algebraic Difference Approach) is the traditional forestry equivalent of what your Env-PINN does — cite both as complementary approaches.

**Sections:** Ch. 2.2 (growth modelling), Ch. 4 (justification for site-specific $y_{\max}$)

---

### Manso, Davidson & McLean (2022) — Dynamic Sitka spruce height increment model
**Full ref:** Manso R, Davidson R, McLean JP. Diameter, height and volume increment single tree models for improved Sitka spruce in Great Britain. *Forestry: An International Journal of Forest Research*, 95(3):391–404, 2022. DOI: 10.1093/forestry/cpab049
**URL:** https://academic.oup.com/forestry/article/95/3/391/6427502

**Why cite:** Forest Research paper on Sitka spruce specifically. Uses Zeide's (1993) difference equation framework where climate (SPEI, spring minimum temperature) modifies the age-related growth decline parameter — the temporal equivalent of your environmental conditioning of $y_{\max}$. Key finding: site effects were clearly present between their two Scottish sites (Kintyre and Brecon) but they *could not explain them* because they only had two sites. That unresolved site effect is exactly your research question. Also confirms SPEI (drought index) significantly modifies height increment — directly supporting your use of SMD as a temporal climate index.

**Sections:** Ch. 2.2 (growth modelling), Ch. 2.3 (factors affecting growth), Introduction (gap your dissertation fills)

---

### CR-H hybrid model for pine in Mexico (2022) — Site-specific asymptote derivation
**Full ref:** [Authors]. A dynamical model based on the Chapman-Richards growth equation for fitting growth curves for four pine species in Northern Mexico. *[Journal]*, 2022.
**URL:** https://www.researchgate.net/publication/365200957

**Why cite:** Shows that the inverse relationship between CR's asymptote ($\theta_1$ = max height) and growth rate ($\theta_2$) allows reducing the model to a single site-specific parameter — and that making the asymptote site-specific improves predictions. Direct precedent for your Env-PINN's design decision to let $y_{\max}$ vary while keeping $k$ and $p$ global. Cite as evidence that site-specific growth ceilings are an established forestry modelling approach, not a novelty introduced by your neural network.

**Sections:** Ch. 2.2 (growth modelling), Ch. 4 (Env-PINN design justification)

---

## Remote Sensing / LiDAR in Forestry

---

### Frontiers in Remote Sensing (2025) — LiDAR-based forest attribute mapping
**Full ref:** [Author(s)]. [Title]. *Frontiers in Remote Sensing*, 2025. DOI: 10.3389/frsen.2025.1531097
**URL:** https://www.frontiersin.org/journals/remote-sensing/articles/10.3389/frsen.2025.1531097/full

**Why cite:** Recent (2025) paper on LiDAR-derived forest attribute estimation. Useful for the data description chapter to cite as context for how LiDAR-derived metrics (top height, canopy cover, LAI, volume) are computed from point cloud data — the same process Forest Research used to produce your dataset. Also provides up-to-date methodology reference for the plot-level attribute derivation pipeline.

**Sections:** Ch. 3.1 (LiDAR data description), Ch. 3.4 (feature extraction)

---

### MDPI Remote Sensing (2019) — Remote sensing for forest inventory
**Full ref:** [Author(s)]. [Title]. *Remote Sensing*, 11(20):2407, 2019. DOI: 10.3390/rs11202407
**URL:** https://www.mdpi.com/2072-4292/11/20/2407

**Why cite:** Context for LiDAR-based forest mapping and the challenges of integrating remote sensing data with growth models — relevant to the Introduction and data chapter for situating your work within the broader remote sensing-forestry literature.

**Sections:** Ch. 1 (Introduction context), Ch. 3.1 (LiDAR data)

---

### bioRxiv (2020) — LiDAR time series for forest growth monitoring
**Full ref:** [Author(s)]. [Title]. bioRxiv, 2020.11.13.382515, 2020.
**URL:** https://www.biorxiv.org/content/10.1101/2020.11.13.382515v2.full.pdf

**Why cite:** Preprint on using multi-temporal LiDAR data to track forest growth over time — methodologically close to your use of six LiDAR timestamps. Relevant for citing precedent for multi-temporal LiDAR analysis and the challenges of matching plots across irregular scan intervals. Check whether this is now published in a peer-reviewed journal before citing (preprints should be confirmed).

**Sections:** Ch. 3.2 (plot matching across timestamps)

---

### Nature Communications (2022) — Global forest growth drivers
**Full ref:** [Author(s)]. [Title]. *Nature Communications*, 2022. DOI: 10.1038/s41467-022-29838-9
**URL:** https://www.nature.com/articles/s41467-022-29838-9

**Why cite:** High-impact paper in *Nature Communications* on environmental drivers of forest growth variation. Likely relevant to your Background chapter on factors affecting growth (climate, terrain, competition). A Nature Communications citation carries significant weight. Confirm which environmental drivers they identify as dominant — if climate/water availability is primary, this supports your research framing.

**Sections:** Ch. 2.3 (factors affecting forest growth)

---

### arXiv (2025) — [Recent ML/forestry paper]
**Full ref:** [Author(s)]. [Title]. arXiv:2509.18228, 2025.
**URL:** https://arxiv.org/pdf/2509.18228

**Why cite:** Very recent (2025) preprint — likely relevant to ML applied to forestry or ecological modelling. Confirm content once accessed. If it covers SHAP or feature importance for forest growth attribution, cite in Ch. 2.4. If it covers PINN applications in ecology, cite alongside AgriPINN.

**Sections:** TBC once content confirmed — likely Ch. 2.4 or Ch. 4

---

## From Lynch (2025) Dissertation — Key Citations for PINN Methodology

*These citations appear in Reuben Lynch's dissertation and are directly relevant to your PINN implementation.*

---

### Pienaar & Turnbull (1973) — Chapman-Richards equation
**Full ref:** Pienaar LV, Turnbull KJ. The Chapman-Richards generalization of Von Bertalanffy's growth model for basal area growth and yield in even-aged stands. *Forest Science*, 19(1):2–22, 1973.

**Why cite:** Original paper introducing the CR equation in the forestry context. Required citation whenever you use the CR formula. Establishes the three-parameter sigmoid as a biologically grounded growth baseline.

**Sections:** Ch. 2.2, Ch. 4 (physics loss derivation)

---

### Raissi et al. (2017) — Physics-informed deep learning (Part 1)
**Full ref:** Raissi M, Perdikaris P, Karniadakis GE. Physics informed deep learning (Part I): data-driven solutions of nonlinear partial differential equations. arXiv:1711.10561, 2017.

**Why cite:** The preprint predecessor to the 2019 Journal of Computational Physics paper. Some dissertations cite both — the 2019 published version is preferred, but cite this if referencing the original framework introduction.

**Sections:** Ch. 2.4

---

### Batuwatta-Gamage et al. (2022) — PINN for plant cell shrinkage
**Full ref:** Batuwatta-Gamage CP, Rathnayaka CM, Karunasena HCP, Jeong W, Karim MA, Gu YT. A physics-informed neural network-based surrogate framework to predict moisture concentration and shrinkage of a plant cell during drying. *Journal of Food Engineering*, 332:111137, 2022.

**Why cite:** PINN applied to a biological context (plant cell dynamics) — one of the few biological PINN papers predating Lynch (2025). Shows PINN reduced overfitting by 53% over five timesteps compared to a DNN on a similar sparse data problem. Directly analogous to your small-data forestry context.

**Sections:** Ch. 2.4 (biological PINN precedents)

---

### Nathaniel et al. (2023) — PINN for above-ground biomass
**Full ref:** Nathaniel J, et al. Above ground carbon biomass estimate with physics-informed deep network. *IEEE IGARSS*, 2023.

**Why cite:** PINN outperforming random forest and linear regression for above-ground biomass prediction — the most directly comparable ML benchmark to your study. Evidence that physics parameters in the PINN loss improve accuracy over data-only approaches.

**Sections:** Ch. 2.4, Ch. 5 (comparison with benchmarks)

---

### Karniadakis et al. (2021) — Physics-informed machine learning (Nature Reviews)
*(Already listed above — see PINN methodology section)*

---

### Wong et al. (2022) — PINN robustness to sensor noise
**Full ref:** Wong JC, et al. Robustness of physics-informed neural networks to noise in sensor data. arXiv, 2022.

**Why cite:** Demonstrates that PINNs denoise better than standard DNNs under noisy sensor conditions — directly applicable to your noisy LiDAR data argument. Noisy sensor data significantly degrades DNN accuracy while PINN maintains comparable performance to a DNN trained on clean data.

**Sections:** Ch. 2.4, Ch. 5 (noise robustness discussion)

---

### Kingma & Ba (2015) — Adam optimiser
**Full ref:** Kingma DP, Ba J. Adam: a method for stochastic optimisation. *ICLR*, 2015.

**Why cite:** Standard citation for the Adam optimiser, which you use in PINN training.

**Sections:** Ch. 4 (PINN training procedure)

---

### Breiman (2001) — Random Forests
**Full ref:** Breiman L. Random forests. *Machine Learning*, 45:5–32, 2001.

**Why cite:** Original random forest paper — required if you include RF as a baseline model.

**Sections:** Ch. 4 (benchmark models)

---

### Lundberg & Lee (2017) — SHAP
**Full ref:** Lundberg SM, Lee SI. A unified approach to interpreting model predictions. *NeurIPS*, 2017.

**Why cite:** Original SHAP paper. Required citation for any SHAP analysis.

**Sections:** Ch. 2.4, Ch. 4 (XGBoost + SHAP methodology), Ch. 5.4.4 (convergent validity)

---

### Lundberg et al. (2020) — Tree SHAP
**Full ref:** Lundberg SM, Erion G, Chen H, et al. From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2:56–67, 2020.

**Why cite:** Tree-specific SHAP algorithm enabling exact computation for XGBoost. Required alongside Lundberg & Lee (2017) when using SHAP on tree-based models.

**Sections:** Ch. 4 (SHAP methodology)

---

## Sitka Spruce Ecology & Forestry Context

---

### Worrell (1987) — TOPEX predicts Sitka Yield Class in Scotland
**Full ref:** Worrell R. Predicting the productivity of Sitka spruce on upland sites in Northern Britain. *Forestry Commission Bulletin*, 72, 1987.

**Why cite:** The foundational empirical paper showing that TOPEX (topographic wind exposure) is among the strongest predictors of Sitka spruce Yield Class variation across Scottish upland sites. Yield Class declines 3.2–4.0 m³/ha/yr per 100m elevation increase. This is your ecological prior — if your Env-PINN learns that TOPEX and elevation matter, Worrell (1987) is the independent literature confirmation. Essential for the convergent validity argument.

**Sections:** Ch. 2.1, Ch. 2.3, Ch. 5.4.4 (convergent validity)

---

### Morison et al. (2010) — Soil water deficit limits Sitka growth in Scotland
**Full ref:** Morison J, et al. Understanding the growth of Sitka spruce: soil water deficit is the primary growth limiter. *[Forest Research report]*, 2010.

**Why cite:** Establishes that soil water deficit (SMD), not atmospheric VPD, is the primary climate-related growth limiter for Sitka spruce in Scottish conditions. Directly justifies your choice of SMD as the key temporal climate index and explains why HadUK-derived SMD is ecologically meaningful for your temporal sub-question.

**Sections:** Ch. 2.1, Ch. 3.5 (climate indices selection)

---

### Forest Research (2025) — Sitka spruce ecology and management
**Full ref:** Forest Research. Sitka spruce (*Picea sitchensis*). Forest Research website, 2025. https://www.forestresearch.gov.uk

**Why cite:** Authoritative species description from the same organisation that provided your data. Covers moisture requirements (≥900–1000mm), wind susceptibility, soil preferences, and waterlogging effects. Use as primary species ecology reference.

**Sections:** Ch. 2.1

---

### Telewski (2006) — Thigmomorphogenesis: mechanical effect of wind on tree growth
**Full ref:** Telewski FW. A unified hypothesis of mechanoperception in plants. *American Journal of Botany*, 93(10):1466–1476, 2006.

**Why cite:** Mechanistic explanation for why wind suppresses height growth — thigmomorphogenesis produces shorter, stockier growth forms at the cost of height increment. This is the biological mechanism behind your TOPEX/WASP attribution findings.

**Sections:** Ch. 2.1, Ch. 2.3, Ch. 6 (interpretation of results)

---

### Blyth & MacLeod (various) — Site factors and Sitka growth in northeast Scotland
**Full ref:** Blyth JF, MacLeod DA. The significance of soil nutrient status and site factors in determining the site index of Sitka spruce in northeast Scotland. *Journal of Soil Science*, 32:93–105, 1981.

**Why cite:** Found that growth in northeast Scotland correlates with TOPEX, elevation, and position-on-slope — exactly the terrain features in your feature set. Also found soil temperature during growing season and winter rainfall correlated with growth. Useful for the spatial attribution background.

**Sections:** Ch. 2.3 (factors affecting growth), Ch. 5 (feature importance comparison)

---

*Note: Papers marked as "TBC" (to be confirmed) require you to access the full content before citing. Preprints should be checked for published versions. Always verify DOIs and page numbers before submission.*

---
*Compiled: 9 July 2026. Based on URLs provided and prior work discussed in dissertation planning.*

---

## New Additions — Process-Guided / Hybrid Neural Networks for Ecology and Forestry

---

### Wesselkamp et al. (2024) — Process-Informed Neural Networks for ecology
**Full ref:** Wesselkamp M, Moser N, Kalweit M, Boedecker J, Dormann CF. Process-Informed Neural Networks: A Hybrid Modelling Approach to Improve Predictive Performance and Inference of Neural Networks in Ecology and Beyond. *Ecology Letters*, 27(11):e70012, 2024. DOI: 10.1111/ele.70012

**Why cite:** Probably your single most important methodological citation after Raissi et al. (2019). Published in *Ecology Letters* (high-impact), it evaluates five types of process-informed neural networks for predicting carbon fluxes in temperate forests — exactly the sparse-data ecological context you're working in. Key findings directly relevant to your dissertation: (1) PINNs outperform both pure process-based models and pure neural networks, especially in data-sparse regimes and high-transfer tasks; (2) process-guided networks expose mis- or undetected ecological processes — directly analogous to your goal of using the Env-PINN to identify which environmental factors the CR equation misses; (3) two important constraints on DL in ecology are identified: data sparsity and black-box opacity — exactly the two problems your Env-PINN addresses. Also note Boedecker and Dormann are co-authors on Habenicht et al. (2026) below — this is a research group actively working on the same problem as your dissertation.

**Sections:** Ch. 2.4 (PINN/KGML framework — your most important literature anchor), Ch. 4 (architecture justification), Ch. 6 (discussion of approach)

---

### Habenicht et al. (2026) — Process-Guided Neural Networks for forest carbon flux — transferability and robustness
**Full ref:** Habenicht H, Raum H, Boedecker J, Dormann CF. Evaluating Transferability and Robustness of Process-Guided Neural Networks in Forest Carbon Flux Modelling. bioRxiv 2026.02.24.707715, 2026. DOI: 10.64898/2026.02.24.707715

**Why cite:** Very recent preprint (February 2026) from the same Freiburg group as Wesselkamp et al. (2024) — essentially the follow-up paper. Directly relevant findings: (1) process-guided neural networks outperform naïve neural networks specifically in sparse-data settings and extrapolative scenarios with unseen climate conditions — your exact situation with six timestamps and a 9-year gap; (2) process-guided networks show greater robustness under transferable scenarios, which maps to your test of 2021–2023 as a held-out period; (3) variable importance analyses (accumulated local effects) show that both process-guided and naïve networks learn physically plausible relationships between meteorological drivers and growth responses — parallels your convergent validity test. Cite as the most recent evidence that the approach you're using is state-of-the-art. Note: preprint, not yet peer-reviewed — acknowledge this when citing.

**Sections:** Ch. 2.4 (most recent prior work on process-guided networks in forest ecology), Ch. 5 (robustness discussion), Ch. 6 (future work context)

---

### Pichler & Käber (2025) — FINN: Forest Informed Neural Networks via hybrid gap model
**Full ref:** Pichler M, Käber Y. Inferring Processes within Dynamic Forest Models Using Hybrid Modeling. arXiv:2508.01228 [q-bio], 2025. DOI: 10.48550/arXiv.2508.01228

**Why cite:** Introduces FINN (Forest Informed Neural Networks) — a hybrid approach that replaces specific mechanistic components of a forest gap model (FORMIND) with DNNs, then calibrates them jointly. Key relevance: (1) the growth process sub-network in FINN learns an ecologically plausible improved functional form — this is the same structural idea as your environmental sub-network learning a plot-specific $\hat{y}_{\max}$; (2) they extract what the DNN learned using explainable AI (XAI) — direct parallel to your convergent validity test using SHAP; (3) paper explicitly frames the approach as "replacing processes with DNNs" rather than using DNNs as black boxes — the philosophical framing you want for your Env-PINN architecture. Cite alongside Wesselkamp et al. (2024) as establishing the hybrid model paradigm in forest ecology.

**Sections:** Ch. 2.4 (hybrid forest modelling), Ch. 4 (sub-network architecture justification), Ch. 5.4.4 (XAI comparison)

---

### Jin et al. (2026) — Knowledge-Guided Machine Learning for Global Change Ecology
**Full ref:** Jin Z, Liu L, Yang Q, Jia X, Tao S, Guo Y, Ghosh R, Wang S, Zhu Q, Jung M, Guan K, Kumar V, Reichstein M, Fang J, Luo Y. Knowledge-Guided Machine Learning for Global Change Ecology Research. *Global Change Biology*, 32(2):e70742, 2026. DOI: 10.1111/gcb.70742

**Why cite:** Invited mini-review in *Global Change Biology* (very high impact) establishing the KGML (Knowledge-Guided Machine Learning) framework for ecology. Identifies four integration strategies for combining domain knowledge with ML: (1) domain observations as auxiliary supervision, (2) synthetic pretraining, (3) knowledge-guided structure, (4) knowledge-guided loss functions. Your Env-PINN uses strategies 3 and 4 simultaneously — cite this as the taxonomic framework for positioning your architectural contribution. The review also argues that traditional process-based approaches fail on spatiotemporal parameterisation while pure ML fails on extrapolation and interpretability — the exact two failure modes your dissertation addresses. Use in Introduction and Background to frame your contribution within the broader KGML paradigm.

**Sections:** Ch. 1 (Introduction framing), Ch. 2.4 (KGML taxonomy — replaces/supersedes earlier "GCB review 2025" placeholder in the plan), Ch. 4 (architecture justification)

---

### Qin et al. (2023) — 3PG-MT-LSTM hybrid model for long-term forest growth
**Full ref:** Qin J, Ma M, Zhu Y, Wu B, Su X. 3PG-MT-LSTM: A Hybrid Model under Biomass Compatibility Constraints for the Prediction of Long-Term Forest Growth to Support Sustainable Management. *Forests*, 14(7):1482, 2023. DOI: 10.3390/f14071482

**Why cite:** Directly comparable hybrid architecture to your Env-PINN but using LSTM rather than PINN. Couples a process-based model (3PG) with an LSTM network using biomass compatibility constraints — the same philosophy of embedding process knowledge in neural network training. Key differences from your approach: (1) uses monthly timesteps and climate inputs, making it data-hungry in ways your sparse LiDAR data cannot support; (2) uses LSTM sequence modelling, which requires more temporal observations than your six timestamps; (3) focuses on biomass rather than top height. Cite as a closely related hybrid approach that is architecturally more complex and data-demanding, justifying your choice of the simpler PINN framework with CR physics loss as more appropriate for sparse irregular timestamps.

**Sections:** Ch. 2.4 (why not LSTM/sequence model), Ch. 4 (comparison of hybrid approaches)

---

### Schwartz et al. (2025) — Deep learning for yearly forest growth from satellite data
**Full ref:** Schwartz M, Ciais P, Sean E, et al. Retrieving Yearly Forest Growth from Satellite Data: A Deep Learning Based Approach. *Remote Sensing of Environment*, 330:114959, December 2025. DOI: 10.1016/j.rse.2025.114959

**Why cite:** Very recent (2025) paper on retrieving annual forest growth increments from satellite remote sensing using deep learning — methodologically close to your use of multi-temporal LiDAR for growth measurement. Relevant for: (1) establishing that ML approaches to satellite/remote sensing-based forest growth prediction are an active research area; (2) demonstrating that DL can recover annual growth signals from coarser temporal observations — relevant context for your six-timestamp LiDAR analysis; (3) use in Remote Sensing / LiDAR data section as recent state-of-the-art reference. Check whether they handle irregular temporal gaps, as this would make it more directly comparable to your problem.

**Sections:** Ch. 2.4 (DL for forest growth from remote sensing), Ch. 3.1 (LiDAR data context)

---

*Updated: 9 July 2026. Seven new citations added covering process-guided neural networks in forest ecology (Wesselkamp 2024, Habenicht 2026, Pichler 2025), KGML framework (Jin 2026), hybrid forest growth models (Qin 2023), and satellite-based forest growth DL (Schwartz 2025).*

---

## Additional Citations — 9 July 2026

---

### Tompalski et al. (2021) — Airborne LiDAR for forest attribute change estimation and growth projection
**Full ref:** Tompalski P, Coops NC, White JC, Goodbody TRH, Hennigar CR, Wulder MA, Socha J, Woods ME. Estimating Changes in Forest Attributes and Enhancing Growth Projections: A Review of Existing Approaches and Future Directions Using Airborne 3D Point Cloud Data. *Current Forestry Reports*, 7(1):1–24, 2021. DOI: 10.1007/s40725-021-00135-w

**Why cite:** Review paper covering exactly the data pipeline you are using — airborne LiDAR point clouds processed into plot-level forest attributes (top height, volume, basal area) across multiple timestamps. Directly relevant for: (1) contextualising your six-timestamp LiDAR dataset within established practice for multi-temporal forest attribute estimation; (2) justifying your use of top height as the primary response variable, since it reviews how ALS-derived top height is used as the standard productivity proxy across the literature; (3) citing known challenges with plot matching across irregular scan intervals — which you address in your data cleaning chapter. Wulder and Coops are among the most cited names in operational forest remote sensing; this review carries significant authority. Also note co-author Socha, who appears in the Socha et al. (2021) CR modelling paper already in your citations list — useful to connect the two.

**Sections:** Ch. 3.1 (LiDAR data description and justification), Ch. 3.2 (plot matching across timestamps), Introduction (motivating multi-temporal LiDAR for growth monitoring)

---


---

*Updated: 9 July 2026. One citation added: Tompalski et al. (2021) on multi-temporal ALS for forest attribute estimation. Liu et al. (2026) removed — Boreal3D synthetic point cloud dataset is not relevant to plot-level growth modelling.*
