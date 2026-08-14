# Citation Reference List
% ==============================================================================
% BIBTEX KEY NAMING CONVENTION:
% Format: [lastname][year][FirstTitleWord]_[CategoryCode]
% Example: malcolm1970Site_C1_SS
%
% CATEGORY CODES:
% _C1_SS   = Cat 1: Sitka Spruce, Top Height, Yield Models & Digital Twins
% _C2_CR   = Cat 2: Chapman-Richards & Height-Growth Modelling
% _C3_LID  = Cat 3: Repeated LiDAR & Top-Height Measurement & Data Sources
% _C4_NN   = Cat 4: DNNs, PINNs & Hybrid Ecological Models
% _C5_EVAL = Cat 5: Spatial & General Evaluation (Moran's, LISA, NLME, BlockCV)
% _C6_GWR  = Cat 6: GWR and GNNWR Spatial Models
% _C7_XAI  = Cat 7: Residual Attribution & Interpretable ML (SHAP, ALE)
% _C8_FW   = Cat 8: Future Work & General Extensions
% ==============================================================================

% ==============================================================================
% EXAMPLE CITATION TEMPLATE & CLEANUP RULES
% ==============================================================================
% @article{lastnameYEARWord_CATEGORY,
%   title        = {KEEP: Exact title of the paper, use {{Double Braces}} for acronyms},
%   author       = {KEEP: Lastname, Firstname and Lastname, Firstname},
%   year         = {KEEP: Publication year},
%   journal      = {KEEP: Journal name (for @article)},
%   booktitle    = {KEEP: Conference name (for @inproceedings)},
%   publisher    = {KEEP: Only if a book or institution},
%   volume       = {KEEP: Journal volume (if applicable)},
%   number       = {KEEP: Journal issue (if applicable)},
%   pages        = {KEEP: Page range},
%   doi          = {KEEP: Always keep if available},
%   url          = {KEEP: If no DOI exists},
%   eprint       = {KEEP: Essential for arXiv preprints},
%   archiveprefix= {KEEP: Use 'arXiv' if eprint is used},
%
%   month        = {DELETE/MAYBE: Usually ignored by plain numeric styles},
%   eprinttype   = {DELETE: e.g., {hdl}, clutter},
%   langid       = {DELETE: e.g., {english}, clutter},
%   abstract     = {DELETE: Takes up too much space},
%   file         = {DELETE: Local file paths},
%   keywords     = {DELETE: Internal reference manager tags},
%   issn         = {DELETE: Unnecessary for standard referencing},
%   primaryclass = {DELETE: arXiv subcategory clutter}
% }


% ==============================================================================
% 1. Sitka Spruce, Top Height, Yield Models & Digital Twins (_C1_SS)
% ==============================================================================

% TO ADD:
% - Manso et al. (2021) — Dynamic top height models for several major forest tree species in Great Britain.
% - Edwards and Christie (1981) — Yield Models for Forest Management.
% - Forest Research — Forest Yield.
% - Worrell (1987) — Predicting the Productivity of Sitka Spruce on Upland Sites in Northern Britain.
% - Farrelly et al. (2011) — Sitka spruce site index in response to varying soil moisture and nutrients.
% - Manso et al. (2022) — Diameter, height and volume increment single-tree models.
% - Forest Research — Sitka spruce species account.
% - Cameron (2015) — Building resilience into Sitka spruce plantations in northern Britain.
% - Lee and Watt (2012) — Choosing Sitka Spruce Planting Stock.
% - Buonocore et al. (2022) — A Proposal for a Forest Digital Twin Framework and Its Perspectives.
% - Qiu et al. (2023) — Forest digital twin: A new tool for forest management practices.
% - Jiang et al. (2022) — Forestry Digital Twin With Machine Learning in Landsat 7 Data.

@article{buonocore2022Proposal_C1_SS,
  title = {A {{Proposal}} for a {{Forest Digital Twin Framework}} and {{Its Perspectives}}},
  author = {Buonocore, Luca and Yates, Jim and Valentini, Riccardo},
  year = {2022},
  journal = {Forests},
  volume = {13},
  number = {4},
  pages = {498},
  doi = {10.3390/f13040498}
}

@article{janiec2023Development_C1_SS,
  title = {Development of Regional Height Growth Model for {{Scots}} Pine Using Repeated Airborne Laser Scanning Data},
  author = {Janiec, P. and {Tymi{\'n}ska-Czaba{\'n}ska}, L. and Hawry{\l}o, P. and Socha, J.},
  year = {2023},
  journal = {Frontiers in Environmental Science},
  volume = {11},
  pages = {1260725},
  doi = {10.3389/fenvs.2023.1260725}
}

@article{malcolm1970Site_C1_SS,
  title = {Site Factors and the Growth of {{Sitka}} Spruce},
  author = {Malcolm, Douglas C.},
  year = {1970},
  publisher = {The University of Edinburgh},
  url = {http://hdl.handle.net/1842/35126}
}

@misc{ForestResearch2026Sitka_C1_SS,
  title = {Sitka spruce (SS) - Tree Species Database},
  author = {{Forest Research}},
  year = {2026},
  url = {https://www.forestresearch.gov.uk/tools-and-resources/tree-species-database/131584-sitka-spruce-ss-2/},
  note = {Accessed: 2026-08-11}
}

@article{baskent2026Review_C1_SS,
  title = {A Review and Conceptual Framework for a New Era of Forest Management Planning: Integrating Hybrid Digital Twin Systems toward Sustainable Forest Ecosystems},
  author = {Baskent, Emin Zeki and Bon{\v c}ina, Andrej and Borges, Jos{\'e} Guilherme},
  year = 2026,
  month = jun,
  journal = {Ecosystem Services},
  volume = {79},
  doi = {10.1016/j.ecoser.2026.101851}
}

@misc{ScottishForestry2026Future_C1_SS,
  title = {Future Productive Species List Scotland},
  author = {{Scottish Forestry}},
  year = {2026},
  url = {https://www.forestry.gov.scot/publications/future-productive-species-list-scotland},
  note = {Accessed: 2026-08-11}
}
@article{mason2011Sitka_C1_SS,
  title = {{{Sitka}} Spruce Forests in {{Atlantic Europe}}: Changes and Challenges},
  author = {Mason, William L. and Perks, Mark P.},
  year = {2011},
  journal = {Scandinavian Journal of Forest Research},
  volume = {26},
  number = {sup11},
  pages = {72--81},
  doi = {10.1080/02827581.2011.564383}
}

@techreport{ForestResearch2009Evidence_C1_SS,
  title = {The Evidence Supporting the Use of {{CCF}} in Adapting {{Scotland's}} Forests to the Risks of Climate Change},
  author = {{Forest Research}},
  year = {2009},
  institution = {Forestry Commission Scotland},
  url = {https://cdn.forestresearch.gov.uk/2022/02/ccf_and_climate_change_report-1.pdf}
}

@inproceedings{suarezminguez2008Practical_C1_SS,
  title = {A Practical Application of Airborne {{LiDAR}} for Forestry Management in {{Scotland}}},
  author = {Suarez Minguez, Juan and Rosette, Jacqueline and Nicoll, Bruce and Gardiner, Barry},
  year = {2008},
  booktitle = {Proceedings of SilviLaser 2008},
  pages = {17--19}
}

@article{socha2023Higher_C1_SS,
  title = {Higher Site Productivity and Stand Age Enhance Forest Susceptibility to Drought-Induced Mortality},
  author = {Socha, Jaros{\l}aw and Hawry{\l}o, Pawe{\l} and {Tymi{\'n}ska-Czaba{\'n}ska}, Luiza and Reineking, Bj{\"o}rn and Lindner, Marcus and Netzel, Pawe{\l} and {Grabska-Szwagrzyk}, Ewa and Vallejos, Ronny and Reyer, Christopher P. O.},
  year = {2023},
  journal = {Agricultural and Forest Meteorology},
  volume = {341},
  pages = {109680},
  doi = {10.1016/j.agrformet.2023.109680}
}

@article{cameron2015Building_C1_SS,
  title = {Building {{Resilience}} into {{Sitka Spruce}} ({{Picea}} Sitchensis ({{Bong}}.) {{Carr}}.) {{Forests}} in {{Scotland}} in {{Response}} to the {{Threat}} of {{Climate Change}}},
  author = {Cameron, Andrew},
  year = {2015},
  journal = {Forests},
  volume = {6},
  pages = {398--415},
  doi = {10.3390/f6020398}
}

@techreport{ForestResearch2022Climate_C1_SS,
  title = {Climate-Ready Forestry at {{Queen Elizabeth Forest Park}} ({{Case Study 1}})},
  author = {{Forest Research}},
  year = {2022},
  institution = {Forest Research},
  url = {https://cdn.forestresearch.gov.uk/2022/05/UKFSPG026_CS1_QEFP.pdf}
}


@misc{dtu2019Global_C1_SS,
  title = {Global {{Wind Atlas}} 3.0},
  author = {{Technical University of Denmark (DTU)} and {World Bank Group}},
  year = {2019},
  url = {https://globalwindatlas.info},
  note = {Accessed: 2026-08-13}
}

@techreport{worrell1987Predicting_C1_SS,
  title = {Predicting the Productivity of {{Sitka}} Spruce on Upland Sites in {{Northern Britain}}},
  author = {Worrell, R.},
  year = {1987},
  institution = {Forestry Commission},
  type = {Bulletin},
  number = {72}
}
% ==============================================================================
% 2. Chapman–Richards & Height-Growth Modelling (_C2_CR)
% ==============================================================================

% TO ADD:
% - Richards (1959) — A flexible growth function for empirical use.
% - Chapman (1961) — Statistical problems in dynamics of exploited fisheries populations.
% - Rennolls (1995) — Forest height growth modelling.
% - Cieszewski and Bailey (2000) — Generalized algebraic difference approach.
% - Ercanli, Bolat and Yavuz (2023) — A comparison of artificial neural networks and regression modeling.

@article{pommerening2015Methods_C2_CR,
  title = {Methods of modelling relative growth rate},
  author = {Pommerening, A. and Muszta, A.},
  year = {2015},
  journal = {Forest Ecosystems},
  volume = {2},
  pages = {5},
  doi = {10.1186/s40663-015-0029-4}
}

%MIXED MODELs two stage RS2B
@article{socha2016Assessment_C2_CR,
  title   = {Assessment of Age Bias in Site Index Equations},
  author  = {Socha, Jaros{\l}aw and Coops, Nicholas C. and Ocha{\l}, Wojciech},
  year    = {2016},
  journal = {iForest - Biogeosciences and Forestry},
  volume  = {9},
  number  = {3},
  pages   = {402--408},
  doi     = {10.3832/ifor1548-008}
}
rq3 heith growth chaing ymax
@article{socha2021Regional_C2_CR,
  title   = {Regional Height Growth Models for {{Scots}} Pine in {{Poland}}},
  author  = {Socha, Jaros{\l}aw and Tymi{\'n}ska-Czaba{\'n}ska, Luiza and Bronisz, Karol and Zi{\k{e}}ba, Stanis{\l}aw and Hawry{\l}o, Pawe{\l}},
  year    = {2021},
  journal = {Scientific Reports},
  volume  = {11},
  pages   = {10330},
  doi     = {10.1038/s41598-021-89826-9}
}

% ==============================================================================
% 3. Repeated LiDAR & Top-Height Measurement & Data Sources (_C3_LID)
% ==============================================================================

% TO ADD:
% - Næsset (1997) — Determination of mean tree height of forest stands using airborne laser scanner data.
% - Næsset and Gobakken (2005) — Estimating forest growth using canopy metrics derived from airborne laser scanner data.
% - Hyyppä et al. (2012) — forest inventory using ALS.
% - Riofrío et al. (2022) — Harmonizing multi-temporal airborne laser scanning point clouds.
% - Hawryło et al. (2024) — How to adequately determine the top height of forest stands based on airborne laser scanning.
% - Yu et al. (2024) — reliability of bi-temporal canopy-height change.
% - White et al. (2013) — LiDAR forest-inventory best practice.
@article{suarez2005Use_C3_LID,
  title = {Use of Airborne {{LiDAR}} and Aerial Photography in the Estimation of Individual Tree Heights in Forestry},
  author = {Su{\'a}rez, Juan C. and Ontiveros, Carlos and Smith, Steve and Snape, Stewart},
  year = {2005},
  journal = {Computers \& Geosciences},
  volume = {31},
  number = {2},
  pages = {253--262},
  doi = {10.1016/j.cageo.2004.09.015}
}

@article{moisen2002Comparing_C3_LID,
  title = {Comparing Five Modelling Techniques for Predicting Forest Characteristics},
  author = {Moisen, Gretchen G. and Frescino, Tracey S.},
  year = {2002},
  journal = {Ecological Modelling},
  volume = {157},
  number = {2-3},
  pages = {209--225},
  doi = {10.1016/S0304-3800(02)00197-7}
}
--------- Check below citaons and doi links, find last one, 
@article{poggio2021SoilGrids_C3_LID,
  title = {{SoilGrids} 2.0: Producing Soil Information for the Globe with Quantified Spatial Uncertainty},
  author = {Poggio, Laura and de Sousa, Lu{\'i}s M. and Batjes, Niels H. and Heuvelink, Gerard B. M. and Kempen, Bas and Ribeiro, Eloi and Rossiter, David},
  year = {2021},
  journal = {SOIL},
  volume = {7},
  number = {1},
  pages = {217--240},
  doi = {10.5194/soil-7-217-2021}
}
@article{davis2023GlobalWindAtlas_C3_LID,
  title = {The Global Wind Atlas: A High-Resolution Dataset of Climatologies and Associated Web-Based Application},
  author = {Davis, Neil N. and Badger, Jake and Hahmann, Andrea N. and Hansen, Brian Ohrbeck and Mortensen, Niels Gylling and Kelly, Mark and Lars{\'e}n, Xiaoli Guo and Olsen, Bjarke T. and Floors, Rogier and Lizcano, Gil and Casso, Pau and Lacave, Oriol and Bosch, Albert and Bauwens, Ides and Knight, Oliver James and van Loon, Albertine Potter and Fox, Rachel and Parvanyan, Tigran and Hansen, S{\o}ren Bo Krohn and Heathfield, Duncan and Onninen, Marko and Drummond, Ray},
  year = {2023},
  journal = {Bulletin of the American Meteorological Society},
  volume = {104},
  number = {8},
  pages = {E1507--E1525},
  doi = {10.1175/BAMS-D-21-0075.1}
}
@article{hollis2019HadUKGrid_C3_LID,
  title = {{HadUK-Grid}: A New UK Dataset of Gridded Climate Observations},
  author = {Hollis, Dan and McCarthy, Mark and Kendon, Mike and Legg, Tim and Simpson, Ian},
  year = {2019},
  journal = {Geoscience Data Journal},
  volume = {6},
  number = {2},
  pages = {151--159},
  doi = {10.1002/gdj3.78}
}
@article{brun2022GlobalClimatePredictors_C3_LID,
  title = {Global Climate-Related Predictors at Kilometer Resolution for the Past and Future},
  author = {Brun, Philipp and Zimmermann, Niklaus E. and Hari, Chantal and Pellissier, Lo{\"i}c and Karger, Dirk Nikolaus},
  year = {2022},
  journal = {Earth System Science Data},
  volume = {14},
  number = {12},
  pages = {5573--5603},
  doi = {10.5194/essd-14-5573-2022}
}
@article{munozsabater2021ERA5Land_C3_LID,
  title = {{ERA5-Land}: A State-of-the-Art Global Reanalysis Dataset for Land Applications},
  author = {Mu{\~n}oz-Sabater, Joaqu{\'i}n and Dutra, Emanuel and Agust{\'i}-Panareda, Anna and Albergel, Cl{\'e}ment and Arduini, Gabriele and Balsamo, Gianpaolo and Boussetta, Souhail and Choulga, Margarita and Harrigan, Shaun and Hersbach, Hans and Martens, Brecht and Miralles, Diego G. and Piles, Mar{\'i}a and Rodr{\'i}guez-Fern{\'a}ndez, Nemesio J. and Zsoter, Ervin and Buontempo, Carlo and Th{\'e}paut, Jean-No{\"e}l},
  year = {2021},
  journal = {Earth System Science Data},
  volume = {13},
  number = {9},
  pages = {4349--4383},
  doi = {10.5194/essd-13-4349-2021}
}
@misc{metoffice2019MIDASOpen_C3_LID,
  title = {Met Office {MIDAS} Open: UK Land Surface Stations Data (1853--Current)},
  author = {{Met Office}},
  year = {2019},
  publisher = {Centre for Environmental Data Analysis},
  url = {https://catalogue.ceda.ac.uk/uuid/dbd451271eb04662beade68da43546e1/},
  urldate = {2026-08-13}
}
@misc{ordnancesurveyTerrain50_C3_LID,
  title = {{OS Terrain 50}},
  author = {{Ordnance Survey}},
  year = {2026},
  url = {https://www.ordnancesurvey.co.uk/products/os-terrain-50},
  urldate = {2026-08-13}
}
@misc{ordnancesurveyOpenRoads_C3_LID,
  title = {{OS Open Roads}},
  author = {{Ordnance Survey}},
  year = {2026},
  url = {https://www.ordnancesurvey.co.uk/products/os-open-roads},
  urldate = {2026-08-13}
}
@misc{ordnancesurveyOpenRivers_C3_LID,
  title = {{OS Open Rivers}},
  author = {{Ordnance Survey}},
  year = {2026},
  url = {https://www.ordnancesurvey.co.uk/products/os-open-rivers},
  urldate = {2026-08-13}
}
@misc{forestresearchAberfoylePlots_C3_LID,
  title = {Repeated Forest Plot and Airborne Laser Scanning Data for Aberfoyle Forest},
  author = {{Forest Research}},
  year = {2023},
  note = {Dataset supplied for this study; final dataset title and access details to be confirmed}
}
@misc{ukcehSoilMaps_C3_LID,
  title = {National-Scale Maps of Parent Material Properties, Terrain and Soil Natural Capital Units for Great Britain},
  author = {{UK Centre for Ecology \& Hydrology}},
  year = {2020},
  note = {Exact dataset record and version to be confirmed from the downloaded source files}
}
--------- Check above citaons and doi links, find last one, 

% ==============================================================================
% 4. DNNs, PINNs & Hybrid Ecological Models (_C4_NN)
% ==============================================================================
% TO ADD:
% - Raissi, Perdikaris and Karniadakis (2019) — Physics-informed neural networks.
% - Karniadakis et al. (2021) — Physics-informed machine learning.
% - Willard et al. (2022) — Integrating scientific knowledge with machine learning.
% - Wang, Teng and Perdikaris (2021) — Understanding and mitigating gradient flow pathologies.
% - Rathore et al. (2024) — Challenges in training PINNs: A loss landscape perspective.
% - Wesselkamp et al. (2024) — Process-Informed Neural Networks.

@misc{pichler2025Inferring_C4_NN,
  title = {Inferring Processes within Dynamic Forest Models Using Hybrid Modeling},
  author = {Pichler, Maximilian and K{\"a}ber, Yannek},
  year = {2025},
  eprint = {2508.01228},
  archiveprefix = {arXiv},
  doi = {10.48550/arXiv.2508.01228}
}

@article{karpatne2017Theory_C4_NN,
  title = {Theory-Guided Data Science: A New Paradigm for Scientific Discovery from Data},
  author = {Karpatne, Anuj and Atluri, Gowtham and Faghmous, James H. and Steinbach, Michael and Banerjee, Arindam and Ganguly, Auroop and Shekhar, Shashi and Samatova, Nagiza and Kumar, Vipin},
  year = {2017},
  journal = {IEEE Transactions on Knowledge and Data Engineering}, 
  volume = {29},
  number = {10},
  pages = {2318--2331},
  doi = {10.1109/TKDE.2017.2720168}
}

@mastersthesis{lynch2025Digital_C4_NN,
  title = {Digital Twins in Forestry: A Comparative Study of Physics-Informed Neural Networks and Data-Driven Models for Growth Prediction},
  author = {Lynch, Reuben},
  year = {2025},
  school = {The University of Edinburgh}
}

@article{kovacs2022Conditional_C4_NN,
  title   = {Conditional Physics Informed Neural Networks},
  author  = {Kovacs, Alexander and Exl, Lukas and Kornell, Alexander and Fischbacher, Johann and Hovorka, Markus and Gusenbauer, Markus and Breth, Leoni and Oezelt, Harald and Yano, Masao and Sakuma, Noritsugu and Kinoshita, Akihito and Shoji, Tetsuya and Kato, Akira and Schrefl, Thomas},
  year    = {2022},
  journal = {Communications in Nonlinear Science and Numerical Simulation},
  volume  = {104},
  pages   = {106041},
  doi     = {10.1016/j.cnsns.2021.106041}
}

@article{miao2023VCPINN_C4_NN,
  title   = {{{VC-PINN}}: Variable Coefficient Physics-Informed Neural Network for Forward and Inverse Problems of {{PDEs}} with Variable Coefficient},
  author  = {Miao, Zhengwu and Chen, Yong},
  year    = {2023},
  journal = {Physica D: Nonlinear Phenomena},
  volume  = {456},
  pages   = {133945},
  doi     = {10.1016/j.physd.2023.133945}
}
%Chooisng sets via elastic net 
@article{zou2005Regularization_C4_NN,
  title   = {Regularization and Variable Selection via the Elastic Net},
  author  = {Zou, Hui and Hastie, Trevor},
  year    = {2005},
  journal = {Journal of the Royal Statistical Society: Series B (Statistical Methodology)},
  volume  = {67},
  number  = {2},
  pages   = {301--320},
  doi     = {10.1111/j.1467-9868.2005.00503.x}
}
\choosing set for xgboost? not sure need to check 
@inproceedings{chen2016XGBoost_C4_NN,
  title     = {{{XGBoost}}: A Scalable Tree Boosting System},
  author    = {Chen, Tianqi and Guestrin, Carlos},
  year      = {2016},
  booktitle = {Proceedings of the 22nd {{ACM}} {{SIGKDD}} International Conference on Knowledge Discovery and Data Mining},
  pages     = {785--794},
  doi       = {10.1145/2939672.2939785}
}
% ==============================================================================
% 5. Spatial & General Evaluation (Moran's, LISA, NLME, BlockCV) (_C5_EVAL)
% ==============================================================================

% TO ADD:
% - Roberts et al. (2017) — Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure.
% - Valavi et al. (2019) — blockCV: An R package for generating spatially or environmentally separated folds.
% - Moran (1950) — Notes on continuous stochastic phenomena.
% - Anselin (1995) — Local Indicators of Spatial Association—LISA.
% - Dormann et al. (2007) — Methods to account for spatial autocorrelation.
% - Kim et al. (2021) — Predicting the magnitude of residual spatial autocorrelation in geographical ecology.
% - (Add any simple NLME or other evaluation papers here)

@article{ploton2020Spatial_C5_EVAL,
  title = {Spatial Validation Reveals Poor Predictive Performance of Large-Scale Ecological Mapping Models},
  author = {Ploton, Pierre and Mortier, Fr{\'e}d{\'e}ric and {R{\'e}jou-M{\'e}chain}, Maxime and Barbier, Nicolas and Picard, Nicolas and Rossi, Vivien and Dormann, Carsten and Cornu, Guillaume and Viennois, Ga{\"e}lle and Bayol, Nicolas and Lyapustin, Alexei and {Gourlet-Fleury}, Sylvie and P{\'e}lissier, Rapha{\"e}l},
  year = {2020},
  journal = {Nature Communications},
  volume = {11},
  pages = {4540},
  doi = {10.1038/s41467-020-18321-y}
}
@article{karasiak2022Spatial_C5_EVAL,
  title = {Spatial Dependence between Training and Test Sets: Another Pitfall of Classification Accuracy Assessment in Remote Sensing},
  author = {Karasiak, N. and Dejoux, J.-F. and Monteil, C. and Sheeren, D.},
  year = {2022},
  journal = {Machine Learning},
  volume = {111},
  number = {7},
  pages = {2715--2740},
  doi = {10.1007/s10994-021-05972-1}
}
%Geometric centre of clipped cells 
@book{openshaw1984Modifiable_C5_EVAL,
  title = {The Modifiable Areal Unit Problem},
  author = {Openshaw, Stan},
  year = {1984},
  series = {Concepts and Techniques in Modern Geography (CATMOG)},
  number = {38},
  publisher = {Geo Books},
  address = {Norwich}
}
%spatial splits, 
@article{roberts2017Cross_C5_EVAL,
  title = {Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure},
  author = {Roberts, David R. and Bahn, Volker and Ciuti, Simone and Boyce, and others},
  year = {2017},
  journal = {Ecography},
  volume = {40},
  number = {8},
  pages = {913--929},
  doi = {10.1111/ecog.02881}
}
%VIF
@article{obrien2007Caution_C5_EVAL,
  title = {A Caution Regarding Rules of Thumb for Variance Inflation Factors},
  author = {O'Brien, Robert M.},
  year = {2007},
  journal = {Quality \& Quantity},
  volume = {41},
  number = {5},
  pages = {673--690},
  doi = {10.1007/s11135-006-9018-6}
}

@book{pinheiro2000Mixed_C5_EVAL,
  title     = {Mixed-Effects Models in {{S}} and {{S-PLUS}}},
  author    = {Pinheiro, Jos{\'e} C. and Bates, Douglas M.},
  year      = {2000},
  publisher = {Springer},
  address   = {New York},
  doi       = {10.1007/b98882}
}

% ==============================================================================
% 6. GWR and GNNWR Spatial Models (_C6_GWR)
% ==============================================================================

% TO ADD:
% - Brunsdon, Fotheringham and Charlton (1996) — Geographically weighted regression.
% - Du et al. (2020) — Geographically neural network weighted regression.
% - Yin et al. (2024) — GNNWR: an open-source package.

@article{du2020Geographically_C6_GWR,
  title   = {Geographically Neural Network Weighted Regression for the Accurate Estimation of Spatial Non-Stationarity},
  author  = {Du, Zhenhong and Wang, Zhongyi and Wu, Sensen and Zhang, Feng and Liu, Renyi},
  year    = {2020},
  journal = {International Journal of Geographical Information Science},
  volume  = {34},
  number  = {7},
  pages   = {1353--1377},
  doi     = {10.1080/13658816.2019.1707834}
}
% ==============================================================================
% 7. Residual Attribution & Interpretable ML (SHAP, ALE) (_C7_XAI)
% ==============================================================================

% TO ADD:
% - Lundberg and Lee (2017) — A Unified Approach to Interpreting Model Predictions (SHAP).
% - Apley and Zhu (2020) — Visualizing the effects of predictor variables in black box supervised learning models (ALE).
@inproceedings{lundberg2017Unified_C7_XAI,
  title     = {A Unified Approach to Interpreting Model Predictions},
  author    = {Lundberg, Scott M. and Lee, Su-In},
  year      = {2017},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS 2017)},
  volume    = {30},
  pages     = {4765--4774}
}

% ==============================================================================
% 8. Future Work & General Extensions (_C8_FW)
% ==============================================================================

% TO ADD:
% - (Add papers related to broader future directions not covered by Cat 1)






# OLD citaitons below need to be moved to the correct category with correct naming convention, and format. 
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
