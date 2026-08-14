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
% - Farrelly et al. (2011) — Sitka spruce site index in response to varying soil moisture and nutrients.
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

% -- Merged from the old markdown citation list (see merge log at end of file) --

@article{manso2022Diameter_C1_SS,
  title = {Diameter, Height and Volume Increment Single Tree Models for Improved {{Sitka}} Spruce in {{Great Britain}}},
  author = {Manso, R. and Davidson, R. and McLean, J. P.},
  year = {2022},
  journal = {Forestry: An International Journal of Forest Research},
  volume = {95},
  number = {3},
  pages = {391--404},
  doi = {10.1093/forestry/cpab049}
}

@techreport{morison2010Understanding_C1_SS,
  title = {Understanding the Growth of {{Sitka}} Spruce: Soil Water Deficit Is the Primary Growth Limiter},
  author = {Morison, J. and others},
  year = {2010},
  institution = {Forest Research},
  note = {INCOMPLETE METADATA -- only "Morison J, et al." was recorded, full author list and report number/series not yet confirmed. Verify before citing.}
}

@article{telewski2006Unified_C1_SS,
  title = {A Unified Hypothesis of Mechanoperception in Plants},
  author = {Telewski, F. W.},
  year = {2006},
  journal = {American Journal of Botany},
  volume = {93},
  number = {10},
  pages = {1466--1476}
}

@article{blyth1981Significance_C1_SS,
  title = {The Significance of Soil Nutrient Status and Site Factors in Determining the Site Index of {{Sitka}} Spruce in Northeast {{Scotland}}},
  author = {Blyth, J. F. and MacLeod, D. A.},
  year = {1981},
  journal = {Journal of Soil Science},
  volume = {32},
  pages = {93--105}
}

@misc{todo2022natureCommsGrowthDrivers_C1_SS,
  title = {[TITLE UNCONFIRMED -- environmental drivers of global forest growth variation]},
  year = {2022},
  journal = {Nature Communications},
  doi = {10.1038/s41467-022-29838-9},
  note = {INCOMPLETE METADATA -- author(s) and exact title not recorded in the source note, only the DOI. Confirm which environmental drivers this paper identifies as dominant before citing -- relevant to Ch. 2.3 (factors affecting forest growth) if climate/water availability is primary.}
}
% Note: "Forest Research (2025) -- Sitka spruce ecology and management" from the old list is the
% same resource as ForestResearch2026Sitka_C1_SS above (same website/species page, cited a year
% earlier) -- not duplicated as a separate entry.

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
%rq3 heith growth chaing ymax
@article{socha2021Regional_C2_CR,
  title   = {Regional Height Growth Models for {{Scots}} Pine in {{Poland}}},
  author  = {Socha, Jaros{\l}aw and Tymi{\'n}ska-Czaba{\'n}ska, Luiza and Bronisz, Karol and Zi{\k{e}}ba, Stanis{\l}aw and Hawry{\l}o, Pawe{\l}},
  year    = {2021},
  journal = {Scientific Reports},
  volume  = {11},
  pages   = {10330},
  doi     = {10.1038/s41598-021-89826-9}
}

% -- Merged from the old markdown citation list (see merge log at end of file) --

@article{pienaar1973Chapmanrichards_C2_CR,
  title = {The {{Chapman-Richards}} Generalization of {{Von Bertalanffy's}} Growth Model for Basal Area Growth and Yield in Even-Aged Stands},
  author = {Pienaar, L. V. and Turnbull, K. J.},
  year = {1973},
  journal = {Forest Science},
  volume = {19},
  number = {1},
  pages = {2--22}
}

@misc{todo2022chapmanRichardsPineMexico_C2_CR,
  title = {A Dynamical Model Based on the {{Chapman-Richards}} Growth Equation for Fitting Growth Curves for Four Pine Species in {{Northern Mexico}}},
  year = {2022},
  url = {https://www.researchgate.net/publication/365200957},
  note = {INCOMPLETE METADATA -- authors and journal name not recorded in the source note, only title/year/URL. Relevant point (once confirmed): shows the CR asymptote and growth-rate parameters can be reduced to a single site-specific parameter -- direct precedent for the Env-PINN's design decision to let y_max vary while keeping k/p global. Verify before citing.}
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
%--------- Check below citaons and doi links, find last one,
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
%--------- Check above citaons and doi links, find last one,

% -- Merged from the old markdown citation list (see merge log at end of file) --

@article{tompalski2021Estimating_C3_LID,
  title = {Estimating Changes in Forest Attributes and Enhancing Growth Projections: A Review of Existing Approaches and Future Directions Using Airborne {{3D}} Point Cloud Data},
  author = {Tompalski, P. and Coops, N. C. and White, J. C. and Goodbody, T. R. H. and Hennigar, C. R. and Wulder, M. A. and Socha, J. and Woods, M. E.},
  year = {2021},
  journal = {Current Forestry Reports},
  volume = {7},
  number = {1},
  pages = {1--24},
  doi = {10.1007/s40725-021-00135-w}
}

@article{schwartz2025Retrieving_C3_LID,
  title = {Retrieving Yearly Forest Growth from Satellite Data: A Deep Learning Based Approach},
  author = {Schwartz, M. and Ciais, P. and Sean, E. and others},
  year = {2025},
  journal = {Remote Sensing of Environment},
  volume = {330},
  pages = {114959},
  doi = {10.1016/j.rse.2025.114959},
  note = {Check whether they handle irregular temporal gaps -- would make this more directly comparable to this project's six-timestamp, unevenly-spaced LiDAR data.}
}

@misc{todo2025frontiersRemoteSensingLidar_C3_LID,
  title = {[TITLE UNCONFIRMED -- LiDAR-based forest attribute mapping]},
  year = {2025},
  journal = {Frontiers in Remote Sensing},
  doi = {10.3389/frsen.2025.1531097},
  note = {INCOMPLETE METADATA -- only journal/year/DOI recorded, author(s) and exact title not confirmed. Verify before citing.}
}

@misc{todo2019remoteSensingForestInventory_C3_LID,
  title = {[TITLE UNCONFIRMED -- remote sensing for forest inventory]},
  year = {2019},
  journal = {Remote Sensing},
  volume = {11},
  number = {20},
  pages = {2407},
  doi = {10.3390/rs11202407},
  note = {INCOMPLETE METADATA -- only journal/volume/issue/year/DOI recorded, author(s) and exact title not confirmed. Verify before citing.}
}

@misc{todo2020biorxivLidarTimeSeries_C3_LID,
  title = {[TITLE UNCONFIRMED -- LiDAR time series for forest growth monitoring]},
  year = {2020},
  url = {https://www.biorxiv.org/content/10.1101/2020.11.13.382515v2.full.pdf},
  note = {INCOMPLETE METADATA -- preprint, author(s)/title not confirmed. Check whether this is now published in a peer-reviewed journal before citing.}
}

% ==============================================================================
% 4. DNNs, PINNs & Hybrid Ecological Models (_C4_NN)
% ==============================================================================
% TO ADD:
% - Willard et al. (2022) — Integrating scientific knowledge with machine learning.
% - Wang, Teng and Perdikaris (2021) — Understanding and mitigating gradient flow pathologies.
% - Rathore et al. (2024) — Challenges in training PINNs: A loss landscape perspective.

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
%choosing set for xgboost? not sure need to check 
@inproceedings{chen2016XGBoost_C4_NN,
  title     = {{{XGBoost}}: A Scalable Tree Boosting System},
  author    = {Chen, Tianqi and Guestrin, Carlos},
  year      = {2016},
  booktitle = {Proceedings of the 22nd {{ACM}} {{SIGKDD}} International Conference on Knowledge Discovery and Data Mining},
  pages     = {785--794},
  doi       = {10.1145/2939672.2939785}
}

% -- Merged from the old markdown citation list (see merge log at end of file) --

@article{raissi2019Physicsinformed_C4_NN,
  title = {Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations},
  author = {Raissi, M. and Perdikaris, P. and Karniadakis, G. E.},
  year = {2019},
  journal = {Journal of Computational Physics},
  volume = {378},
  pages = {686--707}
}

@article{karniadakis2021Physicsinformed_C4_NN,
  title = {Physics-Informed Machine Learning},
  author = {Karniadakis, G. E. and Kevrekidis, I. G. and Lu, L. and Perdikaris, P. and Wang, S. and Yang, L.},
  year = {2021},
  journal = {Nature Reviews Physics},
  volume = {3},
  pages = {422--440}
}

@misc{raissi2017Physics_C4_NN,
  title = {Physics Informed Deep Learning (Part {{I}}): Data-Driven Solutions of Nonlinear Partial Differential Equations},
  author = {Raissi, M. and Perdikaris, P. and Karniadakis, G. E.},
  year = {2017},
  eprint = {1711.10561},
  archiveprefix = {arXiv},
  note = {Preprint predecessor to raissi2019Physicsinformed_C4_NN (the published JCP version) -- cite the published version unless specifically referencing the original framework introduction.}
}

@misc{shi2026AgriPINN_C4_NN,
  title = {{{AgriPINN}}: A Process-Informed Neural Network for Interpretable and Scalable Crop Biomass Prediction under Water Stress},
  author = {Shi, Y. and Han, L. and Zhang, X. and Sobeih, T. and Srivastava, A. K. and others},
  year = {2026},
  eprint = {2601.16045},
  archiveprefix = {arXiv}
}

@article{zhang2023Machine_C4_NN,
  title = {Machine Learning versus Crop Growth Models: An Ally, Not a Rival},
  author = {Zhang, N. and Zhou, X. and Kang, M. and Hu, B. G. and Heuvelink, E. and Marcelis, L. F. M.},
  year = {2023},
  journal = {AoB PLANTS},
  volume = {15},
  number = {2},
  pages = {plac061},
  doi = {10.1093/aobpla/plac061}
}

@article{wesselkamp2024Processinformed_C4_NN,
  title = {Process-Informed Neural Networks: A Hybrid Modelling Approach to Improve Predictive Performance and Inference of Neural Networks in Ecology and Beyond},
  author = {Wesselkamp, M. and Moser, N. and Kalweit, M. and Boedecker, J. and Dormann, C. F.},
  year = {2024},
  journal = {Ecology Letters},
  volume = {27},
  number = {11},
  pages = {e70012},
  doi = {10.1111/ele.70012}
}

@misc{habenicht2026Evaluating_C4_NN,
  title = {Evaluating Transferability and Robustness of Process-Guided Neural Networks in Forest Carbon Flux Modelling},
  author = {Habenicht, H. and Raum, H. and Boedecker, J. and Dormann, C. F.},
  year = {2026},
  doi = {10.64898/2026.02.24.707715},
  note = {bioRxiv 2026.02.24.707715 -- preprint, not yet peer-reviewed. Same Freiburg group as wesselkamp2024Processinformed_C4_NN.}
}

@article{jin2026Knowledgeguided_C4_NN,
  title = {Knowledge-Guided Machine Learning for Global Change Ecology Research},
  author = {Jin, Z. and Liu, L. and Yang, Q. and Jia, X. and Tao, S. and Guo, Y. and Ghosh, R. and Wang, S. and Zhu, Q. and Jung, M. and Guan, K. and Kumar, V. and Reichstein, M. and Fang, J. and Luo, Y.},
  year = {2026},
  journal = {Global Change Biology},
  volume = {32},
  number = {2},
  pages = {e70742},
  doi = {10.1111/gcb.70742}
}

@article{qin2023Hybrid_C4_NN,
  title = {{{3PG-MT-LSTM}}: A Hybrid Model under Biomass Compatibility Constraints for the Prediction of Long-Term Forest Growth to Support Sustainable Management},
  author = {Qin, J. and Ma, M. and Zhu, Y. and Wu, B. and Su, X.},
  year = {2023},
  journal = {Forests},
  volume = {14},
  number = {7},
  pages = {1482},
  doi = {10.3390/f14071482}
}

@article{batuwattagamage2022Physicsinformed_C4_NN,
  title = {A Physics-Informed Neural Network-Based Surrogate Framework to Predict Moisture Concentration and Shrinkage of a Plant Cell during Drying},
  author = {Batuwatta-Gamage, C. P. and Rathnayaka, C. M. and Karunasena, H. C. P. and Jeong, W. and Karim, M. A. and Gu, Y. T.},
  year = {2022},
  journal = {Journal of Food Engineering},
  volume = {332},
  pages = {111137}
}

@inproceedings{nathaniel2023Above_C4_NN,
  title = {Above Ground Carbon Biomass Estimate with Physics-Informed Deep Network},
  author = {Nathaniel, J. and others},
  year = {2023},
  booktitle = {IEEE International Geoscience and Remote Sensing Symposium (IGARSS)}
}

@misc{wong2022Robustness_C4_NN,
  title = {Robustness of Physics-Informed Neural Networks to Noise in Sensor Data},
  author = {Wong, J. C. and others},
  year = {2022},
  note = {INCOMPLETE METADATA -- arXiv preprint, exact arXiv ID not recorded in the source note. Verify before citing.}
}

@inproceedings{kingma2015Adam_C4_NN,
  title = {Adam: A Method for Stochastic Optimisation},
  author = {Kingma, D. P. and Ba, J.},
  year = {2015},
  booktitle = {International Conference on Learning Representations (ICLR)}
}

@article{breiman2001Random_C4_NN,
  title = {Random Forests},
  author = {Breiman, L.},
  year = {2001},
  journal = {Machine Learning},
  volume = {45},
  pages = {5--32}
}

% ==============================================================================
% 5. Spatial & General Evaluation (Moran's, LISA, NLME, BlockCV) (_C5_EVAL)
% ==============================================================================

% TO ADD:
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
% - Apley and Zhu (2020) — Visualizing the effects of predictor variables in black box supervised learning models (ALE).
@inproceedings{lundberg2017Unified_C7_XAI,
  title     = {A Unified Approach to Interpreting Model Predictions},
  author    = {Lundberg, Scott M. and Lee, Su-In},
  year      = {2017},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS 2017)},
  volume    = {30},
  pages     = {4765--4774}
}

% -- Merged from the old markdown citation list (see merge log at end of file) --

@article{lundberg2020From_C7_XAI,
  title = {From Local Explanations to Global Understanding with Explainable {{AI}} for Trees},
  author = {Lundberg, S. M. and Erion, G. and Chen, H. and others},
  year = {2020},
  journal = {Nature Machine Intelligence},
  volume = {2},
  pages = {56--67},
  note = {Tree-specific SHAP algorithm enabling exact computation for XGBoost -- cite alongside lundberg2017Unified_C7_XAI when using SHAP on tree-based models.}
}

% ==============================================================================
% 8. Future Work & General Extensions (_C8_FW)
% ==============================================================================

% TO ADD:
% - (Add papers related to broader future directions not covered by Cat 1)

% -- Merged from the old markdown citation list (see merge log at end of file) --
% This one could not be confidently placed in any category above -- content was never
% confirmed in the old list ("[Recent ML/forestry paper]" placeholder), so it's parked here
% rather than guessed into a category. Re-categorize once read.
@misc{todo2025arxivUnidentified_C8_FW,
  title = {[TITLE/CONTENT UNCONFIRMED]},
  year = {2025},
  eprint = {2509.18228},
  archiveprefix = {arXiv},
  note = {INCOMPLETE METADATA -- content never confirmed in the source note. If it covers SHAP/feature importance for forest growth attribution, move to _C7_XAI. If it covers PINN applications in ecology, move to _C4_NN alongside shi2026AgriPINN_C4_NN.}
}

% ==============================================================================
% MERGE LOG
% ==============================================================================
% 2026-08-14 -- merged all "OLD" markdown-format citations (the list that used to follow this
% point in the file) into the categorized BibTeX structure above, per the note that used to sit
% here ("OLD citations below need to be moved to the correct category with correct naming
% convention, and format"). The old markdown block has been removed now that everything in it
% lives above in its proper category.
%
% Duplicates found and NOT re-added (already present above before this merge):
%   - Socha et al. (2021)              -> socha2021Regional_C2_CR
%   - Pichler & K\"aber (2025)         -> pichler2025Inferring_C4_NN
%   - Lundberg & Lee (2017), SHAP      -> lundberg2017Unified_C7_XAI
%   - Worrell (1987)                   -> worrell1987Predicting_C1_SS
%   - Forest Research Sitka species page (cited 2025) -- same resource as ForestResearch2026Sitka_C1_SS
%
% Entries added with INCOMPLETE metadata (source notes only had partial info -- verify against
% the DOI/URL before citing in the actual chapters):
%   - todo2022natureCommsGrowthDrivers_C1_SS   (Nature Communications, DOI only, author/title unknown)
%   - todo2022chapmanRichardsPineMexico_C2_CR  (ResearchGate URL only, author/journal unknown)
%   - todo2025frontiersRemoteSensingLidar_C3_LID (DOI only, author/title unknown)
%   - todo2019remoteSensingForestInventory_C3_LID (DOI only, author/title unknown)
%   - todo2020biorxivLidarTimeSeries_C3_LID    (preprint URL only, author/title unknown, check for a published version)
%   - morison2010Understanding_C1_SS           (first author + "et al." only)
%   - wong2022Robustness_C4_NN                 (first author + "et al." only, no arXiv ID recorded)
%   - todo2025arxivUnidentified_C8_FW          (arXiv ID only -- content was never actually read/confirmed, parked in Cat 8 pending review)
%
% None of these `todo*`/incomplete entries should be cited in the actual dissertation text until
% verified -- they're kept as placeholders so the underlying DOI/URL isn't lost, not as citable
% references yet.
