# Datasets used by Jersey Futures Engine

Generated from `data/manifests/sources.json` and data pack `jfe-pack-1-f11d2e080499` (built 2026-10-03T16:10:26Z). Every machine-readable file is downloaded by `python -m jfe.ingest` and refused if its SHA-256 differs from the pinned value. Nothing is fetched at runtime.

## Machine-readable sources (inputs to the data pack)

| ID | Dataset | Publisher | Resource ID | Vintage | Licence | Used for | Bytes | SHA-256 |
|---|---|---|---|---|---|---|---:|---|
| S06 | [Annual population estimates by age and sex (2011 to present)](https://opendata.gov.je/dataset/199794de-4927-457a-9fc4-569c0d2f7b47/resource/4f0988d8-c5d2-4bff-af72-0a314b95024a/download/annual-population-estimates-by-age-and-sex.csv) | Statistics Jersey via opendata.gov.je | `4f0988d8-c5d2-4bff-af72-0a314b95024a` | sj_population_2026_09 | OGL-J-1.0 | Observed end-2025 baseline (rebased model); observed history 2011-2025; LTC rate denominators (2024) | 25172 | `3bd5ec5dd1a054c9f450485a289672a826da11701933be16710f1fa5b28babba` |
| S07-nil | [Population projections 2025 to 2080 (net nil)](https://opendata.gov.je/dataset/381acb20-cf53-42c1-a011-7b5faea43fbe/resource/234a7695-42af-46d4-ad46-7d226a78ff32/download/population-projections-net-nil-open-data.csv) | Statistics Jersey via opendata.gov.je | `234a7695-42af-46d4-ad46-7d226a78ff32` | sj_projection_2026_02 | OGL-J-1.0 | Official reference; net-nil migration age profile; fertility-shape training | 9881915 | `0223a13e10bebc618cb15de2ea3fb5392cba6316d0731c0ffd11e6370fa2b47a` |
| S07-200 | [Population projections 2025 to 2080 (+200)](https://opendata.gov.je/dataset/381acb20-cf53-42c1-a011-7b5faea43fbe/resource/6e222cd7-d296-429a-abea-09001dcc45f6/download/population-projections-200-open-data.csv) | Statistics Jersey via opendata.gov.je | `6e222cd7-d296-429a-abea-09001dcc45f6` | sj_projection_2026_02 | OGL-J-1.0 | Official reference; held-out test for migration linearity and fertility shape | 9883731 | `6076a3b072e770b6116bd214e7a69c7d320b14222e11cc250538c446ddcdcd1e` |
| S07 | [Population projections 2025 to 2080 (+400)](https://opendata.gov.je/dataset/381acb20-cf53-42c1-a011-7b5faea43fbe/resource/90a0097c-e045-4bf3-83d8-8320ffe180aa/download/population-projections-400-open-data.csv) | Statistics Jersey via opendata.gov.je | `90a0097c-e045-4bf3-83d8-8320ffe180aa` | sj_projection_2026_02 | OGL-J-1.0 | Official mid-range reference; mortality probabilities; migration per-person profile; fertility-shape training | 9885170 | `0d0ac9268aa9d04df7e8925de26cc3eeb21a5679ff9cd6d190906764b19a4653` |
| S07-600 | [Population projections 2025 to 2080 (+600)](https://opendata.gov.je/dataset/381acb20-cf53-42c1-a011-7b5faea43fbe/resource/604b1098-ee19-4e4e-9011-9dd417114b30/download/population-projections-600-open-data.csv) | Statistics Jersey via opendata.gov.je | `604b1098-ee19-4e4e-9011-9dd417114b30` | sj_projection_2026_02 | OGL-J-1.0 | Official reference; held-out test for migration linearity and fertility shape | 9886578 | `40b6e718572ce86d3c63e04ef23175f6e28db821e861bac0bbffc200208f059f` |
| S07-800 | [Population projections 2025 to 2080 (+800)](https://opendata.gov.je/dataset/381acb20-cf53-42c1-a011-7b5faea43fbe/resource/fe77d304-15f3-421b-968d-fc469a8b703e/download/population-projections-800-open-data.csv) | Statistics Jersey via opendata.gov.je | `fe77d304-15f3-421b-968d-fc469a8b703e` | sj_projection_2026_02 | OGL-J-1.0 | Official reference; fertility-shape training | 9886766 | `793e7e0de9ffef319a045f95853af24dd8b1a38bfaa6c4827369e53a336290e2` |
| S08 | [Long-Term Care Demographics (claims at year end)](https://opendata.gov.je/dataset/a92bf540-a7ec-469f-bf67-8935ecbc59c1/resource/55d18461-1433-480c-bab4-a7b4c6947e5e/download/long-term_care_demographics.csv) | Government of Jersey (Customer and Local Services) via opendata.gov.je | `55d18461-1433-480c-bab4-a7b4c6947e5e` | ltc_claims_2024 | OGL-J-1.0 | 2024 open LTC claims by 65+ band and sex -> constant claim rates for the claim-pressure index | 4737 | `27b5e9cc292527b8a8816b8da3d56e7f4dc1e7d2f9cc45515efa256bdbdc2c1a` |
| S35 | [Total population, annual change, natural growth and net migration per year (2000-2025)](https://opendata.gov.je/dataset/600fed19-46d6-453a-aa17-2a7bf111f07d/resource/75c9a5ed-e95e-4052-a7c8-96d5dab8a64a/download/total-population-annual-change-natural-growth-net-migration-per-year.csv) | Statistics Jersey via opendata.gov.je | `75c9a5ed-e95e-4052-a7c8-96d5dab8a64a` | sj_population_2026_09 | OGL-J-1.0 | Retrospective check only: observed births, deaths and net migration 2018-2025 drive the backtest; never used to calibrate scenario parameters | 3704 | `dad597e524c95dcf8730694f5be4a367699f434495b1b70682fe90358a250c8b` |
| S36 | [Education: student numbers by school type (2011-2024)](https://opendata.gov.je/dataset/e9acb214-5778-4a0a-a8b7-2622c51a0a9e/resource/7d16ab4d-e0ff-4b59-bec7-b45db60ea48a/download/total-students-by-school-type.csv) | Government of Jersey (Children, Young People, Education and Skills) via opendata.gov.je | `7d16ab4d-e0ff-4b59-bec7-b45db60ea48a` | school_census_2024 | OGL-J-1.0 | Pupil participation ratios (primary; secondary incl. sixth form) = January pupils / school-age residents at the preceding year end; held at 2024 for the pupil proxy; 2012-2024 used to check ratio stability | 618 | `922812ce877bb573a6e5164c11d80d30a274ced1c6a571e2d3c0076497fea35e` |
| S39 | [Housing needs projections 2025-2040](https://opendata.gov.je/dataset/5f550e45-93e9-4c00-9445-4b6d82c6b027/resource/ead0eb40-523c-4310-8949-4691abcdfcc6/download/housing-needs-projections.csv) | Statistics Jersey via opendata.gov.je | `ead0eb40-523c-4310-8949-4691abcdfcc6` | sj_projection_2026_02 | OGL-J-1.0 | Homes needed by dwelling type for any net migration: exact decomposition H = H_nil + net x H_per_person (linear across the five published scenarios), 2026-2040 | 78691 | `bd4b03bd7089df8fd23be3cbe6d87535f8db67201b1641711c9466450b2dc561` |
| S42 | [Population projections 2023 to 2080 (previous official round)](https://opendata.gov.je/dataset/381acb20-cf53-42c1-a011-7b5faea43fbe/resource/55484fdf-7547-4099-ad5e-e8115656c087/download/population-projections-2023-to-2080.csv) | Statistics Jersey via opendata.gov.je | `55484fdf-7547-4099-ad5e-e8115656c087` | sj_projection_2023 | OGL-J-1.0 | Vintage check only: previous round's +325 projection for end-2023 to end-2025 versus observed estimates | 2459002 | `a378387ab0db09374ade1863dd227352450e2776317851c22be7d40ede02dcec` |

## Documents (definitions, assumptions and one transcribed table)

| ID | Document | Used for | SHA-256 |
|---|---|---|---|
| S34 | [Births and Breastfeeding Profile 2025 (Public Health Intelligence, Government of Jersey, 12 March 2026)](https://www.gov.je/SiteCollectionDocuments/Health%20and%20wellbeing/Births%20and%20Breastfeeding%20Profile%202025.pdf) | Fertility age-shape: Table 2, page 11 (2023-2025 pooled women and births by 5-year age of mother; TFR 1.14). Values transcribed below and verified against the PDF text. | `b19e967cba4f5c6ee1de1ad8e4e54306da7007562ea6ac5d4f5278701ed79852` |

Values transcribed from S34: `{"bands": ["15-19", "20-24", "25-29", "30-34", "35-39", "40-44"], "women": [7930, 7350, 8450, 10130, 10790, 11810], "births": [20, 116, 443, 895, 591, 158], "notes": "15-19 includes a small number of births to mothers under 15; 40-44 includes births at 45+. Published TFR 1.14."}`

| S02 | [Population and migration 2025 - total population (Sept 2026)](https://stats.je/publication/population-and-migration-2025-total-population/) | Opening context: 104,490 provisional end-2025 estimate | `not archived` |
| S03 | [Population projections 2025-2080](https://stats.je/publication/population-projections-2025-2080/) | Official projection context | `not archived` |
| S04 | [Population and household projections methodology (Feb 2026)](https://stats.je/wp-content/uploads/2026/02/R-Population-And-Household-Projections-Methodology-20260225.pdf) | Event order (ageing, deaths, births, emigration, immigration); assumption codes; TFR 1.01 by 2034 (used only as an independent check) | `not archived` |
| S28 | [Population projections report 2025-2080 (Feb 2026)](https://stats.je/wp-content/uploads/2026/02/R-Population-Projections-2025-2080-20260225.pdf) | 16-64 working-age convention; headline results | `not archived` |
| S37 | [Disease Projection Report 2023-2053 (Public Health Intelligence, Government of Jersey, 8 Feb 2024)](https://www.gov.je/SiteCollectionDocuments/Government%20and%20administration/Disease%20Projection%20Report%202023%20to%202053.pdf) | Age/sex weights: Fig. 39 p.25 hospital bed days per person (2018-22, acute medical/surgical/women and children, PAS); Fig. 4 p.6 GP appointments per person (2021-23, surgery appointments). Applied to Sept 2026 estimates they reproduce the report's 2023 totals within 0.3% (bed days 60,488 vs 60,300) and 0.8% (GP 413,949 vs 410,790). | `e27de53a88a95028e8fe1417068d62f275d179f235885b973d2eac12043ae590` |

Values transcribed from S37: `{"bands": ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80-84", "85-89", "90+"], "bed_days_F": [0.9, 0.0, 0.1, 0.2, 0.2, 0.3, 0.4, 0.3, 0.2, 0.2, 0.3, 0.3, 0.4, 0.7, 1.2, 1.8, 2.8, 4.6, 6.0], "bed_days_M": [1.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.3, 0.5, 0.7, 1.3, 1.9, 3.1, 4.3, 6.4], "gp_appointments_F": [4.6, 2.2, 1.5, 2.6, 3.2, 4.2, 4.8, 4.9, 4.3, 4.5, 4.8, 4.8, 5.1, 5.3, 5.8, 6.7, 7.5, 8.2, 8.5], "gp_appointments_M": [5.0, 2.2, 1.4, 1.6, 1.8, 1.9, 2.0, 2.2, 2.5, 2.7, 3.2, 3.7, 4.5, 4.9, 5.6, 6.4, 8.0, 9.1, 10.1], "estimated_from_bar_height": ["bed_days_F 25-29", "bed_days_F 30-34", "bed_days_F 35-39", "bed_days_F 60-64", "bed_days_F 65-69", "bed_days_M 55-59", "bed_days_M 60-64", "bed_days_M 65-69"], "notes": "Values transcribed from the PDF text layer; bars without labels estimated from pixel height (+/-0.1)."}`

| S38 | [Public Sector Staffing Statistics, 31 December 2025 (Government of Jersey)](https://www.gov.je/SiteCollectionDocuments/Government%20and%20administration/Public%20Sector%20Staffing%20Statistics%2031%20December%202025.pdf) | Workforce baseline, Table 1 pp.3-4: Health and Care Jersey FTE at 31 Dec 2025 - nurses and midwives 1,112; doctors and consultants 222; junior and trainee doctors 50. Employed public staff only (excludes agency, GPs, private sector). | `75cc62f3a30d7b40019b4c1abd002a7ed9e45c14cdf27c8b4b62b666d8bd551a` |

Values transcribed from S38: `{"nurses_midwives_fte": 1112, "doctors_fte": 272, "date": "2025-12-31"}`

| S41 | [Data on schools, pupils and pupil characteristics as of 21 September 2026 (Government of Jersey, CYPES)](https://www.gov.je/SiteCollectionDocuments/Education/Data%20on%20schools%2c%20pupils%20and%20pupil%20characteristics%20as%20of%2021%20September%202026.pdf) | Validation of school year-group weighting: pupils by year group, spring 2024/25 (p.5), summing to the p.4 total of 14,325 | `7d1a3f5fa3138e3deaf61a4e5120a29cbc02f9c5800f20fc15e3db4e6a2e326e` |

Values transcribed from S41: `{"school_year": "2024/25", "year_groups": ["Reception", "Y1", "Y2", "Y3", "Y4", "Y5", "Y6", "Y7", "Y8", "Y9", "Y10", "Y11", "Y12", "Y13"], "pupils": [841, 896, 954, 973, 1061, 994, 1040, 1066, 1165, 1059, 996, 1008, 953, 787], "nursery": 532, "total": 14325, "notes": "Chart labels transcribed from the PDF text layer; all settings excluding private nurseries and elective home education; Highlands Y12-13 partly included."}`


## How each dataset enters the model

| Parameter | Provenance class | Derivation |
|---|---|---|
| Baseline population end-2025 | published (+3 imputed cells) | S06 single-year age/sex counts; suppressed male 98/99/100+ cells filled from the official 2025 projected counts (labelled modelling assumption) |
| Death probabilities by year/age/sex/code | derived | S07 Deaths / population aged forward (event order verified against S04) for each life-expectancy code |
| Net migration age/sex profile | derived | Exact decomposition M = M_nil + net x P from S07-nil and S07 (+400); verified on held-out +200/+600/+800 to 4e-5 people |
| Fertility age pattern | published | S34 Table 2 (2023-25 ASFRs, uniform within 5-year bands) |
| Fertility level by year/code | derived | Scale factor calibrated to official births (S07 cubes net 0/400/800); held-out births error <= 0.71/yr; implied 2034 TFR 1.026 vs published 1.01 |
| Birth sex ratio | derived | S07 births by sex (1.0408 M per F, constant) |
| LTC claim rates 2024 | published / derived | S08 2024 open claims by 65+ band and sex / S06 2024 population (same vintage) |
| School participation ratios | derived | S36 January pupils / school-age residents (year-group weights); 2024 ratio held constant |
| Hospital bed days and GP appointments | published rates (transcribed) | S37 Figs. 39 and 4 per-person rates by 5-year age band and sex |
| Hospital nurse and doctor FTE proxies | published baseline (transcribed) | S38 Table 1 Health and Care Jersey FTE at 31 Dec 2025, held constant per age-weighted bed day |
| Homes needed by dwelling type | derived | S39 exact decomposition H = H_nil + net x H_per_person per year and type (verified on held-out +200/+600/+800); change since 2025 |
| Year-group and vintage checks | published | S41 pupils by year group 2024/25; S42 previous-round projection (validation only) |
| Observed history and backtest inputs | published | S06 estimates 2011-2025; S35 births, deaths, net migration 2018-2025 (retrospective check only) |

## Data-quality checks (all must pass for the pack to build)

- PASS: total accounting net=0
- PASS: total accounting net=200
- PASS: total accounting net=400
- PASS: total accounting net=600
- PASS: total accounting net=800
- PASS: death probabilities in [0,1]
- PASS: mortality independent of fertility code
- PASS: migration linear in net (held-out +200/+600/+800)
- PASS: per-person migration profile sums to 1
- PASS: fertility shape predicts held-out scenario births
- PASS: birth sex ratio constant
- PASS: estimates cover 2011-2025
- PASS: 2024 LTC claims for 65+ bands are published (not suppressed)
- PASS: components of change available 2012-2025
- PASS: school participation ratios computed
- PASS: bed-day weights reproduce PHI 2023 total (60,300) within 2%
- PASS: GP weights reproduce PHI 2021-23 total (410,790) within 2%
- PASS: homes needed linear in net migration (held-out +200/+600/+800)
