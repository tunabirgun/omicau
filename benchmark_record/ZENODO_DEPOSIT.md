# Zenodo application fields

Resource type: Publication

Publication subtype: Working paper

Title: omicau benchmark protocol: a leakage-safe, compute-efficient evaluation of multi-omics audit and fusion workflows

Creators:

1. Given names: Tuna
   Family name: Birgün
   Display name: Tuna Birgün
   ORCID: 0009-0009-1827-7933
   ORCID URL: https://orcid.org/0009-0009-1827-7933
   Affiliation 1: Istanbul Technical University, Graduate School, Department of Molecular Biology - Genetics and Biotechnology, Türkiye
   Affiliation 2: Istanbul Yeni Yuzyil University, Faculty of Sciences and Literature, Department of Molecular Biology and Genetics, Türkiye

Description:

This working paper and protocol prospectively specifies an independent, compute-efficient benchmark of omicau, a multi-omics audit and fusion workflow. Simulated datasets are independent experimental units evaluated with one frozen five-fold cross-validation partition. Real cohorts use five-fold, three-repeat group-aware outer cross-validation and three-fold inner cross-validation for tuning. External-method comparisons are restricted to a prespecified simulation subset, and explainability analysis is disabled except for one prespecified case study. Null controls are treated as pipeline sanity checks; group leakage is assessed through the grouping warning and the naive-versus-group-aware performance gap; batch and missingness outputs are interpreted as risk flags rather than causal proof. TCGA-BRCA PAM50 is included only as a secondary positive control, while primary real-cohort endpoints are externally defined. All analyses run before this protocol is frozen and archived are pilot-only and cannot support confirmatory claims. Definitive simulation seeds must pass a zero-overlap audit against retained pilot seeds. Primary interval-coverage claims are outside scope.

Version: 1.0.0

Language: English

Visibility: Public (open access)

License: Creative Commons Attribution 4.0 International (CC BY 4.0)

Keywords:

- multi-omics
- benchmark protocol
- leakage-safe machine learning
- group-aware cross-validation
- reproducible research
- simulation study
- bioinformatics
- model auditing

Related software: https://github.com/tunabirgun/omicau

Additional notes:

Archive SHA-256: {{ARCHIVE_SHA256}}

The uploaded ZIP is the prospective protocol record only. It contains no definitive results, pilot outputs, raw cohort data, subject identifiers, local paths, or machine-specific environment capture. Verify the archive with the accompanying `SHA256SUMS.txt` retained in the local delivery set.
