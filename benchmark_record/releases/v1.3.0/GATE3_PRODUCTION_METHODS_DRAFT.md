# Gate 3 production methods draft

Canonical source: `benchmark_record/releases/v1.3.0/gate3_production_methods.draft.json`.

This document is a human-readable field mirror of the canonical JSON. Values are reproduced exactly; arrays retain canonical order. The canonical JSON remains authoritative.

## Identity, status, activation, and cross-claim contract

- `/schema_version`: `"gate3_production_methods_draft_1"`
- `/protocol_version`: `"1.3.0"`
- `/status`: `"draft_methods_only_not_calibrated_not_authorized"`
- `/zenodo_ready`: `false`
- `/activation/effective`: `false`
- `/activation/freeze_ready`: `false`
- `/activation/methods_complete`: `false`
- `/activation/reason`: `"quantitative calibration, dependency, seed, environment, split, and authoritative scenario bindings remain unresolved"`
- `/authorizations/calibration_execution`: `false`
- `/authorizations/candidate_data_access`: `false`
- `/authorizations/candidate_fitting`: `false`
- `/authorizations/candidate_performance_inspection`: `false`
- `/authorizations/definitive_validation`: `false`
- `/authorizations/feature_outcome_association`: `false`
- `/authorizations/freeze`: `false`
- `/authorizations/manuscript_claim_revision`: `false`
- `/authorizations/publication`: `false`
- `/authorizations/scenario_truth_assignment`: `false`
- `/authorizations/software_release`: `false`
- `/authorizations/zenodo_upload`: `false`
- `/cross_claim_contract/calibration_boundary`: `"only a disjoint G3B prefix may choose quantitative fields and it supplies no editor-facing positive-validation evidence"`
- `/cross_claim_contract/decision_hierarchy`: `"every warning-producing component belongs to exactly one declared primary family and no secondary output may change that primary decision"`
- `/cross_claim_contract/group_contract`: `"the highest exchangeable group receives equal inferential weight and endpoint plus modality-specific batch purity is required unless a distinct preregistered composition estimand exists"`
- `/cross_claim_contract/non_estimable_policy`: `"unexpected eligible non-estimable output fails the relevant claim and no failed or partial coordinate may be selectively replaced"`
- `/cross_claim_contract/operational_alpha_vs_validation_alpha`: `"software warning multiplicity and Gate3 validation confidence bounds are separate bound contracts"`
- `/cross_claim_contract/permutation_contract`: `"whole-group exchangeability blocks are fixed from design variables only and Monte Carlo p-values use the plus-one rule"`
- `/cross_claim_contract/scenario_authority`: `"this artifact assigns no scenario truth expected decision or observed result"`
- `/cross_claim_contract/validation_inference_binding`: `"benchmark_record/releases/v1.3.0/gate3_inference_amendment.draft.json"`

## Claim C03

- `/claims/C03/advertised_scope/claim_language`: `"association between the registered missingness representation and the observed target at the highest exchangeable-group unit"`
- `/claims/C03/advertised_scope/forbidden_language`: `["causal missingness mechanism","independent censoring established","MAR established","MNAR established"]`
- `/claims/C03/advertised_scope/tasks`: `["classification","regression","survival"]`
- `/claims/C03/eligibility/group_contract`: `"one equal-weight highest exchangeable group is one inferential unit"`
- `/claims/C03/eligibility/refusal_codes`: `["c03_endpoint_mixed_within_group","c03_group_id_missing","c03_group_reducer_unregistered","c03_mgc_distance_invalid","c03_primary_nonestimable","c03_survival_event_nonestimable"]`
- `/claims/C03/eligibility/requirements`: `["endpoint is constant within every group","group identifiers are complete","more than one row per group uses a preregistered reducer","survival uses one right-censored record per group","distance matrices are finite symmetric hollow and label aligned"]`
- `/claims/C03/estimand/analysis_unit`: `"highest_exchangeable_group"`
- `/claims/C03/estimand/group_weighting`: `"equal"`
- `/claims/C03/estimand/quantity`: `"population dependence between group-level missingness and the registered observed target"`
- `/claims/C03/method/monte_carlo`: `"exact enumeration when attainable otherwise frozen-B whole-group permutation with plus-one numerator and denominator"`
- `/claims/C03/method/non_survival_primary/categorical_target_distance`: `"delta_distance"`
- `/claims/C03/method/non_survival_primary/continuous_target_distance`: `"absolute_difference_of_normalized_midranks"`
- `/claims/C03/method/non_survival_primary/missingness_distance`: `"normalized_L1_between_per_feature_group_missingness_proportions"`
- `/claims/C03/method/non_survival_primary/statistic`: `"scipy_stats_multiscale_graphcorr_precomputed_distances_compute_distance_none_workers_one"`
- `/claims/C03/method/secondary_localization/aggregate_categorical`: `"whole-group permutation Kruskal-Wallis statistic on group mean missingness burden"`
- `/claims/C03/method/secondary_localization/aggregate_continuous`: `"whole-group permutation absolute Spearman statistic on group mean missingness burden"`
- `/claims/C03/method/secondary_localization/categorical_feature`: `"whole-group permutation Kruskal-Wallis statistic on feature missingness proportions"`
- `/claims/C03/method/secondary_localization/continuous_feature`: `"whole-group permutation absolute Spearman statistic on feature missingness proportions"`
- `/claims/C03/method/secondary_localization/survival_censoring_feature`: `"separate univariate Cox score statistic for observed censoring hazard on feature missingness proportion"`
- `/claims/C03/method/secondary_localization/survival_event_feature`: `"univariate Cox score statistic for observed event hazard on feature missingness proportion"`
- `/claims/C03/method/survival_primary/censoring_component`: `"separate association with the observed censoring hazard and never evidence of independent censoring"`
- `/claims/C03/method/survival_primary/event_component`: `"Cox proportional-hazards score test for group missingness burden with log hazard ratio per frozen one-SD increment"`
- `/claims/C03/method/survival_primary/no_censoring_rule`: `"event component remains eligible and censoring component is not_applicable_no_censoring"`
- `/claims/C03/multiplicity/primary_family`: `"one Holm family across every modality and event_target_censor component allowed to trigger a C03 warning"`
- `/claims/C03/multiplicity/secondary_families`: `["aggregate localization Holm family","feature localization Benjamini-Yekutieli family"]`
- `/claims/C03/multiplicity/secondary_may_trigger_primary`: `false`
- `/claims/C03/oracle/comparison_rule`: `"primary and oracle decisions must agree exactly and statistics must meet a frozen tolerance"`
- `/claims/C03/oracle/independence_rule`: `"oracle constructs distances, permutations, risk sets, and multiplicity adjustments through a separately maintained path"`
- `/claims/C03/oracle/methods`: `["independent distance covariance permutation test","independent group-proportion contrast","independent Cox score and risk-set calculation","independent Holm and Benjamini-Yekutieli adjustment"]`
- `/claims/C03/outputs/forbidden_private_keys`: `["clinical_values","group_ids","labels","local_paths","row_ids","sample_ids","subject_ids"]`
- `/claims/C03/outputs/required_public_keys`: `["claim_id","decision","effect_summary","eligibility_reason","group_count","method_id","modality_count","multiplicity_family_id","oracle_status","permutation_registry_sha256"]`
- `/claims/C03/production_gaps`: `["current implementation uses row-level aggregate burden tests","current target and batch p-values share one pooled correction family","current output contains sample-level missingness material","survival censoring-aware C03 evidence is absent"]`
- `/claims/C03/quantitative_freeze/effect_targets`: `null`
- `/claims/C03/quantitative_freeze/maximum_feature_count`: `null`
- `/claims/C03/quantitative_freeze/minimum_group_count`: `null`
- `/claims/C03/quantitative_freeze/minimum_survival_events`: `null`
- `/claims/C03/quantitative_freeze/multiplicity_alpha_allocation`: `null`
- `/claims/C03/quantitative_freeze/permutation_count`: `null`
- `/claims/C03/quantitative_freeze/statistic_tolerance`: `null`
- `/claims/C03/scenario_truth/authoritative`: `false`
- `/claims/C03/scenario_truth/expected_decisions`: `null`
- `/claims/C03/scenario_truth/source_registry_pointer`: `"/source_templates"`
- `/claims/C03/secondary_evidence`: `["group-level aggregate burden contrasts","feature-level missingness-proportion localization","proportional-hazards diagnostics"]`
- `/claims/C03/watched_failures`: `["apply categorical localization to a continuous or survival endpoint","replace normalized L1 with Hamming on group proportions","reintroduce row-count weighting","use categorical table statistics on fractional group proportions","merge event and censoring decisions","describe association as MAR or MNAR","let secondary localization trigger the primary warning"]`

## Claim C04

- `/claims/C04/advertised_scope/claim_language`: `"association between the registered missingness representation and the recorded batch variable at the highest exchangeable-group unit"`
- `/claims/C04/advertised_scope/forbidden_language`: `["batch caused missingness","outcome-independent batch missingness without a registered supporting design"]`
- `/claims/C04/advertised_scope/tasks`: `["classification","regression","survival"]`
- `/claims/C04/eligibility/group_contract`: `"one equal-weight highest exchangeable group with one modality-specific batch label is one inferential unit"`
- `/claims/C04/eligibility/refusal_codes`: `["c04_batch_mixed_within_group","c04_batch_missing","c04_batch_support_insufficient","c04_group_id_missing","c04_group_reducer_unregistered","c04_mgc_distance_invalid"]`
- `/claims/C04/eligibility/requirements`: `["batch is constant within each group for the tested modality","group identifiers are complete","at least two batches are represented","each batch meets the frozen independent-group support","batch cardinality is inside the calibrated envelope"]`
- `/claims/C04/estimand/analysis_unit`: `"highest_exchangeable_group"`
- `/claims/C04/estimand/group_weighting`: `"equal"`
- `/claims/C04/estimand/quantity`: `"population dependence between group-level missingness and the recorded modality-specific batch variable"`
- `/claims/C04/method/batch_distance`: `"delta_distance"`
- `/claims/C04/method/missingness_distance`: `"normalized_L1_between_per_feature_group_missingness_proportions"`
- `/claims/C04/method/monte_carlo`: `"exact enumeration when attainable otherwise frozen-B whole-group permutation with plus-one numerator and denominator"`
- `/claims/C04/method/primary_statistic`: `"scipy_stats_multiscale_graphcorr_precomputed_distances_compute_distance_none_workers_one"`
- `/claims/C04/method/secondary_localization/aggregate`: `"whole-group permutation rank statistic on group mean missingness burden"`
- `/claims/C04/method/secondary_localization/feature`: `"whole-group permutation Kruskal-Wallis statistic on feature missingness proportions"`
- `/claims/C04/multiplicity/primary_family`: `"one Holm family across all modality omnibus p-values allowed to trigger a C04 warning"`
- `/claims/C04/multiplicity/secondary_families`: `["aggregate localization Holm family","feature localization Benjamini-Yekutieli family"]`
- `/claims/C04/multiplicity/secondary_may_trigger_primary`: `false`
- `/claims/C04/oracle/comparison_rule`: `"primary and oracle decisions must agree exactly and statistics must meet a frozen tolerance"`
- `/claims/C04/oracle/independence_rule`: `"oracle constructs its distance matrix and group permutations independently"`
- `/claims/C04/oracle/methods`: `["independent distance covariance permutation test","independent group-proportion contrast","independent Holm and Benjamini-Yekutieli adjustment"]`
- `/claims/C04/outputs/forbidden_private_keys`: `["batch_labels","group_ids","local_paths","row_ids","sample_ids","subject_ids"]`
- `/claims/C04/outputs/required_public_keys`: `["batch_count","claim_id","decision","effect_summary","eligibility_reason","group_count","method_id","modality_count","multiplicity_family_id","oracle_status","permutation_registry_sha256"]`
- `/claims/C04/production_gaps`: `["current implementation tests row-level aggregate missingness only","current target and batch p-values share one pooled correction family","current implementation has no batch-purity refusal","current implementation lacks pattern-level and feature-level calibrated coverage"]`
- `/claims/C04/quantitative_freeze/effect_targets`: `null`
- `/claims/C04/quantitative_freeze/maximum_batch_cardinality`: `null`
- `/claims/C04/quantitative_freeze/maximum_feature_count`: `null`
- `/claims/C04/quantitative_freeze/minimum_groups_per_batch`: `null`
- `/claims/C04/quantitative_freeze/multiplicity_alpha_allocation`: `null`
- `/claims/C04/quantitative_freeze/permutation_count`: `null`
- `/claims/C04/quantitative_freeze/statistic_tolerance`: `null`
- `/claims/C04/scenario_truth/authoritative`: `false`
- `/claims/C04/scenario_truth/expected_decisions`: `null`
- `/claims/C04/scenario_truth/source_registry_pointer`: `"/source_templates"`
- `/claims/C04/secondary_evidence`: `["group-level aggregate burden contrasts","feature-level missingness-proportion localization"]`
- `/claims/C04/watched_failures`: `["replace normalized L1 with Hamming on group proportions","accept mixed batch labels within a group","reintroduce row-count weighting","merge C03 and C04 primary families","let secondary localization trigger the primary warning","describe association as causal"]`

## Claim C05

- `/claims/C05/advertised_scope/claim_language`: `"batch-associated structure and batch-outcome dependence in the frozen diagnostic representation"`
- `/claims/C05/advertised_scope/forbidden_language`: `["batch caused the outcome","informative censoring established","universal batch effect"]`
- `/claims/C05/advertised_scope/tasks`: `["classification","regression","survival"]`
- `/claims/C05/eligibility/group_contract`: `"one equal-weight highest exchangeable group with pure endpoint and modality-specific batch labels is one inferential unit"`
- `/claims/C05/eligibility/refusal_codes`: `["c05_batch_mixed_within_group","c05_batch_support_insufficient","c05_cox_effect_nonestimable","c05_distance_invalid","c05_endpoint_mixed_within_group","c05_group_reducer_unregistered","c05_primary_nonestimable"]`
- `/claims/C05/eligibility/requirements`: `["endpoint and modality-specific batch are constant within each group","the group value reducer is preregistered","the frozen diagnostic representation is finite and nonconstant","every gate-producing component meets frozen support","survival coding reference convergence separation and PH rules are frozen"]`
- `/claims/C05/estimand/analysis_unit`: `"highest_exchangeable_group"`
- `/claims/C05/estimand/group_weighting`: `"equal"`
- `/claims/C05/estimand/quantity`: `"distributional dependence of the frozen diagnostic representation on batch and dependence of the registered observed endpoint on batch"`
- `/claims/C05/method/monte_carlo`: `"exact enumeration when attainable otherwise frozen-B whole-group permutation with plus-one numerator and denominator"`
- `/claims/C05/method/outcome_components/classification`: `"whole-group fixed-margin permutation Pearson statistic on the group-level batch-by-outcome table"`
- `/claims/C05/method/outcome_components/regression`: `"MGC between group outcome normalized-midrank distance and batch delta distance"`
- `/claims/C05/method/outcome_components/survival_censoring`: `"separate association with the observed censoring hazard"`
- `/claims/C05/method/outcome_components/survival_event`: `"frozen k-sample log-rank score statistic scoped to the registered proportional-hazards alternative"`
- `/claims/C05/method/structure_component/batch_distance`: `"delta_distance"`
- `/claims/C05/method/structure_component/representation_distance`: `"frozen_distance_on_globally_standardized_group_level_diagnostic_values"`
- `/claims/C05/method/structure_component/statistic`: `"scipy_stats_multiscale_graphcorr_precomputed_distances_compute_distance_none_workers_one"`
- `/claims/C05/method/survival_effects`: `"joint unpenalized Cox effect summaries are secondary and require frozen coding global statistic support convergence separation and PH rules"`
- `/claims/C05/multiplicity/primary_family`: `"one Holm family across every modality and structure_outcome_event_censor p-value allowed to trigger any C05 warning"`
- `/claims/C05/multiplicity/secondary_families`: `["location localization Benjamini-Yekutieli family","scale localization Benjamini-Yekutieli family"]`
- `/claims/C05/multiplicity/secondary_may_trigger_primary`: `false`
- `/claims/C05/oracle/comparison_rule`: `"primary and oracle decisions must agree exactly and statistics must meet frozen tolerances"`
- `/claims/C05/oracle/independence_rule`: `"oracle uses separately constructed distances permutations risk sets and adjustment code"`
- `/claims/C05/oracle/methods`: `["independent distance covariance","independent fixed-margin table enumeration","independent log-rank score calculation","PERMANOVA for registered centroid alternatives","PERMDISP for registered scale alternatives","frozen group-CV batch classifier as corroboration only"]`
- `/claims/C05/outputs/forbidden_private_keys`: `["batch_labels","clinical_values","group_ids","local_paths","pca_coordinates","row_ids","sample_ids","subject_ids"]`
- `/claims/C05/outputs/required_public_keys`: `["claim_id","component_decisions","effect_summaries","eligibility_reasons","group_count","method_ids","modality_count","multiplicity_family_id","oracle_status","permutation_registry_sha256"]`
- `/claims/C05/production_gaps`: `["current structure inference emphasizes PC1 and fixed silhouette or eta-squared cutoffs","current survival confounding branch treats observed time as uncensored","current output exposes PCA coordinates","current utility logic can hide a structure-absent outcome-associated factorial cell","current method lacks one explicit family across every warning-producing component"]`
- `/claims/C05/quantitative_freeze/batch_classifier_contract`: `null`
- `/claims/C05/quantitative_freeze/cox_feasibility_and_global_test`: `null`
- `/claims/C05/quantitative_freeze/effect_targets`: `null`
- `/claims/C05/quantitative_freeze/maximum_batch_cardinality`: `null`
- `/claims/C05/quantitative_freeze/minimum_component_support`: `null`
- `/claims/C05/quantitative_freeze/multiplicity_alpha_allocation`: `null`
- `/claims/C05/quantitative_freeze/non_proportional_hazards_component`: `null`
- `/claims/C05/quantitative_freeze/permutation_count`: `null`
- `/claims/C05/quantitative_freeze/representation_distance`: `null`
- `/claims/C05/quantitative_freeze/statistic_tolerances`: `null`
- `/claims/C05/scenario_truth/authoritative`: `false`
- `/claims/C05/scenario_truth/expected_decisions`: `null`
- `/claims/C05/scenario_truth/source_registry_pointer`: `"/source_templates"`
- `/claims/C05/secondary_evidence`: `["Welch location summaries","median-centered Brown-Forsythe scale summaries","PC and silhouette descriptions without decision authority","Cox effect summaries when estimable"]`
- `/claims/C05/watched_failures`: `["restore PC1-only gate","merge location and scale localization","merge event and censoring decisions","split warning-producing components into uncontrolled families","interpret observed censoring hazard as informative censoring","let corroborative classifier evidence trigger the primary warning"]`

## Claim C06

- `/claims/C06/advertised_scope/claim_language`: `"exact feasibility and realized validity of the requested nested group-aware evaluation plan"`
- `/claims/C06/advertised_scope/forbidden_language`: `["clamped fold count is equivalent to requested fold count","solver timeout proves infeasibility"]`
- `/claims/C06/advertised_scope/tasks`: `["classification","regression","survival"]`
- `/claims/C06/eligibility/group_contract`: `"every row belongs to exactly one complete highest exchangeable group and every realized child partition is contained in its parent training partition"`
- `/claims/C06/eligibility/refusal_codes`: `["c06_group_id_missing","c06_group_outcome_mixed","c06_metric_support_insufficient","c06_requested_plan_infeasible","c06_split_manifest_invalid","c06_split_search_unresolved"]`
- `/claims/C06/eligibility/requirements`: `["requested K_outer and K_inner are integers at least two","every requested fold is realized exactly once","train and assessment sets are complete disjoint and group-disjoint","task-specific calibrated training and assessment support holds in every outer and inner fold","performance is scored at the group estimand"]`
- `/claims/C06/estimand/analysis_unit`: `"highest_exchangeable_group"`
- `/claims/C06/estimand/group_weighting`: `"equal"`
- `/claims/C06/estimand/quantity`: `"existence and realized validity of the exact requested nested plan under the frozen metric-support constraints"`
- `/claims/C06/method/assignment_identity`: `"not part of the feasibility estimand; the first independently verified feasible manifest is frozen by SHA-256 and reused exactly"`
- `/claims/C06/method/candidate_solver`: `"scipy_optimize_milp"`
- `/claims/C06/method/neural_early_stop`: `"separate group-disjoint fit and validation partition inside each outer training set"`
- `/claims/C06/method/solver_status_rule`: `"status_zero_plus_independent_manifest_validation_is_feasible_status_two_is_infeasible_all_other_statuses_are_unresolved"`
- `/claims/C06/method/split_rule`: `"joint whole-group assignment with no fold clamping skipping replacement or overlap"`
- `/claims/C06/multiplicity/primary_family`: `"not_applicable_deterministic_invariant"`
- `/claims/C06/multiplicity/secondary_families`: `[]`
- `/claims/C06/multiplicity/secondary_may_trigger_primary`: `false`
- `/claims/C06/oracle/comparison_rule`: `"every index group class event variance and comparable-pair invariant must agree exactly"`
- `/claims/C06/oracle/independence_rule`: `"small fixtures use exhaustive assignment enumeration and all manifests use a separate set-based verifier"`
- `/claims/C06/oracle/methods`: `["exhaustive group-assignment oracle for small fixtures","independent split-manifest verifier","independent survival comparable-pair enumerator"]`
- `/claims/C06/outputs/forbidden_private_keys`: `["group_ids","labels","local_paths","row_indices","sample_ids","subject_ids"]`
- `/claims/C06/outputs/required_public_keys`: `["claim_id","decision","eligibility_reason","group_count","inner_fold_count","outer_fold_count","split_manifest_sha256","support_summary","verifier_status"]`
- `/claims/C06/production_gaps`: `["current split helper silently clamps and forces at least two folds","current survival path silently skips folds and maps zero comparable pairs to chance","current neural early stopping shuffles rows and can reuse fit rows","current outputs lack complete realized partition evidence"]`
- `/claims/C06/quantitative_freeze/K_inner`: `null`
- `/claims/C06/quantitative_freeze/K_outer`: `null`
- `/claims/C06/quantitative_freeze/minimum_assessment_groups`: `null`
- `/claims/C06/quantitative_freeze/minimum_class_support`: `null`
- `/claims/C06/quantitative_freeze/minimum_comparable_pairs`: `null`
- `/claims/C06/quantitative_freeze/minimum_regression_variance`: `null`
- `/claims/C06/quantitative_freeze/minimum_survival_events`: `null`
- `/claims/C06/quantitative_freeze/minimum_training_groups`: `null`
- `/claims/C06/quantitative_freeze/neural_retain_or_refit_choice`: `null`
- `/claims/C06/quantitative_freeze/solver_objective_and_tiebreak`: `"zero objective feasibility search; no assignment tie-break because assignment identity is not the estimand"`
- `/claims/C06/quantitative_freeze/solver_options`: `null`
- `/claims/C06/quantitative_freeze/solver_version`: `null`
- `/claims/C06/scenario_truth/authoritative`: `false`
- `/claims/C06/scenario_truth/expected_decisions`: `null`
- `/claims/C06/scenario_truth/source_registry_pointer`: `"/source_templates"`
- `/claims/C06/secondary_evidence`: `["row-wise minus group-wise optimism experiments"]`
- `/claims/C06/watched_failures`: `["clamp requested folds","skip an infeasible fold","treat one survival pair as calibrated support","treat two regression observations as calibrated support","reuse a group across train and assessment","label solver limit as infeasible","reuse neural fit rows for early-stop validation"]`

## Claim C07

- `/claims/C07/advertised_scope/claim_language`: `"registered negative-control behavior exact-copy detection and calibrated target-recoverability proxy risk"`
- `/claims/C07/advertised_scope/forbidden_language`: `["arbitrary deterministic transform detected","noisy proxy proves leakage","target-derived exchangeability block"]`
- `/claims/C07/advertised_scope/tasks`: `["classification","regression","survival"]`
- `/claims/C07/eligibility/group_contract`: `"one endpoint object per highest exchangeable group is permuted intact inside predeclared target-independent design blocks within outer training only"`
- `/claims/C07/eligibility/refusal_codes`: `["c07_group_outcome_mixed","c07_no_nontrivial_group_permutation","c07_permutation_scope_invalid","c07_proxy_contract_unfrozen","c07_stratum_target_derived"]`
- `/claims/C07/eligibility/requirements`: `["every block is fixed from design variables before outcomes are inspected","at least one block permits more than one distinct assignment","permutation is a blockwise bijection of complete group outcome objects","outer assessment outcomes are unchanged","every target-dependent node is refit downstream of the permuted target"]`
- `/claims/C07/estimand/analysis_unit`: `"highest_exchangeable_group"`
- `/claims/C07/estimand/group_weighting`: `"equal"`
- `/claims/C07/estimand/quantity`: `"behavior of registered controls and target-recoverability checks under valid group-object exchangeability"`
- `/claims/C07/method/exact_copy_scanner`: `"registered classification equality_or_bijection regression equality_or_nonzero_affine and survival observed_time_event_joint_tuple_or_frozen_risk_inversion relations only"`
- `/claims/C07/method/negative_control`: `"within_outer_training blockwise whole-group endpoint-object permutation broadcast to member rows"`
- `/claims/C07/method/proxy_output`: `"target_recoverability_proxy_risk requiring provenance before any leakage conclusion"`
- `/claims/C07/method/target_dependent_refit`: `"feature_selection supervised_representation_threshold_calibration_and_model_nodes_descend_from_the_permuted_target_node"`
- `/claims/C07/multiplicity/primary_family`: `"exact-copy invariants deterministic and stochastic control or proxy components use their frozen validation families"`
- `/claims/C07/multiplicity/secondary_families`: `[]`
- `/claims/C07/multiplicity/secondary_may_trigger_primary`: `false`
- `/claims/C07/oracle/comparison_rule`: `"permutation mechanics and exact-copy relations agree exactly while stochastic decisions follow the inference amendment"`
- `/claims/C07/oracle/independence_rule`: `"oracle reconstructs group objects blocks assignments broadcast and fit ancestry without production control helpers"`
- `/claims/C07/oracle/methods`: `["independent blockwise permutation reconstruction","independent canonical-value exact-copy scanner","independent fit-ancestry validation"]`
- `/claims/C07/outputs/forbidden_private_keys`: `["group_ids","labels","local_paths","permutation_mapping","row_indices","sample_ids","subject_ids"]`
- `/claims/C07/outputs/required_public_keys`: `["attainable_permutation_count","claim_id","control_kind","decision","eligibility_reason","exchangeability_contract_sha256","fit_ancestry_status","method_id","permutation_registry_sha256"]`
- `/claims/C07/production_gaps`: `["current controls permute rows globally before cross-validation","current survival control permutation is not group-object scoped","current warning margin is uncalibrated","current controls do not directly detect registered target copies","current wording overstates generic leakage detection"]`
- `/claims/C07/quantitative_freeze/control_family_alpha`: `null`
- `/claims/C07/quantitative_freeze/control_margin`: `null`
- `/claims/C07/quantitative_freeze/minimum_attainable_permutations`: `null`
- `/claims/C07/quantitative_freeze/noisy_proxy_grid`: `null`
- `/claims/C07/quantitative_freeze/permutation_count`: `null`
- `/claims/C07/quantitative_freeze/proxy_effect_targets`: `null`
- `/claims/C07/quantitative_freeze/proxy_threshold`: `null`
- `/claims/C07/scenario_truth/authoritative`: `false`
- `/claims/C07/scenario_truth/expected_decisions`: `null`
- `/claims/C07/scenario_truth/source_registry_pointer`: `"/source_templates"`
- `/claims/C07/secondary_evidence`: `["control score and interval summaries","proxy-risk effect summaries"]`
- `/claims/C07/watched_failures`: `["permute rows instead of groups","derive strata from the target","allow an identity-only permutation","reuse target-dependent preprocessing fitted on the real target","call a noisy proxy leakage without provenance","claim coverage beyond registered exact-copy transforms"]`

## Claim C08

- `/claims/C08/advertised_scope/claim_language`: `"complete traceable train-only fitting for every registered result-affecting operation"`
- `/claims/C08/advertised_scope/forbidden_language`: `["a digest alone proves absence of hidden fitting","generic survival probability calibration without a frozen horizon contract"]`
- `/claims/C08/advertised_scope/tasks`: `["classification","regression","survival"]`
- `/claims/C08/eligibility/group_contract`: `"every fit ancestry is contained in the appropriate group-disjoint training partition"`
- `/claims/C08/eligibility/refusal_codes`: `["c08_assessment_ancestry_detected","c08_cache_training_digest_mismatch","c08_callsite_unregistered","c08_fit_trace_incomplete","c08_state_uncanonicalizable","c08_static_runtime_inventory_mismatch"]`
- `/claims/C08/eligibility/requirements`: `["static and runtime fit-callsite inventories reconcile exactly","every data-dependent operation emits one canonical trace node","all node parents and index digests are present","no fit ancestry contains an assessment group","stacking and calibration predictions are generated by fits excluding the predicted group"]`
- `/claims/C08/estimand/analysis_unit`: `"registered_fit_operation_with_group_ancestry"`
- `/claims/C08/estimand/group_weighting`: `"not_applicable_deterministic_trace"`
- `/claims/C08/estimand/quantity`: `"completeness and training-only ancestry of every result-affecting learned state"`
- `/claims/C08/method/fit_trace_node`: `"callsite_component_version_parent_split_stage_fold_fit_validation_assessment_digests_input_schema_target_use_parameters_seed_state_output_support_and_parent_nodes"`
- `/claims/C08/method/poison_contract/assessment_feature_poison`: `"learned_state_unchanged_assessment_predictions_may_change"`
- `/claims/C08/method/poison_contract/assessment_outcome_poison`: `"learned_state_and_predictions_unchanged"`
- `/claims/C08/method/poison_contract/unchanged_sentinel`: `"sentinel_predictions_unchanged_when_other_assessment_features_are_poisoned"`
- `/claims/C08/method/stacking`: `"outer_inner group cross-fitting with one excluded-group meta-feature per outer-training group and one untouched outer assessment prediction"`
- `/claims/C08/method/static_runtime_reconciliation`: `"registered AST callsite inventory equals emitted runtime callsite inventory"`
- `/claims/C08/multiplicity/primary_family`: `"not_applicable_deterministic_invariant"`
- `/claims/C08/multiplicity/secondary_families`: `[]`
- `/claims/C08/multiplicity/secondary_may_trigger_primary`: `false`
- `/claims/C08/oracle/comparison_rule`: `"inventory DAG subset disjointness learned-state and poison invariants agree exactly"`
- `/claims/C08/oracle/independence_rule`: `"oracle reads traces and split manifests and independently refits registered train-only states"`
- `/claims/C08/oracle/methods`: `["independent static fit-callsite inventory","independent trace-DAG verifier","independent train-only state refit","held-out outcome feature and sentinel poison tests"]`
- `/claims/C08/outputs/forbidden_private_keys`: `["fitted_state","group_ids","labels","local_paths","row_indices","sample_ids","subject_ids"]`
- `/claims/C08/outputs/required_public_keys`: `["callsite_inventory_sha256","claim_id","decision","fit_trace_sha256","node_count","poison_test_status","split_manifest_sha256","state_digest_count","verifier_status"]`
- `/claims/C08/production_gaps`: `["current package emits no complete fit trace or learned-state digest","current stacking is explicitly not fully nested","current neural early-stop split is row-wise","current benchmark trace does not cover every package fit callsite","current calibration output is metrics-only and has no calibrator trace"]`
- `/claims/C08/quantitative_freeze/calibration_method_and_support`: `null`
- `/claims/C08/quantitative_freeze/callsite_inventory_sha256`: `null`
- `/claims/C08/quantitative_freeze/canonical_state_schema`: `null`
- `/claims/C08/quantitative_freeze/neural_determinism_contract`: `null`
- `/claims/C08/quantitative_freeze/state_digest_tolerance`: `null`
- `/claims/C08/quantitative_freeze/survival_calibration_horizon`: `null`
- `/claims/C08/scenario_truth/authoritative`: `false`
- `/claims/C08/scenario_truth/expected_decisions`: `null`
- `/claims/C08/scenario_truth/source_registry_pointer`: `"/source_templates"`
- `/claims/C08/secondary_evidence`: `["score gaps against intentionally global-fit challenge pipelines"]`
- `/claims/C08/watched_failures`: `["omit one registered fit callsite","insert one assessment group into fit ancestry","reuse a cached state under a different training digest","require assessment-feature prediction invariance","allow assessment-outcome prediction drift","train stacking meta-features on in-group fits","accept a digest without static-runtime inventory reconciliation"]`

## Dependency freeze

- `/dependency_freeze/runtime_candidates/lifelines/authorized`: `false`
- `/dependency_freeze/runtime_candidates/lifelines/callables_and_defaults`: `null`
- `/dependency_freeze/runtime_candidates/lifelines/exact_version`: `null`
- `/dependency_freeze/runtime_candidates/lifelines/purpose`: `"candidate survival diagnostic implementation"`
- `/dependency_freeze/runtime_candidates/scikit_learn/authorized`: `false`
- `/dependency_freeze/runtime_candidates/scikit_learn/callables_and_defaults`: `null`
- `/dependency_freeze/runtime_candidates/scikit_learn/exact_version`: `null`
- `/dependency_freeze/runtime_candidates/scikit_learn/purpose`: `"explicit iterable stacking and calibration splits"`
- `/dependency_freeze/runtime_candidates/scipy/authorized`: `false`
- `/dependency_freeze/runtime_candidates/scipy/callables_and_defaults`: `null`
- `/dependency_freeze/runtime_candidates/scipy/exact_version`: `null`
- `/dependency_freeze/runtime_candidates/scipy/purpose`: `"MGC permutation statistics and candidate MILP solver"`
- `/dependency_freeze/validation_candidates/scikit_bio/authorized`: `false`
- `/dependency_freeze/validation_candidates/scikit_bio/callables_and_defaults`: `null`
- `/dependency_freeze/validation_candidates/scikit_bio/exact_version`: `null`
- `/dependency_freeze/validation_candidates/scikit_bio/purpose`: `"PERMANOVA and PERMDISP oracle"`
- `/dependency_freeze/validation_candidates/statsmodels/authorized`: `false`
- `/dependency_freeze/validation_candidates/statsmodels/callables_and_defaults`: `null`
- `/dependency_freeze/validation_candidates/statsmodels/exact_version`: `null`
- `/dependency_freeze/validation_candidates/statsmodels/purpose`: `"multiplicity and survival oracle"`

## Evidence boundary and public-safe evidence contract

- `/evidence_boundary/current_artifact_role`: `"reviewed design candidate only"`
- `/evidence_boundary/does_not_establish`: `["calibrated error control","implementation correctness","positive validation","real-data performance","reproducible released software"]`
- `/evidence_boundary/may_establish_after_independent_check`: `["consistency of proposed estimands algorithms refusal domains oracle roles and unresolved blockers"]`
- `/public_safe_evidence_contract/forbidden`: `["clinical values","credentials","feature matrices","group identifiers","labels","local paths","row indices","sample identifiers","seeds","split membership","subject identifiers"]`
- `/public_safe_evidence_contract/required`: `["aggregate counts","decision tokens","eligibility reason codes","method identifiers","multiplicity family identifiers","oracle status","SHA-256 digests","verification status"]`

## Production gap inventory

- `/production_gap_inventory/C03`: `"row-level aggregate tests pooled with batch and no censoring-aware event/censor separation"`
- `/production_gap_inventory/C04`: `"row-level aggregate test without group or batch-purity contract"`
- `/production_gap_inventory/C05`: `"PC1-centered structure gate uncensored survival branch and private coordinates"`
- `/production_gap_inventory/C06`: `"fold clamping skipping and row-wise early-stop fallback"`
- `/production_gap_inventory/C07`: `"global row-wise controls and uncalibrated broad leakage wording"`
- `/production_gap_inventory/C08`: `"incomplete fit inventory absent state trace and nonnested stacking"`

## Source bindings

### binomial_design_tool

- `/source_bindings/binomial_design_tool/claim_ids`: `["C03","C04","C05","C06","C07","C08"]`
- `/source_bindings/binomial_design_tool/path`: `"benchmark_record/tools/gate3_binomial_design_v1_3_0.py"`
- `/source_bindings/binomial_design_tool/pointers`: `[]`
- `/source_bindings/binomial_design_tool/sha256`: `"643d3c7ea7b5a3a5b1751d747bda88fd75d145edfc3cb2b43d3b48d6637e71f8"`
- `/source_bindings/binomial_design_tool/size_bytes`: `13443`
- `/source_bindings/binomial_design_tool/source_kind`: `"python"`
- `/source_bindings/binomial_design_tool/symbols`: `["clopper_pearson_lower","clopper_pearson_upper","minimum_all_success_n","minimum_zero_event_n","plan_detection_fixed_n","plan_false_warning_fixed_n"]`

### diagnostic_contract

- `/source_bindings/diagnostic_contract/claim_ids`: `["C03","C04","C05","C06","C07","C08"]`
- `/source_bindings/diagnostic_contract/path`: `"benchmark_record/releases/v1.3.0/gate3_diagnostic_validation_contract.draft.json"`
- `/source_bindings/diagnostic_contract/pointers`: `["/authorizations","/claim_designs/C03","/claim_designs/C04","/claim_designs/C05","/claim_designs/C06","/claim_designs/C07","/claim_designs/C08","/evidence_boundary","/execution_truth_policy","/precision_and_acceptance","/unresolved_freeze_blockers","/zenodo_ready"]`
- `/source_bindings/diagnostic_contract/sha256`: `"d230ba29e0e6c90edc2ea0b1289c0834964220db3310b499c39831a3f453f77c"`
- `/source_bindings/diagnostic_contract/size_bytes`: `35928`
- `/source_bindings/diagnostic_contract/source_kind`: `"canonical_json"`
- `/source_bindings/diagnostic_contract/symbols`: `[]`

### inference_amendment

- `/source_bindings/inference_amendment/claim_ids`: `["C03","C04","C05","C06","C07","C08"]`
- `/source_bindings/inference_amendment/path`: `"benchmark_record/releases/v1.3.0/gate3_inference_amendment.draft.json"`
- `/source_bindings/inference_amendment/pointers`: `["/authorizations","/claim_inference/C03","/claim_inference/C04","/claim_inference/C05","/claim_inference/C06","/claim_inference/C07","/claim_inference/C08","/claim_level_IUT","/deterministic_claim_scope","/multiplicity","/non_estimable_and_degenerate_rules","/planning","/seed_policy","/unresolved_blockers","/zenodo_ready"]`
- `/source_bindings/inference_amendment/sha256`: `"2377311ae1c7193af25ba8444fc97fdb3294b431ab487972d6812c7624b6520b"`
- `/source_bindings/inference_amendment/size_bytes`: `27929`
- `/source_bindings/inference_amendment/source_kind`: `"canonical_json"`
- `/source_bindings/inference_amendment/symbols`: `[]`

### production_alignment

- `/source_bindings/production_alignment/claim_ids`: `["C03","C04","C05","C06","C07","C08"]`
- `/source_bindings/production_alignment/path`: `"omicau/data/alignment.py"`
- `/source_bindings/production_alignment/pointers`: `[]`
- `/source_bindings/production_alignment/sha256`: `"67470dd1f695e53212ffd93ef009d55a82c110c944f8ce3306245b34aaa4da7b"`
- `/source_bindings/production_alignment/size_bytes`: `31424`
- `/source_bindings/production_alignment/source_kind`: `"python"`
- `/source_bindings/production_alignment/symbols`: `["_resolve_group_series","check_grouping"]`

### production_audit_batch

- `/source_bindings/production_audit_batch/claim_ids`: `["C05"]`
- `/source_bindings/production_audit_batch/path`: `"omicau/diagnostics/batch.py"`
- `/source_bindings/production_audit_batch/pointers`: `[]`
- `/source_bindings/production_audit_batch/sha256`: `"ae23bc662797953c00c84eaaea5620808eac2cda6de4743de1394c6b9dd93de0"`
- `/source_bindings/production_audit_batch/size_bytes`: `8482`
- `/source_bindings/production_audit_batch/source_kind`: `"python"`
- `/source_bindings/production_audit_batch/symbols`: `["batch_effect_diagnostics"]`

### production_base_models

- `/source_bindings/production_base_models/claim_ids`: `["C06","C08"]`
- `/source_bindings/production_base_models/path`: `"omicau/models/base.py"`
- `/source_bindings/production_base_models/pointers`: `[]`
- `/source_bindings/production_base_models/sha256`: `"9f51ece025b0a9bb6183ca43462dd6ba24eb8f84b4c136ccca2d0ea6a78dcb6c"`
- `/source_bindings/production_base_models/size_bytes`: `18542`
- `/source_bindings/production_base_models/source_kind`: `"python"`
- `/source_bindings/production_base_models/symbols`: `["cross_validate_estimator","make_pipeline","safe_n_splits"]`

### production_classical_models

- `/source_bindings/production_classical_models/claim_ids`: `["C06","C07","C08"]`
- `/source_bindings/production_classical_models/path`: `"omicau/models/classical.py"`
- `/source_bindings/production_classical_models/pointers`: `[]`
- `/source_bindings/production_classical_models/sha256`: `"d4ec9934de4c8845d5a58f5d9eebe9837b3d2caa469ca2d944aca6750efd172c"`
- `/source_bindings/production_classical_models/size_bytes`: `12437`
- `/source_bindings/production_classical_models/source_kind`: `"python"`
- `/source_bindings/production_classical_models/symbols`: `["_run_stacking","run_classical_benchmarks"]`

### production_missingness

- `/source_bindings/production_missingness/claim_ids`: `["C03","C04"]`
- `/source_bindings/production_missingness/path`: `"omicau/diagnostics/missingness.py"`
- `/source_bindings/production_missingness/pointers`: `[]`
- `/source_bindings/production_missingness/sha256`: `"4dbe4800eb4e5ba3f45a37188e20cdff7c41844d4409657138363982670fc31b"`
- `/source_bindings/production_missingness/size_bytes`: `6893`
- `/source_bindings/production_missingness/source_kind`: `"python"`
- `/source_bindings/production_missingness/symbols`: `["missingness_diagnostics"]`

### production_neural_models

- `/source_bindings/production_neural_models/claim_ids`: `["C06","C08"]`
- `/source_bindings/production_neural_models/path`: `"omicau/models/neural.py"`
- `/source_bindings/production_neural_models/pointers`: `[]`
- `/source_bindings/production_neural_models/sha256`: `"0b0650be83315ac9503f9c852a0540fcd16948bd8ea79ee035f5ca5ec3141f0d"`
- `/source_bindings/production_neural_models/size_bytes`: `15126`
- `/source_bindings/production_neural_models/source_kind`: `"python"`
- `/source_bindings/production_neural_models/symbols`: `["_neural_cv","run_neural_benchmark"]`

### production_survival_models

- `/source_bindings/production_survival_models/claim_ids`: `["C03","C05","C06","C07","C08"]`
- `/source_bindings/production_survival_models/path`: `"omicau/models/survival.py"`
- `/source_bindings/production_survival_models/pointers`: `[]`
- `/source_bindings/production_survival_models/sha256`: `"7d0b0254699fdf5b4b371a68b7165948a59dade5f49fbd33f2c8aad1647139ae"`
- `/source_bindings/production_survival_models/size_bytes`: `11135`
- `/source_bindings/production_survival_models/source_kind`: `"python"`
- `/source_bindings/production_survival_models/symbols`: `["_cv_cindex","_splitter","run_survival_benchmark"]`

### production_utility

- `/source_bindings/production_utility/claim_ids`: `["C05","C07"]`
- `/source_bindings/production_utility/path`: `"omicau/interpretation/utility.py"`
- `/source_bindings/production_utility/pointers`: `[]`
- `/source_bindings/production_utility/sha256`: `"97970d4a53f6c0e7032729e2536ce4fa14f3575e0ba358b061c455a430c5e20d"`
- `/source_bindings/production_utility/size_bytes`: `30249`
- `/source_bindings/production_utility/source_kind`: `"python"`
- `/source_bindings/production_utility/symbols`: `["CONTROL_MARGIN","build_utility_ledger"]`

### scenario_templates

- `/source_bindings/scenario_templates/claim_ids`: `["C03","C04","C05","C06","C07","C08"]`
- `/source_bindings/scenario_templates/path`: `"benchmark_record/releases/v1.3.0/gate3_scenario_templates.draft5.json"`
- `/source_bindings/scenario_templates/pointers`: `["/authoritative_definition_fields","/authorizations","/coverage_summary/claim_counts/C03","/coverage_summary/claim_counts/C04","/coverage_summary/claim_counts/C05","/coverage_summary/claim_counts/C06","/coverage_summary/claim_counts/C07","/coverage_summary/claim_counts/C08","/final_authoritative_row_count","/source_templates","/static_truth_policy","/zenodo_ready"]`
- `/source_bindings/scenario_templates/sha256`: `"c3b4f4da1cf31aa4a108648daae4697e510ce5976e47b74f69caedf12e4c0df6"`
- `/source_bindings/scenario_templates/size_bytes`: `215387`
- `/source_bindings/scenario_templates/source_kind`: `"canonical_json"`
- `/source_bindings/scenario_templates/symbols`: `[]`

## Unresolved blockers

- `/unresolved_blockers`: `["authoritative C03-C08 scenario truth and expected decisions","calibration registry prefix bounds count serialization and SHA-256","claim and component beta allocation table and final sample sizes","C03-C05 effect targets warning cutoffs permutation counts support minima and multiplicity allocations","C05 PH and non-PH survival design coding statistic support convergence and separation rules","C06 requested folds metric-support minima solver version options and neural retain-or-refit choice","C07 exchangeability blocks attainable-count floor exact-copy registry proxy grid threshold and effect targets","C08 complete fit-callsite inventory canonical state schema calibrator contract and survival horizon","exact runtime and validation dependency versions callable contracts defaults and failure behavior","frozen group reducers scoring rules and uncertainty aggregation for repeated-row inputs","locked validation seed registry schema contents serialization and SHA-256","result-affecting code environment split and artifact bindings after implementation","successful independent checker and scientific review of this draft","watched-failure implementation for every declared algorithm refusal and decision branch"]`
