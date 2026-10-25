"""TRIPOD+AI checklist and what a run can document of it automatically.

Reference: Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for
reporting clinical prediction models that use regression or machine learning methods. BMJ
2024;385:e078378. doi:10.1136/bmj-2023-078378 (checklist = its Table 2; items and numbering
checked against the published text, wording abridged here).

Honesty rule: ``documented`` only when everything the item asks for, other than a free-text
rationale, is in the report; ``partial`` when the report covers part of it; ``pending`` when it
needs a human (or data the run does not record). A report never ticks an item it cannot back.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

REFERENCE = ("Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for "
             "reporting clinical prediction models that use regression or machine learning "
             "methods. BMJ 2024;385:e078378.")
DOI_URL = "https://doi.org/10.1136/bmj-2023-078378"

DOCUMENTED, PARTIAL, PENDING = "documented", "partial", "pending"
STATUS_LABEL = {DOCUMENTED: "Documented", PARTIAL: "Partly documented", PENDING: "Pending (human)"}


@dataclass
class Context:
    """Facts about the run that decide each item (built by the report)."""
    has_start: bool = False
    has_config: bool = False
    has_eval: bool = False
    commit: str | None = None
    n_boot: int = 0
    has_system: bool = False


@dataclass
class Item:
    id: str
    section: str
    text: str
    applies: str                  # D, E or D;E
    status: str
    note: str
    anchor: str | None = None     # id of the report section that documents it


_Check = Callable[[Context], tuple[str, str, str | None]]


def _pending(note: str = "") -> _Check:
    return lambda c: (PENDING, note or "Requires the authors; the run does not record it.", None)


def _need_start(note: str, anchor: str) -> _Check:
    def check(c: Context):
        if c.has_start:
            return PARTIAL, note, anchor
        return PENDING, "The events file has no run_start.", None
    return check


def _need_config(note: str, anchor: str) -> _Check:
    def check(c: Context):
        if c.has_config:
            return PARTIAL, note, anchor
        return PENDING, "The events file has no configuration.", None
    return check


def _need_eval(status: str, note: str, anchor: str) -> _Check:
    def check(c: Context):
        if c.has_eval:
            return status, note, anchor
        return PENDING, "No inspection data: train with --monitor (monitor.enabled: true).", None
    return check


def _ci(c: Context):
    if not c.has_eval:
        return PENDING, "No inspection data: train with --monitor (monitor.enabled: true).", None
    return (PARTIAL,
            f"Estimates with percentile bootstrap 95 % CI (n_boot={c.n_boot}), overall and per "
            "class. Key subgroups (sociodemographic) are not covered: the run logs no such "
            "attributes.", "eval-metrics")


def _commit(c: Context):
    if not c.has_start:
        return PENDING, "The events file has no run_start.", None
    note = ("Run commit recorded; " if c.commit else "No git commit recorded in this run; ")
    return PARTIAL, note + "availability of the code is for the authors.", "env"


def _model_spec(c: Context):
    if not c.has_start:
        return PENDING, "The events file has no run_start.", None
    return (PARTIAL, "Architecture, full configuration and code version are recorded; access to "
            "the weights and restrictions on reuse are for the authors.", "env")


# (id, section, abridged text, applies, check)
CHECKLIST: list[tuple[str, str, str, str, _Check]] = [
    ("1", "Title", "Identify the study as developing or evaluating a prediction model; target "
     "population and outcome.", "D;E",
     _need_start("Model, task and outcome classes are in the summary; the study framing and "
                 "target population are for the authors.", "sec-summary")),
    ("2", "Abstract", "Abstract following the TRIPOD+AI for Abstracts checklist.", "D;E",
     _pending()),
    ("3a", "Background", "Healthcare context, diagnostic or prognostic, rationale and existing "
     "models.", "D;E", _pending()),
    ("3b", "Background", "Target population and intended purpose in the care pathway; intended "
     "users.", "D;E", _pending()),
    ("3c", "Background", "Known health inequalities between sociodemographic groups.", "D;E",
     _pending()),
    ("4", "Objectives", "Study objectives: development, validation or both.", "D;E", _pending()),
    ("5a", "Data", "Sources of the development and evaluation datasets, rationale, "
     "representativeness.", "D;E",
     _need_start("Dataset name and configuration are recorded; the rationale for the source and "
                 "its representativeness are for the authors.", "sec-summary")),
    ("5b", "Data", "Dates of data collection (accrual and follow-up).", "D;E", _pending()),
    ("6a", "Participants", "Study setting, number and location of centres.", "D;E", _pending()),
    ("6b", "Participants", "Eligibility criteria.", "D;E", _pending()),
    ("6c", "Participants", "Treatments received and how they were handled.", "D;E", _pending()),
    ("7", "Data preparation", "Pre-processing and quality checks, also across sociodemographic "
     "groups.", "D;E",
     _need_config("Pre-processing and augmentation settings are in the configuration. Quality "
                  "checks (and any across-group comparison) are not recorded.", "config")),
    ("8a", "Outcome", "Outcome definition, time horizon, assessment, rationale.", "D;E",
     _need_start("Task type and outcome classes are recorded; time horizon, how and when the "
                 "outcome was assessed and the rationale are for the authors.", "sec-summary")),
    ("8b", "Outcome", "Qualifications of outcome assessors, if subjective.", "D;E", _pending()),
    ("8c", "Outcome", "Blinding of outcome assessment.", "D;E", _pending()),
    ("9a", "Predictors", "Choice of initial predictors and pre-selection.", "D", _pending()),
    ("9b", "Predictors", "Definition of all predictors, how and when measured.", "D;E",
     _need_config("The input (image size, channels, resizing) comes from the configuration; "
                  "how and when the images were acquired is for the authors.", "config")),
    ("9c", "Predictors", "Qualifications of predictor assessors, if subjective.", "D;E",
     _pending()),
    ("10", "Sample size", "How the study size was arrived at and why it is sufficient.", "D;E",
     _need_eval(PARTIAL, "The size of the evaluation set and the positives per class are "
                "reported; a sample-size justification is for the authors.", "eval-perclass")),
    ("11", "Missing data", "How missing data were handled.", "D;E", _pending()),
    ("12a", "Analytical methods", "How the data were used (development, evaluation, "
     "partitioning).", "D",
     _need_config("The partitioning is defined by the dataset and, if set, the seed in the configuration; "
                  "this report evaluates the validation partition only (no held-out test).",
                  "sec-summary")),
    ("12b", "Analytical methods", "How predictors were handled (rescaling, transformation, "
     "standardisation).", "D",
     _need_config("Rescaling and normalisation are set in the configuration.", "config")),
    ("12c", "Analytical methods", "Type of model, rationale, all model-building steps, "
     "hyperparameter tuning, internal validation.", "D",
     _need_config("Model, optimiser, schedule, selection metric, early stopping and the "
                  "hold-out validation are recorded. The rationale and any hyperparameter search "
                  "(done outside the run) are for the authors.", "sec-training")),
    ("12d", "Analytical methods", "Heterogeneity of estimates across clusters (centres, "
     "countries).", "D;E", _pending("No cluster identifiers are logged.")),
    ("12e", "Analytical methods", "Measures and plots used to evaluate performance "
     "(discrimination, calibration, clinical utility).", "D;E",
     _need_eval(DOCUMENTED, "AUC and AP, F1, sensitivity, specificity, accuracy, ROC and PR "
                "curves, calibration (ECE, slope, intercept, reliability diagram) and operating "
                "points, all with their definitions. Clinical utility (net benefit, decision curves) is "
                "not computed.", "sec-evaluation")),
    ("12f", "Analytical methods", "Model updating (e.g. recalibration) after evaluation.", "E",
     _pending("No updating is performed or logged. The calibration slope and intercept in the "
              "report can inform one.")),
    ("12g", "Analytical methods", "How predictions were calculated (formula, code, API).", "E",
     _commit),
    ("13", "Class imbalance", "Class-imbalance methods, why and how; later recalibration.", "D;E",
     _need_config("Loss settings are in the configuration and the class prevalence in the "
                  "per-class table; whether imbalance was addressed, and why, is for the "
                  "authors.", "config")),
    ("14", "Fairness", "Approaches used to address model fairness and their rationale.", "D;E",
     _pending("No sociodemographic attributes are logged, so no fairness analysis is possible "
              "from the run alone.")),
    ("15", "Model output", "Output of the model, rationale for any classification, how "
     "thresholds were identified.", "D",
     _need_eval(PARTIAL, "Outputs are probabilities (sigmoid or softmax). The threshold stored by "
                "the run (chosen on validation) and thresholds for a target sensitivity are "
                "reported; the clinical rationale is for the authors.", "eval-operating")),
    ("16", "Training vs evaluation", "Differences between development and evaluation data.",
     "D;E", _pending("This report evaluates the validation partition; there is no separate "
                     "evaluation dataset to compare.")),
    ("17", "Ethical approval", "Ethics committee and informed consent or waiver.", "D;E",
     _pending()),
    ("18a", "Open science", "Source of funding and role of funders.", "D;E", _pending()),
    ("18b", "Open science", "Conflicts of interest.", "D;E", _pending()),
    ("18c", "Open science", "Where the study protocol can be accessed.", "D;E", _pending()),
    ("18d", "Open science", "Registration information.", "D;E", _pending()),
    ("18e", "Open science", "Availability of the study data.", "D;E", _pending()),
    ("18f", "Open science", "Availability of the analytical code.", "D;E", _commit),
    ("19", "Patient and public involvement", "Involvement in design, conduct, reporting.", "D;E",
     _pending()),
    ("20a", "Participants (results)", "Flow of participants; numbers with and without the "
     "outcome.", "D;E",
     _need_eval(PARTIAL, "Size of the evaluation set and positives and negatives per class are "
                "reported; the flow from the source population is for the authors.",
                "eval-perclass")),
    ("20b", "Participants (results)", "Characteristics of participants, per data source or "
     "setting.", "D;E",
     _pending("Only class counts are available; demographics and clinical characteristics are "
              "not logged.")),
    ("20c", "Participants (results)", "Comparison of predictor distributions with the "
     "development data.", "E", _pending()),
    ("21", "Model development", "Number of participants and outcome events in each analysis.",
     "D;E",
     _need_eval(PARTIAL, "Evaluation counts are reported; the number of training samples is not "
                "recorded in the events.", "eval-perclass")),
    ("22", "Model specification", "Full model details to allow predictions in new individuals "
     "and third-party evaluation.", "D", _model_spec),
    ("23a", "Model performance", "Performance estimates with confidence intervals, including key "
     "subgroups.", "D;E", _ci),
    ("23b", "Model performance", "Heterogeneity of performance across clusters.", "D;E",
     _pending("No cluster identifiers are logged.")),
    ("24", "Model updating", "Results of any model updating.", "E", _pending()),
    ("25", "Interpretation", "Overall interpretation of the results, including fairness.", "D;E",
     _pending()),
    ("26", "Limitations", "Limitations and their effect on bias, uncertainty, "
     "generalisability.", "D;E", _pending()),
    ("27a", "Usability", "Handling of poor-quality or unavailable input data when "
     "implementing.", "D", _pending()),
    ("27b", "Usability", "Whether users must interact with the input data; required "
     "expertise.", "D", _pending()),
    ("27c", "Usability", "Next steps for future research (applicability, generalisability).",
     "D;E", _pending()),
]


def assess(ctx: Context) -> list[Item]:
    """The checklist with the status each item has for this run."""
    items = []
    for id_, section, text, applies, check in CHECKLIST:
        status, note, anchor = check(ctx)
        items.append(Item(id_, section, text, applies, status, note, anchor))
    return items


def summary(items: list[Item]) -> dict[str, int]:
    out = {DOCUMENTED: 0, PARTIAL: 0, PENDING: 0}
    for it in items:
        out[it.status] += 1
    return out

