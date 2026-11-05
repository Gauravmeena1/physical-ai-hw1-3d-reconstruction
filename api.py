import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import csv
import glob
import hashlib
import json
import os
import re
import sys

import numpy as np
from PIL import Image

from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef, XSD

NS = "http://taica.course/hw1/ontology#"
DATA_NS = "http://taica.course/hw1/data/"
EXP_NS = "http://taica.course/hw1/experiment/"
HW1 = Namespace(NS)
SCHEMA = Namespace("https://schema.org/")
QUDT = Namespace("http://qudt.org/schema/qudt/")
UNIT = Namespace("http://qudt.org/vocab/unit/")
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
PROV = Namespace("http://www.w3.org/ns/prov#")

_HERE = os.path.dirname(os.path.abspath(__file__))

_ONTOLOGY_TTL = os.path.join(_HERE, "ontology", "hw1.ttl")

_EXPERIMENT_DIR = os.path.join(_HERE, "experiments")


def experiment_directory(name):
    """Workspace directory for one experiment's Turtle and derived artifacts."""
    return os.path.join(_EXPERIMENT_DIR, name)


def _workspace_for_experiment_file(path):
    """Workspace directory containing the experiment Turtle, or beside it."""
    name = os.path.splitext(os.path.basename(path))[0]
    parent = os.path.dirname(os.path.abspath(path))
    return parent if os.path.basename(parent) == name else os.path.join(parent, name)


def resolve_experiment_path(path):
    """Accept the canonical experiment directory as shorthand for its Turtle file."""
    if os.path.isdir(path):
        # Flat .ttl next to the workspace dir (pre-canonical layout); the
        # directory shorthand still resolves when plots/queries live beside it.
        if _is_capture_dir(path):
            return path
        name = os.path.basename(os.path.normpath(path))
        candidate = os.path.join(path, name + ".ttl")
        if os.path.isfile(candidate):
            return candidate
        legacy_candidate = os.path.join(os.path.dirname(os.path.abspath(path)), name + ".ttl")
        if os.path.isfile(legacy_candidate):
            return legacy_candidate
    return path


def _stem(path):
    """Integer filename stem of a frame path ('.../17.png' -> 17)."""
    return int(os.path.splitext(os.path.basename(path))[0])


def _pair_frames(data_dir):
    """[(stem, rgb, depth)] paired by int stem, sorted. Full strategy note: docs/triplestore.md."""
    rgb_dir = os.path.join(data_dir, "rgb")
    depth_dir = os.path.join(data_dir, "depth")
    if not os.path.isdir(rgb_dir) or not os.path.isdir(depth_dir):
        raise ValueError(f"data-dir must contain rgb/ and depth/ subdirs: {data_dir!r}")

    rgb = {_stem(p): p for p in glob.glob(os.path.join(rgb_dir, "*.png"))}
    depth = {_stem(p): p for p in glob.glob(os.path.join(depth_dir, "*.png"))}
    common = sorted(set(rgb) & set(depth))
    if not common:
        raise ValueError(f"No frames present in BOTH rgb/ and depth/ under: {data_dir!r}")
    return [(str(s), rgb[s], depth[s]) for s in common]


_EXPNAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def batch_name(data_dir, floor):
    """Floor-qualified batch name floor{floor}_{basename}. Full strategy note: docs/triplestore.md."""
    return f"floor{int(floor)}_{os.path.basename(os.path.normpath(data_dir))}"


def batch_iri(name):
    """Batch node IRI <ns>batch/<name>. Full strategy note: docs/triplestore.md."""
    return URIRef(f"{DATA_NS}batch/{name}")


def frame_iri(name, idx):
    """Frame node IRI, <n> decimal unpadded. Full strategy note: docs/triplestore.md."""
    return URIRef(f"{DATA_NS}batch/{name}/frame/{int(idx)}")


def component_iri(name, idx, kind):
    """Image node IRI (rgb|depth); carries contentUrl ONLY, never observables. Full strategy note: docs/triplestore.md."""
    if kind not in ("rgb", "depth"):
        raise ValueError(f"unknown image kind {kind!r}")
    return URIRef(f"{DATA_NS}batch/{name}/{kind}/{int(idx)}")


def data_setting_iri(expname, factor_local, param_local):
    return URIRef(f"{EXP_NS}{expname}/setting/{factor_local}_{param_local}")


def data_run_iri(expname, mode):
    return URIRef(f"{EXP_NS}{expname}/run/{mode}")


def data_factor_iri(expname, factor_local, current_idx, previous_idx=None):
    local = (f"{factor_local}_{int(previous_idx)}_{int(current_idx)}"
                if previous_idx is not None else f"{factor_local}_{int(current_idx)}")
    return URIRef(f"{EXP_NS}{expname}/factor/{local}")


def frame_index_from_iri(iri):
    """Batch frame IRI -> its integer stem."""
    text = str(iri)
    prefix = f"{DATA_NS}batch/"
    if text.startswith(prefix) and "/frame/" in text:
        tail = text.split("/frame/", 1)[1]
        if tail.isdigit():
            return int(tail)
    raise ValueError(f"not a pipeline-schema frame IRI: {text!r}")


def _batch_name_from_batch_iri(iri):
    """Batch IRI -> its <name>. Private."""
    text = str(iri)
    prefix = f"{DATA_NS}batch/"
    if not text.startswith(prefix):
        raise ValueError(f"not a pipeline-schema batch IRI: {text!r}")
    return text[len(prefix):]


def _experiment_name_from_iri(iri, path=None):
    """Experiment IRI -> <expname>, validated against _EXPNAME_RE. Full strategy note: docs/triplestore.md."""
    text = str(iri)
    data_prefix = EXP_NS
    where = f"{path}: " if path else ""
    if not text.startswith(data_prefix):
        raise ValueError(f"{where}not a pipeline-schema experiment IRI: {text!r}")
    name = text[len(data_prefix):]
    if not _EXPNAME_RE.match(name):
        raise ValueError(
            f"{where}experiment IRI tail {name!r} is not a legal experiment name "
            f"(§2: [A-Za-z0-9_-]+, and it must equal the declaration "
            f"file's stem)")
    return name


    # TBox loaders: parameter/factor names come from ontology/hw1.ttl at runtime.
_SETTING_ROLES = ("ReconstructionSetting",)
_VALUE_KINDS = ("double", "integer", "string")


def _local(term):
    """Local name of a term in the hw1 namespace."""
    text = str(term)
    if text.startswith(NS):
        return text[len(NS):]
    return text.rsplit("#", 1)[-1].rsplit("/", 1)[-1]


def _storable(value):
    """Round a double to the precision the .ttl file holds (rdflib: 7 sig digits); non-finite passes through. Full strategy note: docs/triplestore.md."""
    return float(f"{float(value):e}")


def _double_literal(value):
    """The one way to write xsd:double (S4.6): graded precision, XSD-valid INF/-INF/NaN. Full strategy note: docs/triplestore.md."""
    v = _storable(value)
    # NaN is the only value that is not equal to itself — the cheapest exact test,
    # and it avoids importing `math` for three comparisons.
    if v != v:
        return Literal("NaN", datatype=XSD.double, normalize=False)
    if v == float("inf"):
        return Literal("INF", datatype=XSD.double, normalize=False)
    if v == float("-inf"):
        return Literal("-INF", datatype=XSD.double, normalize=False)
    return Literal(v, datatype=XSD.double)


def _setting_value_literal(value, kind):
    """Typed literal for a setting value by parameter kind. Full strategy note: docs/triplestore.md."""
    if kind == "string":
        return Literal(str(value), datatype=XSD.string)
    if kind == "integer":
        return Literal(int(value), datatype=XSD.integer)
    return _double_literal(value)


def load_parameter_declarations(path=_ONTOLOGY_TTL):
    """TBox authority #1: what may be set and how it parses. Full strategy note: docs/triplestore.md."""
    g = Graph()
    g.parse(path, format="turtle")
    decls = {}
    for param in g.subjects(RDF.type, HW1.Parameter):
        local = _local(param)

        role_term = g.value(param, HW1.paramRole)
        role = _local(role_term) if role_term is not None else None
        if role not in _SETTING_ROLES:
            raise ValueError(
                f"{path}: parameter {local!r} declares hw1:paramRole {role!r}; "
                f"expected one of {', '.join(_SETTING_ROLES)}")

        kind_term = g.value(param, HW1.paramValueKind)
        kind = str(kind_term) if kind_term is not None else None
        if kind not in _VALUE_KINDS:
            raise ValueError(
                f"{path}: parameter {local!r} declares hw1:paramValueKind {kind!r}; "
                f"expected one of {', '.join(_VALUE_KINDS)}")

        # Single-valued and mandatory: it is a segment of the setting IRI and of
        # the digest line, so "several" and "none" are equally unusable.
        primaries = sorted(g.objects(param, HW1.paramPrimaryFactor), key=str)
        if len(primaries) != 1:
            raise ValueError(
                f"{path}: parameter {local!r} has {len(primaries)} "
                f"hw1:paramPrimaryFactor values; exactly one is required "
                f"(§5) because it names the setting IRI and the digest line")
        affects = sorted((URIRef(str(a)) for a in g.objects(param, HW1.paramAffectsFactor)),
                            key=str)

        default_term = g.value(param, HW1.paramDefault)
        if default_term is None:
            default = None
        else:
            raw = default_term.toPython() if hasattr(default_term, "toPython") \
                else str(default_term)
            try:
                default = {"double": float, "integer": int, "string": str}[kind](raw)
            except (TypeError, ValueError):
                raise ValueError(
                    f"{path}: parameter {local!r} declares hw1:paramDefault "
                    f"{str(default_term)!r}, which is not a {kind}") from None

        decls[local] = {
            "iri": URIRef(str(param)),
            "role": role,
            "kind": kind,
            "primary": _local(primaries[0]),
            "primaryIri": URIRef(str(primaries[0])),
            "affects": tuple(_local(a) for a in affects),
            "affectsIris": tuple(affects),
            "default": default,
        }
    return decls


def parse_setting_arg(arg, decls):
    """Parse one NAME=VALUE setting against the TBox declarations. Full strategy note: docs/triplestore.md."""
    if "=" not in arg:
        raise ValueError(f"a setting expects NAME=VALUE, got {arg!r}")
    name, _, text = arg.partition("=")
    name = name.strip()
    if name not in decls:
        raise ValueError(
            f"undeclared parameter {name!r}. Parameter names come from the TBox, not "
            f"from the command line; declared parameters are: "
            f"{', '.join(sorted(decls)) or '(none)'}")
    kind = decls[name]["kind"]
    if kind == "string":
        return name, text
    try:
        return name, (int(text) if kind == "integer" else float(text))
    except (TypeError, ValueError):
        raise ValueError(
            f"parameter {name!r} is declared hw1:paramValueKind {kind!r}; "
            f"{text!r} does not parse as one") from None


def load_quality_factors(path=_ONTOLOGY_TTL):
    """TBox authority #2: what is measured and its observable key. Full strategy note: docs/triplestore.md."""
    g = Graph()
    g.parse(path, format="turtle")
    out = {}
    for f in g.subjects(RDF.type, HW1.QualityFactor):
        local = _local(f)
        over = g.value(f, HW1.overProperty)
        # Occurrences store results through generic hw1:value.
        # Keep the internal observable key for measurement dispatch;
        # it is not serialized as a per-metric RDF predicate.
        if over is None:
            raise ValueError(
                f"{path}: hw1:QualityFactor {local!r} is missing hw1:overProperty.")
        out[local] = {"over": _local(over)}
    return out


    # Experiment file = student declaration + machine section, split at MACHINE_MARKER
    # and sealed by the declaration digest.
MACHINE_MARKER = "# ============ MACHINE SECTION (regenerated by api.py — do not edit) ============"


def _marker_offset(text):
    """Locate MACHINE_MARKER in an experiment file. Full strategy note: docs/triplestore.md."""
    start = 0
    while True:
        idx = text.find(MACHINE_MARKER, start)
        if idx < 0:
            return None
        end = idx + len(MACHINE_MARKER)
        at_line_start = idx == 0 or text[idx - 1] == "\n"
        at_line_end = end == len(text) or text[end] == "\n"
        if at_line_start and at_line_end:
            return idx
        start = idx + 1


def _split_sections(text):
    """Split declaration vs machine section at the marker. Full strategy note: docs/triplestore.md."""
    idx = _marker_offset(text)
    if idx is None:
        return text, None
    rest = text[idx + len(MACHINE_MARKER):]
    if rest.startswith("\n"):
        rest = rest[1:]
    return text[:idx], rest


def _declaration_digest(student_bytes):
    """SHA-256 seal of the declaration half. Full strategy note: docs/triplestore.md."""
    return hashlib.sha256(student_bytes).hexdigest()


def _read_text(path):
    """The whole file as text. Turtle is UTF-8 by specification (and so is this)."""
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _verify_seal(g, exp, student_text, path):
    """Check the stored seal against the declaration; hard error on mismatch. Full strategy note: docs/triplestore.md."""
    stored = g.value(exp, HW1.declarationDigest)
    if stored is None:
        raise ValueError(
            f"{path}: experiment {exp} carries no hw1:declarationDigest. Every "
            f"assessed file is sealed by `reconstruct.py` (§3.1), "
            f"so a machine section without a seal was not written by this program.")
    actual = _declaration_digest(student_text.encode("utf-8"))
    if str(stored) != actual:
        raise ValueError(
            f"{path}: THE DECLARATION WAS EDITED AFTER ASSESSMENT. The recorded "
            f"hw1:declarationDigest ({str(stored)[:12]}…) does not match the bytes "
            f"above the marker ({actual[:12]}…), so the verdicts below the marker "
            f"were computed for a treatment this file no longer declares. An "
            f"edited declaration IS A NEW EXPERIMENT AND NEEDS A NEW FILE: copy "
            f"the declaration to a new name and run `reconstruct.py --experiment "
            f"<newname>.ttl` (§3.1 — experiments are write-once).")


    # read_experiment / write_run: sole reader/writer of run triples.
def _sole_experiment(g, path):
    """Exactly one Experiment subject per file, else error. Full strategy note: docs/triplestore.md."""
    exps = sorted(set(g.subjects(RDF.type, HW1.Experiment)), key=str)
    if len(exps) != 1:
        raise ValueError(
            f"{path}: expected exactly one hw1:Experiment subject, found {len(exps)}"
            + (f": {', '.join(str(e) for e in exps)}" if exps else ""))
    return exps[0]


def _setting_value_to_python(lit, path, param_local):
    """Setting literal -> python value by parameter kind. Full strategy note: docs/triplestore.md."""
    dt = lit.datatype
    if dt == XSD.integer:
        return int(lit)
    if dt in (XSD.double, XSD.decimal, XSD.float):
        return float(lit)
    if dt is None or dt == XSD.string:
        return str(lit)
    raise ValueError(
        f"{path}: hw1:settingValue of {param_local!r} carries datatype {dt}; "
        f"§5 declares exactly one of xsd:double / xsd:integer / "
        f"xsd:string, one per hw1:paramValueKind")


def _experiment_settings(g, exp, path):
    """Experiment's full setting vector (declared + defaulted). Full strategy note: docs/triplestore.md."""
    settings = {}
    for setting in g.objects(exp, HW1.hasFactorSetting):
        param = g.value(setting, HW1.settingParameter)
        if param is None:
            raise ValueError(
                f"{path}: hw1:FactorSetting {setting} has no hw1:settingParameter")
        local = _local(param)
        lit = g.value(setting, HW1.settingValue)
        if lit is None:
            raise ValueError(
                f"{path}: hw1:FactorSetting for {local!r} has no hw1:settingValue")
        value = _setting_value_to_python(lit, path, local)
        if local in settings and settings[local] != value:
            raise ValueError(
                f"{path}: parameter {local!r} is recorded twice with different values "
                f"({settings[local]!r} and {value!r}); one experiment is ONE treatment "
                f"(§3.1: two levels are two experiments, and the second "
                f"one needs its own declaration file)")
        settings[local] = value
    return settings




def query_graph(graph, query_text):
    """Run a read-only SPARQL query against a local graph. Full strategy note: docs/triplestore.md."""
    try:
        return graph.query(query_text)
    except Exception as exc:
        raise ValueError(f"invalid or unsupported SPARQL query: {exc}") from None




def frames_from_sparql_query(exp, query_path):
    """Frame stems selected by a student SPARQL query over an experiment.

    CLI entry: ``--selection-query <file.rq>`` (requires ``--experiment``).
    Runs the query against the assessed experiment graph, binds ``?frame``
    (a frame IRI) or ``?frameIndex`` (an integer), constrains results to
    frames known to this experiment, and returns the sorted stem list to
    feed to ``utils.reconstruct(..., frames=[...])``. Callers cut the flat
    list into contiguous segments before reconstructing.
    """
    with open(query_path, "r", encoding="utf-8") as handle:
        query_text = handle.read()
    result = query_graph(exp["graph"], query_text)
    if getattr(result, "type", "SELECT") != "SELECT":
        raise ValueError("--selection-query must be a SPARQL SELECT query that binds "
                            "?frame or ?frameIndex")
    names = {str(var): var for var in result.vars}
    frame_var = names.get("frame")
    index_var = names.get("frameIndex")
    if frame_var is None and index_var is None:
        raise ValueError("--selection-query must bind ?frame (a frame IRI) or "
                            "?frameIndex (an integer)")
    selected = set()
    for row in result:
        from_frame = None if frame_var is None else row[frame_var]
        from_index = None if index_var is None else row[index_var]
        if from_frame is None and from_index is None:
            continue
        try:
            frame = (frame_index_from_iri(from_frame) if from_frame is not None
                        else int(from_index))
        except (TypeError, ValueError) as exc:
            raise ValueError("--selection-query returned an invalid ?frame or "
                                f"?frameIndex: {from_frame!r}, {from_index!r}") from exc
        if from_frame is not None and from_index is not None:
            try:
                if frame != int(from_index):
                    raise ValueError("disagreeing ?frame and ?frameIndex")
            except (TypeError, ValueError) as exc:
                raise ValueError("--selection-query returned a non-integer ?frameIndex: "
                                    f"{from_index!r}") from exc
        selected.add(frame)
    graph = exp["graph"]
    known = {
        int(index)
        for frame in graph.objects(exp["batch_iri"], HW1.hasFrame)
        for index in (graph.value(frame, HW1.frameIndex),)
        if index is not None
    }
    unknown = sorted(selected - known)
    if unknown:
        raise ValueError("--selection-query selected frame index/indices not present in "
                            f"this experiment: {', '.join(map(str, unknown))}")
    return sorted(selected)


def _semantic_occurrence_report(g, exp, b, path):
    """Validate the complete set of pipeline Factor occurrences."""
    factors = load_quality_factors()
    frames = []
    for frame in g.objects(b, HW1.hasFrame):
        idx = g.value(frame, HW1.frameIndex)
        rgb = g.value(frame, HW1.hasRGBImage)
        depth = g.value(frame, HW1.hasDepthImage)
        if idx is None or rgb is None or depth is None:
            raise ValueError(f"{path}: every Batch Frame needs frameIndex, RGBImage and DepthImage")
        frames.append((int(idx), rgb, depth))
    frames.sort()
    memberships = {image: idx for idx, rgb, depth in frames for image in (rgb, depth)}
    expected = set()
    for factor_local in _ALWAYS_MEASURED_FACTORS:
        info = factors.get(factor_local)
        if info is None:
            raise ValueError(f"{path}: pipeline factor {factor_local!r} is not in the TBox")
        observable = info["over"]
        if observable in _PIPELINE_FRAME_OBSERVABLES:
            kind = _PIPELINE_FRAME_OBSERVABLES[observable]["modality"]
            for idx, rgb, depth in frames:
                expected.add((factor_local, depth if kind == "depth" else rgb, None))
        elif observable in _PIPELINE_PAIR_OBSERVABLES:
            for (_i, _rgb0, d0), (_j, _rgb1, d1) in zip(frames, frames[1:]):
                expected.add((factor_local, d1, d0))
        else:
            raise ValueError(f"{path}: pipeline factor {factor_local!r} has no pipeline observable")

    seen, errors = {}, []
    measured = pending = 0
    for node in g.subjects(HW1.inExperiment, exp):
        definitions = list(g.objects(node, HW1.hasDefinition))
        current = list(g.objects(node, HW1.hasCurrentFrame))
        previous = list(g.objects(node, HW1.hasPrevious))
        if len(definitions) != 1 or len(current) != 1 or len(previous) > 1:
            errors.append(f"{node}: expected one definition/current image and at most one previous image")
            continue
        factor_local = _local(definitions[0])
        current_image = current[0]
        previous_image = previous[0] if previous else None
        if current_image not in memberships or (previous_image is not None and previous_image not in memberships):
            errors.append(f"{node}: image link is outside the declared Batch")
            continue
        key = factor_local, current_image, previous_image
        if key not in expected:
            errors.append(f"{node}: unexpected pipeline factor/image occurrence")
        elif key in seen:
            errors.append(f"{node}: duplicate pipeline factor/image occurrence")
        else:
            seen[key] = node
        states = list(g.objects(node, HW1.evaluationState))
        values = list(g.objects(node, HW1.value))
        if len(states) != 1 or states[0] not in (HW1.Pending, HW1.Measured):
            errors.append(f"{node}: evaluationState must be exactly Pending or Measured")
        elif states[0] == HW1.Measured:
            try:
                if len(values) != 1:
                    raise ValueError
                float(values[0])
                measured += 1
            except (TypeError, ValueError):
                errors.append(f"{node}: Measured occurrences need one numeric value")
        elif values:
            errors.append(f"{node}: Pending occurrences must not carry value")
            pending += 1
        else:
            pending += 1
    errors.extend(f"missing expected occurrence for {key[0]}" for key in sorted(expected - set(seen), key=str))
    return {"complete": bool(expected) and not errors and measured == len(expected),
            "expected": len(expected), "measured": measured, "pending": pending, "errors": errors}

def _read_semantic_experiment(g, exp, b, path):
    """Read pipeline Factor occurrences and their measured values."""
    completion = _semantic_occurrence_report(g, exp, b, path)
    frames = []
    for frame in g.objects(b, HW1.hasFrame):
        idx = g.value(frame, HW1.frameIndex)
        rgb = g.value(frame, HW1.hasRGBImage)
        depth = g.value(frame, HW1.hasDepthImage)
        if idx is None or rgb is None or depth is None:
            raise ValueError(f"{path}: every Batch Frame needs frameIndex, RGBImage and DepthImage")
        frames.append((int(idx), frame, rgb, depth))
    frames.sort(key=lambda row: row[0])
    memberships = {row[2]: row[0] for row in frames} | {row[3]: row[0] for row in frames}
    for node in g.subjects(HW1.inExperiment, exp):
        current = g.value(node, HW1.hasCurrentFrame)
        previous = g.value(node, HW1.hasPrevious)
        if current not in memberships or (previous is not None and previous not in memberships):
            raise ValueError(f"{path}: Factor {node} has foreign or incomplete image links")
    frame_indices = [idx for idx, *_ in frames]
    factor_counts = {}
    factors = load_quality_factors()
    for factor_local in factors:
        nodes = [n for n in g.subjects(HW1.inExperiment, exp)
                    if g.value(n, HW1.hasDefinition) == HW1[factor_local]]
        factor_counts[factor_local] = len(nodes)
    batch_path = g.value(b, HW1.batchPath)
    return {"exp_iri": URIRef(str(exp)), "exp_name": _experiment_name_from_iri(exp, path),
            "batch_name": _batch_name_from_batch_iri(b), "batch_iri": URIRef(str(b)),
            "batch_path": None if batch_path is None else str(batch_path),
            "settings": _experiment_settings(g, exp, path),
            "frame_indices": frame_indices, "factor_counts": factor_counts,
            "complete": completion["complete"], "completion": completion, "graph": g}


def read_experiment(path):
    """Canonical reader: declaration, settings, and measured values. Full strategy note: docs/triplestore.md."""
    text = _read_text(path)
    student_text, machine_text = _split_sections(text)
    if machine_text is None:
        raise ValueError(
            f"{path}: this file carries no MACHINE SECTION marker, so it is a "
            f"DECLARATION that has not been assessed yet — there are no values and "
            f"no verdicts in it to read. Run `reconstruct.py --experiment {path}` first "
            f"(`api.py explore {path}` shows what it declares in the meantime).")
    # ONE GRAPH, BOTH SECTIONS (§3.1). The machine section re-opens
    # the student's Experiment subject, so only the whole file carries the whole
    # experiment: the selection and the student's own settings are above the
    # marker, the measurements and the defaulted settings below it.
    full = Graph()
    full.parse(data=text, format="turtle")
    exp = _sole_experiment(full, path)
    schema_version = full.value(exp, HW1.schemaVersion)
    if schema_version is None or not str(schema_version).startswith("7"):
        raise ValueError(f"{path}: unsupported experiment vocabulary; expected schemaVersion 7.x pipeline occurrences")
    _verify_seal(full, exp, student_text, path)
    b = full.value(exp, HW1.onBatch)
    if b is None:
        raise ValueError(f"{path}: experiment {exp} has no hw1:onBatch")
    return _read_semantic_experiment(full, exp, b, path)


    # The IRI `<mode>` segment of §2 pinned to the hw1:SelectionMode
    # individual of §4.4. One dict, so "baseline" cannot end up meaning GoodSegments in
    # one caller and FullBatch in another.
_SELECTION_MODE = {"baseline": HW1.FullBatch,
                "selected": HW1.GoodSegments}

# Reconstruction-mechanism evidence (§14).  These are
# diagnostics, not quality factors. Keep the allow-list here so a misspelling
# cannot silently mint an RDF predicate that no reader knows about.
_RUN_METADATA = {
"gatedSteps": XSD.integer,
"spliceCount": XSD.integer,
"maxGapLength": XSD.integer,
}


def _write_machine_section(path, student_text, machine_graph):
    """Re-serialize only the machine half (declaration bytes preserved). Full strategy note: docs/triplestore.md."""
    body = _serialize_machine_graph(machine_graph)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(student_text)
        fh.write(MACHINE_MARKER + "\n\n")
        fh.write(body if body.endswith("\n") else body + "\n")
    os.replace(tmp, path)


def write_run(exp_path, mode, values, used_frames=None, frame_count=None,
                metadata=None, diagnostic_file=None):
    """THE one writer of hw1:ReconstructionRun triples (S8.2), idempotent per key. Full strategy note: docs/triplestore.md."""
    if mode not in _SELECTION_MODE:
        raise ValueError(
            f"mode must be one of {sorted(_SELECTION_MODE)}, got {mode!r}; the mode is "
            f"the last segment of the run IRI (§2) and is not free text")
    if used_frames is not None and mode != "selected":
        raise ValueError(
            f"used_frames given with mode={mode!r}: hw1:usedFrame is asserted by "
            f"selected subset runs only. A full-batch run states "
            f"its size with hw1:runFrameCount and lists no frames.")
    metadata = {} if metadata is None else dict(metadata)
    unknown_metadata = sorted(set(metadata) - set(_RUN_METADATA))
    if unknown_metadata:
        raise ValueError(
            f"unknown run metadata {unknown_metadata}; allowed statusless mechanism "
            f"properties: {', '.join(sorted(_RUN_METADATA))}")
    for key, value in metadata.items():
        if isinstance(value, bool) or int(value) != value or int(value) < 0:
            raise ValueError(f"run metadata {key} must be a non-negative integer, "
                                f"got {value!r}")

    text = _read_text(exp_path)
    student_text, machine_text = _split_sections(text)
    if machine_text is None:
        raise ValueError(
            f"{exp_path}: no machine section — a run cannot be written into a "
            f"declaration that was never assessed. Run `reconstruct.py --experiment "
            f"{exp_path}` first.")

    # The WHOLE file answers "which experiment, under which settings" (the
    # student's own settings are above the marker); the MACHINE half is what gets
    # mutated and written back. Two parses of one small file.
    full = Graph()
    full.parse(data=text, format="turtle")
    exp = _sole_experiment(full, exp_path)
    _verify_seal(full, exp, student_text, exp_path)
    expname = _experiment_name_from_iri(exp, exp_path)
    run = data_run_iri(expname, mode)

    planned = []
    for key in values:
        if key not in _RUN_OUTCOMES:
            raise ValueError(f"{key!r} is not a ReconstructionRun outcome")
        stored = _storable(values[key])
        planned.append((HW1[key], stored))

    frames = None
    if used_frames is not None:
        b = full.value(exp, HW1.onBatch)
        if b is None:
            raise ValueError(
                f"{exp_path}: experiment {exp} has no hw1:onBatch, so the frame IRIs "
                f"for hw1:usedFrame cannot be minted")
        name = _batch_name_from_batch_iri(b)
        frames = [frame_iri(name, int(idx)) for idx in used_frames]

    # Everything a run node consists of lives BELOW the marker, so the mutation
    # happens on the machine section alone and the declaration is never re-serialized.
    g = _parse_machine_graph(machine_text, student_text)

    # ── remove exactly what is about to be written, and nothing else ──────────
    for old_run in (run,):
        for value_p, _ in planned:
            g.remove((old_run, value_p, None))
        if frames is not None:
            g.remove((old_run, HW1.usedFrame, None))
        if frame_count is not None:
            g.remove((old_run, HW1.runFrameCount, None))
        if diagnostic_file is not None:
            g.remove((old_run, HW1.diagnosticFile, None))
        for key in metadata:
            g.remove((old_run, HW1[key], None))
        g.remove((old_run, HW1.selectionMode, None))

    # ── write ─────────────────────────────────────────────────────────────────
    g.add((run, RDF.type, HW1.ReconstructionRun))
    g.add((run, HW1.selectionMode, _SELECTION_MODE[mode]))
    g.add((exp, HW1.hasRun, run))
    for value_p, v in planned:
        # `_double_literal` is the one spelling of an xsd:double in this project, and
        # it rounds through `_storable` too.
        g.add((run, value_p, _double_literal(v)))
    if frame_count is not None:
        g.add((run, HW1.runFrameCount, Literal(int(frame_count), datatype=XSD.integer)))
    if frames is not None:
        for f in frames:
            g.add((run, HW1.usedFrame, f))
    if diagnostic_file is not None:
        g.add((run, HW1.diagnosticFile, Literal(str(diagnostic_file))))
    for key, value in metadata.items():
        g.add((run, HW1[key], Literal(int(value), datatype=_RUN_METADATA[key])))

    _write_machine_section(exp_path, student_text, g)


    # batch2ttl: structural graph of one capture.
def build_batch_graph(data_dir, floor, derived_from=None):
    """Batch/file structural graph (frames, images, gaps warned). Full strategy note: docs/triplestore.md."""
    g = Graph()
    name = batch_name(data_dir, floor)
    b = batch_iri(name)

    g.add((b, RDF.type, HW1.Batch))
    g.add((b, HW1.batchName, Literal(name)))
    g.add((b, HW1.batchPath, Literal(data_dir)))
    g.add((b, HW1.floor, Literal(int(floor), datatype=XSD.integer)))

    # prov:wasDerivedFrom connects BATCHES only: "these pixels came from those
    # pixels, corrupted". It is not asserted between experiments.
    if derived_from:
        g.add((b, PROV.wasDerivedFrom, batch_iri(derived_from)))

    frames = _pair_frames(data_dir)
    for stem, rgb_path, depth_path in frames:
        f = frame_iri(name, stem)
        g.add((b, HW1.hasFrame, f))
        g.add((f, RDF.type, HW1.Frame))
        g.add((f, HW1.frameIndex, Literal(int(stem), datatype=XSD.integer)))

        rc = component_iri(name, stem, "rgb")
        g.add((f, HW1.hasRGBImage, rc))
        g.add((rc, RDF.type, HW1.RGBImage))
        g.add((rc, SCHEMA.contentUrl, Literal(rgb_path)))

        dc = component_iri(name, stem, "depth")
        g.add((f, HW1.hasDepthImage, dc))
        g.add((dc, RDF.type, HW1.DepthImage))
        g.add((dc, SCHEMA.contentUrl, Literal(depth_path)))

    _warn_stem_gaps(frames, prefix="[batch2ttl]")
    return g, name, b


def _warn_stem_gaps(frames, prefix):
    """Log consecutive-stem holes in a paired capture. Shared by declare/batch2ttl."""
    gaps = [(int(s0), int(s1))
            for (s0, _, _), (s1, _, _) in zip(frames, frames[1:])
            if int(s1) - int(s0) != 1]
    if not gaps:
        return gaps
    shown = ", ".join(f"{i}->{j}" for i, j in gaps[:12])
    more = f" ... (+{len(gaps) - 12} more)" if len(gaps) > 12 else ""
    print(f"{prefix} WARNING: {len(gaps)} stem gap(s) in the paired sequence: "
            f"{shown}{more}")
    print(f"{prefix} WARNING: every experiment over this batch will pair across "
            f"those gaps, spanning more than one capture step. A stem missing from "
            f"rgb/ or depth/ changes which pairs exist, and the pair observables will "
            f"read as larger motion — correctly, but check it is the capture and not a "
            f"lost file.")
    return gaps


def cmd_batch2ttl(args):
    """Write a structural batch sidecar."""
    g, name, _ = build_batch_graph(args.data_dir, args.floor,
                                    derived_from=args.derived_from)
    out = args.out or os.path.join(args.data_dir, "batch.ttl")
    g.serialize(destination=out, format="turtle")
    n_frames = len(set(g.subjects(RDF.type, HW1.Frame)))
    print(f"[batch2ttl] batch {name!r}: {n_frames} frames -> {out}")
    if args.derived_from:
        print(f"[batch2ttl] prov:wasDerivedFrom {batch_iri(args.derived_from)}")
    print(f"[batch2ttl] wrote {len(g)} triples -> {out}")
    return 0



    # declare: scaffold a declaration Turtle.
def _declare_capture_args(args):
    """Capture args recorded in a declaration."""
    data_dir_arg = getattr(args, "data_dir", None)
    batch_file_arg = getattr(args, "batch_file", None)
    floor = int(getattr(args, "floor", 1) or 1)
    if data_dir_arg and batch_file_arg:
        raise SystemExit(
            "[declare] pass --data-dir (the capture directory) or --batch-file, "
            "not both.")
    if data_dir_arg:
        resolved = _resolve_batch_file(data_dir_arg)
        if not _is_capture_dir(resolved):
            raise SystemExit(
                f"[declare] --data-dir {data_dir_arg!r} does not resolve to a capture "
                f"directory with rgb/ and depth/ subdirs ({resolved!r}; relative "
                f"paths resolve against the current working directory).")
        return resolved, floor, data_dir_arg.replace(os.sep, "/")
    if not batch_file_arg:
        raise SystemExit(
            "[declare] --data-dir is required (the capture directory containing "
            "rgb/ and depth/).")
    print("[declare] using --batch-file; --data-dir is preferred for direct capture "
            "references. The scaffold records the capture directory in hw1:batchFile.",
            file=sys.stderr)
    resolved = _resolve_batch_file(batch_file_arg)
    if _is_capture_dir(resolved):
        return resolved, floor, batch_file_arg.replace(os.sep, "/")
    if os.path.isfile(resolved):
        _g, _batch, _name, stored_path, stored_floor = _parse_batch_ttl(resolved)
        data_dir = _capture_dir(stored_path, batch_file_arg)
        if stored_floor is not None:
            floor = stored_floor
        # Write the capture directory into the declaration, not the sidecar.
        return data_dir, floor, os.path.relpath(data_dir, os.getcwd()).replace(os.sep, "/")
    raise SystemExit(
        f"[declare] --batch-file {batch_file_arg!r} does not resolve to a capture "
        f"directory or a batch.ttl ({resolved!r}). Pass --data-dir <capture-dir>.")


def cmd_declare(args):
    """Scaffold a student declaration Turtle (never assesses, never overwrites). Full strategy note: docs/triplestore.md."""
    name = args.name
    if not _EXPNAME_RE.match(name):
        raise SystemExit(
            f"[declare] {name!r} is not a legal experiment name (§2: "
            f"[A-Za-z0-9_-]+). It becomes both the file stem and the tail of the "
            f"Experiment IRI, which must stay equal.")

    workspace = experiment_directory(name)
    out = args.out or os.path.join(workspace, f"{name}.ttl")
    stem = os.path.splitext(os.path.basename(out))[0]
    if stem != name:
        raise SystemExit(
            f"[declare] --out names the file {stem!r}.ttl but --name says {name!r}. "
            f"The file stem must equal the Experiment IRI tail (§2), so "
            f"the two flags must agree.")
    if os.path.exists(out):
        raise SystemExit(
            f"[declare] {out} already exists and experiments/ is append-only "
            f"(§3.1): a new tuning idea is a NEW declaration under a "
            f"new name, and an existing file — assessed or not — is never "
            f"regenerated over.")

    data_dir, floor, batch_file_text = _declare_capture_args(args)
    try:
        frames = _pair_frames(data_dir)
    except ValueError as exc:
        raise SystemExit(f"[declare] {exc}")
    _warn_stem_gaps(frames, prefix="[declare]")
    name_of_batch = batch_name(data_dir, floor)
    batch = batch_iri(name_of_batch)

    decls = load_parameter_declarations(_ONTOLOGY_TTL)
    factors = load_quality_factors(_ONTOLOGY_TTL)
    lines = [
        "# =============================================================================",
        f"# EXPERIMENT DECLARATION — {name}   (scaffolded by `api.py declare`)",
        "#",
        "# This file is YOURS until `reconstruct.py` seals it: edit the label, trim",
        "# add setting overrides. After assessment the student section is",
        "# digest-sealed — a new idea is a NEW file under a NEW name (§3.1).",
        "#",
        "# PREDICTION (write BEFORE assessing — the seal makes it non-retractable):",
        "#   pipeline: TODO — which automatically measured pipeline factors you",
        "#             expect to fail, and why",
        "#   baseline: TODO — expected mapMeanL2 band and verdict for the full-batch run",
        "# =============================================================================",
        "",
        "@prefix hw1:  <http://taica.course/hw1/ontology#> .",
        f"@prefix batch: <{DATA_NS}batch/> .",
        f"@prefix data:  <{DATA_NS}> .",
        f"@prefix exp:   <{EXP_NS}> .",
        f"@prefix factor: <{EXP_NS}{name}/factor/> .",
        f"@prefix run: <{EXP_NS}{name}/run/> .",
        f"@prefix setting: <{EXP_NS}{name}/setting/> .",
        f"@prefix frame: <{DATA_NS}batch/{name_of_batch}/frame/> .",
        f"@prefix rgb: <{DATA_NS}batch/{name_of_batch}/rgb/> .",
        f"@prefix depth: <{DATA_NS}batch/{name_of_batch}/depth/> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        "@prefix schema: <https://schema.org/> .",
        "@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .",
        "",
        f"exp:{name}",
        "    a hw1:Experiment ;",
        "    hw1:schemaVersion \"7.0.0\" ;",
        f"    rdfs:label \"TODO: one line naming the condition this experiment tests\"@en ;",
        f"    hw1:batchFile \"{batch_file_text}\" ;",
        f"    hw1:onBatch batch:{name_of_batch} .   # {name_of_batch}",
        "",
        "# All pipeline-derived factors are recorded automatically during reconstruction.",
        "#",
        "# OVERRIDES — every setting you do not declare is filled from its TBox default",
        "# at assessment time (preview with `explore` before committing). To override",
        "# one, replace the final \".\" above with \";\" and append a blank node, e.g.:",
        "#",
        "#     hw1:hasFactorSetting [",
        "#         a hw1:FactorSetting ;",
        "#         hw1:settingParameter hw1:icpBackend ;",
        "#         hw1:settingRole hw1:ReconstructionSetting ;",
        "#         hw1:settingForFactor hw1:ReconstructionRun ;",
        "#         hw1:settingValue \"my_icp\"^^xsd:string",
        "#     ] .",
        "#",
        "# hw1:settingRole must match the parameter's TBox hw1:paramRole.",
        "",
    ]

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    artifact_workspace = workspace if args.out is None else _workspace_for_experiment_file(out)
    os.makedirs(os.path.join(artifact_workspace, "plots"), exist_ok=True)
    os.makedirs(os.path.join(artifact_workspace, "queries"), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # The generator's own output must pass the validator it will later be read
    # by; a scaffold this command cannot re-read is a bug here, not user error.
    try:
        decl = read_declaration(out)
    except ValueError as exc:
        os.remove(out)
        raise SystemExit(
            f"[declare] scaffold failed its own validation and was removed — this "
            f"is a bug in `declare`, not in your input: {exc}")

    defaulted = sorted(set(decl["required"]) - set(decl["given"]))
    print(f"[declare] wrote {out}")
    print(f"[declare] experiment workspace: {artifact_workspace}")
    print("[declare] all pipeline-derived factors are measured automatically")
    print(f"[declare] {len(defaulted)} setting(s) WILL BE DEFAULTED at assessment; "
            f"preview them with:  api.py explore {out}")
    print(f"[declare] before assessing: replace the PREDICTION TODOs and the "
            f"rdfs:label — the seal makes them non-retractable. Then:  "
            f"reconstruct.py --experiment {out}")
    return 0


    # Pipeline assessment: one declaration in, one machine section out, once.
_PROGRESS_EVERY = 25

# Observable placement, measurement, and menu tables.
# Pipeline-sourced observables: measured from the reconstruction consumer's
# own intermediate state (downsampled clouds, FPFH descriptors, RANSAC/ICP
# results) via the pipeline callback, not from raw rasters. They ride every
# experiment like run factors: always measured, never selected.
_PIPELINE_FRAME_OBSERVABLES = {
"rawPointCount": {"modality": "depth", "params": ()},
"voxelPointYield": {"modality": "depth", "params": ()},
}

_PIPELINE_PAIR_OBSERVABLES = {
"coarseFitness": {"params": ()},
"coarseRmse": {"params": ()},
"coarseCorrespondences": {"params": ()},
"icpFitness": {"params": ()},
"icpRmse": {"params": ()},
"icpCorrespondences": {"params": ()},
}

# Factor locals always measured on every experiment (pipeline + run outcomes).
# Keyed by TBox local name; the observable key rides via load_quality_factors.
_ALWAYS_MEASURED_FACTORS = (
"RawPointCount",
"VoxelizedPointYield",
"CoarseRANSACFitness",
"CoarseRANSACRMSE",
"CoarseRANSACCorrespondences",
"ICPInlierFitness",
"ICPInlierRMSE",
"ICPInlierCorrespondences",
)


def make_pipeline_measurement_callback(factors):
    """Build the always-measured pipeline-factor callback.

    Consumes the per-frame/pair consumer-state dicts emitted by
    ``utils.reconstruct`` (downsampled clouds, FPFH descriptors, RANSAC/ICP
    results) and records plain ``{"value": v, "count": c}`` entries keyed by
    ``(factor_local, stem)`` for frames and ``(factor_local, i, j)`` for
    pairs. Frame entries arrive on ``kind="frame"`` calls, pair entries on
    ``kind="pair"`` calls; the first frame has no pair half.

    Returns ``(callback, frame_records, pair_records)``.
    """
    frame_records = {}
    pair_records = {}

    def pipeline_callback(kind, payload):
        if kind == "frame":
            stem = int(payload["stem"])
            frame_records[("RawPointCount", stem)] = {
                "value": float(payload["raw_point_count"])}
            frame_records[("VoxelizedPointYield", stem)] = {
                "value": float(payload["point_yield"])}
        elif kind == "pair":
            i, j = int(payload["prev_stem"]), int(payload["stem"])
            pair_records[("CoarseRANSACFitness", i, j)] = {
                "source": i, "target": j,
                "value": float(payload["coarse_fitness"])}
            pair_records[("CoarseRANSACRMSE", i, j)] = {
                "source": i, "target": j,
                "value": float(payload["coarse_rmse"])}
            pair_records[("CoarseRANSACCorrespondences", i, j)] = {
                "source": i, "target": j,
                "value": float(payload["coarse_correspondences"])}
            pair_records[("ICPInlierFitness", i, j)] = {
                "source": i, "target": j,
                "value": float(payload["icp_fitness"])}
            pair_records[("ICPInlierRMSE", i, j)] = {
                "source": i, "target": j,
                "value": float(payload["icp_rmse"])}
            pair_records[("ICPInlierCorrespondences", i, j)] = {
                "source": i, "target": j,
                "value": float(payload["icp_correspondences"])}
        else:
            raise ValueError(f"unknown pipeline payload kind {kind!r}")

    return pipeline_callback, frame_records, pair_records

    # Declaration whitelist: verdicts are computed, never declared.
_DECLARATION_EXPERIMENT_PREDICATES = (
RDF.type, RDFS.label, HW1.schemaVersion, HW1.batchFile, HW1.onBatch,
HW1.hasFactorSetting)
_DECLARATION_SETTING_PREDICATES = (
RDF.type, HW1.settingParameter, HW1.settingRole, HW1.settingForFactor,
HW1.settingValue)


def _fmt_term(term):
    """One RDF term as it would be written in the declaration. For error messages."""
    text = str(term)
    if isinstance(term, URIRef):
        return f"hw1:{text[len(NS):]}" if text.startswith(NS) else f"<{text}>"
    try:
        return term.n3()
    except Exception:                                    # pragma: no cover
        return repr(text)


def _fmt_triple(s, p, o):
    """`<s> <p> <o> .` in prefixed form — the offending triple every §4.2 error names."""
    return f"{_fmt_term(s)} {_fmt_term(p)} {_fmt_term(o)} ."


def _selectable_factors(factors):
    """Pipeline factors are automatic and no longer selectable."""
    return {}

def _run_factor_parameters(decls, factors):
    """Run-factor parameters (icpBackend, ...). Full strategy note: docs/triplestore.md."""
    return sorted(name for name, d in decls.items()
                    if d["role"] in _SETTING_ROLES and d["primary"] == "ReconstructionRun")


def _required_parameters(decls, factors):
    """Required parameter set (S4.2 completeness). Full strategy note: docs/triplestore.md."""
    required = set(_run_factor_parameters(decls, factors))
    for name, d in decls.items():
        if d["role"] == "ReconstructionSetting" and d["primary"] == "ReconstructionRun":
            required.add(name)
    return required


def _resolve_batch_file(batch_file):
    """Resolve hw1:batchFile to a readable capture. Full strategy note: docs/triplestore.md."""
    return batch_file if os.path.isabs(batch_file) else os.path.abspath(batch_file)


def _is_capture_dir(path):
    """True iff `path` is a capture directory: rgb/ and depth/ subdirs exist."""
    return (os.path.isdir(path)
            and os.path.isdir(os.path.join(path, "rgb"))
            and os.path.isdir(os.path.join(path, "depth")))


_FLOOR_BATCH_NAME_RE = re.compile(r"^floor(\d+)_(.+)$")


def _floor_from_batch_name(name):
    """Floor number from a floor-qualified batch name."""
    match = _FLOOR_BATCH_NAME_RE.match(name)
    if not match:
        raise ValueError(
            f"batch name {name!r} is not floor-qualified (expected "
            f"'floor<N>_<capture-dir-basename>', e.g. 'floor1_baseline')")
    return int(match.group(1))


def _sidecar_batch_ttl(data_dir):
    """Optional `<data_dir>/batch.ttl` structural sidecar."""
    path = os.path.join(data_dir, "batch.ttl")
    return path if os.path.isfile(path) else None


def _parse_batch_ttl(path):
    """Read one Batch out of a batch.ttl provenance sidecar.

    Returns (graph, batch_iri, batch_name, batch_path_literal, floor_or_None).
    """
    g = Graph()
    g.parse(path, format="turtle")
    batches = sorted(set(g.subjects(RDF.type, HW1.Batch)), key=str)
    if len(batches) != 1:
        raise ValueError(
            f"{path}: expected exactly one hw1:Batch subject, found {len(batches)}"
            + (f": {', '.join(str(b) for b in batches)}" if batches else "")
            + ". A batch file describes exactly one capture (§4.1).")
    batch = batches[0]
    if not str(batch).startswith(f"{DATA_NS}batch/"):
        raise ValueError(f"{path}: batch sidecar must use the pipeline schema data namespace")
    name_lit = g.value(batch, HW1.batchName)
    path_lit = g.value(batch, HW1.batchPath)
    floor_lit = g.value(batch, HW1.floor)
    if name_lit is None or path_lit is None:
        raise ValueError(
            f"{path}: the batch carries no hw1:batchName and/or no hw1:batchPath, "
            f"so neither the frame IRIs nor the pixels can be reached from it "
            f"(§4.1).")
    floor = int(floor_lit) if floor_lit is not None else None
    return g, URIRef(str(batch)), str(name_lit), str(path_lit), floor


def _capture_dir(batch_path, batch_file):
    """Capture dir for a batch name. Full strategy note: docs/triplestore.md."""
    candidates = [os.path.abspath(batch_path),
                    os.path.dirname(_resolve_batch_file(batch_file))]
    for candidate in candidates:
        if _is_capture_dir(candidate):
            return candidate
    raise ValueError(
        f"the batch's hw1:batchPath {batch_path!r} does not resolve to a capture "
        f"directory with rgb/ and depth/ subdirs. Tried "
        f"{', '.join(repr(c) for c in candidates)} (relative paths resolve against "
        f"the current working directory, {os.getcwd()!r}). Point hw1:batchFile at "
        f"the capture directory itself.")


def _resolve_declared_capture(batch_file, on_batch, path, exp, bf):
    """Resolve the declaration's capture to a data dir. Full strategy note: docs/triplestore.md."""
    resolved = _resolve_batch_file(batch_file)
    if _is_capture_dir(resolved):
        if on_batch is None:
            expected = batch_iri(batch_name(resolved, 1))
            raise ValueError(
                f"{path}: the experiment declares no hw1:onBatch. State the batch "
                f"IRI of the capture at {batch_file!r} — it is checked against the "
                f"directory, and that check is what catches 'measured capture A, "
                f"wrote into capture B'. Add:\n"
                f"    {_fmt_term(exp)} hw1:onBatch {_fmt_term(expected)} .")
        declared_name = _batch_name_from_batch_iri(on_batch)
        try:
            floor = _floor_from_batch_name(declared_name)
        except ValueError as exc:
            raise ValueError(
                f"{path}: hw1:onBatch names {_fmt_term(on_batch)}, which is not "
                f"a floor-qualified batch IRI of this assignment "
                f"(expected {DATA_NS}batch/floor<N>_<capture-dir-basename>). {exc} "
                f"Offending triple:\n    {_fmt_triple(exp, HW1.onBatch, on_batch)}")
        expected_name = batch_name(resolved, floor)
        expected_iri = batch_iri(expected_name)
        if URIRef(str(on_batch)) != expected_iri:
            raise ValueError(
                f"{path}: hw1:onBatch names {_fmt_term(on_batch)} but the capture "
                f"at {batch_file!r} is {_fmt_term(expected_iri)}. That is the "
                f"'measured capture A, wrote into capture B' bug at declaration "
                f"time — an error, not a warning (§4.2). Offending "
                f"triple:\n    {_fmt_triple(exp, HW1.onBatch, on_batch)}")
        return resolved, expected_iri, expected_name

    if os.path.isfile(resolved):
        _g, batch, name, stored_path, _floor = _parse_batch_ttl(resolved)
        data_dir = _capture_dir(stored_path, batch_file)
        if on_batch is None:
            raise ValueError(
                f"{path}: the experiment declares no hw1:onBatch. State the batch "
                f"IRI you believe {batch_file!r} describes — it is checked against "
                f"the file, and that check is what catches 'measured capture A, "
                f"wrote into capture B'. Add:\n"
                f"    {_fmt_term(exp)} hw1:onBatch {_fmt_term(batch)} .")
        if URIRef(str(on_batch)) != batch:
            raise ValueError(
                f"{path}: hw1:onBatch names {_fmt_term(on_batch)} but {batch_file!r} "
                f"describes {_fmt_term(batch)}. That is the 'measured capture A, "
                f"wrote into capture B' bug at declaration time — an error, not a "
                f"warning (§4.2). Offending triple:\n"
                f"    {_fmt_triple(exp, HW1.onBatch, on_batch)}")
        return data_dir, batch, name

    raise ValueError(
        f"{path}: hw1:batchFile {batch_file!r} does not resolve to a capture "
        f"directory with rgb/ and depth/ subdirs, or to a batch.ttl provenance sidecar "
        f"({resolved!r}; relative paths resolve against the current working "
        f"directory, {os.getcwd()!r}). Point it at the capture directory. "
        f"Offending triple:\n    {_fmt_triple(exp, HW1.batchFile, bf)}")


def _declaration_setting_value(node, param_local, decl, lit, path):
    """One declared setting value, parsed by kind. Full strategy note: docs/triplestore.md."""
    kind = decl["kind"]
    dt = lit.datatype
    triple = _fmt_triple(node, HW1.settingValue, lit)
    numeric = (XSD.double, XSD.decimal, XSD.float, XSD.integer)
    if kind == "string":
        if dt is not None and dt != XSD.string:
            raise ValueError(
                f"{path}: hw1:{param_local} is declared hw1:paramValueKind \"string\", "
                f"but its hw1:settingValue carries datatype {dt}. Offending triple:\n"
                f"    {triple}")
        return str(lit)
    if kind == "integer":
        if dt != XSD.integer:
            raise ValueError(
                f"{path}: hw1:{param_local} is declared hw1:paramValueKind \"integer\" "
                f"— write it as a bare integer (e.g. 20) or "
                f"\"20\"^^xsd:integer, not as {dt or 'an untyped literal'}. Offending "
                f"triple:\n    {triple}")
        return int(lit)
    if dt not in numeric:
        raise ValueError(
            f"{path}: hw1:{param_local} is declared hw1:paramValueKind \"double\", but "
            f"its hw1:settingValue is {dt or 'an untyped literal'}. Write "
            f"\"0.05\"^^xsd:double (or the bare Turtle form 0.05). Offending triple:\n"
            f"    {triple}")
    return float(lit)


def read_declaration(path):
    """Validate a student declaration: batch, selection, settings, prediction TODOs. Full strategy note: docs/triplestore.md."""
    text = _read_text(path)
    student_text, _machine_text = _split_sections(text)

    g = Graph()
    try:
        g.parse(data=student_text, format="turtle")
    except Exception as exc:
        raise ValueError(
            f"{path}: the declaration (everything above the machine marker) is not "
            f"valid Turtle: {exc}") from None

    decls = load_parameter_declarations(_ONTOLOGY_TTL)
    factors = load_quality_factors(_ONTOLOGY_TTL)

    # ── the one Experiment subject, and its name ──────────────────────────────
    exp = _sole_experiment(g, path)
    exp_name = _experiment_name_from_iri(exp, path)
    stem = os.path.splitext(os.path.basename(path))[0]
    if exp_name != stem:
        raise ValueError(
            f"{path}: the experiment names itself {exp_name!r} but the file is called "
            f"{stem!r}.ttl. The IRI tail IS the experiment's identity and so is the "
            f"file name (§2/§6); they must agree, and choosing one for "
            f"you would silently rename your experiment. Offending triple:\n"
            f"    {_fmt_triple(exp, RDF.type, HW1.Experiment)}")

    # ── the whitelist: verdicts are computed, never declared ──────────────────
    setting_nodes = set(g.objects(exp, HW1.hasFactorSetting))
    for s, p, o in g:
        if s == exp:
            if p not in _DECLARATION_EXPERIMENT_PREDICATES:
                raise ValueError(
                    f"{path}: {_fmt_term(p)} may not appear on a declared "
                    f"hw1:Experiment. A declaration states its batch, its factor "
                    f"batch and settings; factor occurrences and runs are COMPUTED by "
                    f"`reconstruct.py` and written "
                    f"below the marker (§4.2). Declarable predicates: "
                    f"{', '.join(_fmt_term(q) for q in _DECLARATION_EXPERIMENT_PREDICATES)}. "
                    f"Offending triple:\n    {_fmt_triple(s, p, o)}")
        elif s in setting_nodes:
            if p not in _DECLARATION_SETTING_PREDICATES:
                raise ValueError(
                    f"{path}: {_fmt_term(p)} may not appear on a hw1:FactorSetting. "
                    f"Declarable predicates: "
                    f"{', '.join(_fmt_term(q) for q in _DECLARATION_SETTING_PREDICATES)} "
                    f"(§4.2). Offending triple:\n    {_fmt_triple(s, p, o)}")
        else:
            raise ValueError(
                f"{path}: {_fmt_term(s)} is neither the declared hw1:Experiment nor "
                f"one of its hw1:hasFactorSetting nodes, so nothing in this project "
                f"reads it. A declaration describes ONE experiment and its settings "
                f"and nothing else (§4.2). Offending triple:\n"
                f"    {_fmt_triple(s, p, o)}")

    # ── the batch: declaration -> batchFile -> capture directory -> pixels ──
    bf = g.value(exp, HW1.batchFile)
    if bf is None:
        raise ValueError(
            f"{path}: the experiment declares no hw1:batchFile, so nothing says which "
            f"capture to measure. Add e.g.\n"
            f"    {_fmt_term(exp)} hw1:batchFile \"eval/first_floor\" .\n"
            f"(§4.2: the path to the capture directory, resolved against "
            f"the current working directory when relative)")
    batch_file = str(bf)
    on_batch = g.value(exp, HW1.onBatch)
    schema_term = g.value(exp, HW1.schemaVersion)
    if schema_term is None:
        raise ValueError(f"{path}: pipeline occurrence declarations require hw1:schemaVersion 7.x")
    schema_version = str(schema_term)
    if not schema_version.startswith("7"):
        raise ValueError(f"{path}: unsupported schemaVersion {schema_version!r}; expected 7.x pipeline occurrence vocabulary")
    data_dir, batch, batch_name_str = _resolve_declared_capture(
        batch_file, on_batch, path, exp, bf)

    # ── the settings ──────────────────────────────────────────────────────────
    run_params = set(_run_factor_parameters(decls, factors))
    given = {}
    for node in sorted(setting_nodes, key=str):
        param_term = g.value(node, HW1.settingParameter)
        if param_term is None:
            raise ValueError(
                f"{path}: a hw1:FactorSetting has no hw1:settingParameter, so nothing "
                f"says what it sets. Offending triple:\n"
                f"    {_fmt_triple(exp, HW1.hasFactorSetting, node)}")
        local = _local(param_term)
        if local not in decls:
            raise ValueError(
                f"{path}: {_fmt_term(param_term)} is not a declared hw1:Parameter. "
                f"Parameter names come from the TBox ({_ONTOLOGY_TTL}), not from the "
                f"declaration; a typo surviving as an extra FactorSetting would record "
                f"a treatment nobody ran. Declared parameters: "
                f"{', '.join(sorted(decls))}. Offending triple:\n"
                f"    {_fmt_triple(node, HW1.settingParameter, param_term)}")
        decl = decls[local]

        role_term = g.value(node, HW1.settingRole)
        if role_term is None or _local(role_term) != decl["role"]:
            raise ValueError(
                f"{path}: the setting of hw1:{local} declares hw1:settingRole "
                f"{_fmt_term(role_term) if role_term is not None else '(absent)'}, but "
                f"the TBox declares hw1:paramRole hw1:{decl['role']}. The role says "
                f"WHERE a setting lives and WHAT TO FIX when it is the culprit, so it "
                f"is checked rather than copied. Offending triple:\n"
                f"    {_fmt_triple(node, HW1.settingRole, role_term)}")

        for_factors = list(g.objects(node, HW1.settingForFactor))
        if len(for_factors) != 1 or URIRef(str(for_factors[0])) != decl["primaryIri"]:
            found = ", ".join(_fmt_term(f) for f in for_factors) or "(none)"
            primary = _fmt_term(decl["primaryIri"])
            raise ValueError(
                f"{path}: the setting of hw1:{local} must carry exactly one "
                f"hw1:settingForFactor, its parameter's hw1:paramPrimaryFactor "
                f"{primary} — it is single-valued (§4.5) and "
                f"readers traverse hw1:paramAffectsFactor in the TBox for the rest. "
                f"Found: {found}. Write:\n"
                f"    {_fmt_triple(node, HW1.settingForFactor, decl['primaryIri'])}")

        lit = g.value(node, HW1.settingValue)
        if lit is None or not isinstance(lit, Literal):
            raise ValueError(
                f"{path}: the setting of hw1:{local} carries no hw1:settingValue "
                f"literal, so it records no level at all. Offending triple:\n"
                f"    {_fmt_triple(exp, HW1.hasFactorSetting, node)}")
        value = _declaration_setting_value(node, local, decl, lit, path)

        touched = set((decl["primary"],) + tuple(decl["affects"]))
        if decl["primary"] != "ReconstructionRun" and not touched & set(_ALWAYS_MEASURED_FACTORS):
            raise ValueError(
                f"{path}: hw1:{local} does not affect a pipeline factor or ReconstructionRun outcome; "
                f"drop the setting. Offending triple:\n    {_fmt_triple(node, HW1.settingParameter, param_term)}")

        stored = _storable(value) if decl["kind"] == "double" else value
        if local in given and given[local] != stored:
            raise ValueError(
                f"{path}: hw1:{local} is set twice, to {given[local]!r} and to "
                f"{stored!r}. One parameter has one level per experiment; two levels "
                f"are two experiments, and the second one needs its own declaration "
                f"file (§3.1). Offending triple:\n"
                f"    {_fmt_triple(node, HW1.settingValue, lit)}")
        given[local] = stored

    # ── completeness: the pipeline and run required setting set (§4.2) ────────────────
    # `given` is a subset of this by construction — the scope check above rejects
    # any setting that touches no pipeline factor or run outcome — so the union only makes
    # "student-given wins" total rather than adding anything.
    required_names = _required_parameters(decls, factors) | set(given)
    required = {}
    for param in sorted(required_names):
        if param in given:
            required[param] = given[param]
            continue
        decl = decls[param]
        if decl["default"] is None:
            raise ValueError(
                f"{path}: hw1:{param} is required by the pipeline and reconstruction outcomes "
                f"(§4.2) but declares no hw1:paramDefault, and the declaration does not "
                f"set it. Either add a hw1:FactorSetting for it or declare a default "
                f"in {_ONTOLOGY_TTL}.")
        required[param] = (_storable(decl["default"]) if decl["kind"] == "double"
                            else decl["default"])

    return {"exp_iri": URIRef(str(exp)),
            "exp_name": exp_name,
            "schema_version": schema_version,
            "batch_file": batch_file,
            "batch_iri": URIRef(str(batch)),
            "batch_name": batch_name_str,
            "batch_path": data_dir,
            "given": given,
            "required": required}


def _bind_readable_namespaces(g, batch_name_str, expname):
    """Bind short Turtle prefixes so output stays readable."""
    g.bind("hw1", HW1); g.bind("schema", SCHEMA); g.bind("xsd", XSD)
    g.bind("exp", Namespace(EXP_NS))
    g.bind("data", Namespace(DATA_NS))
    g.bind("batch", Namespace(f"{DATA_NS}batch/"))
    g.bind("frame", Namespace(f"{DATA_NS}batch/{batch_name_str}/frame/"))
    g.bind("rgb", Namespace(f"{DATA_NS}batch/{batch_name_str}/rgb/"))
    g.bind("depth", Namespace(f"{DATA_NS}batch/{batch_name_str}/depth/"))
    g.bind("factor", Namespace(f"{EXP_NS}{expname}/factor/"))
    g.bind("setting", Namespace(f"{EXP_NS}{expname}/setting/"))
    g.bind("run", Namespace(f"{EXP_NS}{expname}/run/"))


def _serialize_machine_graph(graph):
    """Serialize machine triples without duplicating declaration prefixes."""
    body = graph.serialize(format="turtle")
    if isinstance(body, bytes):                      # rdflib < 6 returned bytes
        body = body.decode("utf-8")

    # The declaration section already owns the namespace directives.  The
    # machine section is parsed with those directives prepended by
    # _parse_machine_graph(), so discard rdflib's repeated directives here.
    lines = body.splitlines()
    while lines and (
        not lines[0].strip()
        or lines[0].lstrip().startswith("@prefix ")
        or lines[0].lstrip().startswith("@base ")
    ):
        lines.pop(0)
    return "\n".join(lines).lstrip() + "\n"


def _parse_machine_graph(machine_text, student_text):
    """Parse a relative machine section with declaration directives in scope."""
    directives = re.findall(
        r"^\s*(?:@prefix\s+[A-Za-z][\w-]*:\s*<[^>]+>\s*\.|"
        r"@base\s*<[^>]+>\s*\.)\s*$", student_text, re.MULTILINE)
    graph = Graph()
    source = ("\n".join(line.strip() for line in directives) + "\n" +
                machine_text)
    graph.parse(data=source, format="turtle")
    return graph


def _mint_pipeline_occurrences(g, exp, expname, name, frames,
                                pipeline_frame_records, pipeline_pair_records):
    """Mint always-measured pipeline Factor occurrences into the machine graph."""
    minted = 0
    frame_map = {
        "RawPointCount": "rawPointCount",
        "VoxelizedPointYield": "voxelPointYield",
    }
    for factor_local, over in frame_map.items():
        spec = _PIPELINE_FRAME_OBSERVABLES[over]
        for stem, _rgb_path, _depth_path in frames:
            stem_int = int(stem)
            node = data_factor_iri(expname, factor_local, stem)
            image = component_iri(name, stem, spec["modality"])
            g.add((node, RDF.type, HW1.Factor))
            g.add((node, RDF.type, HW1.SingleImageFactor))
            g.add((node, HW1.inExperiment, exp))
            g.add((node, HW1.hasDefinition, HW1[factor_local]))
            g.add((node, HW1.hasCurrentFrame, image))
            if (factor_local, stem_int) not in pipeline_frame_records:
                raise ValueError(
                    f"missing pipeline frame measurement for factor {factor_local} stem {stem_int}")
            value = pipeline_frame_records[(factor_local, stem_int)]["value"]
            g.add((node, HW1.value, _double_literal(value)))
            g.add((node, HW1.evaluationState, HW1.Measured))
            minted += 1
    pair_list = [
        ("CoarseRANSACFitness", "coarseFitness"),
        ("CoarseRANSACRMSE", "coarseRmse"),
        ("CoarseRANSACCorrespondences", "coarseCorrespondences"),
        ("ICPInlierFitness", "icpFitness"),
        ("ICPInlierRMSE", "icpRmse"),
        ("ICPInlierCorrespondences", "icpCorrespondences"),
    ]
    for factor_local, over in pair_list:
        spec = _PIPELINE_PAIR_OBSERVABLES[over]
        for (s0, _rgb0, _d0), (s1, _rgb1, _d1) in zip(frames, frames[1:]):
            i, j = int(s0), int(s1)
            node = data_factor_iri(expname, factor_local, j, i)
            g.add((node, RDF.type, HW1.Factor))
            g.add((node, RDF.type, HW1.DepthPairFactor))
            g.add((node, HW1.inExperiment, exp))
            g.add((node, HW1.hasDefinition, HW1[factor_local]))
            g.add((node, HW1.hasPrevious, component_iri(name, s0, "depth")))
            g.add((node, HW1.hasCurrentFrame, component_iri(name, s1, "depth")))
            if (factor_local, i, j) not in pipeline_pair_records:
                raise ValueError(
                    f"missing pipeline pair measurement for factor {factor_local} pair ({i}, {j})")
            rec = pipeline_pair_records[(factor_local, i, j)]
            value = rec["value"]
            g.add((node, HW1.value, _double_literal(value)))
            g.add((node, HW1.evaluationState, HW1.Measured))
            minted += 1
    return minted

def build_machine_graph(decl, frames, digest, decls=None, factors=None,
                        pipeline_frame_records=None, pipeline_pair_records=None):
    """Emit the machine graph from measured pipeline-derived records."""
    decls = load_parameter_declarations(_ONTOLOGY_TTL) if decls is None else decls
    factors = load_quality_factors(_ONTOLOGY_TTL) if factors is None else factors

    exp, expname = decl["exp_iri"], decl["exp_name"]
    name, settings = decl["batch_name"], decl["required"]
    data_dir = decl["batch_path"]
    g = Graph()
    _bind_readable_namespaces(g, name, expname)
    b = batch_iri(name)
    g.add((exp, HW1.declarationDigest, Literal(digest))); g.add((exp, HW1.onBatch, b))
    g.add((b, RDF.type, HW1.Batch)); g.add((b, HW1.batchName, Literal(name)))
    g.add((b, HW1.batchPath, Literal(data_dir)))
    for stem, rgb_path, depth_path in frames:
        f = frame_iri(name, stem); rgb = component_iri(name, stem, "rgb"); dep = component_iri(name, stem, "depth")
        g.add((b, HW1.hasFrame, f)); g.add((f, RDF.type, HW1.Frame)); g.add((f, HW1.frameIndex, Literal(int(stem), datatype=XSD.integer)))
        g.add((f, HW1.hasRGBImage, rgb)); g.add((rgb, RDF.type, HW1.RGBImage)); g.add((rgb, SCHEMA.contentUrl, Literal(rgb_path)))
        g.add((f, HW1.hasDepthImage, dep)); g.add((dep, RDF.type, HW1.DepthImage)); g.add((dep, SCHEMA.contentUrl, Literal(depth_path)))
    for param in sorted(set(settings) - set(decl["given"])):
        d = decls[param]; s = data_setting_iri(expname, d["primary"], param)
        for triple in ((exp, HW1.hasFactorSetting, s), (s, RDF.type, HW1.FactorSetting), (s, HW1.settingParameter, d["iri"]),
                        (s, HW1.settingRole, HW1[d["role"]]), (s, HW1.settingForFactor, d["primaryIri"])): g.add(triple)
        g.add((s, HW1.settingValue, _setting_value_literal(settings[param], d["kind"])))

    count = _mint_pipeline_occurrences(
        g, exp, expname, name, frames,
        pipeline_frame_records or {}, pipeline_pair_records or {})
    return g, {"frames": len(frames), "factors": count}


def seal_experiment(path, decl, frames, pipeline_frame_records=None,
                    pipeline_pair_records=None):
    """Seal the experiment declaration by appending the machine section once."""
    text = _read_text(path)
    if _marker_offset(text) is not None:
        raise ValueError(f"{path}: experiment is already sealed")

    decls = load_parameter_declarations(_ONTOLOGY_TTL)
    factors = load_quality_factors(_ONTOLOGY_TTL)

    student_text = text if text.endswith("\n") else text + "\n"
    digest = _declaration_digest(student_text.encode("utf-8"))

    g, counts = build_machine_graph(
        decl, frames, digest, decls=decls, factors=factors,
        pipeline_frame_records=pipeline_frame_records,
        pipeline_pair_records=pipeline_pair_records)

    body = _serialize_machine_graph(g)
    with open(path, "a", encoding="utf-8") as fh:
        if not text.endswith("\n"):
            fh.write("\n")
        fh.write(MACHINE_MARKER + "\n\n")
        fh.write(body if body.endswith("\n") else body + "\n")

    occurrences = [node for node in g.subjects(HW1.inExperiment, decl["exp_iri"])]
    print(f"[experiment] batch {decl['batch_name']!r}: {counts['frames']} frames -> "
            f"{counts['factors']} pipeline factor occurrence(s)")
    print(f"[experiment] experiment IRI  {decl['exp_iri']}")
    print(f"[experiment] declarationDigest {digest}")
    print(f"[experiment] appended {len(g)} triples below the marker -> {path}")
    print(f"[experiment] this file is now SEALED: edit it above the marker and every "
            f"reader will refuse it. Next: `api.py explore {path}`, then "
            f"`reconstruct.py --experiment {path}`.")
    return digest, counts


def is_assessed(path):
    """True when the experiment file already carries the machine marker."""
    return _marker_offset(_read_text(path)) is not None




_ROLE_FIX = {
"ReconstructionSetting": "change the reconstruction setup and re-run",
}
_COL_ABBREV = 10
_INDENT = "  "
_WRAP = 92

def _rule(title):
    """A top-level section heading."""
    print()
    print(title)
    print("=" * len(title))


def _sub(title):
    """A block heading inside a view."""
    print()
    print(title)
    print("-" * len(title))


def _kv(label, value):
    """One `label   value` line of a header block.

    Deliberately NOT wrapped: the values here are IRIs and file paths, and a
    wrapped path cannot be copied out of a terminal in one go.

    The gap is a MINIMUM, not a fixed width: experiment names are student-chosen
    and routinely overrun 20 columns, and a name run flush into the path it labels
    is unreadable exactly where the reader most needs to tell them apart.
    """
    gap = max(20 - len(label), 2)
    print(f"{_INDENT}{label}{' ' * gap}{value}")


def _note(text, indent=_INDENT):
    """Prose, greedy-wrapped to `_WRAP` columns. Tables are never wrapped; notes are.

    A wrapped TABLE loses the column alignment that makes it scannable, so the
    per-frame table stays on one line per frame however wide it gets (see the
    section header). A wrapped SENTENCE loses nothing, and the explanations
    around these tables carry the contract references a reader needs.
    """
    line = ""
    for word in text.split():
        if line and len(indent) + len(line) + 1 + len(word) > _WRAP:
            print(indent + line)
            line = word
        else:
            line = f"{line} {word}" if line else word
    if line:
        print(indent + line)


def _render_table(headers, rows, right=()):
    """Aligned ASCII table -> a list of lines. `right` = column indices to right-align.

    Deliberately hand-rolled: `explore` must run under the same rdflib-only stack
    as everything else (§7/§10 removed the last non-stdlib
    dependency this project had beyond numpy/Pillow/rdflib), and a table is
    twelve lines of `str.ljust`.
    """
    head = [str(h) for h in headers]
    body = [["" if c is None else str(c) for c in row] for row in rows]
    widths = [len(h) for h in head]
    for row in body:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(cells):
        out = [(cell.rjust(widths[i]) if i in right else cell.ljust(widths[i]))
                for i, cell in enumerate(cells)]
        return (_INDENT + "  ".join(out)).rstrip()

    lines = [fmt(head), _INDENT + "  ".join("-" * w for w in widths)]
    lines.extend(fmt(row) for row in body)
    return lines


def _print_table(headers, rows, right=()):
    """`_render_table`, printed. An empty table says so rather than printing nothing."""
    if not rows:
        print(f"{_INDENT}(no rows)")
        return
    for line in _render_table(headers, rows, right):
        print(line)


def _fmt_num(term):
    """One numeric literal as a narrow fixed-shape string. INF / -INF / NaN survive.

    The fail-closed sentinels of §9 are values a reader must SEE — an `inf`
    printed as `1.0e+308` or, worse, silently reformatted to something finite,
    would hide exactly the rows the convention exists to surface.
    """
    if term is None:
        return "-"
    try:
        v = float(term)
    except (TypeError, ValueError):
        return str(term)
    if v != v:
        return "NaN"
    if v == float("inf"):
        return "INF"
    if v == float("-inf"):
        return "-INF"
    if v == 0.0:
        return "0.0000"
    if 1e-3 <= abs(v) < 1e5:
        return f"{v:.4f}"
    return f"{v:.2e}"


def _fmt_setting(value):
    """One recorded setting level, printed the way its Python type reads."""
    if isinstance(value, float):
        return _fmt_num(value)
    return str(value)


def _abbrev(names, width=_COL_ABBREV):
    """{name: short name} for column headers, or identity if truncation collides."""
    short = {n: n[:width] for n in names}
    if len(set(short.values())) != len(short):
        return {n: n for n in names}
    return short



    # ── view dispatch: by CONTENT, never by filename (§7.1) ──────────
def _classify(path):
    """What kind of thing this is: "batch" | "declaration" | "experiment".

    BY CONTENT, not by name. A capture directory (rgb/ + depth/) is a batch;
    a `hw1:Batch` subject is a batch.ttl provenance sidecar; a `hw1:Experiment`
    subject is an experiment file, assessed iff the file carries
    `MACHINE_MARKER` (§3.1). Anything else is an error that NAMES what it
    found, because "unrecognised file" with no evidence is the least useful
    error a reader can get.
    """
    if _is_capture_dir(path):
        return "batch"
    text = _read_text(path)
    g = Graph()
    try:
        g.parse(data=text, format="turtle")
    except Exception as exc:
        raise ValueError(f"{path}: not valid Turtle: {exc}") from None

    batches = sorted(set(g.subjects(RDF.type, HW1.Batch)), key=str)
    exps = sorted(set(g.subjects(RDF.type, HW1.Experiment)), key=str)
    assessed = _marker_offset(text) is not None

    if batches and exps:
        # Assessed files embed the Batch snapshot so a single Turtle file is
        # self-contained for SPARQL. Batch sidecars and experiment files remain
        # mutually exclusive.
        version = g.value(exps[0], HW1.schemaVersion)
        has_factors = any(g.subjects(HW1.inExperiment, exps[0]))
        if (version is not None and str(version).startswith("7")) or has_factors:
            return "experiment" if assessed else "declaration"
        raise ValueError(
            f"{path}: this file declares BOTH a hw1:Batch ({batches[0]}) and a "
            f"hw1:Experiment ({exps[0]}). §3 keeps them in separate "
            f"files — structure in the capture directory, measurement in "
            f"experiments/<expname>.ttl — so `explore` cannot tell which view "
            f"you want.")
    if batches:
        return "batch"
    if exps:
        return "experiment" if assessed else "declaration"

    types = sorted({_local(o) for o in g.objects(None, RDF.type)})
    raise ValueError(
        f"{path}: neither a capture directory nor an experiment file. `explore` "
        f"needs a directory with rgb/ + depth/ (the batch view) or a subject typed "
        f"hw1:Experiment (the declaration / assessed views). This file declares "
        + (f"{len(g)} triple(s) with rdf:type " + ", ".join('hw1:' + t for t in types)
            if types else f"{len(g)} triple(s) and no rdf:type at all")
        + ".")


    # ── the batch view (§7.1 view 1) ────────────────────────────────
def _batch_facts_from_ttl(path):
    """Batch header/table facts from a batch.ttl provenance sidecar."""
    g, b, name, stored_path, floor = _parse_batch_ttl(path)
    frames = []
    for f in g.objects(b, HW1.hasFrame):
        idx_lit = g.value(f, HW1.frameIndex)
        idx = int(idx_lit) if idx_lit is not None else frame_index_from_iri(f)
        rgb = g.value(f, HW1.hasRGBImage)
        depth = g.value(f, HW1.hasDepthImage)
        frames.append((idx,
                        g.value(rgb, SCHEMA.contentUrl) if rgb is not None else None,
                        g.value(depth, SCHEMA.contentUrl) if depth is not None else None))
    frames.sort(key=lambda row: row[0])
    stems = [idx for idx, _, _ in frames]
    gaps = [(i, j) for i, j in zip(stems, stems[1:]) if j - i != 1]
    return {"path": path,
            "iri": b,
            "name": name,
            "batch_path": stored_path,
            "floor": floor,
            "frames": frames,
            "gaps": gaps,
            "derived_from": g.value(b, PROV.wasDerivedFrom)}


def _batch_facts_from_capture(data_dir, declared_name=None):
    """Batch header/table facts from the rasters on disk.

    An optional `<data_dir>/batch.ttl` sidecar supplies batch structure and
    source lineage; the frame list always comes from `_pair_frames`.
    """
    sidecar = _sidecar_batch_ttl(data_dir)
    derived_from, floor, name, iri = None, None, None, None
    if sidecar is not None:
        try:
            sidecar_facts = _batch_facts_from_ttl(sidecar)
            derived_from = sidecar_facts["derived_from"]
            floor = sidecar_facts["floor"]
            name = sidecar_facts["name"]
            iri = sidecar_facts["iri"]
        except Exception:                                # noqa: BLE001 — sidecar is optional
            pass
    frames = [(int(stem), rgb, depth) for stem, rgb, depth in _pair_frames(data_dir)]
    if declared_name is not None:
        name = declared_name
        iri = batch_iri(declared_name)
        try:
            floor = _floor_from_batch_name(declared_name)
        except ValueError:
            pass
    elif name is None:
        floor = 1 if floor is None else int(floor)
        name = batch_name(data_dir, floor)
        iri = batch_iri(name)
    stems = [idx for idx, _, _ in frames]
    gaps = [(i, j) for i, j in zip(stems, stems[1:]) if j - i != 1]
    return {"path": sidecar if sidecar is not None else data_dir,
            "iri": iri,
            "name": name,
            "batch_path": data_dir,
            "floor": floor,
            "frames": frames,
            "gaps": gaps,
            "derived_from": derived_from}


def _batch_facts(path, declared_name=None):
    """Everything the batch header and frame table need.

    Accepts a capture directory (preferred) or a batch.ttl provenance sidecar.
    Shared by view 1 (the batch itself) and view 2 (the capture a declaration
    names in `hw1:batchFile`), so the two print the SAME header from the same
    code.
    """
    resolved = path if os.path.isabs(path) else os.path.abspath(path)
    if _is_capture_dir(resolved):
        return _batch_facts_from_capture(resolved, declared_name=declared_name)
    if os.path.isfile(resolved):
        return _batch_facts_from_ttl(resolved)
    raise ValueError(
        f"{path}: not a capture directory (rgb/ + depth/) and not a batch.ttl "
        f"({resolved!r}).")


def _print_batch_header(facts):
    """Print capture identity plus the metadata needed before reconstruction."""
    _kv("batch", facts["name"])
    _kv("floor", facts["floor"])
    _kv("batchPath", facts["batch_path"])
    if os.path.isfile(str(facts["path"])):
        _kv("batch.ttl", facts["path"])
    stems = [idx for idx, _, _ in facts["frames"]]
    span = f", stems {stems[0]}..{stems[-1]}" if stems else ""
    _kv("frames", f"{len(stems)}{span}")
    capture_dir = facts.get("batch_path")
    if capture_dir and os.path.isdir(capture_dir):
        intrinsics_path = os.path.join(capture_dir, "intrinsics.json")
        if not os.path.isfile(intrinsics_path):
            _kv("intrinsics", "MISSING (reconstruction cannot determine the camera)")
        else:
            try:
                with open(intrinsics_path, encoding="utf-8") as handle:
                    intrinsics = json.load(handle)
                keys = ("width", "height", "hfov")
                if not all(key in intrinsics for key in keys):
                    _kv("intrinsics", "INVALID (requires width, height, hfov)")
                else:
                    _kv("intrinsics", f"{int(intrinsics['width'])}x"
                                        f"{int(intrinsics['height'])}, "
                                        f"hfov={float(intrinsics['hfov']):g} deg")
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                _kv("intrinsics", f"INVALID ({exc})")

        gt_path = os.path.join(capture_dir, "GT_pose.npy")
        if not os.path.isfile(gt_path):
            _kv("GT poses", "MISSING (Mean L2 evaluation unavailable)")
        else:
            try:
                gt = np.load(gt_path, allow_pickle=False)
                valid_shape = gt.ndim == 2 and gt.shape[1] == 7
                _kv("GT poses", f"shape={tuple(gt.shape)}"
                                + ("" if valid_shape else " INVALID; requires (N, 7)"))
            except (OSError, ValueError) as exc:
                _kv("GT poses", f"INVALID ({exc})")
    if facts["derived_from"] is not None:
        _kv("derivedFrom", _fmt_term(facts["derived_from"]))
    gaps = facts["gaps"]
    if gaps:
        shown = ", ".join(f"{i}->{j} ({j - i - 1} stem(s) missing)" for i, j in gaps)
        _kv("stem gaps", f"{len(gaps)}: {shown}")
        _note("every experiment over this batch pairs ACROSS those gaps, so those "
                "pairs span more than one capture step and their observables read as "
                "larger motion.", indent=_INDENT + " " * 20)
    else:
        _kv("stem gaps", "none — the paired stems are consecutive")


def _view_batch(path):
    """View 1: batch header + the frame table, with stem gaps called out."""
    facts = _batch_facts(path)
    _rule(f"BATCH  {facts['name']}")
    _print_batch_header(facts)

    _sub("frames")
    rows = [(idx, rgb, depth) for idx, rgb, depth in facts["frames"]]
    _print_table(("frame", "rgb", "depth"), rows, right=(0,))
    _note(f"{len(rows)} frame(s), all shown. A capture holds no measured value by "
            f"contract (§4.1); write a declaration naming it in hw1:batchFile and run "
            f"`reconstruct.py` to reconstruct one.")
    return 0


    # ── the declaration view (§7.1 view 2) ──────────────────────────
def _settings_rows(names, values, decls, source_of):
    """One row per recorded/required parameter: value, role, primary factor, source."""
    rows = []
    for param in sorted(names):
        d = decls.get(param)
        rows.append((param,
                        _fmt_setting(values[param]),
                        d["role"] if d else "(undeclared)",
                        d["primary"] if d else "-",
                        source_of(param)))
    return rows


def _view_declaration(path):
    """View an unassessed pipeline-only declaration and its setting vector."""
    decl = read_declaration(path)
    decls = load_parameter_declarations(_ONTOLOGY_TTL)
    _rule(f"DECLARATION  {decl['exp_name']}  (not yet assessed)")
    _kv("file", path)
    _kv("experiment IRI", decl["exp_iri"])
    _kv("batch", decl["batch_name"])
    _sub("settings")
    given = decl["given"]
    _print_table(("parameter", "value", "role", "primary", "source"),
                    _settings_rows(decl["required"], decl["required"], decls,
                                lambda p: "declared" if p in given else "defaulted"), right=(1,))
    _note("All pipeline-derived factor occurrences are generated during reconstruction.")
    print(f"{_INDENT}Next: `reconstruct.py --experiment {path}`")
    return 0

def _section_settings(path, exp_iri):
    """({param: value} above the marker, {param: value} below it). The §4.2 split.

    Given-versus-defaulted is recorded by WHICH SIDE OF THE MARKER a setting sits
    on (§4.2) — it is not derivable from the setting node itself, because a
    student's setting may be a blank node and a machine-minted one uses the §2
    scheme IRI, and neither shape is a promise about who wrote it. So the two
    halves are parsed separately and read by the SAME reader
    (`_experiment_settings`), which is also what keeps the "one parameter, one
    level" check applying to both.
    """
    student_text, machine_text = _split_sections(_read_text(path))
    student = Graph()
    student.parse(data=student_text, format="turtle")
    machine = (_parse_machine_graph(machine_text, student_text)
                if machine_text is not None else Graph())
    return (_experiment_settings(student, exp_iri, path),
            _experiment_settings(machine, exp_iri, path))


def _print_factor_occurrences(g, exp_iri, factors):
    """Show every measured pipeline factor at its current image and optional predecessor."""
    indices = {}
    for frame in g.objects(g.value(exp_iri, HW1.onBatch), HW1.hasFrame):
        idx = g.value(frame, HW1.frameIndex)
        for image in list(g.objects(frame, HW1.hasRGBImage)) + list(g.objects(frame, HW1.hasDepthImage)):
            indices[image] = int(idx)
    rows = []
    for node in g.subjects(HW1.inExperiment, exp_iri):
        definition = g.value(node, HW1.hasDefinition)
        current = g.value(node, HW1.hasCurrentFrame)
        previous = g.value(node, HW1.hasPrevious)
        local = _local(definition) if definition is not None else "?"
        info = factors.get(local, {})
        rows.append((local, "-" if previous is None else indices.get(previous, "?"),
                        indices.get(current, "?"),
                        _fmt_num(g.value(node, HW1.value)),
                        info.get("over", "-")))
    rows.sort(key=lambda row: (int(row[1]) if row[1] != "-" and str(row[1]).isdigit() else -1, int(row[2]) if str(row[2]).isdigit() else -1, row[0]))
    _print_table(("factor", "previous", "current", "value", "observable"), rows)


def _factor_counts(g, factors, selected=None):
    """Occurrence counts for each pipeline QualityFactor."""
    rows = []
    experiment_iris = set(g.subjects(RDF.type, HW1.Experiment))
    occurrence_nodes = {node for exp in experiment_iris for node in g.subjects(HW1.inExperiment, exp)}
    for factor_local, info in sorted(factors.items()):
        nodes = [node for node in occurrence_nodes
                    if g.value(node, HW1.hasDefinition) == HW1[factor_local]]
        if not nodes:
            continue
        rows.append((factor_local, "hw1:" + info["over"], str(len(nodes))))
    return rows

def _run_nodes(g, exp_iri):
    """Every hw1:ReconstructionRun of this experiment, in a stable mode order."""
    runs = []
    for run in g.objects(exp_iri, HW1.hasRun):
        mode = g.value(run, HW1.selectionMode)
        runs.append((_local(mode) if mode is not None else "?", run))
    runs.sort(key=lambda row: str(row[1]))
    return runs


_RUN_OUTCOMES = (
"mapMeanL2",
)


def _run_factor_locals(g, factors, run):
    """Run outcome properties present on a ReconstructionRun."""
    return sorted(name for name in _RUN_OUTCOMES if g.value(run, HW1[name]) is not None)


def _print_summary(exp, g, factors, selected=None):
    """Summarize pipeline occurrence counts and run outcomes."""
    _print_table(("factor", "observable", "occurrences"), _factor_counts(g, factors), right=(2,))
    _kv("frames", f"{len(exp['frame_indices'])}")
    _sub("runs")
    runs = _run_nodes(g, exp["exp_iri"])
    if not runs:
        _note("No reconstruction runs recorded yet.")
        return
    headers = ["run", "frames", *_RUN_OUTCOMES]
    rows = []
    for _mode, run in runs:
        count = g.value(run, HW1.runFrameCount)
        row = [str(run).rsplit("/", 1)[-1], "-" if count is None else int(count)]
        for key in _RUN_OUTCOMES:
            value = g.value(run, HW1[key])
            row.append("-" if value is None else _fmt_num(value))
        rows.append(row)
    _print_table(headers, rows, right=(1,))
def _print_verdict(exp, g, factors, decls, given, path):
    """List direct ReconstructionRun outcome values."""
    _rule("RECONSTRUCTION RUN OUTCOMES")
    runs = _run_nodes(g, exp["exp_iri"])
    if not runs:
        _note("No reconstruction runs have been recorded.")
        return
    rows = []
    for _mode, run in runs:
        mode = str(run).rsplit("/", 1)[-1]
        for outcome in _RUN_OUTCOMES:
            value = g.value(run, HW1[outcome])
            if value is not None:
                rows.append((mode, outcome, _fmt_num(value)))
    _print_table(("run", "outcome", "value"), rows)

def _view_experiment(path):
    """Terminal view of pipeline factor occurrences, settings, and run outcomes."""
    exp = read_experiment(path)
    decls = load_parameter_declarations(_ONTOLOGY_TTL)
    factors = load_quality_factors(_ONTOLOGY_TTL)
    g = exp["graph"]
    _rule(f"EXPERIMENT  {exp['exp_name']}  (assessed)")
    _kv("file", path)
    _kv("experiment IRI", exp["exp_iri"])
    _kv("batch", exp["batch_name"])
    _kv("pipeline factors", f"{exp['completion']['measured']}/{exp['completion']['expected']} measured")
    _sub("settings")
    given, defaulted = _section_settings(path, exp["exp_iri"])
    _print_table(("parameter", "value", "role", "primary", "source"),
                    _settings_rows(exp["settings"], exp["settings"], decls,
                                lambda p: "declared" if p in given else "defaulted"), right=(1,))
    _sub("pipeline factor occurrences")
    _print_factor_occurrences(g, exp["exp_iri"], factors)
    _sub("summary")
    _print_summary(exp, g, factors)
    _print_verdict(exp, g, factors, decls, given, path)
    return 0



    # ── the comparison view (§7.1 view 4) ───────────────────────────
def _compare_column(path):
    """One column of the comparison table, from whichever reader the file supports."""
    kind = _classify(path)
    if kind == "batch":
        # Named by hw1:batchName, not by the file name: every batch file in this
        # project is called `batch.ttl` (§3), so a basename would label two
        # columns identically.
        return {"path": path, "kind": "batch", "name": str(_batch_facts(path)["name"]),
                "why": "a batch file carries no experiment factors, settings or reconstruction runs "
                        "(§4.1), so there is nothing to compare it on"}
    if kind == "declaration":
        decl = read_declaration(path)
        return {"path": path, "kind": "declaration", "name": decl["exp_name"],
                "batch": decl["batch_name"],
                "settings": decl["required"], "given": decl["given"],
                "runs": {}, "counts": None,
                "why": "declared but NOT ASSESSED — its settings are the required "
                        "set §4.2 would record, and it has no values or "
                        "runs to compare"}
    exp = read_experiment(path)
    g = exp["graph"]
    factors = load_quality_factors(_ONTOLOGY_TTL)
    runs = {}
    for _mode_local, run in _run_nodes(g, exp["exp_iri"]):
        # Keyed by the run IRI's `<mode>` segment (§2) — "baseline" / "selected",
        # the words the CLI and the student use — not by the hw1:SelectionMode
        # individual, which is the same fact spelled for the ontology.
        mode = str(run).rsplit("/", 1)[-1]
        for outcome in _RUN_OUTCOMES:
            value = g.value(run, HW1[outcome])
            if value is not None:
                runs[(mode, outcome)] = _fmt_num(value)
    given, _defaulted = _section_settings(path, exp["exp_iri"])
    return {"path": path, "kind": "experiment", "name": exp["exp_name"],
            "batch": exp["batch_name"],
            "settings": exp["settings"], "given": given, "runs": runs,
            "factor_counts": exp["factor_counts"],
            "counts": (len(exp["frame_indices"]), sum(exp["factor_counts"].values())),
            "why": None}


def _view_compare(paths):
    """View 4: several files side by side — pipeline factors, settings and runs."""
    columns = [_compare_column(p) for p in paths]
    comparable = [c for c in columns if c["kind"] != "batch"]
    comparable = [c for c in columns if c["kind"] != "batch"]
    for c in columns:
        _kv(c["name"], f"{c['path']}  [{c['kind']}]"
                        + (f" — {c['why']}" if c["why"] else ""))
    if not comparable:
        print(f"{_INDENT}Nothing comparable was given.")
        return 0

    names = [c["name"] for c in comparable]
    _sub("identity")
    _print_table(["", *names],
                    [["batch", *[c["batch"] for c in comparable]],
                    ["assessed", *["yes" if c["kind"] == "experiment" else "no"
                                    for c in comparable]],
                    ["frames", *[str(c['counts'][0])
                                        if c["counts"] else "-" for c in comparable]],
                    ["factor occurrences", *[str(c['counts'][1])
                                        if c["counts"] else "-" for c in comparable]]])

    all_factors = sorted({f for c in comparable for f in c.get("factor_counts", {})})
    _sub("pipeline factor occurrences")
    rows = []
    for factor in all_factors:
        vals = []
        for column in comparable:
            counts = column.get("factor_counts", {}).get(factor)
            vals.append("-" if counts is None else str(counts))
        rows.append([factor, *vals])
    _print_table(["  factor", *names], rows,
                    right=tuple(range(1, len(names) + 1)))

    _sub("settings")
    all_params = sorted({p for c in comparable for p in c["settings"]})
    rows = []
    for p in all_params:
        vals = [_fmt_setting(c["settings"][p]) if p in c["settings"] else "-"
                for c in comparable]
        annotated = [v + ("" if c["kind"] != "experiment" else
                            (" (d)" if p in c["given"] else ""))
                        for v, c in zip(vals, comparable)]
        rows.append([("*" if len(set(vals)) > 1 else " ") + " " + p, *annotated])
    _print_table(["  parameter", *names], rows, right=tuple(range(1, len(names) + 1)))
    _note("* marks a row where the columns disagree — that is the treatment "
            "difference. (d) marks a level the DECLARATION set; everything else was "
            "filled from hw1:paramDefault (§4.2).")

    _sub("runs")
    all_runs = sorted({k for c in comparable for k in c["runs"]})
    if not all_runs:
        print(f"{_INDENT}no hw1:ReconstructionRun in any of these files.")
    else:
        rows = []
        for mode_local, over in all_runs:
            vals = [c["runs"].get((mode_local, over), "-") for c in comparable]
            rows.append([f"{mode_local} / hw1:{over}", *vals])
        _print_table(["  run / observable", *names], rows,
                        right=tuple(range(1, len(names) + 1)))
    return 0


def _query_rows(result):
    """Return a stable `(headers, rows)` projection for a SPARQL SELECT result."""
    headers = [str(var) for var in result.vars]
    rows = []
    for row in result:
        rows.append(["" if row[var] is None else str(row[var]) for var in result.vars])
    return headers, rows


def cmd_query(args):
    """Run a read-only SPARQL query against one .ttl file. Full strategy note: docs/triplestore.md."""
    args.path = resolve_experiment_path(args.path)
    if not os.path.isfile(args.path):
        raise ValueError(f"{args.path!r}: query needs a local Turtle file or experiment directory")
    query_text = args.query_text if args.query_text is not None else _read_text(args.query_file)
    graph = Graph()
    try:
        graph.parse(data=_read_text(args.path), format="turtle")
    except Exception as exc:
        raise ValueError(f"{args.path}: not valid Turtle: {exc}") from None
    result = query_graph(graph, query_text)

    if result.type == "SELECT":
        headers, rows = _query_rows(result)
        if args.format == "csv":
            writer = csv.writer(sys.stdout)
            writer.writerow(headers)
            writer.writerows(rows)
        elif args.format == "json":
            print(json.dumps([dict(zip(headers, row)) for row in rows], indent=2))
        else:
            _print_table(headers, rows)
        return 0
    if result.type == "ASK":
        answer = bool(result.askAnswer)
        print(json.dumps({"boolean": answer}) if args.format == "json"
                else str(answer).lower())
        return 0

    # CONSTRUCT and DESCRIBE results are graphs. Turtle is the useful default;
    # `--format json` selects a standards-friendly JSON-LD serialization.
    serialization = "json-ld" if args.format == "json" else "turtle"
    print(result.graph.serialize(format=serialization), end="")
    return 0


def cmd_explore(args):
    """Summarize captures/experiments and save interactive plots for assessed experiments."""
    paths = [resolve_experiment_path(path) for path in args.paths]
    missing = [p for p in paths if not (os.path.isfile(p) or _is_capture_dir(p))]
    if missing:
        raise ValueError(
            f"no such capture directory or file: {', '.join(repr(p) for p in missing)}. "
        f"`explore` reads a capture directory, an experiment directory, a declaration or an assessed "
            f"experiment — and several experiment files at once print the "
            f"comparison view (§7.1).")
    if len(paths) > 1:
        result = _view_compare(paths)
        assessed = [path for path in paths if os.path.isfile(path)
                    and _classify(path) == "experiment"]
        if not assessed:
            print("[explore] no assessed experiments to plot; declarations and captures have no measured factor values")
            return result
        from plot import generate_many
        items = [(path, os.path.join(_workspace_for_experiment_file(path), "plots"))
                 for path in assessed]
        artifacts = generate_many(items, launch_dashboard=getattr(args, "launch_dashboard", False))
        if not getattr(args, "launch_dashboard", False):
            for artifact in artifacts:
                print(f"[explore] wrote plot: {artifact}")
        return result
    kind = _classify(paths[0])
    if kind == "batch":
        result = _view_batch(paths[0])
        if getattr(args, "launch_dashboard", False):
            print("[explore] capture has no measured factors yet; reconstruct it through a declared experiment first")
        return result
    if kind == "declaration":
        result = _view_declaration(paths[0])
        if getattr(args, "launch_dashboard", False):
            print("[explore] declaration has no measured factors yet; run reconstruction before plotting")
        return result
    result = _view_experiment(paths[0])
    exp = read_experiment(paths[0])
    name = exp["exp_name"]
    workspace = _workspace_for_experiment_file(paths[0])
    output_dir = os.path.join(workspace, "plots")
    try:
        from plot import generate
    except ImportError as exc:
        raise RuntimeError("Panel/Bokeh are required for `explore`; install the pixi habitat environment") from exc
    launch = getattr(args, "launch_dashboard", False)
    artifacts = generate(paths[0], output_dir, launch_dashboard=launch)
    if not launch:
        for artifact in artifacts:
            print(f"[explore] wrote plot: {artifact}")
    return result


    # Frame selection: links -> maximal contiguous segments.
def cut_contiguous_segments(links_in):
    """Links -> maximal contiguous segments (no length floor). Full strategy note: docs/triplestore.md."""
    links = sorted(set((int(i), int(j)) for i, j in links_in))
    segments = []
    current = []
    prev_j = None
    for i, j in links:
        if current and prev_j == i:
            current.append(j)                 # the chain continues: one more frame
        else:
            if current:
                segments.append(current)      # hole (or first link): start a segment
            current = [i, j]                  # ONE link is already TWO frames
        prev_j = j
    if current:
        segments.append(current)
    return segments


    # CLI.
def _build_parser():
    p = argparse.ArgumentParser(
        description="HW1 experiment CLI (rdflib + local SPARQL). Scaffold a declaration "
                    "over a capture directory (declare), measure it during the first "
                    "reconstruction pass, and inspect or plot it (explore). "
                    "batch2ttl writes an optional structural sidecar. SPARQL runs "
                    "locally over .ttl files; no server or daemon is required.",
        epilog="A measured value means nothing without the settings it was "
                "measured and judged under, so the settings live in the same file "
                "as the values — the student's own declaration, which is "
                "WRITE-ONCE.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    b2t = sub.add_parser(
        "batch2ttl",
        help="Optional sidecar: write batch structure to "
                "<data_dir>/batch.ttl. declare, reconstruct, and explore read the capture "
                "directory directly.")
    b2t.add_argument("--data-dir", required=True,
                        help="Directory containing rgb/ and depth/ subdirs of integer-stem .png "
                            "frames. A semantic/ subdir, if present, is tolerated but not "
                            "ingested. batch.ttl is written HERE as an optional structural "
                            "sidecar for --derived-from. declare and reconstruct can also "
                            "use the capture directory directly.")
    b2t.add_argument("--floor", type=int, default=1,
                        help="Floor this capture is from (default 1). Stored on the Batch node AND "
                            "prefixed onto the batch name, so floor1_mixed_dev and floor2_mixed_dev "
                            "are distinct batches rather than one batch that overwrites itself.")
    b2t.add_argument("--derived-from", default=None, metavar="BATCHNAME",
                        help="The hw1:batchName of the capture this one was derived from, e.g. "
                            "floor1_baseline for a corrupted copy of it; emits one "
                            "prov:wasDerivedFrom. The only surviving prov: term in the "
                            "project (§6) and the only one that spans two "
                            "batches — there are no derived EXPERIMENTS.")
    b2t.add_argument("--out", default=None,
                        help="Output Turtle path (default <data_dir>/batch.ttl).")
    b2t.set_defaults(func=cmd_batch2ttl)

    dec = sub.add_parser(
        "declare",
        help="Scaffold a declaration Turtle (prefixes, Experiment node, batch join, "
                "pipeline prediction TODOs) so nobody starts from a blank "
                "page. Writes the STUDENT section only; never assesses, never "
                "overwrites.")
    dec.add_argument("--name", required=True,
                        help="Experiment name: the file stem AND the tail of the "
                            "Experiment IRI, which must stay equal (§2). "
                            "[A-Za-z0-9_-]+ — naming your experimental conditions is "
                            "part of designing them.")
    dec.add_argument("--data-dir", default=None,
                        help="Capture directory containing rgb/ and depth/ subdirs of "
                            "integer-stem .png frames. Written into hw1:batchFile "
                            "verbatim and resolved against the CWD; hw1:onBatch is "
                            "derived from --floor + the directory basename.")
    dec.add_argument("--floor", type=int, default=1,
                        help="Floor this capture is from (default 1). Prefixed onto the "
                            "batch name, so floor1_mixed_dev and floor2_mixed_dev are "
                            "distinct batches.")
    dec.add_argument("--batch-file", default=None,
                        help="Path to a capture directory or a batch.ttl provenance "
                            "sidecar. Prefer --data-dir; the scaffold writes the "
                            "capture directory into hw1:batchFile either way.")
    dec.add_argument("--out", default=None,
                        help=f"Output Turtle path (default {_EXPERIMENT_DIR}/<name>/<name>.ttl). The "
                            f"file stem must equal --name. An existing file is a hard "
                            f"error: the notebook is append-only.")
    dec.set_defaults(func=cmd_declare)

    exl = sub.add_parser(
        "explore",
        help="Explore captures and experiments; assessed experiments also save Panel plots.")
    exl.add_argument("paths", nargs="+", metavar="PATH",
                        help="One capture directory, one declaration, one assessed "
                        "experiment directory/Turtle — or SEVERAL experiment files, which prints "
                        "the comparison view (§7.1). Does not edit Turtle or measure; assessed "
                        "experiments save Panel plot HTML files.")
    exl.add_argument("--launch-dashboard", action="store_true",
                     help="After saving assessed experiment plots, serve the Panel dashboard on localhost.")
    exl.set_defaults(func=cmd_explore)

    qry = sub.add_parser(
        "query",
        help="Run a read-only SPARQL query over one local Turtle file. Use a .rq "
                "file for reusable student queries; no triplestore is required.")
    qry.add_argument("path", help="Declaration, assessed experiment directory/Turtle file, or batch Turtle file.")
    source = qry.add_mutually_exclusive_group(required=True)
    source.add_argument("--query-file", metavar="FILE",
                        help="UTF-8 .rq file containing a SPARQL SELECT, ASK, CONSTRUCT, or DESCRIBE query.")
    source.add_argument("--query-text", metavar="SPARQL",
                        help="Inline SPARQL query (quote it in the shell).")
    qry.add_argument("--format", choices=("table", "csv", "json"), default="table",
                        help="SELECT: table (default), csv, or json; graph results: Turtle or JSON-LD.")
    qry.set_defaults(func=cmd_query)

    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
