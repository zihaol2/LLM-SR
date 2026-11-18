"""Bounded frontier search with checkpointed planning and explicit final selection."""

import random
import time
from pathlib import Path

from . import core as C
from . import feedback, hypothesis
from .api import once
from .compression import make_prompt
from .config import field_budget
from .final_selection import cards_for, validate
from .fitting import (
    final_equation,
    fit,
    make_data,
    restore_models,
)
from .fitting import (
    statistics as numerical_statistics,
)
from .frontier import update
from .graph import Registry
from .memory import Archive, fit_version
from .recombine import Probes
from .refinement import FullRefiner
from .wire import expand_response_expressions

CHECKPOINT_SCHEMA = "bounded-frontier-v1"


def execute(archive, proposal, number, progress=None):
    model, row = feedback.evaluate(archive, proposal)
    row["state"] = archive.record(model, proposal, number)
    row["model"] = model
    if progress is not None and archive.data.fit_calls % 25 == 0:
        progress()
    return row


def public(row):
    return {key: value for key, value in row.items() if key != "model"}


def _tuples(value):
    return tuple(_tuples(item) for item in value) if isinstance(value, list) else value


class Search:
    """One run's mutable state; numerical rounds and paid planning commit separately."""

    def __init__(self, case, out):
        self.case, self.out = case, Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        path = self.out / "checkpoint.json"
        checkpoint = C.read(path) if path.exists() else None
        self.started = checkpoint["started_unix"] if checkpoint else time.time()
        self.data = make_data(case)
        self.registry = Registry(case, checkpoint["graph"] if checkpoint else None)
        self.archives = {
            field: Archive(self.data, field, self.registry) for field in case["fields"]
        }
        self.rng = random.Random(C.SETTINGS["seed"])
        self.probes = Probes()
        self.history, self.stats = [], {}
        self.pending, self.plan, self.selected = None, None, None
        if checkpoint:
            self.restore(checkpoint)
        else:
            self.groups = hypothesis.partition(self.registry.graph.candidates, self.rng)
            for field, archive in self.archives.items():
                archive.add(
                    fit(self.data, field, [], self.registry),
                    None,
                    dict(operation="root"),
                    0,
                )
            C.dump(self.out / "initial_graph.json", self.registry.export())
        self.phase = dict(stage="initializing", round=len(self.history) + 1, field=None)

    def restore(self, checkpoint):
        if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("Unsupported checkpoint schema; start in a new output directory")
        self.groups = checkpoint["rotation_groups"]
        self.history, self.stats = checkpoint["history"], checkpoint["stats"]
        self.pending, self.plan = checkpoint["pending"], checkpoint["next_plan"]
        self.selected = checkpoint.get("final_selection")
        for field, archive in self.archives.items():
            archive.restore(checkpoint["archives"][field])
        self.rng.setstate(_tuples(checkpoint["rng_state"]))
        for key, value in checkpoint["counters"].items():
            setattr(self.data, key, value)
        restore_models(
            self.data,
            self.archives,
            (checkpoint.get("numerical_backend_stats") or {}).get("counts"),
        )

    def progress(self, **updates):
        self.phase.update(updates)
        C.dump(
            self.out / "progress.json",
            dict(
                **self.phase,
                completed_rounds=len(self.history),
                fit_calls=self.data.fit_calls,
                working_set_sizes={f: len(a.active_ids) for f, a in self.archives.items()},
                updated_unix=time.time(),
                numerical_backend_stats=numerical_statistics(self.data),
            ),
        )

    def save(self):
        C.dump(
            self.out / "checkpoint.json",
            dict(
                schema=CHECKPOINT_SCHEMA,
                started_unix=self.started,
                graph=self.registry.export(),
                history=self.history,
                stats=self.stats,
                pending=self.pending,
                next_plan=self.plan,
                final_selection=self.selected,
                rotation_groups=self.groups,
                rng_state=self.rng.getstate(),
                counters={
                    k: getattr(self.data, k) for k in ("fit_calls", "fit_nfev", "feature_calls")
                },
                archives={f: a.export() for f, a in self.archives.items()},
                numerical_backend_stats=numerical_statistics(self.data),
            ),
        )

    def prompt_after(self, number, event=None):
        sampled = (
            hypothesis.group_for(self.groups, number + 1) if number < C.SETTINGS["rounds"] else []
        )
        preview = {}
        if sampled:
            for field, archive in self.archives.items():
                queues, info = hypothesis.history_candidates(archive, sampled, [], self.probes)
                preview[field] = dict(
                    history_mechanisms=info["history_mechanisms"],
                    examples=[
                        dict(
                            support=list(item["support"]),
                            relative_probe_MSE=item["probe_relative_MSE"],
                        )
                        for queue in queues
                        for item in list(queue)[:2]
                    ],
                )
        text, packet = make_prompt(
            self.case,
            self.registry,
            self.archives,
            sampled,
            self.stats,
            self.history + ([event] if event else []),
            number,
            preview,
        )
        C.dump(self.out / "inputs" / f"{number:02d}_graph.json", packet)
        return text

    def working_snapshot(self, number, audits=None):
        C.dump(
            self.out / "working_sets" / f"{number:02d}.json",
            dict(
                round=number,
                fields={
                    field: [
                        dict(
                            state=s["id"],
                            model_version=s["model_version"],
                            model=s["model"],
                        )
                        for s in archive.view()
                    ]
                    for field, archive in self.archives.items()
                },
                audits=audits,
            ),
        )

    def bootstrap(self):
        if self.plan is not None or self.history:
            return
        if self.pending is None:
            self.working_snapshot(0)
            self.pending = dict(phase="bootstrap", round=0, prompt=self.prompt_after(0))
            self.save()
        if self.pending["phase"] != "bootstrap":
            raise RuntimeError("Invalid bootstrap checkpoint")
        self.progress(stage="bootstrap_planning", round=0)
        raw, record = once(self.pending["prompt"], self.out / "api", 0)
        self.plan = raw if record["status"] == "accepted" else {}
        self.pending = None
        self.save()

    def fit_field(self, field, policy, sampled, new, input_ids, number):
        archive = self.archives[field]
        current_ids, coverage, historical, revisions, edit_audits = [], [], [], [], []

        def record_trial(proposal, category):
            row = execute(archive, proposal, number, self.progress)
            category.append(public(row))
            if row["state"] is not None:
                current_ids.append(row["state"])
            return row

        def evaluate(proposal):
            return record_trial(proposal, revisions)

        self.progress(stage="coverage", round=number, field=field)
        proposals, selection = hypothesis.build_tests(policy, archive, sampled, self.rng, number)
        for proposal in proposals:
            if proposal["parent"] not in input_ids:
                raise RuntimeError("Undisplayed coverage parent")
            proposal["kind"] = "coverage_" + proposal["source"]
            record_trial(proposal, coverage)
        tested = {mid for row in coverage for mid in row["support"]} & set(sampled)
        if not set(selection["selected"]) <= tested:
            raise RuntimeError("Coverage quota not executed")
        selection["actually_tested_sampled"] = sorted(tested)

        self.progress(stage="LLM_skeleton_edits")
        explicit = []
        for proposal in hypothesis.llm_revisions(
            archive, policy, new, audit=edit_audits, number=number
        ):
            row = evaluate(proposal)
            explicit.append(row)
            edit_audits[proposal["edit_audit_index"]].update(
                status="fitted" if row["valid"] else "fit_invalid",
                state=row["state"],
                score=row["score"],
            )

        self.progress(stage="history_recombination")
        queues, _ = hypothesis.history_candidates(archive, sampled, new["ids"], self.probes)
        stream = hypothesis.next_unseen(queues, archive)
        budget = field_budget(C.SETTINGS["history_fits_per_round"], self.case["fields"], field)
        for _ in range(budget):
            proposal = next(stream, None)
            if proposal is None:
                break
            proposal["kind"] = "historical_" + proposal["origin"]
            record_trial(proposal, historical)

        self.progress(stage="full_refinement")
        refiner = FullRefiner(archive, number, evaluate)
        for row in explicit:
            if row["valid"]:
                refiner.reduce(row)
        preliminary = update(archive, input_ids, current_ids, C.SETTINGS["frontier_limit"])
        family = refiner.run()
        C.dump(
            self.out / "refinement" / f"round{number:02d}_{field}.json",
            dict(preliminary=preliminary, **family),
        )
        frontier = update(archive, input_ids, current_ids, C.SETTINGS["frontier_limit"])
        return dict(
            coverage=coverage,
            historical=historical,
            revisions=revisions,
            edits=edit_audits,
            selection=selection,
            frontier=frontier,
        )

    def fit_round(self, number):
        begin = time.time()
        sampled = hypothesis.group_for(self.groups, number)
        input_ids = {f: list(a.active_ids) for f, a in self.archives.items()}
        for mid in sampled:
            self.stats.setdefault(mid, dict(sampled=0, tested=0))["sampled"] += 1
        packet = C.read(self.out / "inputs" / f"{number - 1:02d}_graph.json")
        expanded, template_audit = expand_response_expressions(
            self.plan, packet.get("wire_packet", {})
        )
        policy = hypothesis.normalize_policy(expanded, self.archives)
        new = hypothesis.register_mechanisms(policy, self.registry, number)
        fields = {
            f: self.fit_field(f, policy, sampled, new, input_ids[f], number) for f in self.archives
        }
        batches = {
            key: {f: result[key] for f, result in fields.items()}
            for key in (
                "coverage",
                "historical",
                "revisions",
                "edits",
                "selection",
                "frontier",
            )
        }
        rows = [
            row
            for key in ("coverage", "historical", "revisions")
            for batch in batches[key].values()
            for row in batch
        ]
        for mid in {mid for row in rows for mid in row["support"]}:
            self.stats.setdefault(mid, dict(sampled=0, tested=0))["tested"] += 1
        edits = [item for batch in batches["edits"].values() for item in batch]
        event = dict(
            round=number,
            input_working_ids=input_ids,
            output_working_ids={f: list(a.active_ids) for f, a in self.archives.items()},
            frontier_audits=batches["frontier"],
            coverage_fits=sum(map(len, batches["coverage"].values())),
            history_fits=sum(map(len, batches["historical"].values())),
            revision_fits=sum(map(len, batches["revisions"].values())),
            valid_fits=sum(row["valid"] for row in rows),
            ordinary_local_fits=0,
            edit_counts={
                status: sum(item["status"] == status for item in edits)
                for status in ("fitted", "rejected", "fit_invalid")
            },
            policy_normalization_errors=policy["errors"],
            numerical_seconds=time.time() - begin,
            template_response_expansion=template_audit,
        )
        event["edit_counts"]["proposed"] = len(edits)
        if event["history_fits"] > C.SETTINGS["history_fits_per_round"]:
            raise RuntimeError("History budget exceeded")
        C.dump(
            self.out / "rounds" / f"{number:02d}.json",
            dict(
                event=event,
                selection=batches["selection"],
                coverage=batches["coverage"],
                historical=batches["historical"],
                revisions=batches["revisions"],
                LLM_edit_audit=batches["edits"],
            ),
        )
        self.working_snapshot(number, batches["frontier"])
        final = number == C.SETTINGS["rounds"]
        self.pending = dict(
            phase="final_selection" if final else "planning",
            round=number,
            event=event,
            cards=({f: cards_for(a) for f, a in self.archives.items()} if final else None),
            prompt=self.prompt_after(number, event),
        )
        self.save()  # Freeze numeric work and the exact prompt before dispatch.

    def select_final(self, raw, record):
        if record["status"] != "accepted":
            self.progress(stage="final_selection_unavailable")
            raise RuntimeError(
                "Final LLM unavailable; working set saved; no implicit Score fallback or retry"
            )
        self.selected = validate(raw, self.pending["cards"])
        for field, decision in self.selected.items():
            archive = self.archives[field]
            if (
                decision["state"] not in archive.active_ids
                or fit_version(archive.states[decision["state"]]["model"])
                != decision["model_version"]
            ):
                raise RuntimeError("Final choice is not a frozen working-set fit")
        C.dump(
            self.out / "final_selection.json",
            dict(
                policy=raw,
                decisions=self.selected,
                fit_calls=self.data.fit_calls,
                reference_access=False,
                final_fits=0,
            ),
        )

    def plan_round(self, number):
        if self.pending["round"] != number:
            raise RuntimeError("Checkpoint round mismatch")
        self.progress(stage=self.pending["phase"], round=number, field=None)
        raw, record = once(self.pending["prompt"], self.out / "api", number)
        if self.pending["phase"] == "final_selection":
            self.select_final(raw, record)
        event = self.pending["event"]
        event.update(API_status=record["status"], reason=raw.get("reason", ""))
        self.history.append(event)
        self.plan = raw if record["status"] == "accepted" else {}
        self.pending = None
        self.save()
        print(
            self.case["id"],
            "ROUND COMPLETE",
            number,
            "working set",
            {f: len(a.active_ids) for f, a in self.archives.items()},
            "coverage/history/revision",
            event["coverage_fits"],
            event["history_fits"],
            event["revision_fits"],
            flush=True,
        )

    def coverage(self):
        roots = {mid for group in self.groups for mid in group}
        coverage = {
            f: len(roots & a.tested_ids()) / len(roots) if roots else 1.0
            for f, a in self.archives.items()
        }
        if C.SETTINGS["rounds"] == 20:
            required = 0.2 if C.SETTINGS["coverage_schedule"] == "global_1pct" else 1.0
            if any(value < required for value in coverage.values()):
                raise RuntimeError("Incomplete initial-root coverage")
        return coverage

    def result(self):
        if self.selected is None:
            raise RuntimeError("Missing explicit final selection")
        coverage = self.coverage()
        records = [C.read(path) for path in sorted((self.out / "api").glob("*_record.json"))]
        before = self.data.fit_calls
        equations = {
            f: final_equation(self.data, f, self.archives[f].states[d["state"]]["model"])
            for f, d in self.selected.items()
        }
        result = dict(
            method="GEM-GSS bounded frontier iteration",
            completed_rounds=len(self.history),
            anonymous_id=self.case["id"],
            equations=equations,
            selected_states={f: d["state"] for f, d in self.selected.items()},
            final_working_ids={f: list(a.active_ids) for f, a in self.archives.items()},
            final_selection=self.selected,
            final_selection_rule="One LLM selects an existing final-frontier equation by MSE/complexity marginal tradeoff",
            final_API_calls=records[-1]["http_attempts"],
            search_API_calls=sum(r["http_attempts"] for r in records[:-1]),
            total_API_calls=sum(r["http_attempts"] for r in records),
            bootstrap_API_calls=records[0]["http_attempts"],
            round_planning_API_calls=sum(r["http_attempts"] for r in records[1:-1]),
            terminal_fit_calls=0,
            unavailable_LLM_records=[r["round"] for r in records if r["status"] != "accepted"],
            fit_calls=self.data.fit_calls,
            loss_evaluations=self.data.fit_nfev,
            fit_breakdown=dict(
                initial_empty=len(self.archives),
                coverage=sum(r["coverage_fits"] for r in self.history),
                history=sum(r["history_fits"] for r in self.history),
                online_revision=sum(r["revision_fits"] for r in self.history),
                ordinary_local=0,
                terminal=0,
            ),
            initial_root_attempt_coverage=coverage,
            coverage_schedule=C.SETTINGS["coverage_schedule"],
            guaranteed_initial_root_fraction=(
                0.2 if C.SETTINGS["coverage_schedule"] == "global_1pct" else 1.0
            ),
            MSE=sum(e["MSE"] for e in equations.values()) / len(equations),
            reference_access=False,
            frozen_unix=time.time(),
            seconds=time.time() - self.started,
            numerical_backend_stats=numerical_statistics(self.data),
        )
        if (
            self.data.fit_calls != before
            or sum(result["fit_breakdown"].values()) != self.data.fit_calls
        ):
            raise RuntimeError("Fit accounting mismatch")
        C.dump(self.out / "final_graph.json", self.registry.export())
        C.dump(self.out / "final_result.json", result)
        self.progress(stage="complete", round=len(self.history))
        return result

    def run(self):
        self.bootstrap()
        for number in range(len(self.history) + 1, C.SETTINGS["rounds"] + 1):
            if self.pending is None:
                self.fit_round(number)
            self.plan_round(number)
        return self.result()


def run(case, out):
    out = Path(out)
    if (out / "final_result.json").exists():
        return C.read(out / "final_result.json")
    return Search(case, out).run()
