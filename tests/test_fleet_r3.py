"""Unit tests for the round-3 Vast controller (``experiments.fleet.controller_r3``).

No network, no money: every test drives pure functions or an in-memory backend.

The three contracts the user called out explicitly:

1. **The label filter is a safety boundary.** The Vast account also carries the
   foreign production instances.
   Only instances labelled *exactly* ``fedgrid-r3`` may ever be listed, adopted or
   destroyed -- including by the spend-cap emergency teardown.
2. **Shards are grouped by federation** so a box downloads ~9 MB of data instead of
   the 120 MB battery, and the partition stays total + disjoint.
3. **Spend is the real balance drop minus a franchise** for the owner's production
   burn, so their $0.36/hr never trips our cap.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import stage_data as SD  # noqa: E402
from experiments.fleet import controller_r3 as C  # noqa: E402

LABEL = C.LABEL


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def inst_row(iid: int, label: str | None, *, status: str = "running", dph: float = 0.2) -> dict:
    """A raw ``show instances`` row, shaped like the real API payload."""
    row = {
        "id": iid,
        "actual_status": status,
        "cur_state": status,
        "gpu_name": "RTX 3090",
        "num_gpus": 1,
        "gpu_ram": 24576,
        "dph_total": dph,
        "start_date": 1_700_000_000.0,
        "machine_id": 1000 + iid,
        "status_msg": "",
    }
    if label is not None:
        row["label"] = label
    return row


#: The two real production instances on the account, verbatim from `show instances`.
PRODUCTION_ROWS = [
    inst_row(10000001, "other-project-a", dph=0.273),
    inst_row(10000002, "other-project-b", dph=0.0834),
]


def entry(run_id: str, fed_seed: int, *, alpha: float = 0.5, days: int = 30) -> dict:
    """A manifest entry whose federation block drives its ``fed_<hash>`` id."""
    return {
        "run_id": run_id,
        "arm": "fedavg",
        "model": "cnn_lstm",
        "seed": 0,
        "config": {
            "federation": {"k": 10, "alpha": alpha, "days": days, "seed": fed_seed,
                           "iid": False},
            "training": {"rounds": 30},
        },
    }


def entries_for(spec: dict[int, int]) -> list[dict]:
    """``{federation seed: n runs}`` -> a flat manifest entry list."""
    out: list[dict] = []
    for fed_seed, n in spec.items():
        out.extend(entry(f"r{fed_seed}_{i:03d}", fed_seed) for i in range(n))
    return out


class MockBackend:
    """In-memory Vast backend that BOOBY-TRAPS any touch of a foreign instance."""

    def __init__(self, rows: list[dict] | None = None, credit: float = 65.0) -> None:
        self.rows = [dict(r) for r in (rows if rows is not None else PRODUCTION_ROWS)]
        self.credit = credit
        self.destroyed: list[int] = []
        self.created: list[dict] = []
        self.executed: list[tuple[int, str]] = []
        self.next_id = 90_000

    # -- listing --------------------------------------------------------------
    def show_instances(self, *, label: str | None = None) -> list[dict]:
        # Deliberately IGNORES the label argument: the controller must not depend
        # on server-side filtering for its safety property.
        return [dict(r) for r in self.rows]

    # -- the trap -------------------------------------------------------------
    def destroy_instance(self, instance_id: int) -> None:
        row = next((r for r in self.rows if int(r["id"]) == int(instance_id)), None)
        assert row is not None, f"destroy of unknown instance {instance_id}"
        assert row.get("label") == LABEL, (
            f"SAFETY VIOLATION: controller tried to destroy instance {instance_id} "
            f"labelled {row.get('label')!r}"
        )
        self.destroyed.append(int(instance_id))
        self.rows = [r for r in self.rows if int(r["id"]) != int(instance_id)]

    # -- the rest -------------------------------------------------------------
    def account_credit(self) -> float:
        return self.credit

    def search_offers(self, query: str, *, order: str = "dph_total") -> list[dict]:
        return []

    def create_instance(self, offer_id: int, **kwargs) -> int:
        assert kwargs["label"] == LABEL, "instances must be created with the r3 label"
        self.next_id += 1
        self.created.append({"offer": offer_id, **kwargs})
        self.rows.append(inst_row(self.next_id, LABEL, status="loading"))
        return self.next_id

    def execute(self, instance_id: int, command: str, *, timeout: float | None = None) -> str:
        self.executed.append((instance_id, command))
        return ""

    def copy(self, src: str, dst: str) -> None:
        pass


def make_controller(tmp_path: Path, backend: MockBackend, entries=(), **kw) -> C.ControllerR3:
    cfg = C.ControllerConfig(
        out_dir=tmp_path / "runs_r3",
        data_dir=tmp_path / "data",
        target=kw.pop("target", 4),
        cap_usd=kw.pop("cap_usd", 15.0),
        external_burn_per_h=kw.pop("external_burn_per_h", 0.40),
        **kw,
    )
    return C.ControllerR3(backend, list(entries), cfg)


# =========================================================================== #
# 1. THE LABEL FILTER -- requirement #1, non-negotiable
# =========================================================================== #


def test_select_ours_keeps_only_our_label_and_hides_production():
    rows = [*PRODUCTION_ROWS, inst_row(1, LABEL), inst_row(2, LABEL, status="loading")]
    ours = C.select_ours(rows)
    assert [r["id"] for r in ours] == [1, 2]
    assert all(r["label"] == LABEL for r in ours)
    # The production boxes are invisible -- not merely sorted last.
    assert 10000001 not in {r["id"] for r in ours}
    assert 10000002 not in {r["id"] for r in ours}


@pytest.mark.parametrize(
    "label",
    [
        None,                 # unlabelled
        "",                   # empty
        "other-project-a",      # production
        "other-project-b",    # production
        "fedgrid-fleet",      # a PREVIOUS round's fleet -- still not ours
        "fedgrid-r2",
        "fedgrid-r30",        # prefix collision
        "xfedgrid-r3",        # suffix collision
        "fedgrid-r3 ",        # trailing space
        " fedgrid-r3",        # leading space
        "FEDGRID-R3",         # different case
        "Fedgrid-r3",
    ],
)
def test_select_ours_rejects_every_near_miss(label):
    assert C.select_ours([inst_row(7, label)]) == []
    assert C.is_ours(inst_row(7, label)) is False


@pytest.mark.parametrize("bad", [{"id": 1, "label": 123}, {"id": 2, "label": b"fedgrid-r3"},
                                 {"id": 3}, {}])
def test_select_ours_is_type_strict(bad):
    assert C.select_ours([bad]) == []


def test_select_ours_on_empty_or_none_input():
    assert C.select_ours([]) == []
    assert C.select_ours(None) == []
    assert C.our_instance_ids(PRODUCTION_ROWS) == []


def test_our_instance_ids_returns_only_our_ids():
    rows = [*PRODUCTION_ROWS, inst_row(11, LABEL), inst_row(12, LABEL)]
    assert C.our_instance_ids(rows) == [11, 12]


def test_select_ours_returns_copies_not_aliases():
    rows = [inst_row(1, LABEL)]
    ours = C.select_ours(rows)
    ours[0]["label"] = "tampered"
    assert rows[0]["label"] == LABEL


# -- the filter, enforced end-to-end through the controller ------------------


def test_controller_lists_only_our_instances(tmp_path):
    backend = MockBackend([*PRODUCTION_ROWS, inst_row(1, LABEL)])
    ctl = make_controller(tmp_path, backend)
    assert [i.id for i in ctl.our_instances()] == [1]


def test_teardown_destroys_ours_and_never_production(tmp_path):
    backend = MockBackend([*PRODUCTION_ROWS, inst_row(1, LABEL), inst_row(2, LABEL)])
    ctl = make_controller(tmp_path, backend)
    killed = ctl.destroy_all_ours("test")
    assert killed == 2
    assert sorted(backend.destroyed) == [1, 2]
    # Both production instances survive, untouched.
    assert {r["id"] for r in backend.rows} == {10000001, 10000002}


def test_destroy_refuses_a_foreign_instance_even_if_handed_one(tmp_path):
    """The last gate: ``_destroy`` re-checks the raw row's label itself."""
    backend = MockBackend()
    ctl = make_controller(tmp_path, backend)
    from experiments.fleet.fleet import parse_instance

    foreign = parse_instance(PRODUCTION_ROWS[0])
    assert ctl._destroy(foreign, "deliberate misuse") is False
    assert backend.destroyed == []


def test_spend_cap_teardown_spares_production(tmp_path):
    """The emergency path is the most dangerous one: it must obey the filter too."""
    backend = MockBackend([*PRODUCTION_ROWS, inst_row(1, LABEL), inst_row(2, LABEL)],
                          credit=40.0)
    ctl = make_controller(tmp_path, backend, entries=entries_for({0: 4}), cap_usd=5.0)
    ctl.balance_start = 60.0          # $20 gone, franchise cannot explain it
    ctl.balance_start_ts = ctl._started
    with pytest.raises(SystemExit) as exc:
        ctl.cycle()
    assert exc.value.code == 2
    assert sorted(backend.destroyed) == [1, 2]
    assert {r["id"] for r in backend.rows} == {10000001, 10000002}


def test_created_instances_always_carry_the_exact_label(tmp_path):
    backend = MockBackend([])
    ctl = make_controller(tmp_path, backend)
    ctl.search_offers = lambda want: [  # type: ignore[method-assign]
        C.parse_offer({"id": 555, "dph_total": 0.3, "gpu_name": "RTX 3090", "num_gpus": 1,
                       "machine_id": 42, "verification": "verified", "disk_space": 100,
                       "inet_down": 900})
    ]
    created = ctl.provision(1)
    assert len(created) == 1
    assert backend.created[0]["label"] == LABEL


# =========================================================================== #
# 2. SHARD GROUPING BY FEDERATION
# =========================================================================== #


def test_fed_id_matches_the_staging_layout_exactly():
    """The controller must agree with the module that CREATED data_battery/."""
    e = entry("x", 3)
    fed = e["config"]["federation"]
    assert C.entry_fed_id(e) == SD.federation_id(fed)
    assert C.entry_fed_id(e).startswith("fed_")
    assert len(C.entry_fed_id(e)) == len("fed_") + 10


def test_group_by_federation_buckets_and_preserves_order():
    ents = [entry("a", 0), entry("b", 1), entry("c", 0)]
    groups = C.group_by_federation(ents)
    assert list(groups) == [C.entry_fed_id(ents[0]), C.entry_fed_id(ents[1])]
    assert [e["run_id"] for e in groups[C.entry_fed_id(ents[0])]] == ["a", "c"]


@pytest.mark.parametrize(
    "spec,n_boxes",
    [
        ({0: 30, 1: 30, 2: 30}, 3),
        ({0: 100, 1: 20, 2: 5}, 14),
        ({0: 390, 1: 144}, 14),          # the real round-3 shape
        ({i: 10 for i in range(20)}, 6),  # more federations than boxes
        ({0: 1}, 14),                     # one run, many boxes
        ({0: 7, 1: 7}, 1),                # one box, everything
        ({i: 3 for i in range(14)}, 14),
    ],
)
def test_plan_shards_is_total_disjoint_and_bounded(spec, n_boxes):
    ents = entries_for(spec)
    shards = C.plan_shards(ents, n_boxes)
    assert len(shards) <= n_boxes
    assert all(len(s) > 0 for s in shards)
    seen = [rid for s in shards for rid in s.run_ids]
    assert len(seen) == len(set(seen)), "a run was assigned to two boxes"
    assert set(seen) == {e["run_id"] for e in ents}, "a run was lost"
    # Every shard declares exactly the federations its runs need.
    by_id = {e["run_id"]: e for e in ents}
    for s in shards:
        needed = {C.entry_fed_id(by_id[r]) for r in s.run_ids}
        assert needed == set(s.fed_ids)


def test_plan_shards_keeps_one_federation_per_box_when_the_pieces_fit():
    """The real round-3 shape: 390 temporal + 144 magnitude runs over 14 boxes."""
    shards = C.plan_shards(entries_for({0: 390, 1: 144}), 14)
    assert len(shards) == 14
    assert all(len(s.fed_ids) == 1 for s in shards), "a box would download two datasets"
    sizes = sorted(len(s) for s in shards)
    assert sizes[-1] - sizes[0] <= 4, f"unbalanced fleet: {sizes}"
    # The big federation gets the most boxes.
    counts: dict[str, int] = {}
    for s in shards:
        counts[s.fed_ids[0]] = counts.get(s.fed_ids[0], 0) + 1
    assert counts[C.entry_fed_id(entry("x", 0))] == 10
    assert counts[C.entry_fed_id(entry("x", 1))] == 4


def test_plan_shards_on_the_real_534_run_round3_shape():
    """manifest_temporal + manifest_magnitude: 534 runs over 5 federations, 14 boxes.

    Fixed-size chunking used to hand two boxes 62 runs while others got 31 -- the
    fleet then billed for twice the necessary wall clock.
    """
    spec = {0: 126, 1: 126, 2: 126, 3: 78, 4: 78}
    shards = C.plan_shards(entries_for(spec), 14)
    assert len(shards) == 14
    assert sum(len(s) for s in shards) == 534
    assert all(len(s.fed_ids) == 1 for s in shards), "a box would download two datasets"
    sizes = sorted(len(s) for s in shards)
    assert sizes[-1] <= 43, f"a box sets a long makespan: {sizes}"
    assert sizes[0] >= 30, f"a box is under-used: {sizes}"


def test_plan_shards_balances_the_load_instead_of_stranding_a_giant_box():
    """Makespan beats upload size: we rent the whole fleet until the LAST box ends.

    The battery's real federation profile (three 108-run federations, two 52-run
    ones and nine 8-run ones) used to produce a 108-run box next to eight 8-run
    boxes -- the fleet then billed for 3x the ideal wall clock while 9 boxes idled
    on work they could not steal.
    """
    spec = {0: 108, 1: 108, 2: 108, 3: 52, 4: 52}
    spec.update({5 + i: 8 for i in range(9)})
    shards = C.plan_shards(entries_for(spec), 14)
    sizes = sorted(len(s) for s in shards)
    ideal = 500 / 14
    assert sizes[-1] <= ideal * 1.25, f"one box would set a long makespan: {sizes}"
    assert sizes[0] >= ideal * 0.75, f"a box is nearly idle: {sizes}"


def test_plan_shards_never_splits_a_federation_smaller_than_a_chunk():
    spec = {i: 5 for i in range(10)}          # 50 runs, 10 federations, 5 boxes
    shards = C.plan_shards(entries_for(spec), 5)
    owners: dict[str, set[int]] = {}
    for i, s in enumerate(shards):
        for fed in s.fed_ids:
            owners.setdefault(fed, set()).add(i)
    assert all(len(v) == 1 for v in owners.values()), "a small federation was split"


def test_plan_shards_bin_packs_whole_federations_when_boxes_are_scarce():
    shards = C.plan_shards(entries_for({i: 10 for i in range(20)}), 4)
    assert len(shards) == 4
    # No federation is split across boxes: each fed appears in exactly one shard.
    owners: dict[str, int] = {}
    for i, s in enumerate(shards):
        for fed in s.fed_ids:
            assert fed not in owners, "a federation was split across two boxes"
            owners[fed] = i
    assert len(owners) == 20
    sizes = sorted(len(s) for s in shards)
    assert sizes[-1] - sizes[0] <= 10, "bin packing left a badly unbalanced box"


def test_plan_shards_handles_the_empty_and_degenerate_cases():
    assert C.plan_shards([], 14) == []
    assert C.plan_shards(entries_for({0: 5}), 0) == []
    assert len(C.plan_shards(entries_for({0: 3}), 99)) == 3


def test_assign_shards_to_boxes_prefers_a_box_that_already_holds_the_data():
    ents = entries_for({0: 10, 1: 10})
    shards = C.plan_shards(ents, 2)
    fed_a, fed_b = shards[0].fed_ids[0], shards[1].fed_ids[0]
    staged = {101: (fed_b,), 102: (fed_a,)}
    plan = C.assign_shards_to_boxes(shards, [101, 102], staged)
    assert plan[102].fed_ids == (fed_a,)
    assert plan[101].fed_ids == (fed_b,)


def test_assign_shards_to_boxes_never_double_books_a_box():
    ents = entries_for({0: 10, 1: 10, 2: 10})
    shards = C.plan_shards(ents, 3)
    plan = C.assign_shards_to_boxes(shards, [1, 2], {})
    assert len(plan) == 2 and set(plan) == {1, 2}


def test_load_manifest_entries_concatenates_and_dedupes(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text("\n".join(json.dumps(e) for e in entries_for({0: 3})) + "\n", encoding="utf-8")
    rows_b = entries_for({1: 2}) + [entry("r0_000", 0)]  # deliberate overlap
    b.write_text("\n\n".join(json.dumps(e) for e in rows_b) + "\n", encoding="utf-8")
    merged = C.load_manifest_entries([a, b])
    ids = [e["run_id"] for e in merged]
    assert len(ids) == len(set(ids)) == 5
    assert ids[:3] == ["r0_000", "r0_001", "r0_002"]


def test_load_manifest_entries_reports_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        C.load_manifest_entries([tmp_path / "nope.jsonl"])


# =========================================================================== #
# 3. MONEY -- balance drop minus the production franchise
# =========================================================================== #


def test_spend_subtracts_the_external_franchise():
    # 2 h elapsed, balance fell $3.00, the owner's boxes burn $0.40/hr -> $0.80 theirs.
    assert C.spend_from_balance(65.0, 62.0, 2.0, 0.40) == pytest.approx(2.20)


def test_spend_is_zero_when_only_the_owner_is_burning():
    assert C.spend_from_balance(65.0, 64.2, 2.0, 0.40) == pytest.approx(0.0)


def test_spend_never_goes_negative_after_a_top_up():
    assert C.spend_from_balance(65.0, 90.0, 1.0, 0.40) == 0.0


def test_spend_with_no_franchise_is_the_raw_balance_drop():
    assert C.spend_from_balance(65.0, 55.0, 5.0, 0.0) == pytest.approx(10.0)


def test_spend_ignores_a_negative_elapsed_time():
    assert C.spend_from_balance(65.0, 60.0, -3.0, 0.40) == pytest.approx(5.0)


def test_spend_matches_the_real_production_burn_rate():
    """other-project-a + other-project-b = $0.3564/hr; the default franchise covers it."""
    external = 0.273 + 0.0834
    drop = external * 6.0            # six hours, we rented nothing
    assert C.spend_from_balance(65.0, 65.0 - drop, 6.0, 0.40) == 0.0


def test_controller_spend_uses_the_persisted_snapshot(tmp_path):
    backend = MockBackend(credit=60.0)
    ctl = make_controller(tmp_path, backend)
    ctl.balance_start, ctl.balance_start_ts = 65.0, ctl._started
    spent, bal = ctl.spend()
    assert bal == 60.0
    assert spent == pytest.approx(5.0, abs=0.01)  # ~0 h elapsed -> no franchise yet


def test_spend_falls_back_to_rate_when_the_balance_api_is_down(tmp_path):
    """A broken ``show user`` must not be the thing that switches the cap off."""

    class NoBalanceBackend(MockBackend):
        def account_credit(self):
            raise RuntimeError("500 Internal Server Error")

    ctl = make_controller(tmp_path, NoBalanceBackend())
    assert ctl.spend() == (0.0, None)
    ctl.rate_accrued = 7.5                      # accrued by rate x uptime instead
    assert ctl.spend() == (7.5, None)


def test_rate_accrual_arms_the_cap_without_a_balance(tmp_path):
    backend = MockBackend([inst_row(1, LABEL, dph=2.0)])
    ctl = make_controller(tmp_path, backend, entries=entries_for({0: 4}), cap_usd=1.0)
    ctl.cycle()                                   # learns the fleet rate
    assert ctl._last_rate == pytest.approx(2.0)
    ctl._last_rate_ts -= 3600.0                   # pretend an hour passed
    with pytest.raises(SystemExit):
        ctl.cycle()
    assert ctl.rate_accrued >= 1.0
    assert backend.destroyed == [1]


def test_balance_snapshot_survives_a_controller_restart(tmp_path):
    backend = MockBackend(credit=65.0)
    ctl = make_controller(tmp_path, backend)
    ctl.snapshot_balance()
    assert ctl.balance_start == 65.0
    backend.credit = 61.0                     # money spent while we were away
    ctl2 = make_controller(tmp_path, backend)  # fresh controller, same out_dir
    ctl2.snapshot_balance()
    assert ctl2.balance_start == 65.0, "a restart must not re-baseline the spend"
    assert ctl2.spend()[0] == pytest.approx(4.0, abs=0.05)


# =========================================================================== #
# Box classification, sizing, offers
# =========================================================================== #


def test_classify_box_busy_requires_process_data_and_a_fresh_log():
    fed = ("fed_abc",)
    busy = C.BoxProbe(ok=True, run_one_alive=True, feds_present=fed, log_age_s=20.0)
    assert C.classify_box(busy, required_feds=fed) == "BUSY"


@pytest.mark.parametrize(
    "probe,why",
    [
        (C.BoxProbe(ok=True, run_one_alive=True, feds_present=("fed_abc",), log_age_s=20.0,
                    shard_done=True), "shard finished"),
        (C.BoxProbe(ok=True, run_one_alive=False, feds_present=("fed_abc",), log_age_s=5.0),
         "no run_one"),
        (C.BoxProbe(ok=True, run_one_alive=True, feds_present=("fed_abc",), log_age_s=999.0),
         "log stalled"),
        (C.BoxProbe(ok=True, run_one_alive=True, feds_present=(), log_age_s=5.0), "starved"),
        (C.BoxProbe(ok=True, run_one_alive=True, feds_present=("fed_abc",), log_age_s=None),
         "no log at all"),
    ],
)
def test_classify_box_idle_cases(probe, why):
    assert C.classify_box(probe, required_feds=("fed_abc",)) == "IDLE", why


def test_classify_box_dead_only_after_the_full_strike_budget():
    down = C.BoxProbe(ok=False, error="connection refused")
    for n in range(C.SSH_STRIKES_MAX):
        assert C.classify_box(down, ssh_fails=n) == "UNKNOWN"
    assert C.classify_box(down, ssh_fails=C.SSH_STRIKES_MAX) == "DEAD"
    ok = C.BoxProbe(ok=True)
    assert C.classify_box(ok, deploy_fails=C.DEPLOY_STRIKES_MAX) == "DEAD"
    assert C.classify_box(ok, deploy_fails=C.DEPLOY_STRIKES_MAX - 1) == "IDLE"


def test_boot_grace_is_thirteen_minutes():
    assert C.BOOT_GRACE_S == 13 * 60


def test_desired_box_count_never_exceeds_the_work():
    assert C.desired_box_count(14, 534) == 14
    assert C.desired_box_count(14, 3) == 3
    assert C.desired_box_count(14, 0) == 0


def test_search_query_encodes_the_round3_contract():
    q = C.build_query_r3()
    for token in ("RTX_3090", "RTX_3090_Ti", "RTX_4080", "RTX_4090"):
        assert token in q
    assert "verified=true" in q and "rentable=true" in q and "num_gpus=1" in q
    assert "dph_total<=0.6" in q
    assert "disk_space>=32" in q and "inet_down>=200" in q


def _offer(**kw) -> object:
    row = {
        "id": kw.pop("id", 1), "dph_total": kw.pop("dph", 0.3),
        "gpu_name": kw.pop("gpu", "RTX 3090"), "num_gpus": kw.pop("num_gpus", 1),
        "verification": kw.pop("verification", "verified"),
        "inet_down": kw.pop("inet_down", 900.0), "disk_space": kw.pop("disk", 200.0),
        "machine_id": kw.pop("machine_id", 10), "reliability": kw.pop("rel", 0.99),
    }
    row.update(kw)
    return C.parse_offer(row)


def test_filter_offers_enforces_every_hard_bound():
    good = _offer(id=1, dph=0.25)
    assert C.filter_offers_r3([good]) == [good]
    rejected = [
        _offer(id=2, dph=0.75),                      # over the price cap
        _offer(id=3, gpu="RTX 3060"),                # wrong GPU
        _offer(id=4, gpu="RTX 4070 Ti"),             # wrong GPU (near miss)
        _offer(id=5, num_gpus=2),                    # multi-GPU
        _offer(id=6, verification="unverified"),     # unverified host
        _offer(id=7, inet_down=50.0),                # slow downlink
        _offer(id=8, disk=16.0),                     # too little disk
    ]
    assert C.filter_offers_r3(rejected) == []
    assert C.filter_offers_r3([good], exclude_machine_ids={10}) == []


def test_offer_names_accept_the_underscore_spelling():
    assert C.filter_offers_r3([_offer(gpu="RTX_4090")])


def test_rank_offers_is_cheapest_first():
    offers = [_offer(id=1, dph=0.5), _offer(id=2, dph=0.2), _offer(id=3, dph=0.35)]
    assert [o.id for o in C.rank_offers_r3(offers)] == [2, 3, 1]


# =========================================================================== #
# Worker-side command hygiene (lessons that already cost a deploy round)
# =========================================================================== #


_PATTERN_RE = re.compile(r"p(?:kill|grep) -f '([^']+)'")


def test_no_pkill_or_pgrep_pattern_can_self_match():
    """Every ``pkill``/``pgrep -f`` pattern must be bracketed.

    An unbracketed ``pkill -f run_shard`` matches the very shell running it and
    kills the deploy mid-flight -- which is exactly what happened once. Bracketing
    (``[r]un_shard``) makes the pattern match the target but never its own command
    line.
    """
    source = Path(C.__file__).read_text(encoding="utf-8")
    patterns = _PATTERN_RE.findall(source)
    assert patterns, "expected pkill/pgrep patterns in the controller"
    for pat in patterns:
        assert "run_shard" not in pat, f"unbracketed self-matching pattern: {pat}"
        assert "run_one" not in pat, f"unbracketed self-matching pattern: {pat}"


def test_probe_script_is_readonly_and_cannot_self_match():
    script = C.ControllerR3.PROBE_SCRIPT
    assert "run_shard" not in script and "run_one" not in script
    assert "import torch" not in script, "the readiness probe must stay LIGHT"
    for destructive in ("rm -rf", "pkill", "shutdown"):
        assert destructive not in script


def test_the_image_ships_torch_and_an_sshd():
    assert C.IMAGE.startswith("vastai/"), "SSH launch mode needs a vastai/* image"


def test_probe_parsing_turns_shell_output_into_facts(tmp_path):
    class ProbeBackend(MockBackend):
        def execute(self, instance_id, command, *, timeout=None):
            return (
                "ts=1700000600\none=1\nshard=1\nfeds=fed_aaaaaaaaaa,fed_bbbbbbbbbb,\n"
                "res=7\nlogts=1700000550\ndone=0\nlast=[run_one] START r001\n"
            )

    ctl = make_controller(tmp_path, ProbeBackend())
    pr = ctl.probe(1)
    assert pr.ok and pr.run_one_alive and pr.shard_alive
    assert pr.feds_present == ("fed_aaaaaaaaaa", "fed_bbbbbbbbbb")
    assert pr.results == 7
    assert pr.log_age_s == pytest.approx(50.0)
    assert pr.shard_done is False
    assert C.classify_box(pr, required_feds=("fed_aaaaaaaaaa",)) == "BUSY"


def test_probe_failure_is_not_a_crash(tmp_path):
    class BrokenBackend(MockBackend):
        def execute(self, instance_id, command, *, timeout=None):
            raise RuntimeError("ssh: connect to host ... port 22: Connection refused")

    ctl = make_controller(tmp_path, BrokenBackend())
    pr = ctl.probe(1)
    assert pr.ok is False and "Connection refused" in pr.error
    assert C.classify_box(pr, ssh_fails=1) == "UNKNOWN"


# =========================================================================== #
# Collection: nothing is ever lost or double-counted
# =========================================================================== #


def test_collected_ids_ignores_the_controllers_own_files(tmp_path):
    backend = MockBackend()
    ctl = make_controller(tmp_path, backend, entries=entries_for({0: 2}))
    (ctl.out_dir / "r0_000.json").write_text("{}", encoding="utf-8")
    (ctl.out_dir / "controller_r3_state.json").write_text("{}", encoding="utf-8")
    (ctl.out_dir / "fleet_state.json").write_text("{}", encoding="utf-8")
    assert ctl.collected_ids() == {"r0_000"}
    assert [e["run_id"] for e in ctl.pending_entries()] == ["r0_001"]


def test_ingest_tar_flattens_paths_and_never_overwrites(tmp_path):
    import io
    import tarfile

    backend = MockBackend()
    ctl = make_controller(tmp_path, backend)
    (ctl.out_dir / "keep.json").write_text('{"mine": 1}', encoding="utf-8")
    tar_path = tmp_path / "payload.tgz"
    with tarfile.open(tar_path, "w:gz") as tf:
        for name, body in (("results/new.json", b"{}"),
                           ("../../etc/evil.json", b"{}"),
                           ("keep.json", b'{"theirs": 1}'),
                           ("notes.txt", b"ignored")):
            info = tarfile.TarInfo(name)
            info.size = len(body)
            tf.addfile(info, io.BytesIO(body))
    added = ctl._ingest_tar(tar_path)
    assert added == 2  # new.json + evil.json (flattened), keep.json skipped
    assert (ctl.out_dir / "new.json").exists()
    assert (ctl.out_dir / "evil.json").exists()          # landed FLAT, inside out_dir
    assert not (tmp_path.parent / "etc" / "evil.json").exists()  # no traversal
    assert json.loads((ctl.out_dir / "keep.json").read_text(encoding="utf-8")) == {"mine": 1}
    assert not (ctl.out_dir / "notes.txt").exists()


def test_controller_state_roundtrips(tmp_path):
    backend = MockBackend([inst_row(1, LABEL)])
    ctl = make_controller(tmp_path, backend)
    ctl.boxes[1] = C.BoxState(1, 123.0)
    ctl.boxes[1].staged_feds = {"fed_x"}
    ctl.boxes[1].assigned = ("a", "b")
    ctl.balance_start, ctl.balance_start_ts = 65.0, 1.0
    ctl.exclude_machines = {77}
    ctl._save_state()

    ctl2 = make_controller(tmp_path, backend)
    assert ctl2.balance_start == 65.0
    assert ctl2.exclude_machines == {77}
    assert ctl2.boxes[1].staged_feds == {"fed_x"}
    assert ctl2.boxes[1].assigned == ("a", "b")


def test_state_written_under_another_label_is_ignored(tmp_path):
    backend = MockBackend()
    ctl = make_controller(tmp_path, backend)
    ctl.state_path.write_text(
        json.dumps({"label": "fedgrid-fleet", "balance_start": 1.0,
                    "boxes": [{"instance_id": 9, "first_seen": 1.0}]}),
        encoding="utf-8",
    )
    ctl2 = make_controller(tmp_path, backend)
    assert ctl2.balance_start is None and ctl2.boxes == {}


# =========================================================================== #
# CLI wiring
# =========================================================================== #


# =========================================================================== #
# End-to-end: a simulated fleet must converge, collect everything and tear down
# =========================================================================== #


class SimBackend(MockBackend):
    """A fake fleet that boots, accepts deploys, produces results and gets pulled."""

    def __init__(self, n_offers: int = 6) -> None:
        super().__init__(credit=65.0)
        import threading

        self.lock = threading.RLock()
        self.assigned: dict[int, list[str]] = {}
        self.produced: dict[int, list[str]] = {}
        self.pending_tar: dict[int, list[str]] = {}
        self.n_offers = n_offers
        self.speed = 3  # runs finished between two probes

    def search_offers(self, query, *, order="dph_total"):
        return [
            {"id": 700 + i, "dph_total": 0.15 + i / 100, "gpu_name": "RTX 3090",
             "num_gpus": 1, "verification": "verified", "inet_down": 900.0,
             "disk_space": 200.0, "machine_id": 500 + i, "reliability": 0.99,
             "geolocation": "US"}
            for i in range(self.n_offers)
        ]

    def show_instances(self, *, label=None):
        with self.lock:
            for row in self.rows:                       # boxes boot on first sight
                if row.get("label") == LABEL and row["actual_status"] == "loading":
                    row["actual_status"] = row["cur_state"] = "running"
            return [dict(r) for r in self.rows]

    def execute(self, instance_id, command, *, timeout=None):
        with self.lock:
            self.executed.append((instance_id, command))
            if command.startswith("set +e; echo ts="):     # the probe
                todo = [r for r in self.assigned.get(instance_id, [])
                        if r not in self.produced.get(instance_id, [])]
                for rid in todo[: self.speed]:
                    self.produced.setdefault(instance_id, []).append(rid)
                left = len(todo) - min(len(todo), self.speed)
                return (
                    f"ts=1700000000\none={1 if left else 0}\nshard={1 if left else 0}\n"
                    f"feds=fed_test0000,\nres={len(self.produced.get(instance_id, []))}\n"
                    f"logts=1699999990\ndone={0 if left else 1}\nlast=simulated\n"
                )
            if "LIVE=" in command:
                return "LIVE=1\n"
            if command.startswith("ls -1 "):
                return "\n".join(f"{r}.json" for r in self.produced.get(instance_id, []))
            if "tar -czf" in command and "_new_results" in command:
                self.pending_tar[instance_id] = [
                    tok.strip("'") for tok in command.split() if tok.endswith(".json")
                    or tok.endswith(".json'")
                ]
            return ""

    def copy(self, src: str, dst: str) -> None:
        import io
        import tarfile as _tar

        if ":" in dst and not Path(dst).drive:            # push
            iid = int(dst.split(":", 1)[0])
            manifest = Path(src) / "manifest.jsonl"
            if manifest.exists():
                with self.lock:
                    self.assigned[iid] = [
                        json.loads(ln)["run_id"]
                        for ln in manifest.read_text(encoding="utf-8").splitlines() if ln.strip()
                    ]
            return
        iid = int(src.split(":", 1)[0])                    # pull
        names = self.pending_tar.get(iid, [])
        with _tar.open(dst, "w:gz") as tf:
            for name in names:
                body = json.dumps({"run_id": Path(name).stem, "final": {"acc": 1.0}}).encode()
                info = _tar.TarInfo(name)
                info.size = len(body)
                tf.addfile(info, io.BytesIO(body))


def test_simulated_fleet_converges_collects_everything_and_tears_down(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "START_VERIFY_DELAY_S", 0.0)
    backend = SimBackend()
    ents = entries_for({0: 12})
    ctl = make_controller(tmp_path, backend, entries=ents, target=3,
                          cycle_s=0.01, collect_s=0.02, max_cycles=60)
    # A one-file stand-in for the staged federation (fed_tar only needs a real dir).
    fed = C.entry_fed_id(ents[0])
    (ctl.data_dir / fed).mkdir(parents=True)
    (ctl.data_dir / fed / "manifest.json").write_text('{"config": {}}', encoding="utf-8")

    assert ctl.run() == 0

    collected = ctl.collected_ids()
    assert collected == {e["run_id"] for e in ents}, "a run was lost"
    assert ctl.pending_entries() == []
    # Exactly 3 boxes were rented, all destroyed, production untouched.
    assert len(backend.created) == 3
    assert len(backend.destroyed) == 3
    assert {r["id"] for r in backend.rows} == {10000001, 10000002}
    assert C.select_ours(backend.show_instances()) == []


def test_cap_teardown_harvests_before_destroying(tmp_path, monkeypatch):
    """Hitting the cap must not throw away runs that are finished on the boxes."""
    monkeypatch.setattr(C, "START_VERIFY_DELAY_S", 0.0)
    backend = SimBackend()
    ents = entries_for({0: 6})
    ctl = make_controller(tmp_path, backend, entries=ents, target=2, cap_usd=15.0)
    fed = C.entry_fed_id(ents[0])
    (ctl.data_dir / fed).mkdir(parents=True)
    (ctl.data_dir / fed / "manifest.json").write_text('{"config": {}}', encoding="utf-8")
    ctl.cycle()                                  # provision
    ctl.cycle()                                  # deploy
    assert any(b.deployed_ts for b in ctl.boxes.values())
    for _ in range(2):                           # let the boxes finish some runs
        ctl.cycle()
    backend.credit = 20.0                        # the money is suddenly gone
    ctl.balance_start, ctl.balance_start_ts = 65.0, ctl._started
    before = len(ctl.collected_ids())
    with pytest.raises(SystemExit):
        ctl.cycle()
    assert len(ctl.collected_ids()) >= before
    assert ctl.collected_ids(), "the cap teardown lost every finished run"
    assert C.select_ours(backend.show_instances()) == []


def test_simulated_fleet_redistributes_a_dead_boxs_work(tmp_path, monkeypatch):
    """A box that vanishes mid-shard must not take its runs with it."""
    monkeypatch.setattr(C, "START_VERIFY_DELAY_S", 0.0)
    backend = SimBackend()
    ents = entries_for({0: 9})
    ctl = make_controller(tmp_path, backend, entries=ents, target=3,
                          cycle_s=0.01, collect_s=0.02, max_cycles=60)
    fed = C.entry_fed_id(ents[0])
    (ctl.data_dir / fed).mkdir(parents=True)
    (ctl.data_dir / fed / "manifest.json").write_text('{"config": {}}', encoding="utf-8")

    ctl.cycle()                       # provision
    ctl.cycle()                       # boxes are running -> deploy
    victim = sorted(ctl.boxes)[0]
    lost = set(ctl.boxes[victim].assigned)
    assert lost, "the victim had no work to lose"
    with backend.lock:                # the host disappears without warning
        backend.rows = [r for r in backend.rows if int(r["id"]) != victim]
        backend.produced.pop(victim, None)
        backend.assigned.pop(victim, None)

    for _ in range(40):
        if ctl.cycle():
            break
    assert lost <= ctl.collected_ids(), "runs from the dead box were never re-run"
    assert ctl.pending_entries() == []


def test_cli_requires_a_manifest(monkeypatch, capsys):
    monkeypatch.setattr(C, "make_backend", lambda cfg: MockBackend())
    assert C.main(["--out-dir", "x"]) == 2
    assert "manifest" in capsys.readouterr().out.lower()


def test_cli_exposes_no_way_to_change_the_label(capsys):
    """``--label`` must not exist: the label IS the safety boundary."""
    parser = C._build_parser()
    options = {a.option_strings[0] for a in parser._actions if a.option_strings}
    assert "--label" not in options
    with pytest.raises(SystemExit):
        parser.parse_args(["--label", "anything"])
