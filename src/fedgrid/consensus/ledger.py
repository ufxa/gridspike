"""Hash-chained ledger of per-round consensus decisions (auditability, not mining).

The claim this structure supports is narrow and worth stating precisely, because
"blockchain" in a federated-learning paper usually means more than it delivers.
It is not a consensus mechanism (PBFT is), it has no proof of work, no incentive
layer, and no protection against a dishonest writer -- an operator who rewrites a
block and rehashes every successor produces a chain that verifies.

What :meth:`Ledger.verify` does buy is *interior* tamper-evidence, in O(n) and
without the auditor holding a reference copy: editing a stored record, editing a
block's own metadata, reordering blocks, or deleting a block that is followed by
another block all break recomputation, because some later block's ``prev_hash``
still commits to the value that was changed or removed.

What it does **not** buy, and the reason it is not a complete audit trail on its
own: **truncating the tail is undetectable.** Dropping the last k blocks (up to
and including all of them) leaves a prefix that is itself a perfectly valid
chain -- every link, digest and index recomputes -- so ``verify()`` returns
``True`` on it. The same hole covers truncate-then-re-append: a writer may
discard the last k rounds and record different ones in their place. Nothing in
the stored structure commits to *how long* the chain is supposed to be, and a
hash chain cannot, from the inside, detect the removal of its own most recent
state.

The missing ingredient is a commitment to the head held outside the chain: the
record count and final digest, signed by the validators or anchored somewhere the
writer does not control, and checked against the chain at audit time. That is
deliberately out of scope in this simulation -- there is one ledger rather than
one replica per validator, and no validator keys -- so the honest statement of
the property is "interior tamper-evidence, given an independently known length
and head". :meth:`Ledger.verify` accepts optional ``expected_len`` /
``expected_head`` arguments so that a caller who *does* hold such a commitment
can check it in the same pass; the console's verify badge does not, and shows
only structural self-consistency.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import numpy as np

__all__ = ["GENESIS_HASH", "Ledger"]

GENESIS_HASH = "0" * 64


def _json_default(o: Any) -> Any:
    """Numpy scalars/arrays are the only non-JSON types the pipeline records."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"{type(o).__name__} is not ledger-serialisable")


def _canonical(record: dict) -> str:
    """Canonical JSON: sorted keys, no whitespace, no NaN.

    Two records that differ only in key insertion order must hash identically or
    the chain would depend on how Python happened to build the dict. NaN/Inf are
    rejected rather than emitted as the non-standard ``NaN`` token: a chain whose
    payload cannot be re-parsed by a third party is not an audit trail, and a NaN
    score is a bug upstream that should surface loudly here.
    """
    return json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
        ensure_ascii=False,
        allow_nan=False,
    )


def _block_hash(prev_hash: str, payload: str) -> str:
    return hashlib.sha256((prev_hash + payload).encode("utf-8")).hexdigest()


class Ledger:
    """Append-only sha256 hash chain over the per-round consensus records.

    Blocks are ``{"index", "prev_hash", "hash", "record"}``. ``record`` is stored
    in its JSON-normalised form (numpy scalars become Python numbers), so a block
    that came back from :meth:`to_list`, a JSON file or the API re-hashes to the
    same digest as the block that was appended.

    Tamper-evidence is *interior only*. Editing any block, reordering blocks or
    deleting an interior block breaks :meth:`verify`; **truncating the tail does
    not**, because the surviving prefix is a valid chain and nothing stored
    commits to the chain's length. See :meth:`verify` for the precise statement
    and for the optional ``expected_len`` / ``expected_head`` arguments that let a
    caller supply the missing commitment.
    """

    def __init__(self, genesis_hash: str = GENESIS_HASH) -> None:
        self._genesis = genesis_hash
        self._blocks: list[dict] = []

    @classmethod
    def from_list(cls, blocks: list[dict], genesis_hash: str = GENESIS_HASH) -> Ledger:
        """Adopt an existing block list -- the auditor's entry point.

        This is how a chain that arrived from somewhere else (API response, saved
        run, another validator) gets checked: adopt, then :meth:`verify`. The
        blocks are copied, so the auditor cannot mutate the source.
        """
        led = cls(genesis_hash=genesis_hash)
        led._blocks = copy.deepcopy(list(blocks))
        return led

    def __len__(self) -> int:
        return len(self._blocks)

    @property
    def head(self) -> str:
        """Hash of the last block, or the genesis hash for an empty chain.

        This digest plus ``len(self)`` is the pair a caller must record *outside*
        the chain if truncation is to be detectable later; see :meth:`verify`.
        """
        return self._blocks[-1]["hash"] if self._blocks else self._genesis

    def append(self, record: dict) -> str:
        """Append ``record`` and return the new chain-head sha256 hash."""
        payload = _canonical(record)
        prev = self.head
        digest = _block_hash(prev, payload)
        self._blocks.append(
            {
                "index": len(self._blocks),
                "prev_hash": prev,
                "hash": digest,
                "record": json.loads(payload),
            }
        )
        return digest

    def verify(
        self,
        *,
        expected_len: int | None = None,
        expected_head: str | None = None,
    ) -> bool:
        """Recompute the chain; ``True`` iff every stored block is self-consistent.

        Checks all four invariants -- link, digest, index and payload -- at every
        block, because a tamper that only fixes one of them (rewriting a record
        and its own hash, say) must still fail on the next block's ``prev_hash``.

        **Caught.** Any edit to a stored record; any edit to a block's ``index``,
        ``prev_hash`` or ``hash``; any reordering; and any deletion of a block
        that is followed by another block. Each leaves some later ``prev_hash``
        committing to a value that is no longer there.

        **Not caught: truncation of the tail.** Dropping the last k blocks -- up
        to and including all of them -- yields a shorter chain that recomputes
        perfectly, and this method returns ``True``. So does discarding the last k
        blocks and appending different records in their place. Each block commits
        to its predecessors and nothing commits to the chain's *length*, so the
        loss of the chain's most recent state is invisible from the inside.
        Closing that gap needs a commitment held outside the chain: the record
        count and the final digest, signed by the validators or anchored where the
        writer cannot rewrite them. This module does not implement one.

        Args:
          expected_len: if given, additionally require ``len(self) ==
            expected_len``. This is the caller's half of the missing commitment
            and is the only way this method can reject a truncated tail.
          expected_head: if given, additionally require ``self.head ==
            expected_head``.

        Both default to ``None``, which is exactly the historical behaviour --
        structural self-consistency only. Neither is a substitute for a signed
        head: a dishonest writer who controls the chain *and* the expected values
        still verifies.
        """
        if expected_len is not None and len(self._blocks) != expected_len:
            return False
        if expected_head is not None and self.head != expected_head:
            return False
        prev = self._genesis
        for i, block in enumerate(self._blocks):
            try:
                payload = _canonical(block["record"])
            except (TypeError, ValueError, KeyError):
                return False
            if block.get("index") != i or block.get("prev_hash") != prev:
                return False
            if block.get("hash") != _block_hash(prev, payload):
                return False
            prev = block["hash"]
        return True

    def to_list(self) -> list[dict]:
        """The recorded blocks in append order (for the API and the console).

        Deep copies: the console and the REST layer hand these dicts around, and
        an accidental mutation there must not be able to corrupt the live chain.
        """
        return copy.deepcopy(self._blocks)
