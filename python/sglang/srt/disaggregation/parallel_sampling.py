"""Shared P/D identities for tokenizer-side parallel sampling."""

import hashlib
import json


def assign_parallel_sample_bootstrap(
    obj, tokenized_obj, *, prompt_index: int, sample_index: int
) -> None:
    """Give copied requests a separate room and pair key on both P/D workers.

    P and D may have different RIDs and expand at different times. Derive only
    from the router's shared identity and stable indices; -1 denotes warmup.
    Include the pair key so a retried bootstrap attempt gets fresh child rooms.
    Call after tokenization has validated the parent's bootstrap credentials.
    """
    room = getattr(tokenized_obj, "bootstrap_room", None)
    if room is None:
        return
    pair_key = tokenized_obj.bootstrap_pair_key
    identity = json.dumps(
        ["sglang-parallel-sampling-v1", room, pair_key, prompt_index, sample_index],
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(identity).digest()
    # Zero is the unwritten metadata sentinel. Stay within signed int64 as
    # well as uint64, since some transfer backends use signed metadata buffers.
    child_room = int.from_bytes(digest[:8], "big") % ((1 << 63) - 1) + 1
    # Backends without pair keys retain that contract. Do not manufacture a
    # credential that would bypass Mooncake's missing-parent-key validation.
    child_key = digest.hex() if pair_key is not None else None
    obj.bootstrap_room = tokenized_obj.bootstrap_room = child_room
    obj.bootstrap_pair_key = tokenized_obj.bootstrap_pair_key = child_key
