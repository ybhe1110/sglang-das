from __future__ import annotations

import importlib.metadata as importlib_metadata
import json
import logging
import os
import threading
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from sglang.srt.environ import envs
from sglang.srt.runtime_context import (
    get_disagg,
    get_exec,
    get_memory,
)
from sglang.srt.utils.network import NetworkAddress, get_free_port, get_local_ip_auto

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

# Module-level shared engine instance, set by init_mooncake_transfer_engine().
_mooncake_transfer_engine: Optional[MooncakeTransferEngine] = None
_mooncake_transfer_engine_lock = threading.RLock()


def parse_ib_device_config(
    ib_device_str: Optional[str],
) -> Optional[Union[str, Dict[int, str]]]:
    """Parse IB device config from a shared string, JSON mapping, or JSON file."""
    if ib_device_str is None or not ib_device_str.strip():
        return None

    normalized_input = ib_device_str.strip()
    if not normalized_input.endswith(".json") and not normalized_input.startswith("{"):
        return normalized_input

    if normalized_input.endswith(".json"):
        if not os.path.isfile(normalized_input):
            raise RuntimeError(f"File {normalized_input} does not exist.")
        try:
            with open(normalized_input, "r", encoding="utf-8") as file:
                mapping = json.load(file)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Failed to parse JSON content from file {normalized_input}"
            ) from exc
        except (IOError, OSError) as exc:
            raise RuntimeError(
                f"Failed to read JSON file {normalized_input}: {exc}"
            ) from exc
    else:
        try:
            mapping = json.loads(normalized_input)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON mapping: {normalized_input}") from exc

    if not isinstance(mapping, dict):
        raise ValueError(
            "Invalid format: expected a mapping from GPU id to IB device string"
        )

    normalized_mapping: Dict[int, str] = {}
    for gpu_key, ib_devices in mapping.items():
        normalized_key = int(gpu_key) if str(gpu_key).isdigit() else None
        if normalized_key is None or not isinstance(ib_devices, str):
            raise ValueError(
                "Invalid format: keys must be integers (or string "
                "representations of integers) and values must be strings"
            )
        normalized_mapping[normalized_key] = ib_devices.strip()

    if not normalized_mapping:
        raise ValueError("No valid GPU mappings found in JSON")

    return normalized_mapping


def get_ib_devices_for_gpu(ib_device_str: Optional[str], gpu_id: int) -> Optional[str]:
    """
    Parse IB device string and get IB devices for a specific GPU ID.

    Supports all the following formats:
    1. Old format: "ib0, ib1, ib2"
    2. New format: {0: "ib0, ib1", 1: "ib2, ib3", 2: "ib4"}
    3. JSON file: path to a JSON file containing the mapping

    Args:
        ib_device_str: The original IB device string or path to JSON file
        gpu_id: The GPU ID to get devices for

    Returns:
        IB devices string for the GPU, or None if not available
    """
    parsed_config = parse_ib_device_config(ib_device_str)
    if parsed_config is None:
        return None

    if isinstance(parsed_config, str):
        return parsed_config

    if gpu_id in parsed_config:
        return parsed_config[gpu_id]

    raise ValueError(
        f"No IB devices configured for GPU {gpu_id}. "
        f"Available GPUs: {list(parsed_config.keys())}"
    )


def configure_mooncake_transfer_timeout() -> None:
    """Set Mooncake's native transfer deadline without overriding operators."""
    # Mooncake reads MC_TRANSFER_TIMEOUT while constructing the native
    # TransferEngine.  This must happen before that construction; a Python
    # Future timeout would only abandon the caller while the native WR could
    # still access the registered KV buffers.
    transfer_timeout_s = envs.SGLANG_DISAGGREGATION_MOONCAKE_TRANSFER_TIMEOUT.get()
    if transfer_timeout_s > 0 and "MC_TRANSFER_TIMEOUT" not in os.environ:
        os.environ["MC_TRANSFER_TIMEOUT"] = str(transfer_timeout_s)
        logger.info(
            "Configured MC_TRANSFER_TIMEOUT=%.3fs for Mooncake transfers",
            transfer_timeout_s,
        )


def get_mooncake_package_version() -> str:
    """Return the installed Mooncake distribution and version for diagnostics."""
    # CUDA 13 images use the suffixed distribution name; CUDA 12 images use
    # the generic package.  The import name is ``mooncake`` for both.
    for distribution in (
        "mooncake-transfer-engine-cuda13",
        "mooncake-transfer-engine",
    ):
        try:
            return f"{distribution}=={importlib_metadata.version(distribution)}"
        except importlib_metadata.PackageNotFoundError:
            continue
    return "unknown"


def log_mooncake_runtime_configuration(engine) -> None:
    """Log the effective native timeout and API surface once at startup."""
    configured_timeout = envs.SGLANG_DISAGGREGATION_MOONCAKE_TRANSFER_TIMEOUT.get()
    native_timeout = os.environ.get("MC_TRANSFER_TIMEOUT", "<unset>")
    logger.info(
        "Mooncake runtime configuration: package=%s, "
        "sglang_timeout=%.3fs, MC_TRANSFER_TIMEOUT=%s, "
        "batch_transfer_sync_write=%s, MC_FORCE_TCP=%s",
        get_mooncake_package_version(),
        configured_timeout,
        native_timeout,
        hasattr(engine, "batch_transfer_sync_write"),
        os.environ.get("MC_FORCE_TCP", "0"),
    )


class MooncakeTransferEngine:
    """Shared Mooncake transfer engine for RDMA/transfer operations."""

    def __init__(
        self,
        hostname: str,
        gpu_id: Optional[int] = None,
        ib_device: Optional[str] = None,
    ):
        configure_mooncake_transfer_timeout()
        try:
            from mooncake.engine import TransferEngine
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://kvcache-ai.github.io/Mooncake/getting_started/build.html "
                "to run SGLang with MooncakeTransferEngine."
            ) from e

        self.engine = TransferEngine()
        self._shutdown = False
        self._registered_regions = set()
        # Peer endpoint ids are learned from transfer/probe calls.  Keeping a
        # process-local set lets shutdown explicitly tear down every session
        # before destroying the native endpoint and its QPs.
        self._peer_sessions = set()
        self._region_lock = threading.RLock()
        self._session_lock = threading.RLock()
        log_mooncake_runtime_configuration(self.engine)
        self.hostname = hostname
        self.gpu_id = gpu_id if gpu_id is not None else 0
        # MC_FORCE_TCP=1 makes mooncake install TcpTransport instead of RDMA,
        # in which case RDMA HCA selection is irrelevant; pass empty device.
        if os.environ.get("MC_FORCE_TCP") == "1":
            self.ib_device = ""
        else:
            self.ib_device = get_ib_devices_for_gpu(ib_device, self.gpu_id)

        self.initialize(
            hostname=self.hostname,
            device_name=self.ib_device,
        )
        self.session_id = NetworkAddress(
            self.hostname, self.engine.get_rpc_port()
        ).to_host_port_str()

    def register(self, ptr, length):
        try:
            ret_value = self.engine.register_memory(ptr, length)
        except Exception:
            # Mark register as failed
            ret_value = -1

        if ret_value != 0:
            logger.debug("Mooncake memory registration %s failed.", ptr)
        else:
            with self._region_lock:
                self._registered_regions.add((ptr, length))

    def deregister(self, ptr):
        try:
            ret_value = self.engine.unregister_memory(ptr)
        except Exception:
            # Mark deregister as failed
            ret_value = -1

        if ret_value != 0:
            logger.debug("Mooncake memory deregistration %s failed.", ptr)
        else:
            with self._region_lock:
                self._registered_regions = {
                    region for region in self._registered_regions if region[0] != ptr
                }

    def batch_register(self, ptrs: List[int], lengths: List[int]) -> int:
        """Batch register multiple memory regions."""
        try:
            ret_value = self.engine.batch_register_memory(ptrs, lengths)
        except Exception:
            # Mark batch register as failed
            ret_value = -1
            if not hasattr(self.engine, "batch_register_memory"):
                raise RuntimeError(
                    "Mooncake's batch register requires a newer version of "
                    "mooncake-transfer-engine. Please upgrade Mooncake."
                )

        if ret_value != 0:
            logger.debug("Mooncake batch memory registration failed.")
        else:
            with self._region_lock:
                self._registered_regions.update(zip(ptrs, lengths))
        return ret_value

    def batch_deregister(self, ptrs: List[int]) -> int:
        """Batch deregister multiple memory regions."""
        try:
            ret_value = self.engine.batch_unregister_memory(ptrs)
        except Exception:
            # Mark batch deregister as failed
            ret_value = -1

        if ret_value != 0:
            logger.debug("Mooncake batch memory deregistration failed.")
        else:
            with self._region_lock:
                ptr_set = set(ptrs)
                self._registered_regions = {
                    region for region in self._registered_regions if region[0] not in ptr_set
                }
        return ret_value

    def shutdown(self, force_sessions: bool = True) -> None:
        """Deregister buffers and destroy the native endpoint/QP.

        Mooncake has exposed several names for the native endpoint teardown
        across releases.  Use the strongest available method exactly once;
        the native implementation owns QP destruction and its completion
        queues.  If an old package has no teardown API, log loudly and rely on
        process teardown instead of pretending the endpoint was destroyed.
        """
        with self._region_lock:
            if self._shutdown:
                return
            self._shutdown = True
            regions = list(self._registered_regions)
        if force_sessions:
            self.force_destroy_sessions()
        if regions:
            self.batch_deregister([ptr for ptr, _ in regions])

        for target in self._native_targets():
            for method_name in ("shutdown", "close", "destroy", "finalize", "stop"):
                method = getattr(target, method_name, None)
                if not callable(method):
                    continue
                try:
                    method()
                    logger.info(
                        "Mooncake Transfer Engine endpoint shut down via %s",
                        method_name,
                    )
                except Exception:
                    logger.exception(
                        "Mooncake Transfer Engine endpoint shutdown via %s failed",
                        method_name,
                    )
                return
        logger.warning(
            "Mooncake Transfer Engine exposes no endpoint shutdown API; "
            "native QP cleanup is deferred to process teardown"
        )

    def _track_peer_session(self, session_id: str) -> None:
        if session_id:
            added = False
            with self._session_lock:
                added = session_id not in self._peer_sessions
                self._peer_sessions.add(session_id)
                count = len(self._peer_sessions)
            if added:
                logger.info("Mooncake peer sessions tracked: count=%d", count)

    def peer_session_count(self) -> int:
        """Return the number of unique peer endpoints tracked by this process."""
        with self._session_lock:
            return len(self._peer_sessions)

    def _native_targets(self):
        """Yield wrapper and underlying native endpoint exactly once."""
        targets = [self.engine]
        get_engine = getattr(self.engine, "get_engine", None)
        if callable(get_engine):
            try:
                native = get_engine()
            except Exception:
                native = None
            if native is not None and native is not self.engine:
                targets.append(native)
        return targets

    def destroy_session(self, session_id: str) -> bool:
        """Destroy one native peer session when the installed API supports it.

        Mooncake has used several names for this operation.  Endpoint
        shutdown remains the final fallback because older wheels do not expose
        a per-session method.
        """
        if not session_id:
            return True
        for target in self._native_targets():
            for method_name in (
                "destroy_session",
                "close_session",
                "remove_session",
                "unregister_session",
                "disconnect",
            ):
                method = getattr(target, method_name, None)
                if not callable(method):
                    continue
                try:
                    result = method(session_id)
                except TypeError:
                    # A method with the same name but endpoint-level signature
                    # is not a per-session API; try the next candidate.
                    continue
                except Exception:
                    logger.exception(
                        "Mooncake session %s destroy via %s failed",
                        session_id,
                        method_name,
                    )
                    return False
                if result is None or result == 0 or result is True:
                    with self._session_lock:
                        self._peer_sessions.discard(session_id)
                    logger.debug(
                        "Mooncake session %s destroyed via %s",
                        session_id,
                        method_name,
                    )
                    return True
                logger.warning(
                    "Mooncake session %s destroy via %s returned %r",
                    session_id,
                    method_name,
                    result,
                )
                return False
        return False

    def force_destroy_sessions(self) -> bool:
        """Destroy all tracked peer sessions before endpoint teardown."""
        with self._session_lock:
            sessions = list(self._peer_sessions)
        if not sessions:
            return True
        all_destroyed = True
        for session_id in sessions:
            if not self.destroy_session(session_id):
                all_destroyed = False
        if not all_destroyed:
            logger.warning(
                "Mooncake native wheel has no usable per-session destroy API; "
                "endpoint shutdown will be used for sessions=%s",
                sessions,
            )
        return all_destroyed

    def initialize(
        self,
        hostname: str,
        device_name: Optional[str],
    ) -> None:
        """Initialize the mooncake instance."""
        if envs.ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE.get():
            npu_phy_id = envs.ASCEND_NPU_PHY_ID.get()
            suffix = self.gpu_id if npu_phy_id == -1 else npu_phy_id
            hostname += f":{get_free_port()}:npu_{suffix}"
            protocol = "ascend"
        else:
            # MOONCAKE_PROTOCOL selects the transport (rdma | efa | tcp | ...).
            # Default is "rdma"; set MOONCAKE_PROTOCOL=efa on AWS EFA hardware.
            protocol = envs.MOONCAKE_PROTOCOL.get()

        ret_value = self.engine.initialize(
            hostname,
            "P2PHANDSHAKE",
            protocol,
            device_name if device_name is not None else "",
        )
        if ret_value != 0:
            logger.error("Mooncake Transfer Engine initialization failed.")
            raise RuntimeError("Mooncake Transfer Engine initialization failed.")

    def transfer_sync(
        self, session_id: str, buffer: int, peer_buffer_address: int, length: int
    ) -> int:
        """Synchronously transfer data to the specified address."""
        self._track_peer_session(session_id)
        try:
            ret = self.engine.transfer_sync_write(
                session_id, buffer, peer_buffer_address, length
            )
        except Exception:
            ret = -1

        if ret < 0:
            logger.debug(
                "Failed to transfer data from %s to %s - %s.",
                buffer,
                session_id,
                peer_buffer_address,
            )

        return ret

    def batch_transfer_sync(
        self,
        session_id: str,
        buffers: List[int],
        peer_buffer_addresses: List[int],
        lengths: List[int],
    ) -> int:
        """Synchronously transfer data to the specified addresses in batches."""
        self._track_peer_session(session_id)
        try:
            ret = self.engine.batch_transfer_sync_write(
                session_id, buffers, peer_buffer_addresses, lengths
            )
        except Exception:
            ret = -1
            if not hasattr(self.engine, "batch_transfer_sync_write"):
                raise RuntimeError(
                    "Mooncake's batch transfer requires mooncake-transfer-engine "
                    ">= 0.3.4.post2. Please upgrade Mooncake by "
                    "'pip install mooncake-transfer-engine --upgrade'"
                )

        if ret < 0:
            logger.debug(
                "Failed to batch transfer data. Buffers: %s, Session: %s, "
                "Peer addresses: %s",
                buffers,
                session_id,
                peer_buffer_addresses,
            )
        return ret

    def get_session_id(self):
        return self.session_id

    def send_probe(self, peer_session_id: str) -> int:
        self._track_peer_session(peer_session_id)
        return self.engine.send_probe(peer_session_id)

    def get_engine(self):
        return self.engine.get_engine()

    def get_ib_device(self):
        return self.ib_device


def init_mooncake_transfer_engine(
    hostname: str,
    gpu_id: Optional[int] = None,
    ib_device: Optional[str] = None,
) -> MooncakeTransferEngine:
    """
    Initialize the shared MooncakeTransferEngine. Note: if already
    initialized with the same (hostname, gpu_id, ib_device), returns existing
    instance. Call from parallel_state when model parallel is set up and
    mooncake transfer is needed.
    """
    global _mooncake_transfer_engine
    with _mooncake_transfer_engine_lock:
        if _mooncake_transfer_engine is not None:
            return _mooncake_transfer_engine
        _mooncake_transfer_engine = MooncakeTransferEngine(
            hostname=hostname, gpu_id=gpu_id, ib_device=ib_device
        )
        return _mooncake_transfer_engine


def get_mooncake_transfer_engine() -> Optional[MooncakeTransferEngine]:
    """Return the shared MooncakeTransferEngine if initialized, else None."""
    with _mooncake_transfer_engine_lock:
        return _mooncake_transfer_engine


def shutdown_mooncake_transfer_engine(force: bool = True) -> None:
    """Shutdown the process-level Mooncake engine, if one is initialized."""
    global _mooncake_transfer_engine
    with _mooncake_transfer_engine_lock:
        engine = _mooncake_transfer_engine
        _mooncake_transfer_engine = None
    if engine is not None:
        engine.shutdown(force_sessions=force)


def maybe_init_shared_mooncake_transfer_engine(
    *, server_args: ServerArgs, gpu_id: int
) -> None:
    """
    Need MooncakeTransferEngine when:
    1) PD disaggregation uses mooncake for KV transfer (prefill/decode)
    2) HiCache uses mooncake storage backend
    3) Encoder disaggregation uses mooncake
    """
    use_mooncake_te = (
        (
            get_disagg().disaggregation_mode != "null"
            and get_disagg().disaggregation_transfer_backend == "mooncake"
        )
        or (
            get_memory().enable_hierarchical_cache
            and get_memory().hicache_storage_backend == "mooncake"
            and envs.SGLANG_HICACHE_MOONCAKE_REUSE_TE.get()
        )
        or (
            get_disagg().encoder_only
            and get_disagg().encoder_transfer_backend == "mooncake"
        )
        or (
            get_disagg().language_only
            and get_disagg().encoder_transfer_backend == "mooncake"
        )
        or (
            get_exec().moe.enable_elastic_expert_backup
            and get_exec().moe.elastic_ep_backend is not None
        )
        or get_exec().moe.elastic_ep_backend == "mooncake"
    )

    if use_mooncake_te:
        init_mooncake_transfer_engine(
            hostname=get_local_ip_auto(),
            gpu_id=gpu_id,
            ib_device=(
                get_disagg().disaggregation_ib_device
                or get_exec().moe.mooncake_ib_device
            ),
        )

        if get_exec().moe.elastic_ep_backend == "mooncake":
            try:
                from mooncake.pg import set_transfer_engine
            except ImportError as e:
                raise ImportError(
                    "Failed to import 'set_transfer_engine' from 'mooncake.pg'. "
                    "Please upgrade your 'mooncake-transfer-engine' "
                    "installation to 0.3.11 or above."
                ) from e
            set_transfer_engine(_mooncake_transfer_engine.engine)
