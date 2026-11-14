"""Canonical inference algorithms and their neighborhood policies."""

RTEC_UER = "RTEC-UER"
RTEC_FULL = "RTEC-Full"
RTEC_NS = "RTEC-NS"
NRT_INC = "NrtInc"
METHODS = (NRT_INC, RTEC_NS, RTEC_UER, RTEC_FULL)


def neighborhood_policy(method: str, *, fanout: int = 5, seed: int = 0):
	from lib.graph.neighborhood import NeighborhoodPolicy

	method = validate_method(method)
	if type(fanout) is not int or fanout < 1:
		raise ValueError("Sampling fanout must be a positive integer")
	return NeighborhoodPolicy(
		reuse_intermediates=method in (RTEC_UER, NRT_INC),
		fanout=fanout if method == RTEC_NS else -1,
		seed=seed,
	)


def validate_method(value: str) -> str:
	"""Reject IDs outside the implemented algorithms."""
	method = value
	if method not in METHODS:
		raise ValueError(f"Unsupported method {value!r}; implemented methods: {', '.join(METHODS)}")
	return method


def method_from_switches(switches: dict, *, policy=None) -> str:
	"""Name the serving algorithm independently of its case/configuration ID."""
	if policy is not None and not policy.reuse_intermediates:
		return RTEC_FULL if policy.fanout == -1 else RTEC_NS
	return NRT_INC if switches.get("operator_incremental") or switches.get("incremental_aggregates") else RTEC_UER
