import pytest

from lib.graph.methods import method_from_switches, neighborhood_policy, validate_method


@pytest.mark.parametrize("method", ["RTEC-Full", "RTEC-NS", "RTEC-UER", "NrtInc"])
def test_canonical_method_names(method):
	assert validate_method(method) == method


@pytest.mark.parametrize("method", ["full", "incremental", "RTEC-NS5", "RTEC-NS10"])
def test_removed_launcher_aliases_are_rejected(method):
	with pytest.raises(ValueError, match="Unsupported method"):
		validate_method(method)


def test_unknown_algorithm_cannot_be_mislabeled_as_uer():
	with pytest.raises(ValueError, match="Unsupported method"):
		validate_method("unknown-algorithm")
	assert method_from_switches({"operator_incremental": True}) == "NrtInc"
	assert method_from_switches({"incremental_aggregates": True, "row_refresh": True}) == "NrtInc"
	assert method_from_switches({"retain_aggregates": True}) == "RTEC-UER"
	assert not neighborhood_policy("RTEC-Full").reuse_intermediates
	assert neighborhood_policy("RTEC-NS", fanout=5).fanout == 5
	assert neighborhood_policy("RTEC-NS", fanout=10).fanout == 10
	assert method_from_switches({}, policy=neighborhood_policy("RTEC-Full")) == "RTEC-Full"
	assert method_from_switches({}, policy=neighborhood_policy("RTEC-NS", fanout=1)) == "RTEC-NS"


@pytest.mark.parametrize("fanout", [0, -1, True, 1.5])
def test_sampling_fanout_requires_a_positive_integer(fanout):
	with pytest.raises(ValueError, match="positive integer"):
		neighborhood_policy("RTEC-NS", fanout=fanout)
