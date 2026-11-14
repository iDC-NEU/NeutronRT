"""Direct inference connections with the same bounded envelope at both ends."""


def message_options(max_payload_bytes: int = 64 * 1024**2) -> tuple:
	# Tensor admission retains its own payload budget. The extra MiB covers
	# protobuf metadata; it does not enlarge the admitted tensor allocation.
	if type(max_payload_bytes) is not int or not 0 <= max_payload_bytes < 2**31 - 1024**2:
		raise ValueError("Invalid RPC payload byte limit")
	limit = max_payload_bytes + 1024**2
	return (
		("grpc.max_receive_message_length", limit),
		("grpc.max_send_message_length", limit),
		# Keep graph features and updates on the explicitly configured service
		# connection, independent of ambient HTTP proxy environment variables.
		("grpc.enable_http_proxy", 0),
	)
