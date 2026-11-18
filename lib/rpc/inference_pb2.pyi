from collections.abc import Iterable as _Iterable
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar
from typing import Optional as _Optional
from typing import Union as _Union

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper

DESCRIPTOR: _descriptor.FileDescriptor

class Empty(_message.Message):
	__slots__ = ()
	def __init__(self) -> None: ...

class Tensor(_message.Message):
	__slots__ = ("dtype", "shape", "data")
	class DType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
		__slots__ = ()
		UNSPECIFIED: _ClassVar[Tensor.DType]
		INT64: _ClassVar[Tensor.DType]
		FLOAT32: _ClassVar[Tensor.DType]
		BOOL: _ClassVar[Tensor.DType]

	UNSPECIFIED: Tensor.DType
	INT64: Tensor.DType
	FLOAT32: Tensor.DType
	BOOL: Tensor.DType
	DTYPE_FIELD_NUMBER: _ClassVar[int]
	SHAPE_FIELD_NUMBER: _ClassVar[int]
	DATA_FIELD_NUMBER: _ClassVar[int]
	dtype: Tensor.DType
	shape: _containers.RepeatedScalarFieldContainer[int]
	data: bytes
	def __init__(
		self,
		dtype: _Optional[_Union[Tensor.DType, str]] = ...,
		shape: _Optional[_Iterable[int]] = ...,
		data: _Optional[bytes] = ...,
	) -> None: ...

class Update(_message.Message):
	__slots__ = ("kind", "first", "second", "request_id", "session", "attributes")
	class Kind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
		__slots__ = ()
		UNSPECIFIED: _ClassVar[Update.Kind]
		FEATURES: _ClassVar[Update.Kind]
		INSERT: _ClassVar[Update.Kind]
		DELETE: _ClassVar[Update.Kind]

	UNSPECIFIED: Update.Kind
	FEATURES: Update.Kind
	INSERT: Update.Kind
	DELETE: Update.Kind
	KIND_FIELD_NUMBER: _ClassVar[int]
	FIRST_FIELD_NUMBER: _ClassVar[int]
	SECOND_FIELD_NUMBER: _ClassVar[int]
	REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
	SESSION_FIELD_NUMBER: _ClassVar[int]
	ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
	kind: Update.Kind
	first: Tensor
	second: Tensor
	request_id: str
	session: str
	attributes: Tensor
	def __init__(
		self,
		kind: _Optional[_Union[Update.Kind, str]] = ...,
		first: _Optional[_Union[Tensor, _Mapping]] = ...,
		second: _Optional[_Union[Tensor, _Mapping]] = ...,
		request_id: _Optional[str] = ...,
		session: _Optional[str] = ...,
		attributes: _Optional[_Union[Tensor, _Mapping]] = ...,
	) -> None: ...

class Ticket(_message.Message):
	__slots__ = ("session", "number", "request_id")
	SESSION_FIELD_NUMBER: _ClassVar[int]
	NUMBER_FIELD_NUMBER: _ClassVar[int]
	REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
	session: str
	number: int
	request_id: str
	def __init__(
		self, session: _Optional[str] = ..., number: _Optional[int] = ..., request_id: _Optional[str] = ...
	) -> None: ...

class WaitRequest(_message.Message):
	__slots__ = ("ticket", "poll")
	TICKET_FIELD_NUMBER: _ClassVar[int]
	POLL_FIELD_NUMBER: _ClassVar[int]
	ticket: Ticket
	poll: bool
	def __init__(self, ticket: _Optional[_Union[Ticket, _Mapping]] = ..., poll: _Optional[bool] = ...) -> None: ...

class Completion(_message.Message):
	__slots__ = (
		"ticket",
		"ready",
		"version",
		"affected",
		"input_changes",
		"model_tasks",
		"input_version",
		"admitted_at_ns",
		"execution_started_at_ns",
		"staged_at_ns",
		"publication_observed_at_ns",
		"completion_at_ns",
		"server_reply_prepared_at_ns",
		"queue_wait_ns",
		"execution_ns",
		"delayed_publication_wait_ns",
		"timestamp_scope",
		"publication_source",
		"state",
	)
	TICKET_FIELD_NUMBER: _ClassVar[int]
	READY_FIELD_NUMBER: _ClassVar[int]
	VERSION_FIELD_NUMBER: _ClassVar[int]
	AFFECTED_FIELD_NUMBER: _ClassVar[int]
	INPUT_CHANGES_FIELD_NUMBER: _ClassVar[int]
	MODEL_TASKS_FIELD_NUMBER: _ClassVar[int]
	INPUT_VERSION_FIELD_NUMBER: _ClassVar[int]
	ADMITTED_AT_NS_FIELD_NUMBER: _ClassVar[int]
	EXECUTION_STARTED_AT_NS_FIELD_NUMBER: _ClassVar[int]
	STAGED_AT_NS_FIELD_NUMBER: _ClassVar[int]
	PUBLICATION_OBSERVED_AT_NS_FIELD_NUMBER: _ClassVar[int]
	COMPLETION_AT_NS_FIELD_NUMBER: _ClassVar[int]
	SERVER_REPLY_PREPARED_AT_NS_FIELD_NUMBER: _ClassVar[int]
	QUEUE_WAIT_NS_FIELD_NUMBER: _ClassVar[int]
	EXECUTION_NS_FIELD_NUMBER: _ClassVar[int]
	DELAYED_PUBLICATION_WAIT_NS_FIELD_NUMBER: _ClassVar[int]
	TIMESTAMP_SCOPE_FIELD_NUMBER: _ClassVar[int]
	PUBLICATION_SOURCE_FIELD_NUMBER: _ClassVar[int]
	STATE_FIELD_NUMBER: _ClassVar[int]
	ticket: Ticket
	ready: bool
	version: int
	affected: _containers.RepeatedScalarFieldContainer[int]
	input_changes: int
	model_tasks: int
	input_version: int
	admitted_at_ns: int
	execution_started_at_ns: int
	staged_at_ns: int
	publication_observed_at_ns: int
	completion_at_ns: int
	server_reply_prepared_at_ns: int
	queue_wait_ns: int
	execution_ns: int
	delayed_publication_wait_ns: int
	timestamp_scope: str
	publication_source: str
	state: str
	def __init__(
		self,
		ticket: _Optional[_Union[Ticket, _Mapping]] = ...,
		ready: _Optional[bool] = ...,
		version: _Optional[int] = ...,
		affected: _Optional[_Iterable[int]] = ...,
		input_changes: _Optional[int] = ...,
		model_tasks: _Optional[int] = ...,
		input_version: _Optional[int] = ...,
		admitted_at_ns: _Optional[int] = ...,
		execution_started_at_ns: _Optional[int] = ...,
		staged_at_ns: _Optional[int] = ...,
		publication_observed_at_ns: _Optional[int] = ...,
		completion_at_ns: _Optional[int] = ...,
		server_reply_prepared_at_ns: _Optional[int] = ...,
		queue_wait_ns: _Optional[int] = ...,
		execution_ns: _Optional[int] = ...,
		delayed_publication_wait_ns: _Optional[int] = ...,
		timestamp_scope: _Optional[str] = ...,
		publication_source: _Optional[str] = ...,
		state: _Optional[str] = ...,
	) -> None: ...

class QueryRequest(_message.Message):
	__slots__ = ("nodes", "layer", "max_staleness")
	NODES_FIELD_NUMBER: _ClassVar[int]
	LAYER_FIELD_NUMBER: _ClassVar[int]
	MAX_STALENESS_FIELD_NUMBER: _ClassVar[int]
	nodes: Tensor
	layer: int
	max_staleness: int
	def __init__(
		self,
		nodes: _Optional[_Union[Tensor, _Mapping]] = ...,
		layer: _Optional[int] = ...,
		max_staleness: _Optional[int] = ...,
	) -> None: ...

class QueryResponse(_message.Message):
	__slots__ = ("values", "version", "min_fresh_version", "max_age")
	VALUES_FIELD_NUMBER: _ClassVar[int]
	VERSION_FIELD_NUMBER: _ClassVar[int]
	MIN_FRESH_VERSION_FIELD_NUMBER: _ClassVar[int]
	MAX_AGE_FIELD_NUMBER: _ClassVar[int]
	values: Tensor
	version: int
	min_fresh_version: int
	max_age: int
	def __init__(
		self,
		values: _Optional[_Union[Tensor, _Mapping]] = ...,
		version: _Optional[int] = ...,
		min_fresh_version: _Optional[int] = ...,
		max_age: _Optional[int] = ...,
	) -> None: ...

class RuntimeInfo(_message.Message):
	__slots__ = (
		"session",
		"nodes",
		"widths",
		"graph_version",
		"completed_updates",
		"completed_input_changes",
		"pending_updates",
		"failed_updates",
		"recovery_rows",
		"recovery_seconds",
		"uptime_seconds",
		"incomplete_version",
		"max_staleness",
		"admitted_updates",
		"pending_update_bytes",
	)
	SESSION_FIELD_NUMBER: _ClassVar[int]
	NODES_FIELD_NUMBER: _ClassVar[int]
	WIDTHS_FIELD_NUMBER: _ClassVar[int]
	GRAPH_VERSION_FIELD_NUMBER: _ClassVar[int]
	COMPLETED_UPDATES_FIELD_NUMBER: _ClassVar[int]
	COMPLETED_INPUT_CHANGES_FIELD_NUMBER: _ClassVar[int]
	PENDING_UPDATES_FIELD_NUMBER: _ClassVar[int]
	FAILED_UPDATES_FIELD_NUMBER: _ClassVar[int]
	RECOVERY_ROWS_FIELD_NUMBER: _ClassVar[int]
	RECOVERY_SECONDS_FIELD_NUMBER: _ClassVar[int]
	UPTIME_SECONDS_FIELD_NUMBER: _ClassVar[int]
	INCOMPLETE_VERSION_FIELD_NUMBER: _ClassVar[int]
	MAX_STALENESS_FIELD_NUMBER: _ClassVar[int]
	ADMITTED_UPDATES_FIELD_NUMBER: _ClassVar[int]
	PENDING_UPDATE_BYTES_FIELD_NUMBER: _ClassVar[int]
	session: str
	nodes: int
	widths: _containers.RepeatedScalarFieldContainer[int]
	graph_version: int
	completed_updates: int
	completed_input_changes: int
	pending_updates: int
	failed_updates: int
	recovery_rows: int
	recovery_seconds: float
	uptime_seconds: float
	incomplete_version: int
	max_staleness: int
	admitted_updates: int
	pending_update_bytes: int
	def __init__(
		self,
		session: _Optional[str] = ...,
		nodes: _Optional[int] = ...,
		widths: _Optional[_Iterable[int]] = ...,
		graph_version: _Optional[int] = ...,
		completed_updates: _Optional[int] = ...,
		completed_input_changes: _Optional[int] = ...,
		pending_updates: _Optional[int] = ...,
		failed_updates: _Optional[int] = ...,
		recovery_rows: _Optional[int] = ...,
		recovery_seconds: _Optional[float] = ...,
		uptime_seconds: _Optional[float] = ...,
		incomplete_version: _Optional[int] = ...,
		max_staleness: _Optional[int] = ...,
		admitted_updates: _Optional[int] = ...,
		pending_update_bytes: _Optional[int] = ...,
	) -> None: ...
