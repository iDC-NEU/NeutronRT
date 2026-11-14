"""Private local storage process entrypoint; never import the serving application."""

import sys
from multiprocessing.connection import Connection

from lib.storage.process import _actor


def main() -> None:
	connection = Connection(int(sys.argv[1]))
	_actor(connection, *connection.recv())


if __name__ == "__main__":
	main()
