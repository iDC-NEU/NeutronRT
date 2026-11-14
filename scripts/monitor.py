"""
无需 root 权限，监控 PyTorch Gloo 后端进程的网络通信情况。

优化特性:
- 自动累计发送/接收总字节数（近似）
- 单位自适应（B、KB、MB、GB）
- 更美观的表格化输出
- 灵活指定进程名或 PID

用法:
    python -m scripts.monitor                     # 自动检测 pt_main_thread 进程
    python -m scripts.monitor --name my_trainer   # 按进程名匹配
    python -m scripts.monitor --pids 1234 --pids 5678    # 指定 PID
    python -m scripts.monitor --no-details        # 只显示摘要，不显示连接详情
"""

#!/usr/bin/env python3
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Annotated

import typer

try:
	import psutil
except ImportError:
	print("请先安装 psutil: pip install psutil")
	sys.exit(1)


# ──────────────────────────── 工具函数 ────────────────────────────
def parse_proc_net_tcp(proc_path="/proc/net/tcp"):
	"""解析 /proc/net/tcp 或 tcp6，返回 {inode: (recv_queue, send_queue)}"""
	result = {}
	try:
		with open(proc_path, "r") as f:
			lines = f.readlines()[1:]  # 跳过表头
			for line in lines:
				parts = line.strip().split()
				if len(parts) < 17:
					continue
				queue = parts[4].split(":")
				if len(queue) != 2:
					continue
				rx_queue = int(queue[0], 16)
				tx_queue = int(queue[1], 16)
				inode = int(parts[9])  # 第10列
				result[inode] = (rx_queue, tx_queue)
	except Exception as e:
		print(f"读取 {proc_path} 失败: {e}")
	return result


def get_all_socket_queues():
	"""合并 tcp 和 tcp6 的队列信息"""
	queues = {}
	queues.update(parse_proc_net_tcp("/proc/net/tcp"))
	if os.path.exists("/proc/net/tcp6"):
		queues.update(parse_proc_net_tcp("/proc/net/tcp6"))
	return queues


def get_process_socket_inodes(pid):
	"""
	返回进程的所有 socket inode 及其连接信息。
	返回值: [(inode, psutil.connection), ...]
	"""
	inodes = []
	try:
		proc = psutil.Process(pid)
		for conn in proc.connections(kind="inet"):
			if conn.fd is not None:
				fd_path = f"/proc/{pid}/fdinfo/{conn.fd}"
				if os.path.exists(fd_path):
					with open(fd_path, "r") as f:
						for line in f:
							if line.startswith("ino:"):
								inode = int(line.split()[1])
								inodes.append((inode, conn))
								break
	except (psutil.NoSuchProcess, psutil.AccessDenied):
		pass
	return inodes


def get_all_target_pids(target_names=("pt_main_thread",)):
	"""根据进程名获取 PID 列表（不区分大小写，匹配包含）"""
	pids = []
	for proc in psutil.process_iter(["pid", "name"]):
		try:
			name = proc.info["name"]
			if name and any(target in name.lower() for target in target_names):
				pids.append(proc.info["pid"])
		except (psutil.NoSuchProcess, psutil.AccessDenied):
			continue
	return pids


def format_size(size_bytes: int) -> str:
	"""将字节数转换为带单位的可读字符串（B, KB, MB, GB）"""
	if size_bytes < 1024:
		return f"{size_bytes:.2f} B"
	elif size_bytes < 1024**2:
		return f"{size_bytes / 1024:.2f} KB"
	elif size_bytes < 1024**3:
		return f"{size_bytes / 1024**2:.2f} MB"
	else:
		return f"{size_bytes / 1024**3:.2f} GB"


def format_rate(rate_bytes_per_sec: float) -> str:
	"""将速率转换为带单位的可读字符串（B/s, KB/s, MB/s, GB/s）"""
	if rate_bytes_per_sec < 1024:
		return f"{rate_bytes_per_sec:.2f} B/s"
	elif rate_bytes_per_sec < 1024**2:
		return f"{rate_bytes_per_sec / 1024:.2f} KB/s"
	elif rate_bytes_per_sec < 1024**3:
		return f"{rate_bytes_per_sec / 1024**2:.2f} MB/s"
	else:
		return f"{rate_bytes_per_sec / 1024**3:.2f} GB/s"


# ──────────────────────────── 监控类 ────────────────────────────
class NetworkMonitor:
	def __init__(self, pids, interval=1.0, show_details=True):
		self.pids = pids
		self.interval = interval
		self.show_details = show_details
		self.running = True

		# 累计总字节数: {pid: {'rx': int, 'tx': int}}
		self.total = defaultdict(lambda: {"rx": 0, "tx": 0})

		# 每个 inode 上一次的队列长度: {inode: {'rx': int, 'tx': int, 'pid': int}}
		self.inode_last = {}

		# 第一次采样，建立基准（不计入累计）
		init_data, init_ts = self.sample()
		for pid, conns in init_data.items():
			for inode, info in conns.items():
				self.inode_last[inode] = {"rx": info["rx"], "tx": info["tx"], "pid": pid}
		self.prev_data = init_data
		self.prev_ts = init_ts

	def sample(self):
		"""采样一次，返回 {pid: {inode: {'rx', 'tx', 'raddr', 'laddr', 'status'}}} 和当前时间"""
		global_queues = get_all_socket_queues()
		pid_connections = {}
		for pid in self.pids:
			sock_infos = get_process_socket_inodes(pid)
			pid_data = {}
			for inode, conn in sock_infos:
				if inode in global_queues:
					rx, tx = global_queues[inode]
					pid_data[inode] = {
						"rx": rx,
						"tx": tx,
						"raddr": f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else "0.0.0.0:0",
						"laddr": f"{conn.laddr.ip}:{conn.laddr.port}",
						"status": conn.status,
					}
			pid_connections[pid] = pid_data
		return pid_connections, time.time()

	def update(self, curr_data, curr_ts):
		"""
		对比上一次采样，更新速率和累计总量。
		返回 {pid: (rx_rate, tx_rate)}
		"""
		dt = curr_ts - self.prev_ts
		if dt <= 0:
			return {}

		rates = {}
		for pid, conns in curr_data.items():
			rx_inc = 0  # 本次采样期间接收队列的正增长总量
			tx_inc = 0  # 本次采样期间发送队列的正增长总量

			for inode, info in conns.items():
				last = self.inode_last.get(inode)
				if last is None:
					# 新连接，记录基准，不产生增量
					self.inode_last[inode] = {"rx": info["rx"], "tx": info["tx"], "pid": pid}
					continue

				drx = info["rx"] - last["rx"]
				dtx = info["tx"] - last["tx"]

				# 仅正增量累加到“已发送/已接收”数据量
				if drx > 0:
					rx_inc += drx
					self.total[pid]["rx"] += drx
				if dtx > 0:
					tx_inc += dtx
					self.total[pid]["tx"] += dtx

				# 更新 inode 队列长度记录
				last["rx"] = info["rx"]
				last["tx"] = info["tx"]

			# 计算速率
			rx_rate = rx_inc / dt
			tx_rate = tx_inc / dt
			rates[pid] = (rx_rate, tx_rate)

		self.prev_data = curr_data
		self.prev_ts = curr_ts
		return rates

	def display(self, curr_data, rates):
		"""刷新终端显示"""
		# 清屏
		sys.stdout.write("\033c")
		sys.stdout.flush()

		# 标题
		print(f"PyTorch Gloo 网络监控 | 采样间隔: {self.interval}s | 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
		# 表头
		header = f"{'PID':<8} {'累计接收':<18} {'累计发送':<18} {'接收速率':<18} {'发送速率':<18} {'连接数':<8}"
		print(header)
		print("-" * len(header))

		for pid in self.pids:
			tot = self.total.get(pid, {"rx": 0, "tx": 0})
			total_rx = format_size(tot["rx"])
			total_tx = format_size(tot["tx"])

			rx_rate, tx_rate = rates.get(pid, (0.0, 0.0))
			rx_rate_str = format_rate(rx_rate)
			tx_rate_str = format_rate(tx_rate)

			conn_count = len(curr_data.get(pid, {}))
			print(f"{pid:<8} {total_rx:<18} {total_tx:<18} {rx_rate_str:<18} {tx_rate_str:<18} {conn_count:<8}")

		# 连接详情
		if self.show_details:
			print("\n当前活动连接详情 (每个PID最多显示10条):")
			for pid in self.pids:
				conns = curr_data.get(pid, {})
				if not conns:
					continue
				print(f"\n PID {pid}:")
				for i, (_inode, info) in enumerate(conns.items()):
					if i >= 10:
						print("  ... (更多连接未显示)")
						break
					print(
						f"  {info['laddr']} -> {info['raddr']}  "
						f"[{info['status']}]  rx_q={info['rx']}  tx_q={info['tx']}"
					)

	def run(self):
		print(f"监控 PID: {self.pids}")
		print("按下 Ctrl+C 停止监控\n")
		time.sleep(self.interval)  # 等待第一个间隔
		while self.running:
			curr_data, curr_ts = self.sample()
			rates = self.update(curr_data, curr_ts)
			self.display(curr_data, rates)
			time.sleep(self.interval)


# ──────────────────────────── 入口 ────────────────────────────


@dataclass
class Options:
	pids: list[int] | None
	name: list[str]
	interval: float
	no_details: bool


app = typer.Typer(help=__doc__, no_args_is_help=True, pretty_exceptions_show_locals=False)


@app.command()
def main(
	pids: Annotated[list[int] | None, typer.Option("--pids", help="指定要监控的 PID")] = None,
	name: Annotated[
		list[str] | None, typer.Option("--name", help="匹配的进程名（不区分大小写，默认 pt_main_thread）")
	] = None,
	interval: Annotated[float, typer.Option("--interval", help="采样间隔（秒）")] = 1.0,
	no_details: Annotated[bool, typer.Option("--no-details", help="不显示每个连接的详情")] = False,
):
	args = Options(**{**locals(), "name": name if name is not None else ["pt_main_thread"]})

	# 确定 PID 列表
	if args.pids:
		pids = args.pids
	else:
		# 将输入的进程名统一转小写
		target_names = tuple(name.lower() for name in args.name)
		pids = get_all_target_pids(target_names)
		if not pids:
			print(f"未找到匹配 '{target_names}' 的进程，请手动指定 --pids")
			sys.exit(1)

	monitor = NetworkMonitor(pids, args.interval, show_details=not args.no_details)
	try:
		monitor.run()
	except KeyboardInterrupt:
		print("\n停止监控")


if __name__ == "__main__":
	app()
