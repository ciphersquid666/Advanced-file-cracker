import pyzipper
import multiprocessing
import asyncio
import logging
import os
import sys
import json
import argparse
import itertools
import subprocess
import re
import shutil
import psutil
import signal
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import List, Optional, Generator, Dict, Any
try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import padding
    from argon2 import PasswordHasher
    CRYPTO_AVAILABLE = True
except Exception:
    os.environ["CRYPTOGRAPHY_OPENSSL_NO_LEGACY"] = "1"
    try:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import padding
        from argon2 import PasswordHasher
        CRYPTO_AVAILABLE = True
    except Exception:
        CRYPTO_AVAILABLE = False
import rarfile
try:
    import aiologger
    from aiologger.handlers.files import AsyncFileHandler
    from aiologger.handlers.streams import StreamHandler
    AIOLOGGER_AVAILABLE = True
except Exception:
    AIOLOGGER_AVAILABLE = False
import string
import random
import numpy as np
from gym import Env
from gym.spaces import Discrete, Box
from tenacity import retry, stop_after_attempt, wait_exponential

if AIOLOGGER_AVAILABLE:
    logger = aiologger.Logger.with_default_handlers(
        level=logging.INFO,
        handlers=[
            AsyncFileHandler("archive_cracker.log"),
            StreamHandler(sys.stdout)
        ],
        formatter=logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
else:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("archive_cracker.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger()

LEGAL_DISCLAIMER = """
WARNING: This script is for legal and ethical use only, such as recovering passwords for archives you own. Unauthorized use is illegal and prosecutable.
"""

class Plugin:
    def generate_passwords(self, config: Any) -> Generator[str, None, None]:
        yield from []

class SimplePasswordPlugin(Plugin):
    def generate_passwords(self, config: Any) -> Generator[str, None, None]:
        for _ in range(1000):
            length = random.randint(config.min_len, config.max_len)
            yield "".join(random.choices(config.charset, k=length))
        rules = [
            lambda x: x + str(random.randint(0, 99)),
            lambda x: x.capitalize(),
            lambda x: x + random.choice(["!", "@", "#", "$"]),
            lambda x: "".join(c.upper() if random.random() < 0.3 else c for c in x)
        ]
        for base in config.common_passwords:
            yield base
            for rule in rules:
                yield rule(base)

class WordlistDownloader:
    def __init__(self):
        self.cache_dir = Path("wordlists")
        self.cache_dir.mkdir(exist_ok=True)

    def search_online_wordlists(self, query: str = "password wordlist filetype:txt") -> List[Dict[str, Any]]:
        logger.warning("Online wordlist download not supported in this environment")
        return []

    def download_wordlist(self, url: str, name: str) -> Optional[Path]:
        logger.warning("Online wordlist download not supported in this environment")
        return None

class ResourceEnv(Env):
    def __init__(self, max_workers: int):
        self.action_space = Discrete(3)
        self.observation_space = Box(low=0, high=100, shape=(3,), dtype=np.float32)
        self.max_workers = max_workers
        self.current_workers = max_workers // 2
        self.cpu_history = []
        self.memory_history = []

    def step(self, action):
        cpu = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory().percent
        self.cpu_history.append(cpu)
        self.memory_history.append(memory)
        if action == 0:
            self.current_workers = max(1, self.current_workers - 1)
        elif action == 1:
            self.current_workers = min(self.max_workers, self.current_workers + 1)
        reward = -abs(cpu - 70) - abs(memory - 80)
        done = len(self.cpu_history) > 100
        return np.array([cpu, memory, self.current_workers]), reward, done, {}

    def reset(self):
        self.cpu_history = []
        self.memory_history = []
        self.current_workers = self.max_workers // 2
        return np.array([psutil.cpu_percent(interval=0.1), psutil.virtual_memory().percent, self.current_workers])

class ZipCrackerConfig:
    def __init__(self, config_file: str = "config.json", encryption_key: str = "secret"):
        self.chunk_size = 50
        self.max_retries = 3
        self.retry_delay = 1
        self.timeout = 30
        self.charset = string.ascii_letters + string.digits + "!@#$%^&*()"
        self.min_len = 4
        self.max_len = 8
        self.regex_pattern = None
        self.common_passwords = ["password", "123456", "qwerty", "admin123", "letmein"]
        self.use_gpu = False
        self.use_distributed = False
        self.max_cpu_percent = 80
        self.plugins = ["SimplePasswordPlugin"]
        self.encryption_key = encryption_key
        self.wordlist_sources = []
        self.aws_region = "us-west-2"
        self.load_config(config_file)

    def load_config(self, config_file: str):
        if not Path(config_file).is_file():
            return
        try:
            if CRYPTO_AVAILABLE:
                with open(config_file, "rb") as f:
                    encrypted_data = f.read()
                config = decrypt_data(encrypted_data, self.encryption_key)
            else:
                with open(config_file, "r") as f:
                    config = json.load(f)
            self.chunk_size = config.get("chunk_size", self.chunk_size)
            self.max_retries = config.get("max_retries", self.max_retries)
            self.retry_delay = config.get("retry_delay", self.retry_delay)
            self.timeout = config.get("timeout", self.timeout)
            self.charset = config.get("charset", self.charset)
            self.min_len = config.get("min_len", self.min_len)
            self.max_len = config.get("max_len", self.max_len)
            self.regex_pattern = config.get("regex_pattern", self.regex_pattern)
            self.common_passwords = config.get("common_passwords", self.common_passwords)
            self.use_gpu = config.get("use_gpu", self.use_gpu)
            self.use_distributed = config.get("use_distributed", self.use_distributed)
            self.max_cpu_percent = config.get("max_cpu_percent", self.max_cpu_percent)
            self.plugins = config.get("plugins", self.plugins)
            self.wordlist_sources = config.get("wordlist_sources", self.wordlist_sources)
            self.aws_region = config.get("aws_region", self.aws_region)
        except Exception:
            logger.error(f"Failed to load configuration")

def derive_key(encryption_key: str) -> bytes:
    if not CRYPTO_AVAILABLE:
        raise Exception("Cryptography not available")
    ph = PasswordHasher()
    hash = ph.hash(encryption_key)
    return hashlib.sha256(hash.encode()).digest()

def encrypt_data(data: dict, encryption_key: str) -> bytes:
    if not CRYPTO_AVAILABLE:
        raise Exception("Cryptography not available")
    iv = os.urandom(16)
    key = derive_key(encryption_key)
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(json.dumps(data).encode()) + padder.finalize()
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
    return iv + encrypted_data

def decrypt_data(encrypted_data: bytes, encryption_key: str) -> dict:
    if not CRYPTO_AVAILABLE:
        raise Exception("Cryptography not available")
    iv = encrypted_data[:16]
    encrypted_data = encrypted_data[16:]
    key = derive_key(encryption_key)
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    decrypted_data = decryptor.update(encrypted_data) + decryptor.finalize()
    unpadder = padding.PKCS7(128).unpadder()
    return json.loads(unpadder.update(decrypted_data) + unpadder.finalize())

def is_hashcat_available():
    return shutil.which("hashcat") is not None

async def read_wordlist_async(wordlist_file: str, start_after: Optional[str] = None) -> Generator[str, None, None]:
    loop = asyncio.get_event_loop()
    try:
        with open(wordlist_file, "r", encoding="utf-8", errors="ignore") as f:
            skip = bool(start_after)
            for line in await loop.run_in_executor(None, lambda: list(f)):
                password = line.strip()
                if not password:
                    continue
                if skip and password == start_after:
                    skip = False
                    continue
                if not skip:
                    yield password
    except Exception:
        logger.error(f"Failed to read wordlist")

async def try_password_zip(zip_file: str, password: str, config: ZipCrackerConfig) -> Optional[str]:
    for attempt in range(config.max_retries):
        try:
            with pyzipper.AESZipFile(zip_file, "r", compression=pyzipper.ZIP_DEFLATED, encryption=pyzipper.WZ_AES) as zf:
                zf.extractall(pwd=password.encode())
            return password
        except (pyzipper.BadZipFile, RuntimeError, pyzipper.LargeZipFile, ValueError):
            try:
                with pyzipper.ZipFile(zip_file, "r", compression=pyzipper.ZIP_DEFLATED) as zf:
                    zf.extractall(pwd=password.encode())
                return password
            except (pyzipper.BadZipFile, ValueError):
                return None
        except Exception:
            if attempt < config.max_retries - 1:
                await asyncio.sleep(config.retry_delay * (2 ** attempt))
    return None

async def try_password_rar(rar_file: str, password: str, config: ZipCrackerConfig) -> Optional[str]:
    for attempt in range(config.max_retries):
        try:
            with rarfile.RarFile(rar_file, "r") as rf:
                rf.extractall(pwd=password.encode())
            return password
        except (rarfile.BadRarFile, rarfile.PasswordRequired, rarfile.NoValidEntries, ValueError):
            if attempt < config.max_retries - 1:
                await asyncio.sleep(config.retry_delay * (2 ** attempt))
        except Exception:
            return None
    return None

def try_password_gpu(zip_file: str, passwords: List[str], config: ZipCrackerConfig) -> Optional[str]:
    if not is_hashcat_available():
        return None
    temp_file = f"temp_passwords_{hashlib.md5(zip_file.encode()).hexdigest()}.txt"
    try:
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write("\n".join(passwords))
        result = subprocess.run(
            ["hashcat", "-m", "17210", "-a", "0", zip_file, temp_file, "--quiet"],
            capture_output=True,
            text=True,
            timeout=config.timeout,
        )
        if result.returncode == 0:
            with open("hashcat.potfile", "r") as f:
                for line in f:
                    if ":" in line:
                        return line.split(":")[-1].strip()
    except Exception:
        pass
    finally:
        if Path(temp_file).exists():
            os.remove(temp_file)
    return None

def check_archive_file(archive_file: str) -> Optional[str]:
    try:
        with pyzipper.ZipFile(archive_file, "r") as zf:
            zf.testzip()
        return "zip"
    except (pyzipper.BadZipFile, ValueError):
        try:
            with rarfile.RarFile(archive_file, "r") as rf:
                rf.testrar()
            return "rar"
        except Exception:
            return None

def generate_passwords(config: ZipCrackerConfig) -> Generator[str, None, None]:
    regex = re.compile(config.regex_pattern) if config.regex_pattern else None
    yield from config.common_passwords
    for length in range(config.min_len, config.max_len + 1):
        for pwd in itertools.product(config.charset, repeat=length):
            password = "".join(pwd)
            if regex and not regex.match(password):
                continue
            yield password

def adjust_workers(config: ZipCrackerConfig, max_workers: int, env: ResourceEnv) -> int:
    try:
        cpu = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory().percent
        action = 1 if cpu < config.max_cpu_percent and memory < 85 else 0
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()
        return max(1, min(env.current_workers, multiprocessing.cpu_count()))
    except Exception:
        return max(1, min(max_workers, multiprocessing.cpu_count()))

def save_state(state_file: str, state: Dict[str, Any], encryption_key: str):
    try:
        if CRYPTO_AVAILABLE:
            encrypted_state = encrypt_data(state, encryption_key)
            with open(state_file, "wb") as f:
                f.write(encrypted_state)
        else:
            with open(state_file, "w") as f:
                json.dump(state, f)
    except Exception:
        logger.error(f"Failed to save state")

def load_state(state_file: str, encryption_key: str) -> Optional[Dict[str, Any]]:
    if not Path(state_file).is_file():
        return None
    try:
        if CRYPTO_AVAILABLE:
            with open(state_file, "rb") as f:
                encrypted_state = f.read()
            return decrypt_data(encrypted_state, encryption_key)
        else:
            with open(state_file, "r") as f:
                return json.load(f)
    except Exception:
        logger.error(f"Failed to load state")
        return None

def generate_report(result: Optional[str], stats: Dict[str, Any], report_file: str = "report.json"):
    report = {
        "timestamp": datetime.now().isoformat(),
        "success": result is not None,
        "password": result,
        "stats": stats,
    }
    try:
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
    except Exception:
        logger.error(f"Failed to generate report")

async def interactive_mode():
    pause_event = asyncio.Event()
    pause_event.set()
    async def handle_input():
        while True:
            try:
                cmd = await asyncio.get_event_loop().run_in_executor(None, lambda: input("Enter command (pause/resume/exit): ").lower())
                if cmd == "pause":
                    pause_event.clear()
                    await logger.info("Paused")
                elif cmd == "resume":
                    pause_event.set()
                    await logger.info("Resumed")
                elif cmd == "exit":
                    await logger.info("Exiting")
                    os._exit(0)
            except Exception:
                await logger.info("Interrupt in interactive mode, exiting")
                os._exit(0)
    asyncio.create_task(handle_input())
    return pause_event

def cleanup_temp_files():
    for file in Path.cwd().glob("temp_passwords_*.txt"):
        try:
            file.unlink()
        except Exception:
            pass

async def process_chunk(archive_file: str, chunk: List[str], chunk_idx: int, config: ZipCrackerConfig, archive_type: str) -> Optional[str]:
    logger.info(f"Processing chunk {chunk_idx + 1} of {len(chunk)} passwords")
    if config.use_gpu:
        return try_password_gpu(archive_file, chunk, config)
    for password in chunk:
        result = await (
            try_password_zip(archive_file, password, config)
            if archive_type == "zip"
            else try_password_rar(archive_file, password, config)
        )
        if result:
            return result
    return None

async def crack_archive(
    archive_file: str,
    wordlist_file: Optional[str] = None,
    max_workers: Optional[int] = None,
    use_dynamic: bool = False,
    state_file: str = "state.json",
    config_file: str = "config.json",
    report_file: str = "report.json",
    encryption_key: str = "secret",
) -> Optional[str]:
    await logger.info(LEGAL_DISCLAIMER)
    start_time = datetime.now()
    stats = {"passwords_tested": 0, "chunks_processed": 0, "duration": 0, "wordlists_used": []}
    config = ZipCrackerConfig(config_file, encryption_key)
    archive_type = check_archive_file(archive_file)
    if not archive_type:
        return None
    downloader = WordlistDownloader()
    wordlist_files = [Path(wordlist_file)] if wordlist_file and Path(wordlist_file).is_file() else []
    if config.wordlist_sources:
        try:
            wordlists = downloader.search_online_wordlists()
            for wl in wordlists:
                if path := downloader.download_wordlist(wl["url"], wl["name"]):
                    wordlist_files.append(path)
                    stats["wordlists_used"].append(wl["name"])
        except Exception:
            logger.error(f"Failed to retrieve wordlists")
    max_workers = max_workers or min(1, multiprocessing.cpu_count())
    env = ResourceEnv(max_workers)
    max_workers = adjust_workers(config, max_workers, env)
    await logger.info(f"Using {max_workers} workers for {archive_type} archive")
    state = load_state(state_file, encryption_key) or {}
    last_password = state.get("last_password")
    if last_password:
        await logger.info(f"Resuming from last password: {last_password}")
    def password_generator():
        plugins = [globals()[p]() for p in config.plugins if p in globals()]
        for wl_file in wordlist_files:
            loop = asyncio.get_event_loop()
            gen = read_wordlist_async(str(wl_file), last_password)
            try:
                for pwd in loop.run_until_complete(asyncio.gather(*[gen.__anext__() for _ in range(500)])):
                    if pwd:
                        yield pwd
            except StopAsyncIteration:
                pass
        if use_dynamic or not wordlist_files:
            for plugin in plugins:
                yield from plugin.generate_passwords(config)
            yield from generate_passwords(config)
    chunk = []
    chunk_idx = 0
    total_passwords = 0
    pause_event = await interactive_mode()
    success_rates = []
    async def process_chunks(executor: Optional[ProcessPoolExecutor] = None):
        nonlocal chunk, chunk_idx, total_passwords
        futures = []
        with tqdm(desc="Password attempts", unit="pwd") as pbar:
            for password in password_generator():
                await pause_event.wait()
                chunk.append(password)
                total_passwords += 1
                if len(chunk) >= config.chunk_size:
                    task = executor.submit(process_chunk, archive_file, chunk, chunk_idx, config, archive_type)
                    futures.append(task)
                    chunk = []
                    chunk_idx += 1
                    stats["chunks_processed"] += 1
                    stats["passwords_tested"] += config.chunk_size
                    pbar.update(config.chunk_size)
                    save_state(state_file, {"last_password": password, "stats": stats}, encryption_key)
                    success_rates.append(0)
                    if len(success_rates) > 20 and np.mean(success_rates[-20:]) < 0.005:
                        config.chunk_size = min(config.chunk_size * 2, 5000)
                        await logger.info(f"Adjusted chunk size to {config.chunk_size}")
                    max_workers = adjust_workers(config, max_workers, env)
                if len(futures) >= max_workers:
                    for future in as_completed(futures):
                        result = await future.result()
                        if result:
                            await logger.info(f"Found password: {result}")
                            stats["duration"] = (datetime.now() - start_time).total_seconds()
                            save_state(state_file, {"last_password": result, "stats": stats}, encryption_key)
                            generate_report(result, stats, report_file)
                            return result
                        futures.remove(future)
            if chunk:
                task = executor.submit(process_chunk, archive_file, chunk, chunk_idx, config, archive_type)
                futures.append(task)
                stats["chunks_processed"] += 1
                stats["passwords_tested"] += len(chunk)
                pbar.update(len(chunk))
                save_state(state_file, {"last_password": chunk[-1] if chunk else last_password, "stats": stats}, encryption_key)
            for future in as_completed(futures):
                result = await future.result()
                if result:
                    await logger.info(f"Found password: {result}")
                    stats["duration"] = (datetime.now() - start_time).total_seconds()
                    save_state(state_file, {"last_password": result, "stats": stats}, encryption_key)
                    generate_report(result, stats, report_file)
                    return result
                futures.remove(future)
        return None
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            result = await process_chunks(executor)
    except Exception:
        logger.error(f"Cracking process failed")
        result = None
    finally:
        cleanup_temp_files()
    stats["duration"] = (datetime.now() - start_time).total_seconds()
    if not result:
        await logger.info("No password found")
    generate_report(result, stats, report_file)
    return result

def signal_handler(sig, frame):
    logger.info("Interrupt received, saving state and exiting")
    cleanup_temp_files()
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    parser = argparse.ArgumentParser(description="Advanced archive password recovery tool")
    parser.add_argument("archive_file", help="Path to encrypted ZIP or RAR file")
    parser.add_argument("-w", "--wordlist", help="Path to local wordlist file")
    parser.add_argument("-c", "--config", help="Path to JSON configuration file", default="config.json")
    parser.add_argument("-s", "--state", help="Path to state file", default="state.json")
    parser.add_argument("-r", "--report", help="Path to report file", default="report.json")
    parser.add_argument("-p", "--dynamic", action="store_true", help="Use dynamic password generation")
    parser.add_argument("-m", "--max-workers", type=int, help="Maximum number of workers")
    parser.add_argument("-k", "--encryption-key", help="Encryption key for config and state files", default="secret")
    args = parser.parse_args()
    asyncio.run(
        crack_archive(
            archive_file=args.archive_file,
            wordlist_file=args.wordlist,
            max_workers=args.max_workers,
            use_dynamic=args.dynamic,
            state_file=args.state,
            config_file=args.config,
            report_file=args.report,
            encryption_key=args.encryption_key,
        )
    )

if __name__ == "__main__":
    main()
