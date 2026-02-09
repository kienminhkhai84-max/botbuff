"""
TikTok View Bot - Educational & Research Purpose Only
=======================================================
DISCLAIMER: This tool is created solely for educational and research purposes.
The author does not encourage or support any misuse of this tool.
Use responsibly and in compliance with TikTok's Terms of Service.
"""

import os
# Tắt aiohappyeyeballs để tránh lỗi "Task was destroyed but it is pending"
# Phải set trước khi import aiohttp
os.environ['AIOHTTP_NO_EXTENSIONS'] = '1'

import sys
# Kiểm tra Python version - hỗ trợ Python 3.8 đến 3.14+
if sys.version_info < (3, 8):
    print("❌ Python 3.8+ is required!")
    print(f"   Bạn đang dùng Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    sys.exit(1)

# Hiển thị Python version khi khởi động (tùy chọn)
PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
if sys.version_info >= (3, 12):
    # Python 3.12+ được hỗ trợ đầy đủ
    pass
elif sys.version_info >= (3, 8):
    # Python 3.8-3.11 được hỗ trợ
    pass

import aiohttp
import asyncio
import random
import requests
import re
import time
import secrets
import signal
from hashlib import md5
from time import time as T
from typing import Dict, Tuple, Optional, Union
from dataclasses import dataclass
from collections import deque
from urllib.parse import urlencode
import logging
from enum import Enum
import socket  # Để xử lý gaierror
import json  # Để parse response JSON
import nest_asyncio
nest_asyncio.apply()

# Try to import psutil, fallback if not available
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Configure logging with UTF-8 encoding for Windows compatibility
def setup_logging():
    """Setup logging with UTF-8 encoding"""
    # Create file handler with UTF-8 encoding
    file_handler = logging.FileHandler('tiktok_bot.log', encoding='utf-8')
    file_handler.setLevel(logging.WARNING)

    # Create stream handler with error handling for emoji
    class SafeStreamHandler(logging.StreamHandler):
        def emit(self, record):
            try:
                super().emit(record)
            except UnicodeEncodeError:
                # Replace emoji with text if encoding fails
                record.msg = str(record.msg).encode('ascii', 'ignore').decode('ascii')
                super().emit(record)

    stream_handler = SafeStreamHandler()
    stream_handler.setLevel(logging.WARNING)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    # Tắt logging của aiohttp cho gaierror và DNS errors
    aiohttp_logger = logging.getLogger('aiohttp')
    aiohttp_logger.setLevel(logging.ERROR)  # Chỉ log ERROR, không log WARNING

    # Filter để bỏ qua gaierror
    class GaierrorFilter(logging.Filter):
        def filter(self, record):
            # Bỏ qua các log có chứa gaierror hoặc getaddrinfo
            msg = str(record.getMessage())
            if 'gaierror' in msg.lower() or 'getaddrinfo' in msg.lower():
                return False
            return True

    # Áp dụng filter cho tất cả handlers
    gaierror_filter = GaierrorFilter()
    file_handler.addFilter(gaierror_filter)
    stream_handler.addFilter(gaierror_filter)
    aiohttp_logger.addFilter(gaierror_filter)

    return logging.getLogger(__name__)

logger = setup_logging()

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class DeviceInfo:
    model: str
    version: str
    api_level: int

class DeviceGenerator:
    """Generate unlimited random device information to avoid detection"""

    # Base device templates for realistic generation
    BRANDS = [
        "Google", "Samsung", "Xiaomi", "Oppo", "OnePlus", "Realme", "Vivo",
        "Honor", "Huawei", "Motorola", "Nokia", "Sony", "Asus", "Tecno",
        "Infinix", "TCL", "Nothing", "Redmi", "Poco", "Meizu", "Lenovo"
    ]

    BRAND_MODELS = {
        "Google": ["Pixel", "Pixel Pro", "Pixel XL", "Pixel a"],
        "Samsung": ["Galaxy S", "Galaxy Note", "Galaxy A", "Galaxy Z", "Galaxy M"],
        "Xiaomi": ["Mi", "Redmi Note", "Redmi", "Poco", "Mi Mix"],
        "Oppo": ["Reno", "Find X", "A", "F", "K"],
        "OnePlus": ["OnePlus", "Nord", "Nord CE"],
        "Realme": ["GT", "Realme", "Narzo", "C"],
        "Vivo": ["X", "V", "Y", "S", "T"],
        "Honor": ["Honor", "Magic", "X"],
        "Huawei": ["P", "Mate", "Nova", "Y"],
        "Motorola": ["Edge", "Moto G", "Moto E", "Razr"],
        "Nokia": ["Nokia", "X", "G"],
        "Sony": ["Xperia"],
        "Asus": ["Zenfone", "ROG Phone"],
        "Tecno": ["Camon", "Spark", "Pova"],
        "Infinix": ["Note", "Hot", "Zero"],
        "TCL": ["TCL", "20"],
        "Nothing": ["Phone"],
        "Redmi": ["Note", "Redmi"],
        "Poco": ["Poco X", "Poco F", "Poco M"],
        "Meizu": ["Meizu", "Note"],
        "Lenovo": ["Legion", "K", "A"]
    }

    ANDROID_VERSIONS = {
        "10": 29,
        "11": 30,
        "12": 31,
        "13": 33,
        "14": 34
    }

    USER_AGENTS = [
        "com.ss.android.ugc.trill/400304",
        "com.ss.android.ugc.trill/400305",
        "com.ss.android.ugc.trill/400306",
        "com.ss.android.ugc.trill/400307",
        "com.zhiliaoapp.musically/400304",
        "com.zhiliaoapp.musically/400305",
        "com.ss.android.ugc.aweme/400304",
        "com.ss.android.ugc.aweme/400305",
        "com.ss.android.ugc.aweme/400306",
        "com.ss.android.ugc.aweme.lite/400304",
    ]

    # Cache for generated devices to avoid duplicates
    _generated_devices = set()
    _max_cache_size = 10000

    # Device pool để pre-generate 30,000 devices
    _device_pool = deque(maxlen=50000)  # Pool lớn hơn để đảm bảo đủ devices

    @classmethod
    def _generate_device_name(cls) -> str:
        """Generate a random device name"""
        brand = random.choice(cls.BRANDS)

        # Get model prefix for this brand
        if brand in cls.BRAND_MODELS:
            model_prefix = random.choice(cls.BRAND_MODELS[brand])
        else:
            model_prefix = brand

        # Generate model number
        if random.random() < 0.3:  # 30% chance of having a variant
            variants = ["Pro", "Ultra", "Max", "Plus", "SE", "Lite", "FE"]
            variant = random.choice(variants)
            model_number = random.randint(5, 15)
            device_name = f"{brand} {model_prefix} {model_number} {variant}"
        else:
            model_number = random.randint(5, 15)
            device_name = f"{brand} {model_prefix} {model_number}"

        return device_name

    @classmethod
    def _generate_android_version(cls) -> Tuple[str, int]:
        """Generate random Android version and API level"""
        version = random.choice(list(cls.ANDROID_VERSIONS.keys()))
        api_level = cls.ANDROID_VERSIONS[version]
        return version, api_level

    @classmethod
    def generate_device(cls) -> DeviceInfo:
        """Generate a unique random device (unlimited)"""
        max_attempts = 100

        for _ in range(max_attempts):
            device_name = cls._generate_device_name()
            version, api_level = cls._generate_android_version()

            # Create device signature
            device_signature = f"{device_name}|{version}|{api_level}"

            # Check if we've generated this device before
            if device_signature not in cls._generated_devices:
                cls._generated_devices.add(device_signature)

                # Clean cache if too large
                if len(cls._generated_devices) > cls._max_cache_size:
                    # Remove oldest 20% of cache
                    items_to_remove = list(cls._generated_devices)[:cls._max_cache_size // 5]
                    for item in items_to_remove:
                        cls._generated_devices.discard(item)

                return DeviceInfo(device_name, version, api_level)

        # If all attempts failed (very unlikely), return a random one
        device_name = cls._generate_device_name()
        version, api_level = cls._generate_android_version()
        return DeviceInfo(device_name, version, api_level)

    @classmethod
    def random_device(cls) -> DeviceInfo:
        """Get a random device (uses generator for unlimited devices)"""
        # 70% chance to generate new device, 30% to use existing pattern
        if random.random() < 0.7:
            return cls.generate_device()
        else:
            # Fallback to some common devices for compatibility
            common_devices = [
                DeviceInfo("Pixel 6", "12", 31),
                DeviceInfo("Samsung Galaxy S21", "13", 33),
                DeviceInfo("Xiaomi Mi 11", "12", 31),
                DeviceInfo("Oppo Reno 8", "12", 31),
                DeviceInfo("OnePlus 9", "12", 31),
            ]
            return random.choice(common_devices)

    @classmethod
    def random_user_agent(cls) -> str:
        """Get random user agent"""
        return random.choice(cls.USER_AGENTS)

    @classmethod
    def generate_batch_devices(cls, count: int = 20000) -> None:
        """Generate batch devices và thêm vào pool"""
        print(f"🔄 Đang tạo {count:,} devices...")
        for i in range(count):
            device = cls.generate_device()
            cls._device_pool.append(device)
            if (i + 1) % 5000 == 0:
                print(f"✅ Đã tạo {i + 1:,}/{count:,} devices...")
        print(f"✅ Hoàn thành! Đã tạo {len(cls._device_pool):,} devices trong pool\n")

    @classmethod
    def get_device_from_pool(cls) -> DeviceInfo:
        """Lấy device từ pool hoặc generate mới nếu pool rỗng"""
        if cls._device_pool:
            return cls._device_pool.popleft()
        else:
            # Nếu pool rỗng, generate mới
            return cls.generate_device()

    @classmethod
    def get_stats(cls) -> Dict:
        """Get device generation statistics"""
        return {
            "unique_devices_generated": len(cls._generated_devices),
            "max_cache_size": cls._max_cache_size,
            "devices_in_pool": len(cls._device_pool)
        }

class Signature:
    """Generate TikTok API signature (X-Gorgon)"""
    KEY = [0xDF, 0x77, 0xB9, 0x40, 0xB9, 0x9B, 0x84, 0x83, 0xD1, 0xB9,
           0xCB, 0xD1, 0xF7, 0xC2, 0xB9, 0x85, 0xC3, 0xD0, 0xFB, 0xC3]

    def __init__(self, params: str, data: str, cookies: str):
        self.params = params
        self.data = data
        self.cookies = cookies

    def _md5_hash(self, data: str) -> str:
        return md5(data.encode()).hexdigest()

    def _reverse_byte(self, n: int) -> int:
        return int(f"{n:02x}"[1:] + f"{n:02x}"[0], 16)

    def generate(self) -> Dict[str, str]:
        g = self._md5_hash(self.params)
        g += self._md5_hash(self.data) if self.data else "0" * 32
        g += self._md5_hash(self.cookies) if self.cookies else "0" * 32
        g += "0" * 32

        unix_timestamp = int(T())
        payload = []

        for i in range(0, 12, 4):
            chunk = g[8 * i:8 * (i + 1)]
            for j in range(4):
                payload.append(int(chunk[j * 2:(j + 1) * 2], 16))

        payload.extend([0x0, 0x6, 0xB, 0x1C])
        payload.extend([
            (unix_timestamp & 0xFF000000) >> 24,
            (unix_timestamp & 0x00FF0000) >> 16,
            (unix_timestamp & 0x0000FF00) >> 8,
            (unix_timestamp & 0x000000FF)
        ])

        encrypted = [a ^ b for a, b in zip(payload, self.KEY)]

        for i in range(0x14):
            C = self._reverse_byte(encrypted[i])
            D = encrypted[(i + 1) % 0x14]
            F = int(bin(C ^ D)[2:].zfill(8)[::-1], 2)
            H = ((F ^ 0xFFFFFFFF) ^ 0x14) & 0xFF
            encrypted[i] = H

        signature = "".join(f"{x:02x}" for x in encrypted)

        return {
            "X-Gorgon": "840280416000" + signature,
            "X-Khronos": str(unix_timestamp)
        }

class CircuitBreaker:
    """Simple circuit breaker - no optimization"""
    def __init__(self):
        pass

    def record_success(self):
        pass

    def record_failure(self):
        pass

    def can_proceed(self) -> bool:
        return True

class RateLimiter:
    """Simple rate limiter - no optimization"""
    def __init__(self):
        pass

    def should_throttle(self) -> bool:
        return False

    def record_request(self):
        pass

    def adjust_rate(self, success: bool):
        pass

class TikTokViewBot:
    """Main TikTok View Bot class"""

    def __init__(self):
        self.count = 0
        self.like_count = 0  # Đếm số tim đã buff
        self.start_time = 0
        self.is_running = False
        self.session = None
        self.successful_requests = 0
        self.failed_requests = 0
        self.successful_likes = 0  # Đếm số like thành công
        self.failed_likes = 0  # Đếm số like thất bại
        self.peak_speed = 0
        self.peak_like_speed = 0  # Tốc độ like cao nhất
        self.circuit_breaker = CircuitBreaker()
        self.rate_limiter = RateLimiter()
        self.batch_count = 0  # Đếm số view trong mỗi batch (100 view)
        self.like_batch_count = 0  # Đếm số like trong mỗi batch (100 like)
        self.total_batches = 0  # Tổng số batch đã hoàn thành
        self.total_like_batches = 0  # Tổng số like batch đã hoàn thành
        self.used_devices = set()  # Track devices đã sử dụng
        self.recent_success_rate = deque(maxlen=100)  # Track success rate gần đây
        self.request_delays = deque(maxlen=50)  # Track delays để điều chỉnh
        self.last_request_time = {}  # Track thời gian request cuối cùng cho mỗi endpoint
        self.endpoint_rotation_index = 0  # Index để rotate endpoints
        self.endpoint_success_count = {}  # Track số lần thành công của mỗi endpoint
        self.endpoint_failure_count = {}  # Track số lần thất bại của mỗi endpoint
        self.batch_count = 0

    async def init_session(self):
        """Initialize aiohttp session với DNS error suppression"""
        # Timeout tối ưu - cân bằng giữa tốc độ và ổn định
        timeout = aiohttp.ClientTimeout(
            total=10,  # Giảm từ 15 xuống 10 để giải phóng connections nhanh hơn
            connect=3,  # Giảm từ 5 xuống 3
            sock_read=7  # Giảm từ 10 xuống 7
        )

        # Connection pooling
        # Tắt aiohappyeyeballs để tránh lỗi "Task was destroyed but it is pending"
        # Python 3.14 compatible resolver
        resolver = None
        try:
            # Thử dùng DefaultResolver nếu có
            if hasattr(aiohttp.resolver, 'DefaultResolver'):
                resolver = aiohttp.resolver.DefaultResolver()
        except (AttributeError, ImportError):
            # Nếu không có, dùng None (aiohttp sẽ tự xử lý)
            resolver = None

        # Suppress aiohttp warnings về gaierror
        import warnings
        warnings.filterwarnings('ignore', category=RuntimeWarning, module='aiohttp')

        # Tối ưu connection pooling để giảm tải mạng
        connector = aiohttp.TCPConnector(
            limit=3000,  # Giảm từ 10000 xuống 3000 để tiết kiệm băng thông
            limit_per_host=500,  # Giảm từ 1000 xuống 500
            ttl_dns_cache=3600,  # Tăng DNS cache TTL lên 1 giờ để giảm DNS queries
            use_dns_cache=True,
            keepalive_timeout=120,  # Tăng keepalive để tái sử dụng connections
            enable_cleanup_closed=True,
            force_close=False,
            resolver=resolver  # Tắt aiohappyeyeballs
        )

        user_agent = DeviceGenerator.random_user_agent()

        # Python 3.14 compatible cookie jar
        try:
            cookie_jar = aiohttp.DummyCookieJar()
        except (AttributeError, TypeError):
            # Fallback nếu DummyCookieJar không khả dụng
            cookie_jar = None

        session_kwargs = {
            'timeout': timeout,
            'connector': connector,
            'headers': {'User-Agent': user_agent},
            'skip_auto_headers': {'User-Agent'},
            'connector_owner': True  # Session tự quản lý connector
        }

        # Chỉ thêm cookie_jar nếu có
        if cookie_jar is not None:
            session_kwargs['cookie_jar'] = cookie_jar

        self.session = aiohttp.ClientSession(**session_kwargs)

    async def close_session(self):
        """Close aiohttp session - improved cleanup - Python 3.14 compatible"""
        if self.session:
            try:
                # Đóng session với timeout - Python 3.14 compatible
                await asyncio.wait_for(self.session.close(), timeout=3.0)
            except (asyncio.TimeoutError, RuntimeError):
                # Python 3.14 có thể raise RuntimeError
                logger.warning("[WARNING] Session close timeout, force closing...")
            except Exception as e:
                logger.error(f"[ERROR] Error closing session: {e}")
            finally:
                self.session = None

    def get_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from TikTok URL"""
        try:
            # Method 1: Direct pattern matching
            patterns_url = [
                r'/video/(\d+)',
                r'tiktok\.com/@[^/]+/(\d+)',
                r'(\d{18,19})'
            ]

            for pattern in patterns_url:
                match = re.search(pattern, url)
                if match:
                    video_id = match.group(1)
                    logger.info(f"[OK] Found Video ID from URL: {video_id}")
                    return video_id

            # Method 2: Request page and extract
            response = requests.get(
                url,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                },
                timeout=15
            )
            response.raise_for_status()

            patterns_page = [
                r'"video":\{"id":"(\d+)"',
                r'"id":"(\d+)"',
                r'video/(\d+)',
                r'(\d{19})',
                r'"aweme_id":"(\d+)"'
            ]

            for pattern in patterns_page:
                match = re.search(pattern, response.text)
                if match:
                    video_id = match.group(1)
                    logger.info(f"[OK] Found Video ID from page: {video_id}")
                    return video_id

            logger.error("[ERROR] No video ID found")
            return None

        except Exception as e:
            logger.error(f"[ERROR] Error getting video ID: {e}")
            return None

    def generate_request_data(self, video_id: str) -> Tuple[str, Dict, Dict, Dict]:
        """Generate randomized request data - mỗi request dùng device khác nhau"""
        # Luôn generate device mới để đảm bảo mỗi request dùng device khác nhau
        device = DeviceGenerator.get_device_from_pool()  # Lấy từ pool đã pre-generate
        user_agent = DeviceGenerator.random_user_agent()

        # Track device đã sử dụng (tối ưu memory: chỉ lưu signature)
        device_signature = f"{device.model}|{device.version}|{device.api_level}"
        # Giới hạn số lượng devices track để tránh memory leak - giảm từ 50000 xuống 10000
        if len(self.used_devices) < 10000:  # Chỉ track tối đa 10000 devices để tiết kiệm memory
            self.used_devices.add(device_signature)

        # Randomize parameters
        version_codes = ["400304", "400305", "400306", "400307"]
        version_code = random.choice(version_codes)
        device_id = random.randint(600000000000000, 999999999999999)
        aids = ["1233", "1234", "1235", "1180"]
        aid = random.choice(aids)
        channels = ["googleplay", "appstore", "tiktok_ads"]
        channel = random.choice(channels)

        # Randomize location data
        iid = random.randint(7000000000000000000, 7999999999999999999)
        device_brand = random.choice(['Google', 'Samsung', 'Xiaomi', 'Oppo', 'OnePlus', 'Realme', 'Vivo'])
        app_language = random.choice(['vi', 'en', 'zh'])
        region = random.choice(['VN', 'US', 'SG', 'MY', 'TH'])
        tz_name = random.choice([
            'Asia%2FHo_Chi_Minh', 'America%2FNew_York', 'Asia%2FSingapore',
            'Asia%2FBangkok', 'Asia%2FKuala_Lumpur'
        ])

        # Build URL parameters - thêm các tham số quan trọng để TikTok tính view
        params = (
            f"channel={channel}&aid={aid}&app_name=musical_ly&version_code={version_code}"
            f"&device_platform=android&device_type={device.model.replace(' ', '+')}"
            f"&os_version={device.version}&device_id={device_id}"
            f"&os_api={device.api_level}&app_language={app_language}&tz_name={tz_name}"
            f"&iid={iid}&device_brand={device_brand}"
            f"&language={app_language}&region={region}"
            f"&manifest_version_code={version_code}"
            f"&update_version_code={version_code}"
            f"&ac=wifi&channel_source=ads"
            f"&is_my_cn=0&fp={secrets.token_hex(16)}"
            f"&cdid={secrets.token_hex(16)}"
        )

        # Randomize endpoint
        endpoints = [
            "https://api16-core-c-alisg.tiktokv.com/aweme/v1/aweme/stats/",
            "https://api16-core-c-useast1a.tiktokv.com/aweme/v1/aweme/stats/",
            "https://api16-core-c.tiktokv.com/aweme/v1/aweme/stats/",
            "https://api16-va.tiktokv.com/aweme/v1/aweme/stats/",
            "https://api16-va-alisg.tiktokv.com/aweme/v1/aweme/stats/",
            "https://api16-core.tiktokv.com/aweme/v1/aweme/stats/",
        ]
        base_url = random.choice(endpoints)
        url = f"{base_url}?{params}"

        # Generate request data
        current_time = int(time.time())
        action_time = current_time + random.randint(-1, 1)

        # Data payload đầy đủ để TikTok nhận diện view ngay lập tức
        play_time = random.randint(5, 15)  # Tăng thời gian xem video (5-15 giây) để realistic hơn
        video_play_duration = play_time * 1000  # Chuyển sang milliseconds

        data = {
            "item_id": video_id,
            "play_delta": 1,  # Luôn là 1
            "action_time": action_time,
            "stats_channel": "video",
            "play_time": play_time,  # Thời gian xem video (giây)
            "enter_from": random.choice(["homepage_hot", "homepage_follow", "search", "other"]),
            "from_prefetch": 0,
            "video_play_duration": video_play_duration,  # Thời gian xem (milliseconds)
            "is_play": 1,  # Đánh dấu đã play
            "play_progress": random.randint(30, 80),  # Phần trăm đã xem (30-80%)
            "stay_time": play_time,  # Thời gian ở lại
            "scroll_count": random.randint(0, 3),  # Số lần scroll
            "pause_count": random.randint(0, 1),  # Số lần pause
        }

        # Generate cookies
        cookies = {
            "sessionid": secrets.token_hex(16),
            "sid_guard": secrets.token_hex(32),
            "uid_tt": str(random.randint(1000000000000000000, 9999999999999999999)),
            "sid_tt": secrets.token_hex(16),
        }

        # Generate headers
        timestamp_ms = int(time.time() * 1000)
        req_ticket = timestamp_ms + random.randint(0, 1000)

        headers = {
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "User-Agent": user_agent,
            "Accept-Encoding": random.choice(["gzip, deflate, br", "gzip, deflate"]),
            "Accept-Language": random.choice([
                "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
                "en-US,en;q=0.9",
                "vi,en-US;q=0.9,en;q=0.8",
            ]),
            "Connection": "keep-alive",
            "X-SS-REQ-TICKET": str(req_ticket),
            "X-Tt-Token": "",
            "sdk-version": random.choice(["1", "2", "3"]),
            "X-VC-BDT": str(timestamp_ms),
            "X-Tt-Trace-Id": secrets.token_hex(16),
            "X-Argus": secrets.token_hex(32),
            "X-Ladon": secrets.token_hex(32),
            "sdk_name": "aweme",
            "sdk_version": "2",
        }

        return url, data, cookies, headers

    def generate_like_request_data(self, video_id: str) -> Tuple[str, Dict, Dict, Dict]:
        """Generate request data for TikTok like (digg)"""
        # Luôn generate device mới để đảm bảo mỗi request dùng device khác nhau
        device = DeviceGenerator.get_device_from_pool()
        user_agent = DeviceGenerator.random_user_agent()

        # Randomize parameters
        version_codes = ["400304", "400305", "400306", "400307"]
        version_code = random.choice(version_codes)
        device_id = random.randint(600000000000000, 999999999999999)
        aids = ["1233", "1234", "1235", "1180"]
        aid = random.choice(aids)
        channels = ["googleplay", "appstore", "tiktok_ads"]
        channel = random.choice(channels)

        # Randomize location data
        iid = random.randint(7000000000000000000, 7999999999999999999)
        device_brand = random.choice(['Google', 'Samsung', 'Xiaomi', 'Oppo', 'OnePlus', 'Realme', 'Vivo'])
        app_language = random.choice(['vi', 'en', 'zh'])
        region = random.choice(['VN', 'US', 'SG', 'MY', 'TH'])
        tz_name = random.choice([
            'Asia%2FHo_Chi_Minh', 'America%2FNew_York', 'Asia%2FSingapore',
            'Asia%2FBangkok', 'Asia%2FKuala_Lumpur'
        ])

        # Build URL parameters cho like endpoint
        params = (
            f"channel={channel}&aid={aid}&app_name=musical_ly&version_code={version_code}"
            f"&device_platform=android&device_type={device.model.replace(' ', '+')}"
            f"&os_version={device.version}&device_id={device_id}"
            f"&os_api={device.api_level}&app_language={app_language}&tz_name={tz_name}"
            f"&iid={iid}&device_brand={device_brand}"
            f"&language={app_language}&region={region}"
            f"&manifest_version_code={version_code}"
            f"&update_version_code={version_code}"
            f"&ac=wifi&channel_source=ads"
            f"&is_my_cn=0&fp={secrets.token_hex(16)}"
            f"&cdid={secrets.token_hex(16)}"
        )

        # Like endpoints (digg = like)
        endpoints = [
            "https://api16-core-c-alisg.tiktokv.com/aweme/v1/commit/item/digg/",
            "https://api16-core-c-useast1a.tiktokv.com/aweme/v1/commit/item/digg/",
            "https://api16-core-c.tiktokv.com/aweme/v1/commit/item/digg/",
            "https://api16-va.tiktokv.com/aweme/v1/commit/item/digg/",
            "https://api16-va-alisg.tiktokv.com/aweme/v1/commit/item/digg/",
            "https://api16-core.tiktokv.com/aweme/v1/commit/item/digg/",
        ]
        base_url = random.choice(endpoints)
        url = f"{base_url}?{params}"

        # Generate request data cho like
        current_time = int(time.time())
        action_time = current_time + random.randint(-1, 1)

        # Data payload cho like (digg)
        data = {
            "item_id": video_id,
            "type": 1,  # 1 = like, 0 = unlike
            "action_time": action_time,
        }

        # Generate cookies
        cookies = {
            "sessionid": secrets.token_hex(16),
            "sid_guard": secrets.token_hex(32),
            "uid_tt": str(random.randint(1000000000000000000, 9999999999999999999)),
            "sid_tt": secrets.token_hex(16),
        }

        # Generate headers
        timestamp_ms = int(time.time() * 1000)
        req_ticket = timestamp_ms + random.randint(0, 1000)

        headers = {
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "User-Agent": user_agent,
            "Accept-Encoding": random.choice(["gzip, deflate, br", "gzip, deflate"]),
            "Accept-Language": random.choice([
                "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
                "en-US,en;q=0.9",
                "vi,en-US;q=0.9,en;q=0.8",
            ]),
            "Connection": "keep-alive",
            "X-SS-REQ-TICKET": str(req_ticket),
            "X-Tt-Token": "",
            "sdk-version": random.choice(["1", "2", "3"]),
            "X-VC-BDT": str(timestamp_ms),
            "X-Tt-Trace-Id": secrets.token_hex(16),
            "X-Argus": secrets.token_hex(32),
            "X-Ladon": secrets.token_hex(32),
            "sdk_name": "aweme",
            "sdk_version": "2",
        }

        return url, data, cookies, headers

    async def send_like_request(self, video_id: str, semaphore: asyncio.Semaphore) -> bool:
        """Send like request to TikTok"""
        async with semaphore:
            try:
                url, data, cookies, base_headers = self.generate_like_request_data(video_id)

                # Generate signature
                params_str = url.split('?')[1] if '?' in url else ''
                data_str = f"item_id={data['item_id']}&type={data['type']}&action_time={data['action_time']}"
                cookies_str = f"sessionid={cookies.get('sessionid', '')}"

                sig = Signature(params_str, data_str, cookies_str).generate()
                headers = {**base_headers, **sig}

                # Send request
                form_data = urlencode(data)

                try:
                    async with self.session.post(
                        url,
                        data=form_data,
                        headers=headers,
                        cookies=cookies,
                        ssl=False,
                        allow_redirects=True
                    ) as response:
                        # Cải thiện logic: Chấp nhận nhiều trường hợp thành công cho like
                        status = response.status

                        # Chấp nhận HTTP 200, 204, hoặc bất kỳ status 200-299 nào
                        if 200 <= status < 300:
                            # Thành công - đếm like ngay lập tức
                            self.like_count += 1
                            self.successful_likes += 1
                            self.like_batch_count += 1

                            if self.like_batch_count >= 100:
                                self.total_like_batches += 1
                                self.like_batch_count = 0

                            return True
                        else:
                            # Thử parse JSON để kiểm tra status_code (fallback)
                            try:
                                response_text = await response.text()
                                if response_text:
                                    try:
                                        response_json = json.loads(response_text)
                                        status_code = response_json.get('status_code', -1)

                                        # Nếu status_code = 0 thì thành công
                                        if status_code == 0:
                                            self.like_count += 1
                                            self.successful_likes += 1
                                            self.like_batch_count += 1

                                            if self.like_batch_count >= 100:
                                                self.total_like_batches += 1
                                                self.like_batch_count = 0

                                            return True
                                    except (json.JSONDecodeError, KeyError, TypeError):
                                        pass
                            except Exception:
                                pass

                            # Nếu không thành công
                            self.failed_likes += 1
                            return False
                except (aiohttp.ClientConnectorError, aiohttp.ClientConnectorDNSError) as e:
                    self.failed_likes += 1
                    return False
                except Exception as e:
                    error_name = type(e).__name__
                    if 'gaierror' in error_name.lower() or 'getaddrinfo' in str(e).lower():
                        self.failed_likes += 1
                        return False

            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                self.failed_likes += 1
                error_name = type(e).__name__
                is_dns_error = (
                    isinstance(e, (aiohttp.ClientConnectorError, aiohttp.ClientConnectorDNSError, asyncio.TimeoutError, aiohttp.ServerTimeoutError, aiohttp.ClientTimeoutError)) or
                    'gaierror' in error_name.lower() or
                    'getaddrinfo' in str(e).lower() or
                    (hasattr(e, '__cause__') and isinstance(e.__cause__, socket.gaierror))
                )
                if not is_dns_error:
                    if self.failed_likes % 100 == 0:
                        logger.warning(f"[WARNING] Like request error: {type(e).__name__}")
                return False
            except Exception as e:
                error_name = type(e).__name__
                is_dns_error = (
                    'gaierror' in error_name.lower() or
                    'getaddrinfo' in str(e).lower() or
                    isinstance(e, socket.gaierror) or
                    (hasattr(e, '__cause__') and isinstance(e.__cause__, socket.gaierror))
                )
                if is_dns_error:
                    self.failed_likes += 1
                    return False
                if self.failed_likes % 100 == 0:
                    logger.error(f"[ERROR] Unexpected like error: {type(e).__name__}: {e}")
                self.failed_likes += 1
                return False

    async def send_view_request(self, video_id: str, semaphore: asyncio.Semaphore) -> bool:
        """Send view request - no optimization"""
        async with semaphore:
            try:
                url, data, cookies, base_headers = self.generate_request_data(video_id)

                # Generate signature với đầy đủ data để TikTok nhận diện
                params_str = url.split('?')[1] if '?' in url else ''
                # Include tất cả data fields trong signature
                data_str = f"item_id={data['item_id']}&play_delta={data['play_delta']}&action_time={data['action_time']}"
                if 'play_time' in data:
                    data_str += f"&play_time={data['play_time']}&video_play_duration={data['video_play_duration']}&is_play={data['is_play']}"
                cookies_str = f"sessionid={cookies.get('sessionid', '')}"

                sig = Signature(params_str, data_str, cookies_str).generate()
                headers = {**base_headers, **sig}

                # Send request
                form_data = urlencode(data)

                try:
                    async with self.session.post(
                        url,
                        data=form_data,
                        headers=headers,
                        cookies=cookies,
                        ssl=False,
                        allow_redirects=True
                    ) as response:
                        # Cải thiện logic: Chấp nhận nhiều trường hợp thành công
                        status = response.status

                        # Chấp nhận HTTP 200, 204, hoặc bất kỳ status 200-299 nào
                        if 200 <= status < 300:
                            # Thành công - đếm view ngay lập tức
                            self.count += 1
                            self.successful_requests += 1
                            self.batch_count += 1

                            if self.batch_count >= 100:
                                self.total_batches += 1
                                self.batch_count = 0

                            return True
                        else:
                            # Thử parse JSON để kiểm tra status_code (fallback)
                            try:
                                response_text = await response.text()
                                if response_text:
                                    try:
                                        response_json = json.loads(response_text)
                                        status_code = response_json.get('status_code', -1)

                                        # Nếu status_code = 0 thì thành công
                                        if status_code == 0:
                                            self.count += 1
                                            self.successful_requests += 1
                                            self.batch_count += 1

                                            if self.batch_count >= 100:
                                                self.total_batches += 1
                                                self.batch_count = 0

                                            return True
                                    except (json.JSONDecodeError, KeyError, TypeError):
                                        pass
                            except Exception:
                                pass

                            # Nếu không thành công
                            self.failed_requests += 1
                            return False
                except (aiohttp.ClientConnectorError, aiohttp.ClientConnectorDNSError) as e:
                    # Bỏ qua DNS và connection errors - không log để tránh spam
                    self.failed_requests += 1
                    return False
                except Exception as e:
                    # Xử lý gaierror và các DNS errors khác
                    error_name = type(e).__name__
                    if 'gaierror' in error_name.lower() or 'getaddrinfo' in str(e).lower():
                        # DNS resolution failed - bỏ qua
                        self.failed_requests += 1
                        return False

            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                # Track timeout và client errors
                self.failed_requests += 1
                # Bỏ qua DNS errors và các lỗi kết nối - không log để tránh spam
                error_name = type(e).__name__
                is_dns_error = (
                    isinstance(e, (aiohttp.ClientConnectorError, aiohttp.ClientConnectorDNSError, asyncio.TimeoutError, aiohttp.ServerTimeoutError, aiohttp.ClientTimeoutError)) or
                    'gaierror' in error_name.lower() or
                    'getaddrinfo' in str(e).lower() or
                    (hasattr(e, '__cause__') and isinstance(e.__cause__, socket.gaierror))
                )
                if not is_dns_error:
                    # Chỉ log các lỗi khác
                    if self.failed_requests % 100 == 0:
                        logger.warning(f"[WARNING] Request error: {type(e).__name__}")
                return False
            except Exception as e:
                # Xử lý gaierror và các DNS errors khác
                error_name = type(e).__name__
                is_dns_error = (
                    'gaierror' in error_name.lower() or
                    'getaddrinfo' in str(e).lower() or
                    isinstance(e, socket.gaierror) or
                    (hasattr(e, '__cause__') and isinstance(e.__cause__, socket.gaierror))
                )
                if is_dns_error:
                    # DNS resolution failed - bỏ qua, không log
                    self.failed_requests += 1
                    return False
                # Log exception để debug (chỉ các lỗi khác)
                if self.failed_requests % 100 == 0:
                    logger.error(f"[ERROR] Unexpected error: {type(e).__name__}: {e}")
                self.failed_requests += 1
                return False

    async def view_sender(self, video_id: str, task_id: int, semaphore: asyncio.Semaphore):
        """Send views continuously - tối ưu để giảm tải mạng"""
        # Stagger initial delay
        initial_delay = (task_id % 100) * 0.001  # Tăng từ 0.0001 lên 0.001
        if initial_delay > 0:
            await asyncio.sleep(initial_delay)

        while self.is_running:
            try:
                await self.send_view_request(video_id, semaphore)
            except Exception:
                # Bỏ qua lỗi để tiếp tục gửi requests
                pass
            # Delay tối ưu: 0.005s - cân bằng giữa tốc độ và tải mạng
            await asyncio.sleep(0.005)

    async def like_sender(self, video_id: str, task_id: int, semaphore: asyncio.Semaphore):
        """Send likes continuously - tối ưu để giảm tải mạng"""
        # Stagger initial delay
        initial_delay = (task_id % 100) * 0.001  # Tăng từ 0.0001 lên 0.001
        if initial_delay > 0:
            await asyncio.sleep(initial_delay)

        while self.is_running:
            try:
                await self.send_like_request(video_id, semaphore)
            except Exception:
                # Bỏ qua lỗi để tiếp tục gửi requests
                pass
            # Delay tối ưu: 0.005s - cân bằng giữa tốc độ và tải mạng
            await asyncio.sleep(0.005)

    def calculate_stats(self) -> Dict[str, float]:
        """Calculate statistics"""
        elapsed = time.time() - self.start_time
        if elapsed <= 0:
            return {
                "total_views": self.count,
                "total_likes": self.like_count,
                "elapsed_time": 0,
                "views_per_second": 0,
                "likes_per_second": 0,
                "views_per_minute": 0,
                "likes_per_minute": 0,
                "views_per_hour": 0,
                "likes_per_hour": 0,
                "success_rate": 0,
                "like_success_rate": 0,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "successful_likes": self.successful_likes,
                "failed_likes": self.failed_likes,
                "peak_speed": self.peak_speed,
                "peak_like_speed": self.peak_like_speed
            }

        views_per_second = self.count / elapsed
        likes_per_second = self.like_count / elapsed

        if views_per_second > self.peak_speed:
            self.peak_speed = views_per_second

        if likes_per_second > self.peak_like_speed:
            self.peak_like_speed = likes_per_second

        # Tính toán đơn giản hơn
        views_per_minute = views_per_second * 60
        views_per_hour = views_per_minute * 60
        likes_per_minute = likes_per_second * 60
        likes_per_hour = likes_per_minute * 60

        total_requests = self.successful_requests + self.failed_requests
        success_rate = (self.successful_requests / total_requests * 100) if total_requests > 0 else 0

        total_likes = self.successful_likes + self.failed_likes
        like_success_rate = (self.successful_likes / total_likes * 100) if total_likes > 0 else 0

        return {
            "total_views": self.count,
            "total_likes": self.like_count,
            "elapsed_time": elapsed,
            "views_per_second": views_per_second,
            "likes_per_second": likes_per_second,
            "views_per_minute": views_per_minute,
            "likes_per_minute": likes_per_minute,
            "views_per_hour": views_per_hour,
            "likes_per_hour": likes_per_hour,
            "success_rate": success_rate,
            "like_success_rate": like_success_rate,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "successful_likes": self.successful_likes,
            "failed_likes": self.failed_likes,
            "peak_speed": self.peak_speed,
            "peak_like_speed": self.peak_like_speed
        }

    def display_stats(self):
        """Display statistics"""
        stats = self.calculate_stats()

        print(f"\n{'='*60}")
        print(f"📊 THỐNG KÊ HIỆU SUẤT")
        print(f"{'='*60}")
        print(f"👀 Tổng view: {stats['total_views']:,}")
        print(f"❤️  Tổng tim: {stats['total_likes']:,}")
        print(f"⏰ Thời gian: {stats['elapsed_time']:.1f}s")
        print(f"🚀 Tốc độ view hiện tại: {stats['views_per_second']:.1f} view/s")
        print(f"💖 Tốc độ tim hiện tại: {stats['likes_per_second']:.1f} tim/s")
        print(f"🏆 Tốc độ view cao nhất: {stats['peak_speed']:.1f} view/s")
        print(f"💝 Tốc độ tim cao nhất: {stats['peak_like_speed']:.1f} tim/s")
        print(f"📈 Dự kiến view: {stats['views_per_minute']:,.0f} view/phút | {stats['views_per_hour']:,.0f} view/giờ")
        print(f"💕 Dự kiến tim: {stats['likes_per_minute']:,.0f} tim/phút | {stats['likes_per_hour']:,.0f} tim/giờ")
        print(f"✅ View thành công: {stats['successful_requests']:,} | ❌ Thất bại: {stats['failed_requests']:,}")
        print(f"💚 Tim thành công: {stats['successful_likes']:,} | ❌ Thất bại: {stats['failed_likes']:,}")
        print(f"🎯 Tỷ lệ view thành công: {stats['success_rate']:.1f}%")
        print(f"💗 Tỷ lệ tim thành công: {stats['like_success_rate']:.1f}%")
        print(f"📱 Device đã sử dụng: {len(self.used_devices):,} devices")
        print(f"📦 View batch: {self.total_batches:,} batches (100 views/batch) | Hiện tại: {self.batch_count}/100")
        print(f"💝 Tim batch: {self.total_like_batches:,} batches (100 tim/batch) | Hiện tại: {self.like_batch_count}/100")
        print(f"{'='*60}")

    async def run(self, video_url: str, buff_likes: bool = False):
        """Main run method"""
        print("\n" + "="*60)
        print("🚀 TIKTOK VIEW BOT - Educational & Research Purpose Only")
        print("="*60)
        print("⚠️  DISCLAIMER: For educational and research purposes only!")
        print(f"🐍 Python Version: {PYTHON_VERSION} (Hỗ trợ: 3.8 - 3.14+)")
        print("="*60 + "\n")

        print("🔄 Đang lấy Video ID...")
        video_id = self.get_video_id(video_url)

        if not video_id:
            print("❌ Không thể lấy Video ID. Kiểm tra lại URL!")
            return

        # Tối ưu số workers để giảm tải mạng - giảm từ 20000 xuống 6000
        optimal_workers = 6000

        print(f"✅ Video ID: {video_id}")

        # Pre-generate devices - giảm từ 20000 xuống 6000 để tiết kiệm memory
        DeviceGenerator.generate_batch_devices(6000)

        print(f"🎯 Số tasks view: {optimal_workers:,}")
        if buff_likes:
            print(f"💖 Số tasks tim: {optimal_workers:,}")
        print(f"📱 Mỗi task sẽ dùng 1 device khác nhau (tổng {optimal_workers:,} devices)")
        print(f"📦 Mỗi batch sẽ buff 100 view (từ 0 lên 100, rồi tiếp tục)")
        if buff_likes:
            print(f"💝 Mỗi batch sẽ buff 100 tim (từ 0 lên 100, rồi tiếp tục)")
        print("\n💡 TỐI ƯU ĐÃ ÁP DỤNG:")
        print("   ✅ Giảm workers: 20,000 → 6,000 (giảm 70% tải mạng)")
        print("   ✅ Giảm connections: 10,000 → 3,000 (tiết kiệm băng thông)")
        print("   ✅ Delay tối ưu: 0.005s (cân bằng tốc độ và tải mạng)")
        print("   ✅ Tối ưu memory: Giảm device tracking và pre-generation")
        print("   ✅ Tối ưu connection pooling và keepalive")
        print("   ✅ Cải thiện logic xử lý response để buff view tốt hơn")
        print("\n⚡ Đang khởi chạy với cấu hình tối ưu... (Nhấn Ctrl+C để dừng)\n")

        await asyncio.sleep(0.1)

        await self.init_session()
        self.is_running = True
        self.start_time = time.time()

        # Semaphore tối ưu: 3000 - cân bằng giữa tốc độ và tải mạng
        semaphore = asyncio.Semaphore(3000)

        tasks = []
        try:
            batch_size = 1000  # Giảm từ 2000 xuống 1000 để khởi động nhẹ nhàng hơn

            print("🔄 Đang khởi tạo tasks...")
            for i in range(0, optimal_workers, batch_size):
                batch_end = min(i + batch_size, optimal_workers)
                for j in range(i, batch_end):
                    task = asyncio.create_task(self.view_sender(video_id, j, semaphore))
                    tasks.append(task)

                    # Thêm like tasks nếu được bật
                    if buff_likes:
                        like_task = asyncio.create_task(self.like_sender(video_id, j + optimal_workers, semaphore))
                        tasks.append(like_task)

                if batch_end < optimal_workers:
                    await asyncio.sleep(0.05)  # Tăng delay từ 0.01 lên 0.05 để giảm tải khi khởi động

            logger.info(f"[OK] Da khoi tao {len(tasks):,} tasks")
            print(f"✅ Đã khởi tạo {len(tasks):,} tasks thành công! ({optimal_workers:,} view tasks" + (f" + {optimal_workers:,} like tasks" if buff_likes else "") + ")\n")

            last_display = 0
            while self.is_running:
                await asyncio.sleep(1.0)  # Update stats

                current_time = time.time()
                if current_time - last_display >= 1.0:
                    stats = self.calculate_stats()
                    device_count = len(self.used_devices)
                    if buff_likes:
                        print(
                            f"\r✅ Views: {stats['total_views']:,} ({stats['views_per_second']:.0f}/s) | "
                            f"❤️  Tim: {stats['total_likes']:,} ({stats['likes_per_second']:.0f}/s) | "
                            f"Batch V:{self.total_batches + 1}({self.batch_count}/100) L:{self.total_like_batches + 1}({self.like_batch_count}/100) | "
                            f"Devices: {device_count:,}",
                            end="", flush=True
                        )
                    else:
                        print(
                            f"\r✅ Views: {stats['total_views']:,} | "
                            f"Speed: {stats['views_per_second']:.0f}/s | "
                            f"Batch {self.total_batches + 1}: {self.batch_count}/100 views | "
                            f"Devices: {device_count:,}",
                            end="", flush=True
                        )
                    last_display = current_time

        except KeyboardInterrupt:
            print("\n\n🛑 Đang dừng bot...")
        except Exception as e:
            logger.error(f"[ERROR] Loi: {e}")
        finally:
            self.is_running = False

            # Cải thiện cleanup tasks để tránh lỗi "Task was destroyed but it is pending"
            logger.info("[STOP] Dang dung cac tasks...")

            # Cancel tất cả tasks với delay để tránh quá tải
            if tasks:
                # Bước 1: Cancel tất cả tasks
                cancelled_tasks = []
                for task in tasks:
                    if not task.done():
                        task.cancel()
                        cancelled_tasks.append(task)

                # Bước 2: Đợi một chút để tasks nhận cancel signal
                if cancelled_tasks:
                    await asyncio.sleep(0.5)

                # Bước 3: Đợi tất cả tasks hoàn thành hoặc bị cancel (với timeout)
                try:
                    # Python 3.14 compatible gather
                    await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=3.0  # Giảm timeout xuống 3 giây
                    )
                except (asyncio.TimeoutError, RuntimeError):
                    # Python 3.14 có thể raise RuntimeError thay vì TimeoutError
                    logger.warning("[WARNING] Một số tasks chưa dừng kịp...")
                    # Không force cancel lại, để Python tự cleanup

            # Đợi một chút trước khi đóng session
            await asyncio.sleep(0.2)

            # Đóng session
            try:
                await self.close_session()
            except Exception as e:
                logger.error(f"[ERROR] Lỗi khi đóng session: {e}")

            # Đợi một chút để đảm bảo cleanup hoàn tất
            await asyncio.sleep(0.1)

            self.display_stats()

def signal_handler(sig, frame):
    print("\n\n🛑 Nhận tín hiệu dừng...")
    sys.exit(0)

def main():
    """Main entry point"""
    signal.signal(signal.SIGINT, signal_handler)

    print("\n" + "="*60)
    print("🎯 TIKTOK VIEW BOT")
    print("="*60)
    print("⚠️  EDUCATIONAL & RESEARCH PURPOSE ONLY")
    print(f"🐍 Python {PYTHON_VERSION} (Hỗ trợ: 3.8 - 3.14+, bao gồm Python 3.12)")
    print("="*60)

    video_url = input("\n📥 Nhập URL video TikTok: ").strip()

    # Nếu người dùng nhập text thay vì URL, thử tìm URL trong text
    if video_url and not video_url.startswith(('http://', 'https://')):
        # Tìm URL trong text
        url_pattern = r'https?://[^\s]+'
        match = re.search(url_pattern, video_url)
        if match:
            video_url = match.group(0)
        else:
            print("❌ URL không hợp lệ! Vui lòng nhập URL đầy đủ (ví dụ: https://www.tiktok.com/@user/video/1234567890)")
            return

    if not video_url:
        print("❌ URL không được để trống!")
        return

    # Hỏi có muốn buff tim không
    print("\n💖 Bạn có muốn buff tim (like) cùng lúc với view không?")
    print("   📱 Khi bật, TẤT CẢ các device/phone sẽ tự động TIM (like) video này!")
    buff_likes_input = input("   Nhập 'y' hoặc 'yes' để bật, Enter hoặc 'n' để tắt: ").strip().lower()
    buff_likes = buff_likes_input in ['y', 'yes']

    if buff_likes:
        print("✅ Đã bật tính năng buff tim!")
        print("   📱 Tất cả các device sẽ tự động TIM (like) video này!")
        print("   💖 Bot sẽ buff cả view và tim cùng lúc.\n")
    else:
        print("ℹ️  Chỉ buff view. (Nhấn Enter để tiếp tục)\n")

    # Test internet connection
    print("\n🔍 Đang kiểm tra kết nối...", end="")
    try:
        requests.get("https://www.google.com", timeout=5)
        print(" ✅")
    except:
        print(" ❌ Không có kết nối internet!")
        return

    bot = TikTokViewBot()

    try:
        # Python 3.14: WindowsProactorEventLoopPolicy đã deprecated
        # Python 3.14+ sẽ tự động chọn event loop policy phù hợp
        # Không cần set manually nữa
        asyncio.run(bot.run(video_url, buff_likes))
    except KeyboardInterrupt:
        print("\n\n👋 Đã dừng chương trình. Tạm biệt!")
    except Exception as e:
        print(f"\n💥 Lỗi không mong muốn: {e}")

if __name__ == "__main__":
    main()
