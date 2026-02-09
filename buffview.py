import aiohttp
import asyncio
import random
import re
import time
import secrets
import os
import sys
from hashlib import md5
from time import time as T
from typing import Dict
import nest_asyncio
nest_asyncio.apply()

# --- CẤU HÌNH MÀU SẮC CHO TERMINAL ---
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

# --- BANNER ASCII ---
BANNER = r"""

██╗░░░██╗██╗███████╗░██╗░░░░░░░██╗  ░█████╗░██████╗░██████╗░
██║░░░██║██║██╔════╝░██║░░██╗░░██║  ██╔══██╗██╔══██╗██╔══██╗
╚██╗░██╔╝██║█████╗░░░╚██╗████╗██╔╝  ███████║██║░░██║██████╔╝
░╚████╔╝░██║██╔══╝░░░░████╔═████║░  ██╔══██║██║░░██║██╔══██╗
░░╚██╔╝░░██║███████╗░░╚██╔╝░╚██╔╝░  ██║░░██║██████╔╝██║░░██║
░░░╚═╝░░░╚═╝╚══════╝░░░╚═╝░░░╚═╝░░  ╚═╝░░╚═╝╚═════╝░╚═╝░░╚═╝
            [  - ANDROID  ]
              ]
"""

# --- DANH SÁCH THIẾT BỊ (COSMIC FLEET) ---
DEVICES = [
    {"brand": "samsung", "model": "SM-S918B"}, {"brand": "samsung", "model": "SM-S908E"},
    {"brand": "samsung", "model": "SM-G998B"}, {"brand": "samsung", "model": "SM-A546E"},
    {"brand": "xiaomi", "model": "23021RAA2Y"}, {"brand": "xiaomi", "model": "2201116SG"},
    {"brand": "xiaomi", "model": "Redmi Note 12 Pro"}, {"brand": "xiaomi", "model": "Poco X5 Pro"},
    {"brand": "google", "model": "Pixel 7 Pro"}, {"brand": "google", "model": "Pixel 6a"},
    {"brand": "oneplus", "model": "NE2213"}, {"brand": "oneplus", "model": "OnePlus Nord 2T"},
    {"brand": "oppo", "model": "CPH2449"}, {"brand": "oppo", "model": "Reno8 Pro"},
    {"brand": "vivo", "model": "V2202"}, {"brand": "vivo", "model": "X80 Pro"},
    {"brand": "realme", "model": "RMX3301"}, {"brand": "realme", "model": "GT 2 Pro"},
    {"brand": "huawei", "model": "P50 Pro"}, {"brand": "motorola", "model": "Edge 30 Pro"}
]

# --- LOGIC TẠO CHỮ KÝ (X-GORGON) ---
def generate_signature(params: str, data: str, cookies: str) -> Dict[str, str]:
    KEY = [0xDF, 0x77, 0xB9, 0x40, 0xB9, 0x9B, 0x84, 0x83, 0xD1, 0xB9,
           0xCB, 0xD1, 0xF7, 0xC2, 0xB9, 0x85, 0xC3, 0xD0, 0xFB, 0xC3]

    def _md5(s: str) -> str:
        return md5(s.encode()).hexdigest()

    g = _md5(params)
    g += _md5(data) if data else "0" * 32
    g += _md5(cookies) if cookies else "0"*32
    g += "0" * 32

    unix_timestamp = int(T())
    payload = []
    for i in range(0, 12, 4):
        chunk = g[8 * i:8 * (i + 1)]
        for j in range(4):
            payload.append(int(chunk[j * 2:(j + 1) * 2], 16))

    payload.extend([0x0, 0x6, 0xB, 0x1C])
    payload.extend([
        (unix_timestamp & 0xFF000000) >> 24, (unix_timestamp & 0x00FF0000) >> 16,
        (unix_timestamp & 0x0000FF00) >> 8, (unix_timestamp & 0x000000FF)
    ])

    encrypted = [a ^ b for a, b in zip(payload, KEY)]
    for i in range(0x14):
        C = int(f"{encrypted[i]:02x}"[1:] + f"{encrypted[i]:02x}"[0], 16)
        D = encrypted[(i + 1) % 0x14]
        F = int(bin(C ^ D)[2:].zfill(8)[::-1], 2)
        H = ((F ^ 0xFFFFFFFF) ^ 0x14) & 0xFF
        encrypted[i] = H

    signature = "".join(f"{x:02x}" for x in encrypted)
    return {"X-Gorgon": "840280416000" + signature, "X-Khronos": str(unix_timestamp)}

# --- LOGIC TẠO THIẾT BỊ (NÂNG CAO) ---
def generate_persistent_device() -> dict:
    """
    Tạo thông số thiết bị cố định cho một phiên làm việc.
    Giúp giả lập người dùng thật tốt hơn là đổi device liên tục.
    """
    device = random.choice(DEVICES)

    OS_API_MAP = {'13': 33, '12': 32, '11': 30, '10': 29}
    os_version = random.choice(list(OS_API_MAP.keys()))
    os_api = OS_API_MAP[os_version]

    build_prefix = random.choice(['SP1A', 'RKQ1', 'TP1A'])
    build_date = f"{random.randint(22, 23):02d}{random.randint(1, 12):02d}{random.randint(1, 28):02d}"
    build_version = f"{build_prefix}.{build_date}.{random.randint(1, 999):03d}"

    app_version = f"{random.randint(30, 32)}.{random.randint(0, 9)}.{random.randint(0, 9)}"

    user_agent = (f"com.ss.android.ugc.trill/{app_version} "
                  f"(Linux; U; Android {os_version}; vi_VN; {device['model']}; "
                  f"Build/{build_version}; Cronet/58.0.2991.0)")

    params = (
        f"ssmix=a&aid=1233&app_language=vi&app_name=musical_ly&"
        f"app_version={app_version}&device_brand={device['brand']}&"
        f"device_id={random.randint(10**18, 10**19 - 1)}&device_platform=android&"
        f"device_type={device['model']}&os_api={os_api}&os_version={os_version}&"
        f"region=VN&carrier_region=VN&screen_width=1080&screen_height=2400&"
        f"timezone_name=Asia/Ho_Chi_Minh&channel=googleplay"
    )
    return {"params": params, "user_agent": user_agent}

# --- CORE BOT CLASS ---
class TikTokBotCosmic:
    def __init__(self):
        self.running = False
        self.stats = {"sent": 0, "success": 0, "fail": 0, "429": 0}
        self.lock = asyncio.Lock()

        # Cấu hình hạ nhiệt
        self.is_cooling_down = False
        self.cooldown_duration = 30
        self.consecutive_429 = 0

    def log(self, msg, color=Colors.RESET):
        print(f"{color}[{time.strftime('%H:%M:%S')}] {msg}{Colors.RESET}")

    async def send_view(self, session, video_id, device_cache):
        if not self.running: return

        # Sử dụng device cache để giảm giả lập
        device_info = device_cache
        params = device_info["params"]

        url = f"https://api16-core-c-alisg.tiktokv.com/aweme/v1/aweme/stats/?{params}"
        data = f"item_id={video_id}&play_delta=1"
        cookies = f"sessionid={secrets.token_hex(16)}" # Session giả mới mỗi request

        headers = {
            "User-Agent": device_info["user_agent"],
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "Cookie": cookies,
            "X-SS-REQ-TICKET": str(int(time.time() * 1000)),
            "passport-sdk-version": "19",
            "X-Tt-Token": ""
        }
        headers.update(generate_signature(params, data, cookies))

        try:
            async with session.post(url, data=data, headers=headers, timeout=10) as resp:
                async with self.lock:
                    self.stats["sent"] += 1
                    if resp.status == 200:
                        self.stats["success"] += 1
                    elif resp.status == 429:
                        self.stats["429"] += 1
                        self.stats["fail"] += 1
                    else:
                        self.stats["fail"] += 1
        except Exception:
            async with self.lock:
                self.stats["fail"] += 1

    async def worker(self, video_id, sem):
        # Tạo một thiết bị "bám" theo worker này
        my_device = generate_persistent_device()
        change_device_counter = 0

        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False, limit=0)) as session:
            while self.running:
                if self.is_cooling_down:
                    await asyncio.sleep(1)
                    continue

                async with sem:
                    await self.send_view(session, video_id, my_device)
                    # Jitter ngẫu nhiên để nhìn tự nhiên
                    await asyncio.sleep(random.uniform(0.2, 0.6))

                    # Đổi thiết bị sau mỗi 50 request để tránh bị "dính" quá lâu trên 1 fingerprint
                    change_device_counter += 1
                    if change_device_counter > 50:
                        my_device = generate_persistent_device()
                        change_device_counter = 0

    async def stats_printer(self):
        """In bảng thống kê real-time"""
        last_update = time.time()
        while self.running:
            await asyncio.sleep(1)
            # Xóa dòng hiện tại và in đè (ANSI escape code)
            sys.stdout.write("\033[F") # Back to previous line
            # Dòng trạng thái
            status_color = Colors.GREEN if not self.is_cooling_down else Colors.YELLOW
            status_text = "🟢 ĐANG CHẠY" if not self.is_cooling_down else f"🟠 HẠ NHIỆT ({int(self.cooldown_duration)}s)"

            stat_line = (
                f"{Colors.BOLD}Trạng thái: {status_color}{status_text}{Colors.RESET} | "
                f"Đã gửi: {Colors.CYAN}{self.stats['sent']}{Colors.RESET} | "
                f"Thành công: {Colors.GREEN}{self.stats['success']}{Colors.RESET} | "
                f"Lỗi (429): {Colors.RED}{self.stats['429']}{Colors.RESET} | "
                f"Tổng lỗi: {Colors.YELLOW}{self.stats['fail']}{Colors.RESET}     "
            )
            print(stat_line)

    async def main_loop(self, url, threads):
        try:
            vid_match = re.search(r'video/(\d+)', url)
            if not vid_match: vid_match = re.search(r'v=(\d+)', url)
            vid = vid_match.group(1) if vid_match else None

            if not vid:
                async with aiohttp.ClientSession() as session:
                    async with session.head(url, allow_redirects=True, timeout=5) as r:
                        vid = re.search(r'video/(\d+)', str(r.url)).group(1)
            self.log(f"Target Video ID: {vid}", Colors.CYAN)
        except Exception as e:
            self.log(f"Không lấy được ID video: {e}", Colors.RED)
            return

        self.running = True
        self.stats = {k: 0 for k in self.stats}
        self.log(f"Khởi động {threads} luồng vũ trụ...", Colors.GREEN)

        sem = asyncio.Semaphore(threads)
        tasks = [asyncio.create_task(self.worker(vid, sem)) for _ in range(threads)]

        # Task in stats
        monitor_task = asyncio.create_task(self.stats_printer())

        # Logic hạ nhiệt
        while self.running:
            await asyncio.sleep(1)

            # Kiểm tra 429
            current_429 = self.stats["429"]
            # Nếu tỷ lệ lỗi cao -> tăng cooldown
            if current_429 > 0:
                self.consecutive_429 += 1
            else:
                self.consecutive_429 = 0

            if self.consecutive_429 > (threads // 2) and not self.is_cooling_down:
                self.is_cooling_down = True
                self.log(f"Cảnh báo: Tần suất cao! Kích hoạt hạ nhiệt {int(self.cooldown_duration)}s...", Colors.YELLOW)
                await asyncio.sleep(self.cooldown_duration)
                self.is_cooling_down = False
                self.consecutive_429 = 0
                self.stats["429"] = 0 # Reset để theo dõi chu kỳ mới

        for t in tasks: t.cancel()
        monitor_task.cancel()

# --- MAIN ENTRY POINT ---
def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"{Colors.CYAN}{BANNER}{Colors.RESET}")

    print(f"{Colors.YELLOW}[!] Lưu ý: Tool này buff nát tiktok nha ae.{Colors.RESET}")
    print(f"{Colors.YELLOW}[!] buff view tiktok.{Colors.RESET}\n")

    url = input(f"{Colors.BLUE}Nhập link video TikTok: {Colors.RESET}").strip()
    if not url:
        print(f"{Colors.RED}Link không được để trống!{Colors.RESET}")
        return

    try:
        threads = int(input(f"{Colors.BLUE}Nhập số luồng (Khuyến nghị 50-200): {Colors.RESET}"))
        if threads < 1: threads = 1
    except ValueError:
        threads = 100

    print("\n" + "-"*50)

    bot = TikTokBotCosmic()

    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(bot.main_loop(url, threads))
    except KeyboardInterrupt:
        print(f"\n\n{Colors.RED}Đã dừng bot theo yêu cầu người dùng.{Colors.RESET}")
        bot.running = False
    except Exception as e:
        print(f"\n{Colors.RED}Lỗi hệ thống: {e}{Colors.RESET}")

if __name__ == "__main__":
    main()
