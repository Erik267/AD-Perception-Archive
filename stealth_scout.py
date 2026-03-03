import urllib.request
import re
import time
from datetime import datetime, timedelta

def scout_papers():
    # 关键词锁定 2026 先行研究
    keywords = ['autonomous%20driving', 'end-to-end', 'occupancy', 'world%20model']
    base_url = "http://export.arxiv.org/api/query?search_query="
    
    report_path = "/home/lichong.i/code/paper/PRE_SCAN_REPORT_03_03.md"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# 2026-03-03 AD-Perception 预扫描报告
")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

")
        f.write("> **NEGA 侦察兵提示**: 以下为自动检索到的最新 2026 自动驾驶论文，待用户唤醒后执行 10 步极致拆解。

")

        for kw in keywords:
            url = f"{base_url}all:{kw}&start=0&max_results=5&sortBy=submittedDate&sortOrder=descending"
            try:
                with urllib.request.urlopen(url) as response:
                    xml = response.read().decode('utf-8')
                    titles = re.findall(r'<title>(.*?)</title>', xml, re.DOTALL)[1:]
                    links = re.findall(r'<link href="(.*?)" rel="alternate"', xml)
                    summaries = re.findall(r'<summary>(.*?)</summary>', xml, re.DOTALL)
                    
                    f.write(f"## 🔍 领域: {kw.replace('%20', ' ')}
")
                    for t, l, s in zip(titles, links, summaries):
                        f.write(f"### [{t.strip().replace('
', ' ')}]({l})
")
                        f.write(f"- **摘要摘要**: {s.strip()[:200].replace('
', ' ')}...

")
            except Exception as e:
                f.write(f"Error scanning {kw}: {str(e)}
")

if __name__ == "__main__":
    # 计算距离明天 08:00 的秒数
    now = datetime.now()
    target = now.replace(hour=8, minute=0, second=0, microsecond=0)
    if now >= target:
        target += timedelta(days=1)
    
    wait_seconds = (target - now).total_seconds()
    print(f"Stealth Scout 已启动。将在 {wait_seconds} 秒后（明早 08:00）执行扫描。")
    
    time.sleep(wait_seconds)
    scout_papers()
    print("扫描完成。预扫描报告已生成。")
