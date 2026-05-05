#!/usr/bin/env python
"""DeepResearch-MultiAgent 每日任务执行脚本

用于定时任务调度 (cron / Task Scheduler)，自动执行完整的每日论文处理流水线。

示例:
  # 手动执行
  python scripts/run_daily_task.py

  # Linux cron (每天早上8点)
  0 8 * * * cd /path/to/DeepResearch-MultiAgent && python scripts/run_daily_task.py >> logs/daily.log 2>&1

  # Windows Task Scheduler
  程序: python
  参数: scripts/run_daily_task.py
  起始于: E:\html_code\DeepResearch-MultiAgent
"""
from __future__ import annotations

import sys
import os
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(str(project_root))

from src.main import run_full_pipeline
from src.utils.logger import setup_logging, get_logger
from config.settings import get_settings, Settings


class DailyTaskArgs:
    query = ""
    input = ""


def main():
    settings = get_settings()

    log_dir = settings.log_dir
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"daily_task_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    setup_logging(
        log_level=settings.log_level,
        log_file=str(log_file),
    )

    logger = get_logger("DailyTask")
    logger.info("=" * 70)
    logger.info("DeepResearch-MultiAgent 每日例行任务")
    logger.info(f"日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"日志文件: {log_file}")
    logger.info("=" * 70)

    try:
        run_full_pipeline(DailyTaskArgs())
        logger.info("\n【成功】每日任务全部完成。")
    except Exception as e:
        logger.error(f"\n【失败】每日任务异常: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info(f"任务结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
