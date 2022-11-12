import logging
import os
from pathlib import Path

logger = logging.getLogger()
# level을 반드시 설정해야함
# 예를 들어 .info를 사용하고 싶다면 setLevel(logging.INFO) 반드시 필요
logger.setLevel(logging.INFO)
# 폴더가 반드시 존재해야 함
# 아니면 에러가 남
# logger.addHandler(logging.FileHandler('./test/run.log'))


log_path = Path('./test')
log_path.mkdir(parents=True, exist_ok=True)
logger.addHandler(logging.FileHandler(log_path.joinpath('run.log')))
# logger.addHandler(logging.StreamHandler())
logger.info('testset')