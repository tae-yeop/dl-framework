# from accelerate.logging import get_logger
# import logging
# import os
# from pathlib import Path

# filehandler = logging.FileHandler(f'./training.log')
# logger = get_logger(__name__)

# print(logger.hasHandlers())
# # logger.logger.addHandler(filehandler)
# # logger.logger.info('dsad')
# logger.info('My log', main_process_only=True)
# logger.info('test', main_process_only=False)

# # print(logger)


from accelerate.logging import get_logger
from pathlib import Path
import logging
from accelerate import Accelerator
logger = get_logger(__name__)


if __name__ == '__main__':
    accelerator = Accelerator()
    # setLevel()을 이용하면 이상하게 반영이 안된다.
    # getEffectiveLevel() 상에는 변경이 되는데 실제로 .info() 내용이 나오지 않음
    # logger.setLevel(0)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    print(logger.getEffectiveLevel())

    # logger = logging.getLogger('')
    # logging.logger를 wrapping해서 만들었음
    logger = get_logger(__name__)
    # logger.setLevel(logging.INFO)
    log_path = Path('./test')
    log_path.mkdir(parents=True, exist_ok=True)
    # 기존의 filehandler로 적용하려면 내부에 logger의 method를 호출해서 사용하도록 함
    logger.logger.addHandler(logging.FileHandler(log_path.joinpath('run.log')))
    # logger.logger.addHandler(logging.StreamHandler())

    logger.warning('dsad', main_process_only=True)
    logger.info("***** Running training *****", main_process_only=True)
    logger.info("My log", main_process_only=True)
    logger.debug("My log", main_process_only=True)
    logger.critical('dasd', main_process_only=True)
