import os
from datetime import datetime
import tensorflow as tf
import logging
from pathlib import Path, PurePath


class Logger:
    def __init__(self,
                 log_interval: int = 100,
                 log_name: str = "logfile.log",
                 tb_name: str = "",
                 log_dir: str = "logs",
                 tb_dir: str = "tensorboard",
                 use_tb: bool = True):
        """
        :param log_interval (int): log progress every N steps
        :param log_name (str): file to save log data
        :param tb_name (str): file to save tensorboard data
        :param log_dir (str): path to log_name
        :param tb_dir (str): path to tb_name
        :param use_tb (bool): use tensorboard or not
        """
        self.time_ext = datetime.now().strftime("%Y%m%d-%H%M")
        self.log_interval = log_interval
        self.use_tb = use_tb

        # create logger
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        # create file handler
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, log_name))
        fh.setLevel(logging.DEBUG)

        # create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # create formatter
        formatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s - %(message)s')

        # add formatter to ch
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        # add ch and fh to logger
        self.logger.addHandler(ch)
        self.logger.addHandler(fh)

        if self.use_tb:
            tb_dir = PurePath(__file__).parts[0] if tb_dir is None else dir
            self.tb_dir = Path(tb_dir) / "tensorboard" / tb_name.format(self.time_ext)
            Path(self.tb_dir).mkdir(parents=True, exist_ok=True)
            self.tb_writer = tf.summary.create_file_writer(self.tb_dir)

    def set_lvl(self):
        pass

    def tb_write(self, val, step, title="Average Training Reward"):
        with self.tb_writer.as_default():
            tf.summary.scalar(title, val, step=step)

    def info(self, info):
        self.logger.info(info)

    def _end_train(self, step):
        print("Reached the end of training with {} training steps".format(step))
