
import logging
import os
import time
from torch.utils.tensorboard import SummaryWriter

class Logger():
    def __init__(self, logdir, run_name):
        self.log_name = logdir + '/' + run_name
        self.tf_writer = None
        self.start_time = time.time()

        if not os.path.exists(self.log_name):
            os.makedirs(self.log_name)

        self.writer = SummaryWriter(self.log_name)

        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.log_name + '/logger.log'),
                ],
            datefmt='%Y/%m/%d %I:%M:%S %p'
            )

        logging.getLogger().setLevel(logging.INFO)

    def log_value(self, step, value, input_type):
        self.writer.add_scalar(tag=input_type, scalar_value=value, global_step=step)
