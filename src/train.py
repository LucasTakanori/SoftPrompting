from arg_parser import ArgsParser
import logging


#region logging
# Logging
# -------
# Set logging config
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger_formatter = logging.Formatter(
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt = '%y-%m-%d %H:%M:%S',
    )

# Set a logging stream handler
logger_stream_handler = logging.StreamHandler()
logger_stream_handler.setLevel(logging.INFO)
logger_stream_handler.setFormatter(logger_formatter)

# Add handlers
logger.addHandler(logger_stream_handler)
#endregion


class Trainer():
    def __init__(self, trainer_params):
        self.trainer_params = trainer_params

    def main(self):
        print("Training with the following parameters:")
        for key, value in self.trainer_params.items():
            print(f"{key}: {value}")

def main():
    args_parser = ArgsParser()
    args_parser()
    trainer_params = args_parser.arguments

    trainer = Trainer(trainer_params)
    trainer.main()

if __name__=="__main__":
    main()