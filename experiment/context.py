import os


class ExperimentContext:
    log_output_dir = None

    def init(self, path):
        self.log_output_dir = path
        os.system(f'mkdir -p {self.log_output_dir}')

experiment_context = ExperimentContext()
