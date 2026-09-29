from __future__ import annotations

import os.path
from typing import List, Dict
import logging
import json
import code_manipulation
from torch.utils.tensorboard import SummaryWriter


class Profiler:
    def __init__(
            self,
            log_dir: str | None = None,
            pkl_dir: str | None = None,
            max_log_nums: int | None = None,
    ):

        logging.getLogger().setLevel(logging.INFO)
        self._log_dir = log_dir
        if log_dir:
            self._writer = SummaryWriter(log_dir=log_dir)
            self._json_file_path = os.path.join(log_dir, 'all_samples_history.jsonl')
            with open(self._json_file_path, 'w', encoding='utf-8') as f:
                pass
        else:
            self._json_file_path = None
        # ======================================================================

        self._max_log_nums = max_log_nums
        self._num_samples = 0
        self._cur_best_program_sample_order = None
        self._cur_best_program_score = -99999999
        self._cur_best_program_str = None
        self._evaluate_success_program_num = 0
        self._evaluate_failed_program_num = 0
        self._tot_sample_time = 0
        self._tot_evaluate_time = 0
        self._all_sampled_functions: Dict[int, code_manipulation.Function] = {}

        if log_dir:
            self._writer = SummaryWriter(log_dir=log_dir)

        self._each_sample_best_program_score = []
        self._each_sample_evaluate_success_program_num = []
        self._each_sample_evaluate_failed_program_num = []
        self._each_sample_tot_sample_time = []
        self._each_sample_tot_evaluate_time = []

    def _write_tensorboard(self):
        if not self._log_dir:
            return

        self._writer.add_scalar(
            'Best Score of Function',
            self._cur_best_program_score,
            global_step=self._num_samples
        )
        self._writer.add_scalars(
            'Legal/Illegal Function',
            {
                'legal function num': self._evaluate_success_program_num,
                'illegal function num': self._evaluate_failed_program_num
            },
            global_step=self._num_samples
        )
        self._writer.add_scalars(
            'Total Sample/Evaluate Time',
            {'sample time': self._tot_sample_time, 'evaluate time': self._tot_evaluate_time},
            global_step=self._num_samples
        )
        if self._cur_best_program_str is not None:
            self._writer.add_text(
                'Best Function String',
                self._cur_best_program_str,
                global_step=self._num_samples
            )
        if self._cur_best_program_str is not None:
            self._writer.add_text(
                'Best Function String',
                self._cur_best_program_str,
                global_step=self._num_samples
            )
        # # Log the function_str
        # self._writer.add_text(
        #     'Best Function String',
        #     self._cur_best_program_str,
        #     global_step=self._num_samples
        # )

    def _write_json(self, programs: code_manipulation.Function):
        if not self._json_file_path:
            return

        sample_order = programs.global_sample_nums
        sample_order = sample_order if sample_order is not None else 0

        mse = getattr(programs, 'mse', None)
        complexity = getattr(programs, 'complexity', None)

        content = {
            'sample_order': sample_order,
            'mse': mse,
            'complexity': complexity,
            'score': programs.score,
            'function': str(programs).strip()
        }

        with open(self._json_file_path, 'a', encoding='utf-8') as json_file:
            json.dump(content, json_file, ensure_ascii=False)
            json_file.write('\n')

    def register_function(self, programs: code_manipulation.Function):
        if self._max_log_nums is not None and self._num_samples >= self._max_log_nums:
            return

        sample_orders: int = programs.global_sample_nums
        if sample_orders not in self._all_sampled_functions:
            self._num_samples += 1
            self._all_sampled_functions[sample_orders] = programs
            self._record_and_verbose(sample_orders)
            self._write_tensorboard()
            self._write_json(programs)

    def _record_and_verbose(self, sample_orders: int):
        function = self._all_sampled_functions[sample_orders]
        function_str = str(function).strip('\n')
        mse_str = f"{function.mse:.2e}" if hasattr(function, 'mse') and function.mse is not None else "N/A"
        comp_str = str(function.complexity) if hasattr(function,
                                                       'complexity') and function.complexity is not None else "N/A"
        sample_time = function.sample_time
        evaluate_time = function.evaluate_time
        score = function.score
        print(f'================= Evaluated Function =================')
        print(f'{function_str}')
        print(f'------------------------------------------------------')
        print(f'MSE          : {mse_str}')
        print(f'Complexity   : {comp_str}')
        print(f'Sample time  : {str(sample_time)}')
        print(f'Evaluate time: {str(evaluate_time)}')
        print(f'Sample orders: {str(sample_orders)}')
        print(f'======================================================\n\n')

        if function.score is not None and score > self._cur_best_program_score:
            self._cur_best_program_score = score
            self._cur_best_program_sample_order = sample_orders
            self._cur_best_program_str = function_str

        if score:
            self._evaluate_success_program_num += 1
        else:
            self._evaluate_failed_program_num += 1

        if sample_time:
            self._tot_sample_time += sample_time
        if evaluate_time:
            self._tot_evaluate_time += evaluate_time

