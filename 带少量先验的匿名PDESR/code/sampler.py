"""The LLM client used by the search and by the two design steps."""
from __future__ import annotations
import http.client
import json
import os
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Collection, Sequence, Type
from urllib.parse import urlparse

import numpy as np
import requests

import core





class LLM(ABC):
    def __init__(self, samples_per_prompt: int) -> None:
        self._samples_per_prompt = samples_per_prompt

    def _draw_sample(self, prompt: str) -> str:
        raise NotImplementedError('Must provide a language model.')

    @abstractmethod
    def draw_samples(self, prompt: str) -> Collection[str]:
        return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]



class Sampler:
    _global_samples_nums: int = 0

    def __init__(
            self,
            database: core.ExperienceBuffer,
            evaluators: Sequence[evaluator.Evaluator],
            samples_per_prompt: int,
            config: core.Config,
            max_sample_nums: int | None = None,
            llm_class: Type[LLM] = LLM,
    ):
        self._samples_per_prompt = samples_per_prompt
        self._database = database
        self._evaluators = evaluators
        self._llm = llm_class(samples_per_prompt)
        self._max_sample_nums = max_sample_nums
        self.config = config

    
    def sample(self, **kwargs):
        while True:
            if self._max_sample_nums and self.__class__._global_samples_nums >= self._max_sample_nums:
                break
            
            prompt = self._database.get_prompt()
            
            reset_time = time.time()
            samples = self._llm.draw_samples(prompt.code,self.config)
            sample_time = (time.time() - reset_time) / self._samples_per_prompt

            for sample in samples:
                self._global_sample_nums_plus_one()
                cur_global_sample_nums = self._get_global_sample_nums()
                chosen_evaluator: evaluator.Evaluator = np.random.choice(self._evaluators)
                chosen_evaluator.analyse(
                    sample,
                    prompt.island_id,
                    prompt.version_generated,
                    **kwargs,
                    global_sample_nums=cur_global_sample_nums,
                    sample_time=sample_time
                )

    def _get_global_sample_nums(self) -> int:
        return self.__class__._global_samples_nums

    def set_global_sample_nums(self, num):
        self.__class__._global_samples_nums = num

    def _global_sample_nums_plus_one(self):
        self.__class__._global_samples_nums += 1






def _default_base_url(api_model: str) -> str:
    """Original hard-coded providers, kept as the fallback."""
    if "glm" in api_model.lower():
        return "https://open.bigmodel.cn/api/paas/v4"
    return "https://api.deepseek.com"


def _extract_body(sample: str, config: core.Config) -> str:

    # Models often wrap the answer in markdown code fences; those lines break
    # the indentation-aware parsing below, so drop them first.
    sample = '\n'.join(
        line for line in sample.splitlines() if not line.strip().startswith('```')
    )

    lines = sample.splitlines()
    func_body_lineno = 0
    find_def_declaration = False
    
    for lineno, line in enumerate(lines):
        if line[:3] == 'def':
            func_body_lineno = lineno
            find_def_declaration = True
            break
    
    if find_def_declaration:
        if config.use_api:
            code = ''
            for line in lines[func_body_lineno + 1:]:
                code += line + '\n'
        
        else:
            code = ''
            indent = '    '
            for line in lines[func_body_lineno + 1:]:
                if line[:4] != indent:
                    line = indent + line
                code += line + '\n'
        
        return code
    
    return sample



class LocalLLM(LLM):
    def __init__(self, samples_per_prompt: int, batch_inference: bool = True, trim=True) -> None:
        super().__init__(samples_per_prompt)

        url = "http://127.0.0.1:5000/completions"
        instruction_prompt = ("You are a helpful assistant tasked with discovering mathematical function structures for scientific systems. \
                             Complete the 'equation' function below, considering the physical meaning and relationships of inputs.\n\n")
        self._batch_inference = batch_inference
        self._url = url
        self._instruction_prompt = instruction_prompt
        self._trim = trim


    def draw_samples(self, prompt: str, config: core.Config) -> Collection[str]:
        if config.use_api:
            return self._draw_samples_api(prompt, config)
        else:
            return self._draw_samples_local(prompt, config)

    def _draw_samples_api(self, prompt: str, config: core.Config) -> Collection[str]:
        prompt = '\n'.join([self._instruction_prompt, prompt])

        base_url = getattr(config, 'api_base_url', '') or _default_base_url(config.api_model)
        parsed = urlparse(base_url)
        host = parsed.netloc
        path = parsed.path.rstrip('/') + '/chat/completions'

        key_env = getattr(config, 'api_key_env', 'API_KEY') or 'API_KEY'
        api_key = os.environ[key_env]
        max_tokens = getattr(config, 'max_tokens', 512)

        def draw_one(index: int) -> str:
            empty_retries = 0
            consecutive_failures = 0
            while True:
                try:
                    conn = http.client.HTTPSConnection(host, timeout=600)

                    request_body = {
                        "max_tokens": max_tokens,
                        "model": config.api_model,
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    }

                    reasoning_effort = getattr(config, 'reasoning_effort', '')
                    if reasoning_effort and 'deepseek' in config.api_model.lower():
                        request_body['reasoning_effort'] = reasoning_effort

                    thinking = getattr(config, 'thinking', '')
                    if thinking and 'deepseek' in config.api_model.lower():
                        request_body['thinking'] = {'type': thinking}

                    payload = json.dumps(request_body)

                    headers = {
                        'Authorization': f"Bearer {api_key}",
                        'Content-Type': 'application/json'
                    }

                    conn.request("POST", path, payload, headers)

                    res = conn.getresponse()
                    data = json.loads(res.read().decode("utf-8"))

                    if 'choices' not in data:
                        consecutive_failures += 1
                        print(f"🚨 API Error ({config.api_model}) [sample {index}] "
                              f"[{consecutive_failures}/30]: {data}")
                        if consecutive_failures >= 30:
                            raise RuntimeError(
                                f"Giving up after {consecutive_failures} consecutive API "
                                f"errors for model {config.api_model}: {data}")
                        time.sleep(2)
                        continue

                    response = data['choices'][0]['message']['content']

                    if self._trim:
                        response = _extract_body(response, config)

                    if not response.strip():
                        empty_retries += 1
                        finish_reason = data['choices'][0].get('finish_reason')
                        print(f"⚠️ Empty completion [sample {index}] "
                              f"(finish_reason={finish_reason}, max_tokens={max_tokens}), "
                              f"retry {empty_retries}/3")
                        if empty_retries >= 3:
                            return ''
                        continue

                    return response

                except Exception as e:
                    consecutive_failures += 1
                    print(f"❌ API Request failed ({config.api_model}) [sample {index}] "
                          f"[{consecutive_failures}/30]: {e}")
                    if consecutive_failures >= 30:
                        raise
                    time.sleep(2)
                    continue

        workers = getattr(config, 'concurrent_requests', 1) or 1
        workers = max(1, min(workers, self._samples_per_prompt))
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                return list(pool.map(draw_one, range(self._samples_per_prompt)))
        return [draw_one(i) for i in range(self._samples_per_prompt)]
    
    def _do_request(self, content: str) -> str:
        content = content.strip('\n').strip()
        repeat_prompt: int = self._samples_per_prompt if self._batch_inference else 1
        
        data = {
            'prompt': content,
            'repeat_prompt': repeat_prompt,
            'params': {
                'do_sample': True,
                'temperature': None,
                'top_k': None,
                'top_p': None,
                'add_special_tokens': False,
                'skip_special_tokens': True,
            }
        }
        
        headers = {'Content-Type': 'application/json'}
        response = requests.post(self._url, data=json.dumps(data), headers=headers)
        
        if response.status_code == 200:
            response = response.json()["content"]
            
            return response if self._batch_inference else response[0]
