import os
import re
import time
import json
import numpy as np
import argparse
import torch
from tqdm import tqdm
import fire
from typing import List

from bigcode_eval.tasks.custom_metrics.code_eval import compute_code_eval
from llama import Llama


def eval_gsm8k(reference: str, result: str, verbose: bool = False):
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            pass
        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass
        return False
    
    def extract_answer_number(completion):
        text = completion.split('The answer is: ')
        if len(text) > 1:
            extract_ans = text[-1].strip()
            match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
            if match:
                if '/' in match.group():
                    denominator = match.group().split('/')[1]
                    numerator = match.group().split('/')[0]
                    if is_number(denominator) == True and is_number(numerator) == True:
                        if denominator == '0':
                            return round(float(numerator.replace(',', '')))
                        else:
                            frac = Fraction(match.group().replace(',', ''))
                            num_numerator = frac.numerator
                            num_denominator = frac.denominator
                            return round(float(num_numerator / num_denominator))
                    else:
                        return None
                else:
                    if float(match.group().replace(',', '')) == float('inf'):
                        return None
                    return round(float(match.group().replace(',', '')))
            else:
                return None
        else:
            return None

    if verbose:
        print(result)
        print("--"*10)
        print(reference)
        print("=="*10)

    gt = reference.split("The answer is: ")
    out = result.split("The answer is: ")
    
    if len(out) != 2:
        return gt[-1], out[-1], 0
    
    gt = gt[-1].replace(" ", "").replace(",", "")
    out = out[-1].replace(" ", "").replace(",", "")

    try:
        gt = eval(gt)
        out = eval(out)
        acc = 1 if abs(gt - out) < 1e-5 else 0
    except:
        acc = 0

    return gt, out, acc


def eval_math(reference: str, result: str, verbose: bool = False):
    def is_equiv(str1, str2, verbose=False):
        if str1 is None and str2 is None:
            print("WARNING: Both None")
            return True
        if str1 is None or str2 is None:
            return False

        return str1 == str2

    if verbose:
        print(result)
        print("--"*10)
        print(reference)
        print("=="*10)

    split_ans = result.split('The answer is: ')
    split_gt = reference.split('The answer is: ')
    answer = split_gt[-1]
    if len(split_ans) > 1:
        ans = split_ans[-1]
        extract_ans_temp = ans.split('.\n')[0]
        extract_ans_temp = extract_ans_temp.strip()
        if len(extract_ans_temp)>0 and extract_ans_temp[-1] == '.':
            extract_ans = extract_ans_temp[0:-1]
        else:
            extract_ans = extract_ans_temp
        extract_ans = extract_ans.strip()

        answer = answer.replace(" ", "")
        extract_ans = extract_ans.replace(" ", "")

        if is_equiv(extract_ans, answer):
            return answer, extract_ans, 1
        else:
            return answer, extract_ans, 0
    else:
        return answer, "", 0


def eval_humaneval(datas, k=[1], verbose=False):
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    def process_output(prompt: str, output: str, entry_point: str) -> str:
        pattern = rf"^def\s+{entry_point}\s*\("  # 匹配行首 def entry_point(
        m = re.search(pattern, prompt, flags=re.M)
        if not m:
            raise ValueError(f"在 prompt 中未找到名为 '{entry_point}' 的函数定义")

        # 保留 prompt 中入口函数定义之前的所有内容
        prefix = prompt[:m.start()].rstrip() + "\n\n"

        # 合并：prefix + output
        merged = prefix + output.strip() + "\n"
        return merged
    
    eval_references = []
    eval_predictions = []
    for data in datas:
        prompt = data["instruction"]
        # prompt = prompt.strip()
        # generations = [prompt + data["response"]]
        # generations = data["response"]
        generations = [process_output(prompt, x, data['entry_point']) for x in data["response"]]
        test_func = data["test"]
        entry_point = f"check({data['entry_point']})"
        references = "\n" + test_func + "\n" + entry_point

        if verbose:
            print(generations[0])
            print("--"*10)
            print(references)
            print("=="*10)

        eval_references.append(references)
        eval_predictions.append(generations)

    result, _ = compute_code_eval(
        predictions=eval_predictions,
        references=eval_references,
        k=k,
        num_workers=30,
        timeout=3.0
    )

    return result


def eval_mbpp(datas, k=[1], verbose=False):
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    
    eval_references = []
    eval_predictions = []
    for data in datas:
        generations = data["response"]
        references = "\n".join(data["test_list"])

        if verbose:
            print(generations[0])
            print("--"*10)
            print(references)
            print("=="*10)

        eval_references.append(references)
        eval_predictions.append(generations)

    result, _ = compute_code_eval(
        predictions=eval_predictions,
        references=eval_references,
        k=k,
        num_workers=30,
        timeout=3.0
    )

    return result


def eval_mmlu(reference: str, result: str):
    gt = reference.strip().lower()
    pred = result.strip().lower()
    acc = 1 if gt == pred else 0

    return gt, pred, acc


def eval_piqa(reference: str, result: str):
    gt = reference.strip().lower()
    pred = result.strip().lower()
    acc = 1 if gt == pred else 0

    return gt, pred, acc


def eval_siqa(reference: str, result: str):
    gt = reference.strip().lower()
    pred = result.strip().lower()
    acc = 1 if gt == pred else 0

    return gt, pred, acc


def eval_obqa(reference: str, result: str):
    gt = reference.strip().lower()
    pred = result.strip().lower()
    acc = 1 if gt == pred else 0

    return gt, pred, acc


eval_handlers = {
    "gsm8k":    eval_gsm8k,
    "math":     eval_math,
    "humaneval":eval_humaneval,
    "mbpp":     eval_mbpp,
    "mmlu":     eval_mmlu,
    "piqa":     eval_piqa,
    "siqa":     eval_siqa,
    "obqa":     eval_obqa,
}


def main(
    dataset_name: str,
    dataset_path: str,
    hf_model_path: str,
    lora_model_path: str,
    finetune_ckpt_path: str,
    model_args_path: str,
    max_seq_len: int = 1024,
    max_gen_len: int = 512,
    max_batch_size: int = 32,
    fixed_point_solver: str = "simple",
    fixed_point_max_iters: int = 8,
    fixed_point_ee_threshold: int = 0.0,
    temperature: float = 0.0,
    top_p: float = 0.9,
    n_samples: int = 1,
    pass_k: List[int] = [1,5,10],
):
    generator = Llama.build(
        hf_model_path=hf_model_path,
        lora_model_path=lora_model_path,
        finetune_ckpt_path=finetune_ckpt_path,
        model_args_path=model_args_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        fixed_point_solver=fixed_point_solver,
        fixed_point_max_iters=fixed_point_max_iters,
        fixed_point_ee_threshold=fixed_point_ee_threshold,
    )

    dataset_name = dataset_name.lower()

    # Load test datasets
    data_path = os.path.join(dataset_path, "test.json")
    with open(data_path, "r") as f:
        datas = json.load(f)
    # datas = datas[:10]

    # Load result handlers for option datasets
    if dataset_name in eval_handlers:
        eval_handler = eval_handlers[dataset_name]
    else:
        raise ValueError(f"Unsupport dataset evaluator for task {dataset_name}!")

    # test
    dataset_size = len(datas)
    cnt = 0
    correct_cnt = 0
    steps_cnt = 0

    if dataset_name in ["gsm8k", "math", "mmlu", "piqa", "siqa", "obqa"]:
        log_array = { "prompt":[], "response":[], "gt":[], "pred":[] }    
        with torch.no_grad():
            with tqdm(range(dataset_size), desc=f"【{dataset_name}】", ncols=160) as pbar:
                for i in pbar:
                    cnt += 1
                                    
                    prompt = datas[i]["instruction"]
                    reference = datas[i]["output"]

                    result = generator.text_completion(
                        [prompt],
                        max_gen_len=max_gen_len,
                        temperature=temperature,
                        top_p=top_p,
                    )[0]

                    # llama.model.clear_kvcache()

                    response = result["generation"]
                    steps = result["logsteps"]

                    gt, pred, acc = eval_handler(reference, response)
                    
                    log_array["prompt"].append(prompt)
                    log_array["response"].append(response)
                    log_array["gt"].append(gt)
                    log_array["pred"].append(pred)

                    correct_cnt += acc

                    if len(steps) > 0:
                        steps_cnt += sum(steps) / len(steps)

                    pbar.set_postfix({
                        'Pred':     f"[{pred}]",
                        'GT':       f"[{gt}]",
                        'Accuracy': f"{'%.3f' % (correct_cnt / cnt)}",
                        'Steps':    f"{'%.3f' % (steps_cnt / cnt)}",
                    })

                    pbar.update(1)

        print(f"Accuracy = {'%.3f' % (correct_cnt / dataset_size)}")
    
    elif dataset_name in ["humaneval", "mbpp"]:
        with torch.no_grad():
            with tqdm(range(dataset_size), desc=f"【{dataset_name}】", ncols=160) as pbar:
                for i in pbar:
                    cnt += 1
                                    
                    prompt = datas[i]["instruction"]

                    result = generator.text_completion(
                        [prompt] * n_samples,
                        max_gen_len=max_gen_len,
                        temperature=temperature,
                        top_p=top_p,
                    )

                    steps = [x["logsteps"] for x in result]

                    # llama.model.clear_kvcache()

                    if len(steps) > 0:
                        steps_mean = [sum(x) / len(x) for x in steps]
                        steps_cnt += sum(steps_mean) / len(steps_mean)

                    response = [r["generation"] for r in result]

                    datas[i]['response'] = response

                    pbar.set_postfix({
                        'Steps':    f"{'%.3f' % (steps_cnt / cnt)}",
                    })

                    pbar.update(1)
            
            result = eval_handler(datas, pass_k, verbose=False)

            print(result)



if __name__ == "__main__":
    fire.Fire(main)