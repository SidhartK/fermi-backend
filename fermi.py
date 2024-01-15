import argparse
from dotenv import load_dotenv
import os
from tqdm import tqdm
import numpy as np

# init hugging face
from openai import OpenAI

from prompts.v1 import QUERY_PROMPT, CONTEXT_PROMPT, ASSISTANT_FIRST_RESP_PROMPT, FEW_SHOT_PROMPTS, SYSTEM_PROMPT_STRING

import json

load_dotenv()

# DATASET_PATH = "./data/verified_data/dataset.json"
# OUTPUT_STORE_PATH = "./results/expmt2/output_store.json"
MODEL_NAME = "gpt-3.5-turbo"
client = OpenAI(
    # This is the default and can be omitted
    api_key=os.getenv("OPENAI_API_KEY"),
)

class SamplePredictor:
    def __init__(self, model_name=MODEL_NAME, temperature=0):
        self.model_name = model_name
        self.temperature = temperature

    def chat(self, messages):
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            temperature=self.temperature,
        )
        return chat_completion.choices[0].message.content

    def eval(self, question, units, preds, answer, context=None, num_tokens=None, verbose=True):
        prediction = {
            "question": question,
            "units": units,
            "correct_answer": answer,
            "raw_outputs": preds,
            "context": "- " + "\n- ".join(context) if context is not None else "",
        }

        try:
            summary = preds.split("SUMMARY:=")[-1].split("PROGRAM:=")[0].strip('\n')
            program = preds.split("PROGRAM:=")[-1].strip('\n')
            program_lines = program.split("\n")
            program_lines = [line if line[0] != 'Q' else "# " + line for line in program_lines]
            program_lines = [line.replace(",", "") if line[0] != "#" else line for line in program_lines]
            program = "\n".join(program_lines)
            loc = {}
            exec(program, globals(), loc)
            compiled_out = loc["A0"]

            prediction = {
                    "summary": summary,
                    "program": program,
                    "compiled_answer": compiled_out,
                    **prediction,
                }

            # compiled_acc = accuracy_metric(answer, compiled_out)
            compiled_acc = 1
            parsable = 1
        except:
            compiled_acc = 0
            parsable = 0
            verbose = True
        if verbose:
            print("Question: {}; \nContext: {}\nCorrect Answer: {}".format(prediction["question"], prediction["context"], prediction["correct_answer"]))
            if num_tokens is not None:
                print("Number of Tokens in Response: ", num_tokens)

            if "compiled_answer" in prediction:
                print("Compiled Answer is: {} \nSummarized Problem is: {}\nProgram:\n```python\n{}\n```".format(prediction['compiled_answer'], prediction['summary'], prediction['program']))
            else:
                print("Unable to parse output; Outputting Raw Output:\n{}".format(prediction["raw_outputs"]))
        return compiled_acc, parsable, prediction


    def parse_output(self, raw_output):
        return {
            "accuracy": raw_output[0],
            "parsable": raw_output[1],
            "pred": raw_output[2],
        }

    def ask(self, question, units, answer=1.0, context=None, distractor_context=None, verbose=False):

        messages = [{"role": "system", "content": SYSTEM_PROMPT_STRING}]
        for few_shot_prompt in FEW_SHOT_PROMPTS:
            user_question = few_shot_prompt.split("User:")[1].split("Assistant:")[0].strip('\n')
            messages.append({"role": "user", "content": user_question})
            assistant_response1 = "\n" + few_shot_prompt.split("Assistant:")[1].split("User:")[0].strip('\n')
            messages.append({"role": "assistant", "content": assistant_response1})
            user_context = few_shot_prompt.split("User:")[2].split("Assistant:")[0].strip('\n')
            messages.append({"role": "user", "content": user_context})
            assistant_response2 = "\n" + few_shot_prompt.split("Assistant:")[2].strip('\n')
            messages.append({"role": "assistant", "content": assistant_response2})
        messages.append({"role": "user", "content": QUERY_PROMPT.format(question=question, units=units)})

        if verbose:
            print(f"\n{'-' * 105}\n{'-' * 45} FIRST MESSAGE {'-' * 45}\n{'-' * 105}\n\n")

        preds = self.chat(messages)

        no_ctxt_raw_output = self.eval(question, units, preds, answer, verbose=verbose)
        noctxt_pred = no_ctxt_raw_output[2]
        reg_ctxt_raw_output,  dstr_ctxt_raw_output = (0, 0, {}), (0, 0, {})

        if context is not None:
            if verbose:
                print(f"\n{'-' * 106}\n{'-' * 45} SECOND MESSAGE {'-' * 45}\n{'-' * 106}\n\n")
            messages.append({"role": "assistant", "content": ASSISTANT_FIRST_RESP_PROMPT.format(summary=noctxt_pred.get("summary", ""), program=noctxt_pred.get("program", ""))})
            messages.append({"role": "user", "content": CONTEXT_PROMPT.format(context="- " + "\n- ".join(context))})
            preds = self.chat(messages)
            reg_ctxt_raw_output = self.eval(question, units, preds, answer, context=context, verbose=verbose)

        if distractor_context is not None:
            if verbose:
                print(f"\n{'-' * 106}\n{'-' * 45} SECOND MESSAGE {'-' * 45}\n{'-' * 106}\n\n")
            messages.append({"role": "assistant", "content": ASSISTANT_FIRST_RESP_PROMPT.format(summary=noctxt_pred.get("summary", ""), program=noctxt_pred.get("program", ""))})
            messages.append({"role": "user", "content": CONTEXT_PROMPT.format(context="- " + "\n- ".join(distractor_context))})
            preds = self.chat(messages)
            dstr_ctxt_raw_output = self.eval(question, units, preds, answer, context=distractor_context, verbose=verbose)

        return (self.parse_output(raw_output) for raw_output in (no_ctxt_raw_output, reg_ctxt_raw_output, dstr_ctxt_raw_output))


    def run(self, dataset, N=None, output_store_path=None, verbose=False):
        N = N if N is not None else len(dataset)

        output_store = []

        for i in tqdm(range(N)):
            entry = dataset[i]

            question = entry["question"]
            units = entry["units"]
            answer = entry["answer"]
            context = entry["context"].split('=')[1:]
            distract_context = entry["distract_context"].split('=')[1:]

            if verbose:
                print("About to go into ask")
            noctxt, regctxt, dstrctxt = self.ask(question, units, answer, context, distract_context, verbose=verbose)

            output_store.append({
                "no-context": noctxt,
                "regular-context": regctxt,
                "distractor-context": dstrctxt,
            })

            if output_store_path is not None:
                with open(output_store_path, 'w') as f:
                    json.dump(output_store, f)

        accuracy = {
            "no-context": 0,
            "regular-context": 0,
            "distractor-context": 0,
        }
        parsable = {
            "no-context": 0,
            "regular-context": 0,
            "distractor-context": 0,
        }

        for output in output_store:
            for key in output:
                accuracy[key] += output[key]["accuracy"]
                parsable[key] += output[key]["parsable"]

        return {k: v/N for k, v in accuracy.items()}, {k: v/N for k, v in parsable.items()}, output_store



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=100, help='Random seed')
    parser.add_argument('--model', type=str, default="gpt-4", help="OpenAI model to use")
    parser.add_argument('--temp', type=float, default=0.0, help="Temperature for sampling")
    parser.add_argument('--question', type=str, default="How many tennis balls are sold each year in New York City?", help="What's your question?")
    parser.add_argument('--units', type=str, default="tennis balls", help="What are the units of the response? Or put dimensionles")
    parser.add_argument('--verbose', action='store_true', help='Should the model be verbose?')
    args = parser.parse_args()
    print("You asked the question: {} (units: {})".format(args.question, args.units))
    np.random.seed(args.seed)

    predictor = SamplePredictor(model_name=args.model, temperature=args.temp)
    noctxt, _, _ = predictor.ask(question=args.question, units=args.units, verbose=args.verbose)

    prediction = noctxt["pred"]
    print("I think the answer is: {:.2e} {}\n\n".format(prediction["compiled_answer"], args.units))
    print("Here is my reasoning:\n```python\n{}\n```".format(prediction["program"]))