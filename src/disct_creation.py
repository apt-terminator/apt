

import openai
import time
import numpy as np
import pandas as pd
import json
from openai import OpenAI
from openai import OpenAIError, RateLimitError
import json

from openai._exceptions import RateLimitError
input="./data/pandex/clearscope/ProcessAll.csv"
groundtruth="./data/pandex/clearscope/clearscope_pandex_merged.csv"
# Find positions of true positives in the DataFrame
_processes = pd.read_csv(input)

_labels_df = pd.read_csv(groundtruth)
_apt_list = _labels_df.loc[_labels_df["label"] == "AdmSubject::Node"]["uuid"]
APT_positions = _processes[_processes['Object_ID'].isin(_apt_list)].index.tolist()
outlier_indices=APT_positions

pe_df = pd.read_csv("./data/pandex/clearscope/ProcessEvent.csv")  # Binary features: one-hot system events
px_df = pd.read_csv("./data/pandex/clearscope/ProcessExec.csv")  # One-hot encoded executables
pp_df = pd.read_csv("./data/pandex/clearscope/ProcessParent.csv")  # Parent → Child binary matrix
pn_df = pd.read_csv("./data/pandex/clearscope/ProcessNetflow.csv")  # One-hot encoded connections (e.g., ports/IPs)

df = pd.DataFrame(_processes)
list_of_executable_names = list(px_df.columns)[1:]





client = OpenAI(api_key="sk-proj-PUT YOUR KEY HERE")  # or omit this if using env var



# Dictionary to store translations
exec_translation_dict = {}

# Translation function (adapted)
def translate_px_sentence(px_sentence: str, source_os="Windows", target_os="Android", max_retries=3):
    prompt = f"""Translate only the executable name in the following sentence from {source_os} to a semantically equivalent executable in {target_os}.
Do not rewrite or rephrase the rest of the sentence. 
Example:
 "cmd.exe."
 "/bin/bash"


 

Sentence:
"{px_sentence}"
"""

    for _ in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            raw_output = response.choices[0].message.content.strip()

            if "->" in raw_output:
                translated = raw_output.split("->", 1)[-1].strip().strip('"')
            else:
                translated = raw_output.strip().strip('"')

            return translated

        except RateLimitError:
            print("Rate limit hit, retrying...")
            time.sleep(2)

        except APIError as e:
            return f"APIError: {e}"

        except Exception as e:
            return f"Unhandled error: {e}"

    return "TRANSLATION_FAILED"

# Run translation and build dictionary
import re
import os
 


def get_executable_or_original(cmd):
    if 'exe' in cmd.lower():
        match = re.search(r'([\w\d_\-]+\.exe)', cmd, re.IGNORECASE)
        if match:
            return match.group(1)
    return cmd

for exec_name in list_of_executable_names:
    original_exec=exec_name
    print(f"Translating: {original_exec}")
    exec_name = get_executable_or_original(exec_name)
    print(f"Translating: {exec_name}")
    translation = translate_px_sentence(exec_name)
    # Step 1: Extract the executable path
    exe_path = original_exec.split()[0]
    exe_name = os.path.basename(exe_path).lower()
    bsd_exec =translation
    uuid_match = re.search(r'[\d\-]{20,}', original_exec)
        # Step 4: Append to BSD exec path if UUID is found
    if uuid_match:
        uuid = uuid_match.group()
        translation= f"{bsd_exec} /tmp/.X11-unix/{uuid}"
    else:
        translation= bsd_exec

    print(f"So Translating: {original_exec}")
    print(f"Is : {translation}")
    exec_translation_dict[original_exec] = translation
    time.sleep(1)  # optional throttle

# Save the dictionary
with open("./Windows_to_Android_Pandex_exec_translation_dict.json", "w") as f:
    json.dump(exec_translation_dict, f, indent=2)


