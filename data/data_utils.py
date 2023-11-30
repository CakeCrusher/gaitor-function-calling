import random
import json
import os
from gaitor_function_calling.data.prompting_utils import INSTRUCTION, function_calling_tokens, build_prompt
cwd = os.getcwd()
data_dir = os.path.join(cwd, 'data/train_test')



class DataAbstractor():
    def __init__(self, path_or_name, identifier=None):
        self.identifier = identifier
        self.path_or_name = path_or_name
        self.paths = self.get_path()
        self.raw_data = None
        self.train_data = None
        self.test_data = None
        self.get_data()


    def __str__(self):
        str_to_print = "EXAMPLES:\n\n"
        if self.raw_data:
            str_to_print += f"raw_data:\n{json.dumps(self.raw_data[random.randint(0, len(self.raw_data))], indent=4)}\n\n"
        if self.train_data:
            str_to_print += f"train_data:\n{json.dumps(self.train_data[random.randint(0, len(self.train_data))], indent=4)}\n\n"
        if self.test_data:
            str_to_print += f"test_data:\n{json.dumps(self.test_data[random.randint(0, len(self.test_data))], indent=4)}\n\n"
        return str_to_print

    def get_path(self):
        if data_dir in self.path_or_name:
            path = self.path_or_name
        else:
            path = os.path.join(data_dir, self.path_or_name)
        if f"-train-{self.identifier}" in path:
            train_dir = os.path.dirname(path)
        else:
            train_dir = os.path.join(os.path.dirname(path), f"{os.path.basename(path).split('.')[0]}-train-{self.identifier}")
        return {
            "root": os.path.dirname(path),
            "raw": path,
            "train_dir": train_dir,
            "train": os.path.join(train_dir, "train.json"),
            "test": os.path.join(train_dir, "test.json"),
        }

    def get_data(self):
        try:
            with open(self.paths['raw'], 'r') as f:
                self.raw_data = json.load(f)
        except:
            print("No raw data found")
        try:
            with open(self.paths['train'], 'r') as f:
                self.train_data = json.load(f)
        except:
            print("No train data found")
        try:
            with open(self.paths['test'], 'r') as f:
                self.test_data = json.load(f)
        except:
            print("No test data found")

    def build_data(self, instruction = None, randomize=False, train_test_split=0.95, shots = None):
        data = self.raw_data
        modified_data = []
        for idx, instance in enumerate(data):
            try:
                prompt = build_prompt(instance, instruction, shots)
                modified_data.append({"text": prompt})
            except:
                print(f"{idx} Error building prompt")
        if randomize:
            random.shuffle(modified_data)
        split_index = int(len(modified_data) * train_test_split)
        train_data = modified_data[:split_index]
        test_data = modified_data[split_index:]
        print(f"Train data size: {len(train_data)}\nTest data size: {len(test_data)}")
        return train_data, test_data

    def save(self, raw = None, train = None, test = None):
        if not raw and not train and not test:
            print("No data to save")
            return 0
        
        if not os.path.exists(self.paths['train_dir']):
            os.makedirs(self.paths['train_dir'])

        if raw:            
            with open(os.path.join(self.paths['train_dir'], "raw.json"), 'w') as f:
                json.dump(raw, f)
            print(f"Raw data saved to {os.path.join(self.paths['train_dir'], 'raw.json')}")
        if train:
            with open(self.paths['train'], 'w') as f:
                json.dump(train, f)
            print(f"Train data saved to {self.paths['train']}")
        
        if test:
            with open(self.paths['test'], 'w') as f:
                json.dump(test, f)
            print(f"Test data saved to {self.paths['test']}")
        
        return 1
    
def build_data_dpo(data, randomize=False, train_test_split=0.98, skip_callback = None):
    """
    Build the data in the following format:
    {
        "prompt": [str],
        "chosen": [str],
        "rejected": [str],
    }
    """
    modified_data = {"prompt": [], "chosen": [], "rejected": []}
    skipped = []

    if randomize:
        random.shuffle(data)

    for idx, instance in enumerate(data):
        try:
            if skip_callback and skip_callback(instance):
                skipped.append(instance)
                continue

            divided_expected_prompt = instance["expected_str"].split("[/INST]")
            prompt = divided_expected_prompt[0] + "[/INST]"
            expected_output = divided_expected_prompt[1]
            divided_generated_prompt = instance["generated_str"].split("[/INST]")
            generated_output = divided_generated_prompt[1]

            modified_data["prompt"].append(prompt)
            modified_data["chosen"].append(expected_output)
            modified_data["rejected"].append(generated_output)
        except:
            print(f"{idx} Error building prompt")
    print(f"Skipped {len(skipped)} items")

    split_index = int(len(modified_data["prompt"]) * train_test_split)
    train_data = {
        "prompt": modified_data["prompt"][:split_index],
        "chosen": modified_data["chosen"][:split_index],
        "rejected": modified_data["rejected"][:split_index],
    }
    test_data = {
        "prompt": modified_data["prompt"][split_index:],
        "chosen": modified_data["chosen"][split_index:],
        "rejected": modified_data["rejected"][split_index:],
    }
    
    return train_data, test_data

