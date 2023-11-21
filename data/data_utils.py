import random
import json
import os
from gaitor_function_calling.data.prompting_utils import INSTRUCTION, function_calling_tokens, build_prompt
cwd = os.getcwd()
data_dir = os.path.join(cwd, 'data')

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

    def build_data(self, instruction = None, randomize=False, train_test_split=0.95):
        data = self.raw_data
        modified_data = []
        for idx, instance in enumerate(data):
            try:
                prompt = build_prompt(instance, instruction)
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

    def save(self, raw, train, test):
        if not os.path.exists(self.paths['train_dir']):
            os.makedirs(self.paths['train_dir'])
        
        with open(os.path.join(self.paths['train_dir'], "raw.json"), 'w') as f:
            json.dump(raw, f)
        with open(self.paths['train'], 'w') as f:
            json.dump(train, f)
        with open(self.paths['test'], 'w') as f:
            json.dump(test, f)
        
        return 1
    


