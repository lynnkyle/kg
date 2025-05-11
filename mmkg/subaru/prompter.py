import os
import json


class Prompter(object):
    __slot__ = ("template", "_verbose")

    def __init__(self, template_name="", verbose=False):
        self._verbose = verbose
        if not template_name:
            template_name = "alpaca"
        file_name = os.path.join("templates", f"{template_name}.json")
        if not os.path.exists(file_name):
            raise ValueError(f"Template file{template_name} is not found")
        with open(file_name) as fp:
            self.template = json.load(fp)

        if self._verbose:
            print(f"Using prompt template: {template_name}: {self.template['description']}")

    def generate_prompt(self, instruction, input=None, label=None):
        if input:
            res = self.template['prompt_input'].format(instruction=instruction, input=input)
        else:
            res = self.template['prompt_no_input'].format(instruction=instruction)
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res


if __name__ == '__main__':
    prompter = Prompter(template_name="alpaca_short", verbose=True)
    prompter.generate_prompt('Who is my son?', None, "zhz")
