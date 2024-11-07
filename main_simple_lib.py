# General imports and variables, as well as config
import ast
import math
import sys
import time

import requests
import torch.multiprocessing as mp
from joblib import Memory
from rich.console import Console
from rich.live import Live
from rich.padding import Padding
from rich.pretty import pprint
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich import print
from rich.markup import escape as rich_escape

from IPython.display import update_display, clear_output, display
from PIL import Image
import matplotlib.pyplot as plt

from configs import config
from utils import show_single_image

from IPython.display import update_display, clear_output
from IPython.core.display import HTML

cache = Memory('cache/' if config.use_cache else None, verbose=0)

mp.set_start_method('spawn', force=True)
from vision_processes import forward, finish_all_consumers  # This import loads all the models. May take a while
from image_patch import *
from video_segment import *
from local_datasets.or_multimodal import ORmultimodalDataset

console = Console(highlight=True, force_terminal=False)

time_wait_between_lines = 0.5


def inject_saver(code, show_intermediate_steps, syntax=None, time_wait_between_lines=None, console=None):
    """
    Injects code to display intermediate steps during execution.
    """
    injected_function_name = 'show_all'
    if injected_function_name in code:
        return code
    code = code.split("\n")
    newcode = []
    for n, codeline in enumerate(code):
        codeline, indent = split_codeline_and_indent_level(codeline)

        if codeline.startswith('#') or codeline == '':  # this will cause issues if you have lots of comment lines
            continue
        if '#' in codeline:
            codeline = codeline.split('#')[0]

        thing_to_show, code_type = get_thing_to_show_codetype(codeline)

        if code_type in ('assign', 'append', 'if', 'return', 'for', 'sort', 'add'):
            if '\'' in codeline:
                codeline.replace('\'', '\\\'')

            if show_intermediate_steps:
                escape_thing = lambda x: x.replace("'", "\\'")
                injection_string_format = \
                    lambda \
                        thing: f"{indent}{injected_function_name}(lineno={n},value=({thing}),valuename='{escape_thing(thing)}'," \
                               f"fig=my_fig,console_in=console,time_wait_between_lines=time_wait_between_lines); " \
                               f"CodexAtLine({n},syntax=syntax,time_wait_between_lines=time_wait_between_lines)"
            else:
                injection_string_format = lambda thing: f"{indent}CodexAtLine({n},syntax=syntax," \
                                                        f"time_wait_between_lines=time_wait_between_lines)"

            extension_list = []
            if isinstance(thing_to_show, list):
                injection_string_list = [injection_string_format(f"{thing}") for thing in thing_to_show]
                extension_list.extend(injection_string_list)
            elif code_type == 'for':
                injection_string = injection_string_format(f"{thing_to_show}")
                injection_string = " " * 4 + injection_string
                extension_list.append(injection_string)
            else:
                extension_list.append(injection_string_format(f"{thing_to_show}"))

            if code_type in ('if', 'return'):
                extension_list = extension_list + [f"{indent}{codeline}"]
            else:
                extension_list = [f"{indent}{codeline}"] + extension_list

            newcode.extend(extension_list)

        elif code_type == 'elif_else':
            newcode.append(f"{indent}{codeline}")
        else:
            newcode.append(f"{indent}{codeline}")
    return "\n".join(newcode)


def get_thing_to_show_codetype(codeline):
    # can output either a list of things to show, or a single thing to show
    things_to_show = []
    if codeline.startswith("if"):
        condition, rest = codeline[3:].split(":", 1)
        codeline = f"if {condition}:{rest}"
        code_type = "if"

        operators = ['==', '!=', '>=', '<=', '>', '<']
        things_to_show = []
        for op in operators:
            if op in condition:
                things_to_show = [x.strip() for x in condition.split(op)]
                # print(things_to_show)
                break
        # things_to_show.append(thing_to_show)
        thing_to_show = things_to_show + [condition.strip()]

    elif codeline.startswith("for"):
        code_type = 'for'
        thing_to_show = codeline.split("for ")[1].split(" in ")[0]

    elif codeline.startswith("return"):
        thing_to_show = codeline.split("return ")[1]
        code_type = 'return'

    elif ' = ' in codeline:
        code_type = 'assign'
        thing_to_show = codeline.split(' = ')[0]
    elif ' += ' in codeline:
        code_type = 'assign'
        thing_to_show = codeline.split(' += ')[0]
    elif ' -= ' in codeline:
        code_type = 'assign'
        thing_to_show = codeline.split(' -= ')[0]
    elif ' *= ' in codeline:
        code_type = 'assign'
        thing_to_show = codeline.split(' *= ')[0]
    elif ' /= ' in codeline:
        code_type = 'assign'
        thing_to_show = codeline.split(' /= ')[0]

    elif '.append(' in codeline:
        code_type = 'append'
        thing_to_show = codeline.split('.append(')[0] + '[-1]'
    elif '.add(' in codeline:
        code_type = 'add'
        thing_to_show = codeline.split('.add(')[0]

    elif '.sort(' in codeline:
        code_type = 'sort'
        thing_to_show = codeline.split('.sort(')[0]

    elif codeline.startswith("elif") or codeline.startswith("else"):
        thing_to_show = None
        code_type = 'elif_else'
    else:
        thing_to_show = None
        code_type = 'other'

    if isinstance(thing_to_show, list):
        thing_to_show = [thing if not (thing.strip().startswith("'") and thing.strip().endswith("'"))
                         else thing.replace("'", '"') for thing in thing_to_show if thing is not None]
    elif isinstance(thing_to_show, str):
        thing_to_show = thing_to_show if not (thing_to_show.strip().startswith("'") and
                                              thing_to_show.strip().endswith("'")) else thing_to_show.replace("'", '"')
    return thing_to_show, code_type


def split_codeline_and_indent_level(codeline):
    origlen = len(codeline)
    codeline = codeline.lstrip()
    indent = origlen - len(codeline)
    indent = " " * indent
    return codeline, indent


def show_one_image(image, ax):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()
        if image.dtype == torch.float32:
            image = image.clamp(0, 1)
        image = image.squeeze(0).permute(1, 2, 0)
    ax.imshow(image)


def CodexAtLine(lineno, syntax, time_wait_between_lines=1.):
    """
    Highlight a specific line in the syntax object and wait for a specified amount of time.
    """
    syntax._stylized_ranges = []
    syntax.stylize_range('on red', (lineno + 1, 0), (lineno + 1, 80))
    time.sleep(time_wait_between_lines)


def show_all(lineno, value, valuename, fig=None, usefig=True, disp=True, console_in=None, time_wait_between_lines=None,
             lastlineno=[-1]):
    """
    Visualizes the line-by-line outputs when executing the LLM-generated code. 
    """
    time.sleep(0.1)  # to avoid race condition!

    if console_in is None:
        console_in = console

    thing_to_show = value

    if lineno is not None and lineno != lastlineno[0]:
        console_in.rule(f"[bold]Line {lineno}[/bold]", style="chartreuse2")
        lastlineno[0] = lineno  # ugly hack

    if usefig:
        plt.clf()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xticks([])
        ax.set_yticks([])
    if isinstance(thing_to_show, Image.Image):
        if valuename:
            console_in.print(f'{rich_escape(valuename)} = ')
        show_one_image(thing_to_show, ax)
    elif str(type(thing_to_show)) == "<class 'image_patch.ImagePatch'>":
        if valuename:
            console_in.print(f'{rich_escape(valuename)} = ')
        show_one_image(thing_to_show.cropped_image, ax)
    elif isinstance(thing_to_show, list) or isinstance(thing_to_show, tuple):
        if len(thing_to_show) > 0:
            for i, thing in enumerate(thing_to_show):
                disp_ = disp or i < len(thing_to_show) - 1
                show_all(None, thing, f"{rich_escape(valuename)}[{i}]", fig=fig, disp=disp_, usefig=usefig)
            return
        else:
            console_in.print(f"{rich_escape(valuename)} is empty")
    elif isinstance(thing_to_show, dict):
        if len(thing_to_show) > 0:
            for i, (thing_k, thing_v) in enumerate(thing_to_show.items()):
                disp_ = disp or i < len(thing_to_show) - 1
                show_all(None, thing_v, f"{rich_escape(valuename)}['{thing_k}']", fig=fig, disp=disp_, usefig=usefig)
            return
        else:
            console_in.print(f"{rich_escape(valuename)} is empty")
    else:
        console_in.print(f"{rich_escape(valuename)} = {thing_to_show}")
        if time_wait_between_lines is not None:
            time.sleep(time_wait_between_lines / 2)
        return

    # display small
    if usefig:
        fig.set_size_inches(2, 2)
        if disp:
            display(fig)


def load_image(path):
    if path.startswith("http://") or path.startswith("https://"):
        image = Image.open(requests.get(path, stream=True).raw).convert('RGB')
        image = transforms.ToTensor()(image)
    else:
        with open(path, "rb") as f:
            image = Image.open(f)
            image = transforms.ToTensor()(image)
    return image


def get_code(query):
    model_name_codex = config.codex.model
    code = forward(model_name_codex, prompt=query, input_type="image")
    assert code.startswith("def execute_command")
    code = ast.unparse(ast.parse(code))
    code_for_syntax = code.replace("(image, my_fig, time_wait_between_lines, syntax)", "(image)")
    syntax = Syntax(code_for_syntax, "python", theme="monokai", line_numbers=True, start_line=0, word_wrap=True)
    console.print(syntax)
    return code, syntax


def execute_code(code, im, show_intermediate_steps=True, visualize=False):
    code, syntax = code
    # restricted execution environment
    if not visualize:
        restricted_globals = {
            "__builtins__": {
                    "str": str,
                    "int": int,
                    "float": float,
                    "list": list,
                    "dict": dict,
                    "tuple": tuple,
                    "set": set,
                    "bool": bool,
                    "type": type,
                    "isinstance": isinstance,
                    "abs": abs,
                    "max": max,
                    "min": min,
                    "sum": sum,
                    "sorted": sorted,
                    "len": len,
                    "enumerate": enumerate,
                    "range": range,
                    "any": any
                },
                "math": math,
                "ImagePatch": ImagePatch,
                "best_image_match": best_image_match,
                "distance": distance,
                "bool_to_yesno": bool_to_yesno,
                "llm_query": llm_query,
                "process_guesses": process_guesses,
                "coerce_to_numeric": coerce_to_numeric,
                "VideoSegment": VideoSegment,
            }
        exec(compile(code, 'Codex', 'exec'), restricted_globals)
        # execution of LLM-generated code done here
        result = restricted_globals["execute_command"](im)
        return result
    else:
        code_line = inject_saver(code, show_intermediate_steps, syntax, time_wait_between_lines, console)

        display(HTML("<style>.output_wrapper, .output {height:auto !important; max-height:1000000px;}</style>"))

        with Live(Padding(syntax, 1), refresh_per_second=0.5, console=console, auto_refresh=True) as live:
            my_fig = plt.figure(figsize=(4, 4))

            restricted_globals_for_vis = {
                "__builtins__": {
                    "str": str,
                    "int": int,
                    "float": float,
                    "list": list,
                    "dict": dict,
                    "tuple": tuple,
                    "set": set,
                    "bool": bool,
                    "type": type,
                    "isinstance": isinstance,
                    "abs": abs,
                    "max": max,
                    "min": min,
                    "sum": sum,
                    "sorted": sorted,
                    "len": len,
                    "enumerate": enumerate,
                    "range": range,
                    "any": any
                },
                "math": math,
                "ImagePatch": ImagePatch,
                "best_image_match": best_image_match,
                "distance": distance,
                "bool_to_yesno": bool_to_yesno,
                "llm_query": llm_query,
                "process_guesses": process_guesses,
                "coerce_to_numeric": coerce_to_numeric,
                "VideoSegment": VideoSegment,
                "show_all": show_all,
                "CodexAtLine": CodexAtLine,
                "my_fig": my_fig,
                "time_wait_between_lines": time_wait_between_lines,
                "syntax": syntax,
                "console": console,
            }
            # overwrites execute_command() with the LLM-generated code, but does not call the method
            exec(compile(code_line, 'Codex', 'exec'), restricted_globals_for_vis)
            # execution of LLM-generated code done here
            result = restricted_globals_for_vis["execute_command"](im)

            plt.close(my_fig)

        def is_not_fig(x):
            if x is None:
                return True
            elif isinstance(x, str):
                return True
            elif isinstance(x, float):
                return True
            elif isinstance(x, int):
                return True
            elif isinstance(x, list) or isinstance(x, tuple):
                return all([is_not_fig(xx) for xx in x])
            elif isinstance(x, dict):
                return all([is_not_fig(xx) for xx in x.values()])
            return False

        f = None
        usefig = False
        if not is_not_fig(result):
            f = plt.figure(figsize=(4, 4))
            usefig = True

        console.rule(f"[bold]Final Result[/bold]", style="chartreuse2")
        show_all(None, result, 'Result', fig=f, usefig=usefig, disp=False, console_in=console, time_wait_between_lines=0)
        return result


def show_single_image(im):
    im = Image.fromarray((im.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype("uint8"))
    im.copy()
    im.thumbnail((400, 400))
    display(im)


def execute_command(*args, **kwargs):
    raise NotImplementedError("This is a placeholder function. The actual implementation will be provided dynamically at runtime.")