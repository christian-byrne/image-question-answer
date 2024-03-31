import time
from pathlib import Path

from constants import VERBOSE
from utils.path_utils import ProjPaths

from termcolor import colored
from typing import Any, Union


class Logger:
    def __init__(self, caller_name, color="green"):
        self.name = caller_name
        self.color = color
        self.prefix_color = f"light_{color}"

        # Create logfile
        hour_time = time.strftime("%H:%M:%S")
        self.filename = f"{caller_name}_{hour_time}.log"
        try:
            self.logfile_path = ProjPaths.get_logs(self.filename)
        except FileNotFoundError:
            open(ProjPaths.get_logs() / Path(self.filename), "w").close()
            self.logfile_path = ProjPaths.get_logs(self.filename)

    def log(
        self,
        message: Any,
        print_only: bool = False,
        color_override: Union[str, bool] = False,
        inline: bool = False,
    ):
        if VERBOSE:
            if isinstance(message, dict):
                message = self.prettify_dict(message)
            elif isinstance(message, list):
                message = self.prettify_list(message)
            self.__print(message, color_override, inline)
        if not print_only:
            self.__write_to_log(message)

    def preview_dataframe(self, df):
        if VERBOSE:
            print(
                colored(
                    f"[{self.name}]", self.prefix_color, "on_black", attrs=["bold"]
                ),
                colored("Previewing the first 5 rows of the dataframe\n", self.color),
            )
            print(df.head())

    def prettify_dict(self, dictionary: dict) -> str:
        return (
            "{\n" + "    \n".join([f"{k}: {v}" for k, v in dictionary.items()]) + "\n}"
        )

    def prettify_list(self, lst: list) -> str:
        return "[\n" + "\n    ".join([str(item) for item in lst]) + "\n]"

    def __write_to_log(self, message):
        with open(self.logfile_path, "a") as f:
            f.write(str(message) + "\n")

    def __print(self, message, color_override=False, inline=False):
        if VERBOSE:
            print(
                colored(
                    f"[{self.name}]", self.prefix_color, "on_black", attrs=["bold"]
                ),
                "\n" if inline else "",
                colored(f"{str(message).strip()}", color_override or self.color),
            )
