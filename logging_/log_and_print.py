import time
from termcolor import colored

from constants import VERBOSE


class Logger:
    def __init__(self, caller_name, color="green"):
        self.name = caller_name

        hour_time = time.strftime("%H:%M:%S")
        self.log_file = f"logs/{caller_name}_{hour_time}.log"

        self.color = color
        self.prefix_color = f"light_{color}"

    def log(self, message, print_only=False, color_override="green", inline=False):
        if VERBOSE:
            self.__print(message, color_override, inline)
        if not print_only:
            self.__write_to_log(message)

    def preview_df(self, df):
        if VERBOSE:
            print(
                colored(
                    f"[{self.name}]", self.prefix_color, "on_black", attrs=["bold"]
                ),
                colored("Previewing the first 5 rows of the dataframe\n", self.color),
            )
            print(df.head())

    def __write_to_log(self, message):
        with open(self.log_file, "a") as f:
            f.write(str(message) + "\n")

    def __print(self, message, color_override=None, inline=False):
        if VERBOSE:
            print(
                colored(
                    f"[{self.name}]", self.prefix_color, "on_black", attrs=["bold"]
                ),
                "\n" if inline else "",
                colored(f"{str(message).strip()}", color_override or self.color),
            )
