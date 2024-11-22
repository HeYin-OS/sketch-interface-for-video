import colorama
from datetime import datetime


def getLoggingString(level, info, isTitle):
    if level == 0:
        level_str = f"{colorama.Fore.CYAN}DEBUG{colorama.Style.RESET_ALL}"
    elif level == 1:
        level_str = f"{colorama.Fore.BLUE}INFO{colorama.Style.RESET_ALL}"
    elif level == 2:
        level_str = f"{colorama.Fore.YELLOW}WARNING{colorama.Style.RESET_ALL}]"
    elif level == 3:
        level_str = f"{colorama.Fore.RED}ERROR{colorama.Style.RESET_ALL}"
    elif level == 4:
        level_str = f"{colorama.Fore.GREEN}SUCCESS{colorama.Style.RESET_ALL}"
    else:
        level_str = f"{colorama.Fore.WHITE}NORMAL{colorama.Style.RESET_ALL}"
    if isTitle:
        info_str = f"<<<<<<<<            {info}            >>>>>>>>"
    else:
        info_str = f"{info}"
    current_time = datetime.utcnow()
    log_str = f"{colorama.Fore.LIGHTGREEN_EX}{current_time}{colorama.Style.RESET_ALL} {level_str} {info_str}"
    return log_str


def printLog(level, info, isTitle):
    """
    输出日志。
    Args:
        level (int): 日志级别。
            0 - DEBUG;
            1 - INFO;
            2 - WARNING;
            3 - ERROR;
            4 - SUCCESS;
            else - NORMAL

        info (str): 日志内容。

        isTitle (bool): 是否为标题。

    Returns:
        无。
    """
    print(getLoggingString(level, info, isTitle))
