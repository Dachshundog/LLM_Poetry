import logging
import os
import matplotlib.pyplot as plt
import json
import random


def setup_logging(log_file_path):
    """
    Set up logging configuration to log messages to a specified file and to the console.

    Args:
        log_file_path (str): The path to the log file.

    Returns:
        logging.Logger: Configured logger object.
    """
    logger = logging.getLogger('custom_logger')
    logger.setLevel(logging.DEBUG)

    # File handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    # Adding handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def append_to_file(value, filename='output.txt'):
    with open(filename, 'a') as file:
        file.write(f"{value}\n")

def read_values(filename='output.txt'):
    with open(filename, 'r') as file:
        values = [float(line.strip()) for line in file]
    return values

def save_dict_to_json(data_dict, filename):
    with open(filename, 'w') as f:
        json.dump(data_dict, f, indent=4, default=str)



if __name__ == '__main__':

    # 使用示例
    new_output_dir = create_output_directory()
    log_file = os.path.join(new_output_dir, 'test.log')
    logger = setup_logging(log_file)

    # 记录示例日志
    logger.debug("This is debug")
    logger.info("This is info")
    logger.error("This is error")

    print(f"Log file created at: {log_file}")

    # 模拟迭代过程
    for i in range(10):
        # 假设这里计算出一个值
        value = i ** 2
        append_to_file(value)

    # 读取文件中的值
    values = read_values()

    # 绘图
    plt.plot(values)
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.title('Iteration Values')
    plt.show()
