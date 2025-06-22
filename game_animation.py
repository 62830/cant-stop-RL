import time
import os
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Play a text animation from a file.")
    parser.add_argument("file_path", type=str, help="Path to the text file containing the animation.")
    parser.add_argument("-l", "--lines_per_step", type=int, default=20, help="Number of lines to display per step.")
    parser.add_argument("-d", "--delay", type=float, default=0.5, help="Delay in seconds between steps.")
    return parser.parse_args()

def play_text_animation(file_path, lines_per_step=20, delay=0.5):
    with open(file_path, "r") as f:
        lines = f.readlines()

    for i in range(0, len(lines), lines_per_step):
        os.system("tput clear")  # Clears the terminal
        print("".join(lines[i:i + lines_per_step]))
        time.sleep(delay)

if __name__ == "__main__":
    # usage: python game_animation.py [file_path] [lines_per_step] [delay]
    args = parse_args()
    file_path = args.file_path
    lines_per_step = args.lines_per_step
    delay = args.delay
    play_text_animation(file_path, lines_per_step, delay)
