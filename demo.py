from pynput.keyboard import Controller
import time

keyboard = Controller()
loop_flag = False

def add_command(key):
    if key:
        global loop_flag  # use a global flag variable to control the loop execution
        loop_flag = True  # set the flag to True to start the loop execution
        while loop_flag:  # use the flag variable for loop execution
            keyboard.press(key)
            keyboard.release(key)
            time.sleep(0.1)


def stop_loop():
    global loop_flag
    loop_flag = False


start = time.time()
if ((time.time() - start) > 1):
    stop_loop()
    print("Time out!")
