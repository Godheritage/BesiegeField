import subprocess
import os
import socket
import time
import struct
import signal
import threading
from .besiege_codebook import *
from AgenticCodes.config import SCRIPT_PATH
class BesiegeEnv: 
    def __init__(self,script_path=SCRIPT_PATH,
                 ip="127.0.0.1",
                 port = 6890,
                 no_graphic = True,
                 run_mode = "deafult",
                 use_xvfb=False):
        """
        run_mode: 
            deafult:enter game
            train_mode    
        """

        
        self.receive_queue = []
        self.tcp_plugin_loaded = False

        self.script_path = script_path
        self.ip = ip
        self.port = port
        self.no_graphic = no_graphic
        self.run_mode = run_mode
        self.use_xvfb=use_xvfb

        game_parameters = [
            "--ip", str(self.ip),
            "--port", str(self.port),
        ]

        self.init_reactpoint = []

        if self.no_graphic==True:
            game_parameters.append("-batchmode")
            game_parameters.append("-nographics")
            game_parameters.append("-offline")
        if self.run_mode=="deafult":
            game_parameters.append("--trainmode")
            game_parameters.append("false")
        elif self.run_mode=="trainmode":
            game_parameters.append("--trainmode")
            game_parameters.append("true")
        elif self.run_mode=="costumelevelmode":
            # ToDo
            game_parameters.append("--trainmode")
            game_parameters.append("false")
            #receive key, do value
            self.init_reactpoint.append({"GameLaunched":"Wait 2"})
            self.init_reactpoint.append({"MainMenuLoaded":"Load_Level:MasterSceneMultiplayer"})
            # self.init_reactpoint.append({"BuildingSceneLoaded":f"LoadCustomLevel:{level_full_path}"})
            pass


        
        working_directory = os.path.dirname(self.script_path)
        self.command = [self.script_path] + game_parameters
        self.cwd = working_directory

        # start env

        self.run_thread = threading.Thread(target=self.run)
        self.run_thread.start()

        # launch TCP
        self.tcp_sender()

        # print("Success ")

        #check react point
        response_message_list = []
        while self.init_reactpoint!=[]:
            while self.receive_queue!=[]:
                received_message = self.receive_queue[0]
                if received_message in self.init_reactpoint[0]:
                    received_message = self.receive_queue.pop(0)
                    react_point = self.init_reactpoint.pop(0)
                    response_message = react_point[received_message]
                    response_message_list.append(response_message)
                    break
        for response in response_message_list:
            self.send_message(response)

        while not self.receive_queue:
            time.sleep(0.1)  # Wait receive_queue

        print(f"PORT:{port} Initialization complete and received final message.")
    
    def run(self):
        """Run sub process"""
        if self.use_xvfb:
            env = os.environ.copy()
            env["DISPLAY"] = ":99"
        else:
            env=None
        self.process = subprocess.Popen(self.command, 
                                         cwd=self.cwd, 
                                         stdout=subprocess.PIPE, 
                                         stderr=subprocess.STDOUT, 
                                         text=True,
                                         env=env)
        try:
            # Read Game Output
            for line in self.process.stdout:
                
                # Filter log
                if "BesiegeAgent" in line:  
                    # print(line.strip())
                    self.process_receive(line.strip())

                # print(line.strip())
                
            self.process.wait()
        except KeyboardInterrupt:
            print("Main process interupted, killing game...")
            self.process.terminate()
            self.process.wait()

    def process_receive(self,besiege_stdout):
        
        if "SendMessage:" in besiege_stdout:
            received_code = besiege_stdout.split("SendMessage:")[1].strip()
            # received_message = code_message_map[received_code]  
            received_message = received_code
            self.receive_queue.append(received_message)

        if "[Info   :BesiegeAgent] TCP initalized" in besiege_stdout:
            self.tcp_plugin_loaded = True
            
    
    def tcp_sender(self):
        """Set TCP Connection"""
        while not self.tcp_plugin_loaded:
            time.sleep(0.1)  
        try:
            self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client.connect((self.ip, int(self.port)))
            self.connection_established = True  
        except Exception as e:
            self.connection_established = False

    def send_message(self, command):
        """Send Message"""
        if not self.connection_established:
            return

        try:
            if command=="None":
                
                return
            if "Wait" in command:
                sleep_sec = int(command.split(" ")[1])
                time.sleep(sleep_sec)
                return
            message = command.split(":", 1)
            if len(message) < 2 or message[0] == "" or message[1]=="":
                print("Illegal order, cancel send")
                return
            
            send_header_message = message[0]+":"
            send_tail_message = message[1]
            if message_code_head_map[send_header_message] in ["0","1"]:
                send_code = message_code_head_map[send_header_message]+" "+message_code_tail_map[send_tail_message]
            else:
                send_code = message_code_head_map[send_header_message]+" "+send_tail_message

            message_bytes = send_code.encode('utf-8')
            length = len(message_bytes)
            header = struct.pack('<I', length)
            self.client.send(header + message_bytes)
        except Exception as e:
            pass

    def get_receivequeue(self):
        return self.receive_queue
    def clear_receivequeue(self):
        self.receive_queue=[]
    
    def send_message_waiting_receive(self,command,expected_receive):
        self.send_message(command)
        # print(self.receive_queue)
        while expected_receive not in self.receive_queue:
            # print(self.receive_queue)
            time.sleep(0.1)  
    
    def waiting_receive(self,expected_receive):
        while expected_receive not in self.receive_queue:
            time.sleep(0.1)  
    
    def waiting_stop(self,expected_receive,fail_time = 600):
        init_timer = 0
        sent_fail = False
        while expected_receive not in self.receive_queue:
            time.sleep(0.1)  
            init_timer+=0.1
            if init_timer>=fail_time and (not sent_fail):
                self.send_message("TriggerFail:_")
                sent_fail = True

    def kill_process(self):
        if hasattr(self, 'process') and self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=3)
            except:
                self.process.kill()
                self.process.wait()
            self.process = None

def signal_handler(signum, frame):
    print("Exiting...")
    exit(0)

if __name__ == "__main__":
    pass
    