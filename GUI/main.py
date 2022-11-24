from tkinter import ttk, messagebox
from tkinter import *
from guiutils import GetConfig

import time, random, os, sys, pathlib, subprocess, config_menu, test_train_menu, view_paths_menu, threading

current_dir = pathlib.Path(__file__).parent.resolve()
maindir = pathlib.Path(current_dir).parent.resolve()
datasets_folder = f"{maindir}/datasets"
agents_folder = f"{maindir}/model/data/agents"

sys.path.insert(0, f"{maindir}/model")
from trainer import Trainer, TrainerGUIconnector, main as tr_main, get_gui_values as tr_get_gui
from tester import Tester, TesterGUIconnector, main as tst_main, get_gui_values as tst_get_gui
sys.path.pop(0)

class mainmenu(object):
    def __init__(self):
        # functionality
        self.config = GetConfig(True)

        self.experiments, self.experiment_banners = [], []
        self.tests, self.test_banners = [], []
        self.config_is_open, self.setup_is_open, self.paths_is_open = False, False, False

        # multithread support:
        Tcl().eval('set tcl_platform(threaded)')

        # flags:
        self.is_running = False
        self.running_exp = False
        self.running_tests = False
        
        self.root = Tk()
        self.root.title("Model Generator")
        self.root.resizable(FALSE, FALSE)
        self.root.option_add('*tearOff', FALSE)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # this intercepts the closing button before doing so and can, for example
        # prompt the user to save its changes before proceeding.
        self.root.protocol("WM_DELETE_WINDOW", self.intercept_close)
        # returns x11, win32 or aqua
        self.OSNAME = self.root.tk.call('tk', 'windowingsystem')

        self.mainframe = ttk.Frame(self.root, padding="12 12 12 12")
        self.mainframe.grid(column=0, row=0)

        self.add_styles()
        self.add_elements()

        self.update_connectors()

        self.root.mainloop()

    def update_connectors(self):
        self.tr_conn = TrainerGUIconnector(self.config, self.experiments)
        self.tst_conn = TesterGUIconnector(self.config, self.tests)

    # MAIN ELEMENTS

    def add_elements(self):
        # infotext
        self.infotext = ttk.Label(self.mainframe, text="0 Experiment(s) Loaded, 0 Test(s) Loaded")

        # config block
        self.config_lf = ttk.Labelframe(self.mainframe, text='Configuration')
        self.config_button = ttk.Button(self.config_lf, text="Config", 
        command = lambda: self.open_menu("config"))

        self.setup_button = ttk.Button(self.config_lf, text="Setup", 
        command = lambda: self.open_menu("setup"))

        # runner block
        self.run_lf = ttk.Labelframe(self.mainframe, text='Runner')
        self.train_button = ttk.Button(self.run_lf, text="Train", 
        command = lambda: self.run_experimentation())

        self.test_button = ttk.Button(self.run_lf, text="Test", 
        command = lambda: self.run_tests())

        # progressbar + text
        self.progress_text = ttk.Label(self.mainframe, text="")
        self.progress_bar = ttk.Progressbar(self.mainframe, orient= HORIZONTAL, 
        mode='determinate', length=280, maximum=100)

        # folder buttons

        self.folder_lf = ttk.Labelframe(self.mainframe, text='Folders')
        
        self.datasets_button = ttk.Button(self.folder_lf, text='Datasets', 
        command= lambda: self.open_folder("datasets"))

        self.agents_button = ttk.Button(self.folder_lf, text='Agents', 
        command= lambda: self.open_folder("agents"))

        # view paths

        self.view_paths_button = ttk.Button(self.mainframe, text='View Paths', 
        command= lambda: self.open_menu("paths"))

        # self.view_paths_button = ttk.Button(self.mainframe, text='View Paths', 
        # command= lambda: self.update_all_progress(random.randint(0,99), 100, "text"))

        #error text.
        self.error_text = Label(self.mainframe, text="", fg='red', bg="#33393b")

        self.grid_elements()

    def grid_elements(self):
        #row0
        self.infotext.grid(row=0, column=0, columnspan=2)

        #row1
        self.config_lf.grid(row=1, column=0)
        self.config_button.grid(row=0, column=0)
        self.setup_button.grid(row=1, column=0)
        
        for child in self.config_lf.winfo_children():
            child.grid_configure(padx=15, pady=5)

        self.run_lf.grid(row=1, column=1)
        self.train_button.grid(row=0, column=0)
        self.test_button.grid(row=1, column=0)

        for child in self.run_lf.winfo_children():
            child.grid_configure(padx=15, pady=5)

        #row2
        self.progress_text.grid(row=2, column=0, columnspan=2, pady=3)
        
        #row3
        self.progress_bar.grid(row=3, column=0, columnspan=2)

        #row4
        self.folder_lf.grid(row=4, column=0, columnspan=2)
        self.datasets_button.grid(row=0, column=0, padx=(75,0), pady=(0,5))
        self.agents_button.grid(row=0, column=1, padx=(0,75), pady=(0,5))

        #row5
        self.view_paths_button.grid(row=5, column=0, columnspan=2, pady=3)

        #row6
        self.error_text.grid(row=6, column=0, columnspan=2, pady=3)

    def add_styles(self):
        s = ttk.Style()
        self.root.tk.call('lappend', 'auto_path', f"{current_dir}/awdark/")
        self.root.tk.call('package', 'require', 'awdark')
        s.theme_use('awdark')

    # SUBMENU HANDLING
    def open_menu(self, menutype):
        if(menutype == "config" and not self.config_is_open):
            self.config_is_open = True
            c_menu = config_menu.menu(self.root, self.config)
            c_menu.root.wm_protocol("WM_DELETE_WINDOW", lambda: self.extract_config_on_close(c_menu))

        elif(menutype == "setup" and not self.setup_is_open):
            self.setup_is_open = True
            setup = test_train_menu.menu(self.root, self.experiments, self.tests)
            setup.root.wm_protocol("WM_DELETE_WINDOW", lambda: self.extract_info_on_close(setup))

        elif(menutype == "paths" and not self.paths_is_open):
            self.paths_is_open = True
            paths_menu = view_paths_menu.menu(self.root, self.tests)
            setup.root.wm_protocol("WM_DELETE_WINDOW", lambda: self.extract_info_on_close(paths_menu))

    def extract_config_on_close(self, config_menu):
        savebefore = messagebox.askyesnocancel(
            message='Save changes before closing window?',
            icon='warning', title='Closeing warning')

        if(savebefore is not None):#if user canceled we do nothing
            if(savebefore):
                config_menu.save_config()

            self.config = config_menu.config

            config_menu.root.destroy()
            self.config_is_open = False

            self.update_connectors()

    def extract_info_on_close(self, setup_window):
        self.experiments, self.tests = setup_window.experiments, setup_window.tests
        self.infotext["text"] = f"{len(self.experiments)} Experiment(s) Loaded, {len(self.tests)} Test(s) Loaded"
        setup_window.root.destroy()
        self.setup_is_open = False

        self.update_connectors()

    def pathmenu_teardown(self, pathmenu):
        self.paths_is_open = False
        pathmenu.root.destroy()

    # MISC OPERATIONS
    def open_folder(self, folder:str):
        folder_to_open = datasets_folder if folder == "datasets" else agents_folder

        # x11, win32 or aqua
        if(self.OSNAME == 'x11'): #linux
            try:
                subprocess.run(['xdg-open', os.path.realpath(folder_to_open)])
            except:
                # WSL support
                result = subprocess.run(["wslpath", "-w", folder_to_open], text=True, capture_output=True)
                windows_path = result.stdout
                subprocess.run(["explorer.exe", windows_path])

        elif(self.OSNAME == 'win32'): #windows
            subprocess.run(['explorer', os.path.realpath(folder_to_open)])

        elif(self.OSNAME == 'aqua'): #mac
            subprocess.run(['open', os.path.realpath(folder_to_open)])

        else:
            self.change_error_text("This operation is not supported for this Operatig System.")
        
    def intercept_close(self):
        if self.is_running:
            print("asking the user for permission to close.")
            canclose = messagebox.askyesno(
            message='There is something running, do you want to close anyway?',
            icon='warning', title='Interruption alert.')

            if(canclose):
                self.stop_updater()
                self.kill_all_threads()
                self.root.destroy()
        else:
            self.root.destroy()

    def run_experimentation(self):
        if(not self.is_running):
            self.is_running, self.running_exp = True, True
            self.launch_updater(True)
        else:
            self.showbusyerror()

    def run_tests(self):
        if(not self.is_running):
            self.is_running, self.running_tests = True, True
            self.launch_updater(False)
        else:
            self.showbusyerror()

    def launch_updater(self, is_train):
        self.updater_thread = threading.Thread(name="updaterThread",
        target=lambda: self.update_progress(is_train))
        self.updater_thread.start()

    def kill_all_threads(self):
        mainthread = None
        for thread in threading.enumerate(): 
            if(thread.name != "MainThread"):
                try:
                    thread.daemon = True
                except:pass
            else:
                mainthread = thread
        
        mainthread._stop()


    def stop_updater(self):
        self.change_progtext("Execution is finished.")
        

    def update_progress(self, is_train):
        self.runner_thread = threading.Thread(name="runnerThread",
        target=lambda: self.mainthread(is_train))
        self.runner_thread.start()

        r = True
        while(r):
            try:
                r = not self.tr_conn.active_trainer.is_ready
                print(f"checking if the trainer is ready {r}")
            except:pass
            time.sleep(0.5)

        r = True
        while(r):
            if(self.running_exp):
                tot_it, curr_it, tot_it_step, curr_it_step, curr_prog = tr_get_gui()
            elif(self.running_tests):
                tot_it, curr_it, tot_it_step, curr_it_step, curr_prog = tst_get_gui()

            if(self.is_running):
                t = f"({curr_it}/{tot_it})-({curr_it_step}/{tot_it_step})-{curr_prog}"
                self.update_all_progress(curr_it_step, tot_it_step, t)
         
            if(curr_it == tot_it and curr_it_step == tot_it_step):
                r = False
                self.is_running, self.running_exp, self.running_tests = False, False, False
            
            time.sleep(1)
        
        self.stop_updater()
    
    def mainthread(self, is_train):
        if(is_train):
            tr_main(False, self.tr_conn)
        else:
            tst_main(False, self.tst_conn)

    # errors
    def showbusyerror(self):
        messagebox.showerror(title="Busy Error", message="Something is already running!\n\
Wait for it to finish or abort the execution.")

    def change_error_text(self, newtext:str):
        self.error_text['text'] = newtext

    # progress bar
    def change_progtext(self, newtext:str):
        self.progress_text['text'] = newtext
    
    def set_progbar_value(self, v:int):
        self.progress_bar['value'] = v

    def change_progbar_max(self, qtty:int):
        self.progress_bar['maximum'] = qtty

    def update_all_progress(self, curr, tot, text):
        self.progress_text['text'] = text
        self.progress_bar['value'] = curr
        self.progress_bar['maximum'] = tot

mainmenu()