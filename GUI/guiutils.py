from tkinter import *
from tkinter import ttk

import sys, pathlib, os, shutil

# using pathlib to help with mac and linux compatibility.
current_dir = pathlib.Path(__file__).parent.resolve()
maindir = pathlib.Path(current_dir).parent.resolve()
datasets_folder = pathlib.Path(f"{maindir}/datasets").resolve()
agents_folder = pathlib.Path(f"{maindir}/model/data/agents").resolve()
tests_folder = pathlib.Path(f"{maindir}/model/data/results").resolve()


class ToolTip(object):
    '''
    A class to creat tooltips when hovering on any tkinter element.

    :param widget: widget that generates the tooltip.
    '''
    def __init__(self, widget:ttk.Widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0

    def showtip(self, text:str):
        """
        Creates the tooltip and displays given text.

        :param text: text to display
        """
        self.text = text
        if self.tipwindow or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 57
        y = y + cy + self.widget.winfo_rooty() +27
        self.tipwindow = tw = Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = Label(tw, text=self.text, justify=LEFT,
                      background="#3d86d4", relief=SOLID, borderwidth=1,
                      font=("tahoma", "10", "normal"), fg="white")
        label.pack(ipadx=1)

    def hidetip(self):
        "destroys the tooltip"
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()

class ExperimentBanner(object):
    '''
    A class to create Banners (tkinter labelframes) to add to the setup menu.
    They display information about which tests and experiments are queued to run.

    :param frame: the parent frame to attach the banner to.
    :param bannertext: the text to display on top of the labelframe.
    :param experiment_name: the name of the test or experiment to run.
    :param laps: the number of episodes/laps the experiment/training is goin to run for.
    :param dataset: the dataset which is being relied upon
    :param embeddings: the embeddings that are ging to be used.
    :param single_rel_check: flag for single relation
    :param single_rel_name: name of the relation to test/train.
    :param lapstext: special parameter to indicate if laps should be something else (I.E. \"episodes\")
    '''
    def __init__(self, frame:ttk.Frame, bannertext:str, experiment_name :str, laps : int, 
     dataset : str, embeddings:list, single_rel_check:bool, single_rel_name: str, lapstext = "laps"):
        parent = ttk.Labelframe(frame, text=bannertext)
        namelabel = ttk.Label(parent, text=f'name: {experiment_name}')
        datalabel = ttk.Label(parent, text=f'dataset: {dataset}')
        laplabel = ttk.Label(parent, text=f'{lapstext}: {laps}')
        embeddingslabel = ttk.Label(parent, text=f'embeddings:\n {embeddings}')

        namelabel.grid(row=0, column=0)
        datalabel.grid(row=1, column=0)
        laplabel.grid(row=2, column=0)
        embeddingslabel.grid(row=3, column=0)

        if(single_rel_check):
            relationlabel = ttk.Label(parent, text=f"relation: {single_rel_name}")
            relationlabel.grid(row=4, column=0)


        self.parent = parent

    def getbanner(self):
        '''
        see return

        :returns: the banner object attached to this class.
        '''
        return self.parent

class AgentInfo(object):
    '''
    A holder class for agent information

    :param name: the name of the test or experiment to run.
    :param embeddings: the embeddings that are ging to be used.
    :param dataset: the dataset which is being relied upon.
    :param is_single_rel: flag for single relation.
    :param single_rel_name: name of the relation to test/train.
    '''
    def __init__(self, name:str, embeddings:list, dataset:str, is_single_rel:bool, single_rel_name:str):
        self.name = name
        self.embeddings = embeddings
        self.dataset = dataset
        self.is_single = is_single_rel
        self.single_name = single_rel_name

    def get(self):
        """
        see return

        :returns: all the information that it contains in this order. -> name, embeddings, dataset, single_rel, single_rel_name
        """
        return self.name, self.embeddings, self.dataset, self.is_single, self.single_name

def CreateToolTip(widget:ttk.Widget, text:str):
    """
    Creates a tooltip for the selected widget with the specified text.

    :param widget: widget that generates the tooltip.
    :param text: text to display in the tooltip.
    """
    toolTip = ToolTip(widget)
    def enter(event):
        toolTip.showtip(text)
    def leave(event):
        toolTip.hidetip()
    widget.bind('<Enter>', enter)
    widget.bind('<Leave>', leave)

def GetConfig(is_experiments:bool):
    """
    Imports the configuration information and
    
    :param is_experiments: if we want to retrieve experiments or tests.

    :returns: the configuration dictionary.
    """
    sys.path.insert(0, f"{maindir}/model")
    from config import get_config
    sys.path.pop(0)
    return get_config(is_experiments, only_config = True)

def GetDatasets():
    """
    Gets all available dataset names.

    :returns: all available datasets.
    """
    res = []
    for name in os.listdir(datasets_folder):
        dirpath = pathlib.Path(f"{datasets_folder}/{name}").resolve()
        if os.path.isdir(dirpath):
            res.append(name)

    return res

def GetAgents():
    """
    Gets all agents that have been created and the config used for them, then encapsulates them in a AgentInfo class

    :returns: a list of AgentInfo which contains all generated agents and information about them.
    """
    res = []
    agent_list = os.listdir(agents_folder)
    agent_list.remove('.gitkeep')
    agent_list.remove('TRAINED')
    
    for a in agent_list:
        embeddings = []
        name = a
        dataset = ""
        single_rel_pair = []

        p = f"{agents_folder}/{a}"

        with open(f"{p}/config_used.txt") as c:
            for ln in c:
                if ln.startswith("dataset: "):
                    dataset = ln.lstrip('dataset: ').strip()

                if ln.startswith("single_relation_pair: "):
                    aux = ln.lstrip('single_relation_pair: ')
                    aux = aux.replace("[", "").replace("]","").replace(" ", "").replace("\'", "").strip().split(",")
                    single_rel_pair = [aux[0]=="True", None if aux[1] == "None" else aux[1]]

                if ln.startswith("embeddings: "):
                    aux = ln.lstrip('embeddings: ')
                    embeddings = aux.replace("[", "").replace("]","").replace(" ", "").replace("\'", "").strip().split(",")

        
        # print("\n",embeddings, name, dataset, single_rel_pair, "\n")

        res.append(AgentInfo(name, embeddings, dataset, single_rel_pair[0], single_rel_pair[1]))
    
    return res

def GetTestsPaths():
    """
    Gets all generated tests and their paths and returns them in a dict format.

    :returns: a list of dict objects containing the tests
    """
    res = []

    test_list = os.listdir(tests_folder)
    test_list.remove(".gitkeep")

    for test_name in test_list:
        test_dict = dict()
        test_dict["name"] = test_name
        test_dict["pathdicts"] = []

        p = f"{tests_folder}/{test_name}"

        with open(f"{p}/paths.txt") as pathfile:
            test_dict["dataset"], test_dict["agent_name"] = [x.strip() for x in pathfile.readline().split(",")]
            for ln in pathfile.readlines():
                path_dict = dict()
                pathpair = eval(ln)

                fullpath = pathpair[0]
                target = pathpair[1]

                path_dict["target"] = target
                path = []

                for triple in fullpath:
                    path.append(triple)
                    if(triple[2] == target):
                        break
            
                path_dict["path"] = path

                test_dict["pathdicts"].append(path_dict)

        res.append(test_dict)
        
    return res

def GetExperimentInstance(name:str, dataset:str, embeddings:list, laps:int, single_rel:bool, single_rel_name:str):
    """
    Create an experiment class object with the given information

    :param name: the experiment name
    :param dataset: the dataset which is being relied upon.
    :param embeddings: the embeddings that are ging to be used.
    :param laps: the number of laps to perform
    :param single_rel: flag for single relation.
    :param single_rel_name: name of the relation to test/train.

    :returns: the created Experiment object instance.
    """
    sys.path.insert(0, f"{maindir}/model")
    from config import Experiment
    sys.path.pop(0)

    return Experiment(name, dataset, embeddings, laps, single_rel, relation = single_rel_name)

def GetTestInstance(agentname:str, testname:str, embeddings:list, episodes:int):
    """
    Create a Test class object with the given information

    :param agentname: the name of the agent to test.
    :param testname: the test name.
    :param embeddings: the embeddings that are ging to be used.
    :param episodes: the number of episodes to test for.

    :returns: the created Test object instance.
    """
    sys.path.insert(0, f"{maindir}/model")
    from config import Test
    sys.path.pop(0)
    
    return Test(testname, agentname, embeddings, episodes)

def CheckForRelationInDataset(dataset_name:str, relation_name:str):
    """
    checks for a particular relation in a dataset

    :param dataset_name: the name of the dataset 
    :param relation_name: the name of the relation

    :returns: True if the relation is in the dataset, False otherwise.
    """

    relation_in_graph = False
    filepath = pathlib.Path(f"{datasets_folder}/{dataset_name}/graph.txt").resolve()
    with open(filepath) as d:
        for l in d.readlines():
            if(l.split("\t")[1] == relation_name):
                relation_in_graph = True
                break
    
    return relation_in_graph

def CheckAgentNameColision(name:str):
    """
    checks if an agent with the requested name does already exist.

    :param name: the name of the agent.

    :returns: True if the agent exists, False otherwise.
    """
    subfolders = [ f.name for f in os.scandir(agents_folder) if f.is_dir()]
    subfolders.remove("TRAINED")
    return name in subfolders

def CheckTestCollision(name):
    """
    checks if a test with the requested name does already exist.

    :param name: the name of the test.

    :returns: True if the test exists, False otherwise.
    """
    subfolders = [ f.name for f in os.scandir(tests_folder) if f.is_dir()]
    return name in subfolders

def run_integrity_checks():
    '''
    checks for consistency in folders, in case a test/train suite has faield to complete.
    '''
    print("running integrity checks")
    subfolders = [f.name for f in os.scandir(datasets_folder) if f.is_dir()]
    for s in subfolders:
        try:
            embs_path = f"{datasets_folder}/{s}/embeddings"
            embs_dir = [f.name for f in os.scandir(embs_path) if f.is_dir()]
            for e in embs_dir:
                emb_dir = f"{embs_path}/{e}"
                remove_folders(emb_dir, 0) # removes empty folders
        except:
            print(f"No embeddings have been generated for {s}")

    subfolders = [f.name for f in os.scandir(agents_folder) if f.is_dir()]
    for s in subfolders:
        agent_dir = f"{agents_folder}/{s}"
        remove_folders(agent_dir, 1) # removes folders with only config_used.txt
    
def remove_folders(path_abs:str, filecount:int):
    '''
    deletes a folder with the indicated filecount.
    '''
    files = os.listdir(path_abs)
    if len(files) == filecount:
        print(f"removing path {path_abs}")
        shutil.rmtree(path_abs)
