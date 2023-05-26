import subprocess

from utils.consts import Consts
from engines.engine import BaseEngine
import platform
import os
from utils.latticehelper import LatticeHelper
from utils.particle import Particle
from utils.tracewinout_util import TraceWinOutUtil

class TraceWinEngine(BaseEngine):
    def __init__(self,
                 particle:Particle,
                 bpm_position:list,
                 tracewin_path,
                 ini_name,
                 parent_template_name,
                 son_template_name,
                 final_lattice_name,
                 tracewin_out_path="temp",
                 error_mode=False,
                 error_element=[],
                 error_rate=0.01,
                 max_error = 0):
        '''

        :param particle:
        :param bpm_position:
        :param tracewin_path:
        :param ini_name:
        :param parent_template_name:
        :param son_template_name:
        :param final_lattice_name:
        :param tracewin_out_path:
        :param error_mode:
        :param error_element:
        :param error_rate:
        :param max_error:
        '''

        super(TraceWinEngine, self).__init__("tracewinengine")
        self.bpm_position = bpm_position
        self.parent_template_name = parent_template_name
        self.son_template_name = son_template_name

        self.particle = particle

        #用来生成lattice的模板以及lattice文件
        self.lattice_helper = LatticeHelper(lattice_template = f"{tracewin_path}/{self.parent_template_name}",
                                            lattice_error_template = f"{tracewin_path}/Mn_error_template.dat",
                                            lattice_template_out = f"{tracewin_path}/{self.son_template_name}",
                                            max_error = max_error)

        self.tracewin_path = tracewin_path
        self.ini_name = ini_name

        self.final_lattice_name = final_lattice_name
        self.max_error = max_error
        self.error_mode = error_mode
        self.error_element = error_element
        self.error_rate = error_rate

        self.tracewin_out_path = os.path.join(self.tracewin_path,tracewin_out_path,"tracewin.out")
        # self.tracewinout_helper = TraceWinOutUtil(self.tracewin_out_path)
        if not error_mode:

            self.lattice_helper.generate_lattice_template(self.particle.lattice)    #40Ar 12+
        else:

            self.lattice_helper.generate_lattice_template(self.particle.lattice,  # 40Ar 12+
                error_mode=error_mode,
                error_element=error_element,
                error_rate=error_rate
            )
        os.chdir(self.tracewin_path)

        if platform.system() == 'Windows':
            self.tracewin_name = "TraceWin_noGUI.exe"
        elif platform.system() == 'Linux':
            self.tracewin_name = "./TraceWin_noX11"



    def update_engine_input(self,input):
        # print(f"input = {input}")
        self.input = input
        self.lattice_helper.generate_lattice_file({Consts.D_MAGENET_X: input[::2],Consts.D_MAGENET_Y:input[1::2]},
                                                  lattice_file = os.path.join(self.tracewin_path,self.final_lattice_name))

    def reset_error(self):
        # print(f"max error = {self.max_error}")
        if not self.error_mode:

            self.lattice_helper.generate_lattice_template(self.particle.lattice)

        else:
            self.lattice_helper.generate_lattice_template(self.particle.lattice,
                                                            error_mode=self.error_mode,
                                                            error_element=self.error_element,
                                                            error_rate=self.error_rate
                                                          )

    def execute_engine(self,debug=False):
        if debug:
            cmd = f"echo y|{self.tracewin_name} {self.ini_name} mass1={self.particle.mass} charge1={self.particle.charge} energy1={self.particle.energy}"
        else:
            cmd = f"echo y|{self.tracewin_name} {self.ini_name} mass1={self.particle.mass} charge1={self.particle.charge} energy1={self.particle.energy} > runlog"
        p = subprocess.Popen(cmd, shell=True)
        return_code = p.wait()
        return return_code


    def get_output(self,direction):
        bpm_value = TraceWinOutUtil.read_bpm_from_tracewinout(self.tracewin_out_path,self.bpm_position)
        return bpm_value













