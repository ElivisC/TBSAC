from concurrent.futures.process import ProcessPoolExecutor

import numpy as np
from epics import PV
import time
from engines.engine import BaseEngine


class AcceleratorEngine(BaseEngine):
    def __init__(self):
        super(AcceleratorEngine, self).__init__("accengine")

        '''
        TODO : You need to determine through measurements from the accelerator
         if there are any installation errors in the corrector magnets, 
        such as reverse installation along the X and Y directions. 
        Based on this information, determine the order of the PV values.
        '''

        self.input_setting = {
            'HCM1_PS:DCH_01:ISet':0,
            'HCM1_PS:DCV_01:ISet':0,
            'HCM1_PS:DCH_02:ISet':0,
            'HCM1_PS:DCV_02:ISet':0,
            'HCM1_PS:DCH_03:ISet':0,
            'HCM1_PS:DCV_03:ISet':0,
            'HCM1_PS:DCH_04:ISet':0,
            'HCM1_PS:DCV_04:ISet':0,
            'HCM1_PS:DCH_05:ISet':0,
            'HCM1_PS:DCV_05:ISet':0,
            'HCM1_PS:DCH_06:ISet':0,
            'HCM1_PS:DCV_06:ISet':0,
            'HCM2_PS:DCH_01:ISet':0,
            'HCM2_PS:DCV_01:ISet':0,
            'HCM2_PS:DCH_02:ISet':0,
            'HCM2_PS:DCV_02:ISet':0,
            'HCM2_PS:DCH_03:ISet':0,
            'HCM2_PS:DCV_03:ISet':0,
            'HCM2_PS:DCH_05:ISet':0,
            'HCM2_PS:DCV_05:ISet':0,
        }   #输入的设置值

        self.input_read = {

            'HCM1_PS:DCH_01:IMon':0,
            'HCM1_PS:DCV_01:IMon':0,
            'HCM1_PS:DCH_02:IMon':0,
            'HCM1_PS:DCV_02:IMon':0,
            'HCM1_PS:DCH_03:IMon':0,
            'HCM1_PS:DCV_03:IMon':0,
            'HCM1_PS:DCH_04:IMon':0,
            'HCM1_PS:DCV_04:IMon':0,
            'HCM1_PS:DCH_05:IMon':0,
            'HCM1_PS:DCV_05:IMon':0,
            'HCM1_PS:DCH_06:IMon':0,
            'HCM1_PS:DCV_06:IMon':0,
            'HCM2_PS:DCH_01:IMon':0,
            'HCM2_PS:DCV_01:IMon':0,
            'HCM2_PS:DCH_02:IMon':0,
            'HCM2_PS:DCV_02:IMon':0,
            'HCM2_PS:DCH_03:IMon':0,
            'HCM2_PS:DCV_03:IMon':0,
            'HCM2_PS:DCH_05:IMon':0,
            'HCM2_PS:DCV_05:IMon':0

        }   #输入的回读值

        self.input_read_backup = self.input_read.copy()



        self.output = {
            'Bpm:6-X11': 0,
            'Bpm:6-Y11': 0,
            'Bpm:7-X11': 0,
            'Bpm:7-Y11': 0,
            'Bpm:8-X11': 0,
            'Bpm:8-Y11': 0,
            'Bpm:9-X11': 0,
            'Bpm:9-Y11': 0,
            'Bpm:10-X11': 0,
            'Bpm:10-Y11': 0,
            'Bpm:11-X11': 0,
            'Bpm:11-Y11': 0,
            'Bpm:12-X11': 0,
            'Bpm:12-Y11': 0,
            'Bpm:13-X11': 0,
            'Bpm:13-Y11': 0,
            'Bpm:14-X11': 0,
            'Bpm:14-Y11': 0,
        }

        self.settings_pv = []
        self.input_setting_pv_dict = {}
        self.input_read_pv_dict = {}
        self.output_pv_dict = {}
        self.timeout = 10
        '''
        TODO: you should confirm the polarity of the corrector magnets. 
        And 1.85 is the  is the coefficient between the current value in 
        the simulation environment and the current value in the real 
        environment.
        '''
        self.max_current = np.array([1,-1,-1,-1,-1,1,1,-1,-1,1,1,1])*1.85

        self.num_for_avg = 1      #取多少组bpm的读数求平均
        
        # self.processing_pool = ProcessPoolExecutor(10)

        for key in self.input_setting.keys():
            self.input_setting_pv_dict[key] = PV(key)

        for key in self.input_read.keys():
            self.input_read_pv_dict[key] = PV(key)

        for key in self.output.keys():
            self.output_pv_dict[key] = PV(key)


    def save_init_param(self):
        for key in self.input_read_backup:
            self.input_read_backup[key] =  self.input_read_pv_dict[key].get()
        print(f"init param backup == {self.input_read_backup}")

    def recover_init_param(self):
        for setting_key, read_key in zip(self.input_setting, self.input_read_backup):
        # for key in self.input_read_backup:
            print(f"setting_key = {setting_key},read_key = {read_key},value = {self.input_read_backup[setting_key]}")
            #self.input_setting[setting_key].put(self.input_read_backup[read_key])


    def update_engine_input(self, input):
        assert len(input) == len(self.input_setting.keys())
        assert len(input) == len(self.max_current)

        for index,(value,key) in enumerate(zip(input,self.input_setting)):
            if isinstance(key, float):
                self.input_setting[key] = value * self.max_current[index]
            else:
                print(f"Invild Value {value} for PV key {key}")
                exit(0)

    def execute_engine(self,debug = False):
        for pv_key, setting_key in zip(self.input_setting_pv_dict, self.input_setting):
            print(f"pv_key = {pv_key},setting_key = {setting_key},value = {self.input_setting[setting_key]}")
            self.input_setting_pv_dict[pv_key].put(self.input_setting[setting_key], use_complete=True)
        waiting = True
        total_time = 0
        while waiting:
            time.sleep(0.01)
            total_time += 0.01
            waiting = not all([self.input_setting_pv_dict[pv_key].put_complete for pv_key in self.input_setting_pv_dict])
            if total_time > self.timeout:   #等待时间查过timeout的设置则返回设置失败
                print("=======================PV group set Failed!!!=======================")
                return False
        print("=======================PV group set complete!!!=======================")
        return True


    def get_output(self):
        total_time = 0
        while True:
            all_bool = []
            for key in self.input_read.keys():
                print(f"input read key = {key}")
                self.input_read[key] = self.input_read_pv_dict[key].get()
            for setting_key,read_key in zip(self.input_setting,self.input_read):
                print(f"setting_key = {self.input_setting[setting_key]},read_key = {self.input_read[read_key]}")
                diff = abs(self.input_setting[setting_key] - self.input_read[read_key])
                diff = round(diff,3)
                if diff <= 0.1:
                    all_bool.append(True)
                else:
                    all_bool.append(False)
            print(f"all_bool = {all_bool}")
            print(sum(all_bool) == len(all_bool))
            if sum(all_bool) == len(all_bool):
                tmp_array = np.zeros((len(self.output_pv_dict.keys()),self.num_for_avg))
                for col in range(self.num_for_avg):            #按照采样频率采集self.num_for_avg次BPM数据
                    for row,key in enumerate(self.output):
                        tmp_array[row][col] = self.output_pv_dict[key].get()

                mean_result = np.mean(tmp_array,axis=1)
                print(f"mean_result shape = {mean_result.shape}")
                print(f"mean_result  = {mean_result}")
                return mean_result.tolist()
            else:
                time.sleep(1)
                total_time += 1
                if total_time >= self.timeout:
                    print("read time out !!!")
                    return None



